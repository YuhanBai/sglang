# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat 2.5 media processing using the checkpoint's marker tokens."""

import asyncio
import math
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers.image_transforms import to_pil_image

from sglang.srt.managers.req_types import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.longcat_25 import Longcat25ForConditionalGeneration
from sglang.srt.multimodal.media_io import load_image, load_video
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

MAX_IMAGE_ASPECT_RATIO = 200
Frame = Image.Image | np.ndarray | torch.Tensor


def _resize_image(height: int, width: int, config: dict) -> tuple[int, int]:
    factor = config["patch_size"] * config["spatial_merge_size"]
    if max(height, width) > MAX_IMAGE_ASPECT_RATIO * min(height, width):
        raise ValueError(
            f"LongCat image aspect ratio must be at most {MAX_IMAGE_ASPECT_RATIO}."
        )
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > config["max_pixels"]:
        scale = math.sqrt(height * width / config["max_pixels"])
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < config["min_pixels"]:
        scale = math.sqrt(config["min_pixels"] / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    return h, w


def _media_patches(
    frames: Sequence[Frame], config: dict
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    images = [
        frame if isinstance(frame, Image.Image) else to_pil_image(frame)
        for frame in frames
    ]
    h, w = _resize_image(images[0].height, images[0].width, config)
    patch, merge, temporal = (
        config[key]
        for key in ("patch_size", "spatial_merge_size", "temporal_patch_size")
    )
    mean = np.asarray(config["image_mean"])[:, None, None]
    std = np.asarray(config["image_std"])[:, None, None]
    pixels = []
    for image in images:
        resized = image.convert("RGB").resize((w, h), resample=Image.Resampling.BICUBIC)
        pixels.append((np.asarray(resized).transpose(2, 0, 1) / 255.0 - mean) / std)
    pixels.extend([pixels[-1]] * (-len(pixels) % temporal))
    grid_t = len(pixels) // temporal
    grid_h, grid_w = h // patch, w // patch
    grouped = np.stack(pixels).reshape(
        grid_t,
        temporal,
        3,
        grid_h // merge,
        merge,
        patch,
        grid_w // merge,
        merge,
        patch,
    )
    patches = grouped.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
        grid_t * grid_h * grid_w, -1
    )
    return torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32)), (
        grid_t,
        grid_h,
        grid_w,
    )


def _decode_video(data, config):
    video = load_video(data, use_gpu=False)
    try:
        fps = float(video.avg_fps) if isinstance(video, VideoDecoderWrapper) else 1.0
        if not len(video) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                "LongCat 2.5 video requires at least one frame and a positive FPS"
            )
        # Raw frame sequences are treated as already sampled at 1 FPS.
        stride = max(1.0, fps)
        count = math.floor((len(video) - 1) / stride) + 1
        samples = range(count)
        limit = config["max_frame_num"]
        if count > limit:
            samples = np.linspace(0, count - 1, limit, dtype=int).tolist()
        indices = [round(index * stride) for index in samples]
        frames = (
            video.get_frames_at(indices)
            if isinstance(video, VideoDecoderWrapper)
            else [video[index] for index in indices]
        )
        patches, grid = _media_patches(frames, config)
        seconds = [round(index / fps) for index in indices]
        return patches, grid, seconds[:: config["temporal_patch_size"]]
    finally:
        if isinstance(video, VideoDecoderWrapper):
            video.close()


class LongCat25Processor(BaseMultimodalProcessor):
    models = [Longcat25ForConditionalGeneration]
    prefer_tokenized_input = True
    preserve_processor_input_ids = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.vision = hf_config.visual_config
        self.video = getattr(hf_config, "video_config", None) or {}

    async def process_mm_data_async(
        self, image_data, audio_data, input_text, request_obj, **kwargs
    ):
        if audio_data:
            raise ValueError("LongCat 2.5 supports image and video inputs")
        images = image_data or []
        videos = getattr(request_obj, "video_data", None) or []
        if videos and not self.video.get("enable"):
            raise ValueError("This LongCat 2.5 checkpoint does not enable video")
        if any(self._is_preprocessed_input(data) for data in [*images, *videos]):
            raise ValueError("LongCat 2.5 expects raw image and video inputs")
        return await asyncio.get_running_loop().run_in_executor(
            self.io_executor,
            self._process,
            input_text,
            images,
            videos,
            kwargs.get("max_req_input_len"),
        )

    def _process(self, prompt, images, videos, max_length):
        ids = (
            self._tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        )
        image_start = self.vision["image_start_token_id"]
        image_end = self.vision["image_end_token_id"]
        image_pad = self.vision["image_pad_token_id"]
        video_start = self.video.get("video_start_token_id")
        video_end = self.video.get("video_end_token_id")
        video_pad = self.video.get("video_place_token_id")
        markers = {image_start: (image_end, images, Modality.IMAGE)}
        if video_start is not None:
            markers[video_start] = (video_end, videos, Modality.VIDEO)
        media_indices = dict.fromkeys(markers, 0)
        expanded, items = [], []
        cursor = 0
        while cursor < len(ids):
            token = ids[cursor]
            expanded.append(token)
            cursor += 1
            if token not in markers:
                continue
            end, data, modality = markers[token]
            index = media_indices[token]
            if index >= len(data):
                raise ValueError(
                    f"LongCat 2.5 {modality.name.lower()} marker has no media item"
                )
            try:
                # The published chat template puts the URL inside the marker
                # pair. Use the separately supplied media and discard that text.
                cursor = ids.index(end, cursor) + 1
            except ValueError as error:
                raise ValueError("Unclosed LongCat 2.5 media marker") from error
            media_indices[token] += 1
            spans = []
            if modality == Modality.IMAGE:
                image, _ = load_image(data[index])
                patches, grid = _media_patches([image], self.vision)
                count = math.prod(grid) // self.vision["spatial_merge_size"] ** 2
                spans.append((len(expanded), len(expanded) + count - 1))
                expanded.extend([image_pad] * count)
            else:
                patches, grid, seconds = _decode_video(data[index], self.video)
                count = grid[1] * grid[2] // self.video["spatial_merge_size"] ** 2
                for second in seconds:
                    expanded.append(image_start)
                    spans.append((len(expanded), len(expanded) + count - 1))
                    expanded.extend([video_pad] * count)
                    expanded.append(image_end)
                    if "<TIMEIDX>" in self.video.get("frame_pattern", ""):
                        expanded.extend(
                            self._tokenizer.encode(
                                f"at second[{second}]", add_special_tokens=False
                            )
                        )
            expanded.append(end)
            if max_length is not None and len(expanded) >= max_length:
                raise ValueError(
                    "LongCat 2.5 prompt exceeds the input limit after media expansion"
                )
            items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=spans,
                    feature=patches,
                    model_specific_data={"grid_thw": list(grid)},
                )
            )
        if any(
            media_indices[start] != len(marker[1]) for start, marker in markers.items()
        ):
            raise ValueError("LongCat 2.5 media items do not match the prompt markers")
        return MultimodalProcessorOutput(
            mm_items=items,
            input_ids=expanded,
            im_token_id=image_pad,
            im_start_id=image_start,
            im_end_id=image_end,
            video_token_id=video_pad,
        )
