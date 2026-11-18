"""LongCat 2.5 multimodal model with the existing LongCat language backbone."""

import torch

from sglang.srt.managers.mm_utils import pad_mm_tokens, prepare_mm_inputs
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.longcat_25_vision import LongCat25Vision, LongCat25VisionBridge
from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM
from sglang.srt.runtime_context import get_disagg, get_mm

_VISION_PREFIXES = ("visual_model.", "visual_bridge_model.")
_QUANTIZER_STATE = "visual_model.visual_quantizer.quantize."


class _OEEmbedding:
    def __init__(self, embedding, forward_batch):
        self.embedding = embedding
        self.forward_batch = forward_batch
        self.num_embeddings = embedding.num_embeddings

    def __call__(self, input_ids):
        return self.embedding(input_ids, self.forward_batch)


class Longcat25ForConditionalGeneration(LongcatFlashForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        self.visual_model = None
        if (
            get_mm().enable_multimodal is not False
            and not get_disagg().language_model_only
            and not getattr(config, "language_model_only", False)
        ):
            vision, quantizer = config.visual_config, config.visual_quantizer_config
            self.visual_model = LongCat25Vision(vision, quantizer)
            self.visual_bridge_model = LongCat25VisionBridge(
                vision, quantizer, config.hidden_size, config.rms_norm_eps
            )
        self._visual_weights_initialized = False

    def pad_input_ids(self, input_ids, mm_inputs):
        return pad_mm_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items):
        return torch.cat(
            [
                self.visual_bridge_model(self.visual_model(item.feature, item.grid_thw))
                for item in items
            ],
            dim=0,
        )

    get_video_feature = get_image_feature

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            if self.visual_model is None or input_embeds is not None:
                raise ValueError(
                    "LongCat multimedia prefill requires its vision tower and token IDs"
                )
            input_embeds = prepare_mm_inputs(
                input_ids,
                forward_batch,
                self,
                {},
                multimodal_model=self,
                input_embedding=_OEEmbedding(
                    self.get_input_embeddings(), forward_batch
                ),
            )
        return super().forward(input_ids, positions, forward_batch, input_embeds)

    def load_weights(self, weights):
        if self.visual_model is None:
            return super().load_weights(weights)
        parameters = {
            name: param
            for name, param in self.named_parameters()
            if name.startswith(_VISION_PREFIXES)
        }
        loaded = set()

        def text_weights():
            for name, weight in weights:
                name_without_root = name.removeprefix("model.")
                if not name_without_root.startswith(_VISION_PREFIXES):
                    yield name, weight
                    continue
                if name_without_root == _QUANTIZER_STATE + "codebooks.0.embed":
                    name_without_root = "visual_model.visual_quantizer.codebook"
                    if weight.shape[0] == parameters[name_without_root].shape[0] + 1:
                        weight = weight[:-1]
                elif name_without_root.startswith(_QUANTIZER_STATE):
                    # The shared codebook has repeated entries and EMA state.
                    continue
                param = parameters[name_without_root]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name_without_root)

        super().load_weights(text_weights())
        if not self._visual_weights_initialized:
            if missing := parameters.keys() - loaded:
                raise ValueError(
                    f"LongCat 2.5 missing visual parameters: {sorted(missing)}"
                )
            self._visual_weights_initialized = True


EntryClass = [Longcat25ForConditionalGeneration]
