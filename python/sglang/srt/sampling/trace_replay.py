"""Packed, immutable trace payloads and the batch rows that reference them."""

from dataclasses import dataclass, replace

import torch

from sglang.srt.utils.common import is_pin_memory_available


@dataclass(frozen=True)
class TraceReplayBatch:
    tokens: torch.Tensor
    spans: tuple[tuple[int, int], ...]
    offsets: torch.Tensor
    lengths: torch.Tensor
    steps: torch.Tensor
    prompt_lengths: torch.Tensor
    rows: torch.Tensor

    @staticmethod
    def _copy_to_device(values, device, dtype=torch.long):
        return torch.tensor(
            values, dtype=dtype, pin_memory=is_pin_memory_available(device)
        ).to(device, non_blocking=True)

    @classmethod
    def _from_spans(cls, tokens, spans, steps, prompt_lengths):
        device = tokens.device
        return cls(
            tokens,
            spans,
            cls._copy_to_device([start for start, _ in spans], device),
            cls._copy_to_device([length for _, length in spans], device),
            steps,
            prompt_lengths,
            cls._copy_to_device(
                [row for row, (_, length) in enumerate(spans) if length], device
            ),
        )

    @classmethod
    def from_requests(cls, reqs, device):
        traces = [
            getattr(req.sampling_params, "trace_decode_token_ids", None) for req in reqs
        ]
        if not any(isinstance(trace, list) for trace in traces):
            return None
        tokens = []
        spans = []
        steps = []
        prompt_lengths = []
        for req, trace in zip(reqs, traces, strict=True):
            if not isinstance(trace, list):
                spans.append((0, 0))
                steps.append(0)
                prompt_lengths.append(0)
                continue
            limit = req.sampling_params.max_new_tokens
            if isinstance(limit, int):
                trace = trace[:limit]
            spans.append((len(tokens), len(trace)))
            tokens.extend(trace)
            steps.append(len(req.output_ids))
            prompt_lengths.append(len(req.origin_input_ids))

        if not tokens:
            return None
        return cls._from_spans(
            cls._copy_to_device(tokens, device, torch.int32),
            tuple(spans),
            cls._copy_to_device(steps, device),
            cls._copy_to_device(prompt_lengths, device),
        )

    def filter(self, keep_indices, keep_indices_device):
        spans = tuple(self.spans[i] for i in keep_indices)
        if not any(length for _, length in spans):
            return None
        tokens = self.tokens
        if sum(length for _, length in spans) != tokens.numel():
            # Only copy the payload when a trace leaves. Old snapshots may
            # still be referenced by an in-flight forward.
            pieces = []
            compact = []
            offset = 0
            for start, length in spans:
                compact.append((offset, length))
                if length:
                    pieces.append(tokens.narrow(0, start, length))
                    offset += length
            tokens = torch.cat(pieces)
            spans = tuple(compact)
        return self._from_spans(
            tokens,
            spans,
            self.steps[keep_indices_device],
            self.prompt_lengths[keep_indices_device],
        )

    @classmethod
    def merge(cls, left, right, left_size, right_size):
        if left is None and right is None:
            return None
        trace = left if left is not None else right
        if left is None or right is None:
            # Ordinary requests add metadata only.
            padding = left_size if left is None else right_size
            before = padding if left is None else 0
            after = padding - before
            return cls._from_spans(
                trace.tokens,
                ((0, 0),) * before + trace.spans + ((0, 0),) * after,
                torch.nn.functional.pad(trace.steps, (before, after)),
                torch.nn.functional.pad(trace.prompt_lengths, (before, after)),
            )
        offset = left.tokens.numel()
        return cls._from_spans(
            torch.cat((left.tokens, right.tokens)),
            left.spans
            + tuple((start + offset, length) for start, length in right.spans),
            torch.cat((left.steps, right.steps)),
            torch.cat((left.prompt_lengths, right.prompt_lengths)),
        )

    def advance(self):
        return replace(self, steps=self.steps + (self.lengths > 0))

    def lookup(self, steps):
        """Read one or more trace steps per row, with an explicit validity mask."""
        offsets, lengths = self.offsets, self.lengths
        if steps.ndim == 2:
            offsets, lengths = offsets[:, None], lengths[:, None]
        valid = (steps >= 0) & (steps < lengths)
        indices = torch.where(valid, offsets + steps, 0)
        return self.tokens[indices], valid


def capture_trace_logits(logits, sampling_info, return_logprob):
    """Snapshot only trace rows before sampling mutates the model logits."""
    if return_logprob and sampling_info.trace_replay is not None:
        return logits.index_select(0, sampling_info.trace_replay.rows)
    return None
