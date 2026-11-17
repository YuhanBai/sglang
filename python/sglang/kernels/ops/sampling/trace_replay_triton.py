"""Fused device trace lookup for decode and speculative verification."""

import torch
import triton
import triton.language as tl


@triton.jit
def _trace_replay_kernel(
    table,
    slots,
    lengths,
    steps,
    sampled,
    output,
    valid_output,
    size,
    num_slots: tl.constexpr,
    width: tl.constexpr,
    row_stride: tl.constexpr,
    steps_per_row: tl.constexpr,
    has_sampled: tl.constexpr,
    block_size: tl.constexpr,
):
    idx = tl.program_id(0) * block_size + tl.arange(0, block_size)
    in_batch = idx < size
    row = idx // steps_per_row
    slot = tl.load(slots + row, mask=in_batch, other=0)
    length = tl.load(lengths + row, mask=in_batch, other=0)
    step = tl.load(steps + idx, mask=in_batch, other=0)
    valid = (
        in_batch
        & (slot > 0)
        & (slot < num_slots)
        & (step >= 0)
        & (step < length)
        & (step < width)
    )
    token = tl.load(table + slot * row_stride + step, mask=valid, other=0)
    if has_sampled:
        original = tl.load(sampled + idx, mask=in_batch, other=0)
        token = tl.where(valid, token.to(original.dtype), original)
    else:
        tl.store(valid_output + idx, valid, mask=in_batch)
    tl.store(output + idx, token, mask=in_batch)


def trace_replay_triton(table, slots, lengths, steps, sampled=None):
    size = steps.numel()
    output = (
        torch.empty_like(sampled)
        if sampled is not None
        else torch.empty_like(steps, dtype=table.dtype)
    )
    valid = None if sampled is not None else torch.empty_like(steps, dtype=torch.bool)
    if size:
        block_size = 32
        _trace_replay_kernel[(triton.cdiv(size, block_size),)](
            table,
            slots,
            lengths,
            steps,
            sampled,
            output,
            valid,
            size,
            table.shape[0],
            table.shape[1],
            table.stride(0),
            1 if steps.ndim == 1 else steps.shape[1],
            sampled is not None,
            block_size,
        )
    return output if sampled is not None else (output, valid)
