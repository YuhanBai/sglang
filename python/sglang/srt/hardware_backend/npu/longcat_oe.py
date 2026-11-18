# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 950 LongCat OE kernels, adapted from vLLM-Ascend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_history(
    tokens,
    history,
    ends,
    rows,
    starts,
    count: tl.constexpr,
    batch: tl.constexpr,
    row_stride: tl.constexpr,
    width: tl.constexpr,
    steps: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0) * block + tl.arange(0, block)
    # Upper bound in the packed lengths, including zero-length requests.
    lo = tl.full((block,), 0, tl.int32)
    hi = tl.full((block,), batch, tl.int32)
    for _ in tl.static_range(steps):
        mid = (lo + hi) // 2
        end = tl.load(ends + mid, mid < batch, other=2147483647)
        right = (lo < hi) & (token >= end)
        hi = tl.where((lo < hi) & ~right, mid, hi)
        lo = tl.where(right, mid + 1, lo)
    active = (token < count) & (lo < batch)
    prior_end = tl.load(ends + lo - 1, active & (lo > 0), other=0)
    column = tl.load(starts + lo, active, other=0) + token - prior_end
    row = tl.load(rows + lo, active, other=0).to(tl.int64)
    value = tl.load(tokens + token, active, other=0)
    tl.store(
        history + row * row_stride + column,
        value,
        active & (column >= 0) & (column < width),
    )


def update_token_table(tokens, history, rows, starts, lengths):
    count, batch = tokens.numel(), rows.numel()
    if count == 0 or batch == 0:
        return
    ends = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    _write_history[(triton.cdiv(count, 128),)](
        tokens.contiguous(),
        history,
        ends,
        rows.contiguous(),
        starts.contiguous(),
        count,
        batch,
        history.stride(0),
        history.shape[1],
        batch.bit_length(),
        128,
    )


@triton.jit
def _hash_history(
    history,
    positions,
    rows,
    valid_rows,
    weights,
    moduli,
    offsets,
    output,
    count: tl.constexpr,
    row_stride: tl.constexpr,
    width: tl.constexpr,
    eos: tl.constexpr,
    n: tl.constexpr,
    splits: tl.constexpr,
    tables: tl.constexpr,
    block: tl.constexpr,
):
    lane = tl.arange(0, block)
    # Store token-major IDs in the runner's persistent output buffer.
    for tile in range(
        tl.program_id(0), tl.cdiv(count * tables, block), tl.num_programs(0)
    ):
        linear = tile * block + lane
        token = linear // tables
        table = linear % tables
        active = token < count
        valid = tl.load(valid_rows + token, active, other=0)
        row = tl.load(rows + token, active, other=0).to(tl.int64)
        position = tl.load(positions + token, active, other=0)
        valid = valid & (position >= 0) & (position < width)
        hashed = tl.full((block,), 0, tl.int64)
        for distance in tl.static_range(n):
            column = position - distance
            item = tl.load(
                history + row * row_stride + column,
                active & valid & (column >= 0),
                other=-1,
            )
            valid = valid & (item >= 0)
            if distance > 0:
                valid = valid & (item != eos)
            factor = tl.load(weights + table * n + distance).to(tl.int64)
            hashed += tl.where(
                valid & (distance < table // splits + 2),
                item.to(tl.int64) * factor,
                0,
            )
        modulus = tl.load(moduli + table).to(tl.int64)
        offset = tl.load(offsets + table)
        tl.store(output + linear, hashed % modulus + offset, active)


def compute_ngram_ids(
    history, positions, rows, valid, weights, moduli, offsets, eos, output
):
    count, tables = output.shape
    if count == 0:
        return
    _hash_history[(min(256, triton.cdiv(count * tables, 128)),)](
        history,
        positions.contiguous(),
        rows.contiguous(),
        valid.contiguous(),
        weights,
        moduli,
        offsets,
        output,
        count,
        history.stride(0),
        history.shape[1],
        eos,
        weights.shape[-1],
        weights.shape[1],
        tables,
        128,
    )
