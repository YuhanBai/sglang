# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend 950 LongCat LSA composition, adapted from vLLM-Ascend."""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _gather_candidate_keys(
    cache,
    block_table,
    sequence,
    chosen,
    output,
    rows,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_DIM: tl.constexpr,
    BT_ROW: tl.constexpr,
    BT_COL: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_COLUMNS: tl.constexpr,
    PAGES: tl.constexpr,
    DIM: tl.constexpr,
    COUNT: tl.constexpr,
    TILE: tl.constexpr,
):
    chunks = tl.cdiv(COUNT, TILE)
    for tile in range(tl.program_id(0), rows * chunks, tl.num_programs(0)):
        row = tile // chunks
        column = (tile % chunks) * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, DIM)
        active = column < COUNT
        raw_index = tl.load(chosen + row * C_ROW + column * C_COL, active, other=-1)
        index = tl.maximum(raw_index, 0)
        req = tl.load(sequence + row)
        logical_page = index // PAGE_SIZE
        valid = active & (raw_index >= 0) & (logical_page < PAGE_COLUMNS)
        page = tl.load(
            block_table + req * BT_ROW + logical_page * BT_COL, valid, other=0
        )
        page = tl.maximum(page, 0).to(tl.int64)
        valid = valid & (page < PAGES)
        key = tl.load(
            cache
            + page[:, None] * K_PAGE
            + (index % PAGE_SIZE)[:, None] * K_TOKEN
            + dims[None, :] * K_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        tl.store(
            output + (row * COUNT + column[:, None]) * DIM + dims[None, :],
            key,
            active[:, None],
        )


def gather_candidate_keys_triton(key, block_table, sequence, chosen):
    rows, count = chosen.shape
    dim = key.shape[-1]
    output = torch.empty((rows, count, dim), dtype=torch.float32, device=key.device)
    if rows and count:
        tile = 16
        _gather_candidate_keys[(min(256, rows * triton.cdiv(count, tile)),)](
            key,
            block_table,
            sequence,
            chosen,
            output,
            rows,
            key.stride(0),
            key.stride(1),
            key.stride(3),
            *block_table.stride(),
            *chosen.stride(),
            key.shape[1],
            block_table.shape[1],
            key.shape[0],
            dim,
            count,
            tile,
        )
    return output


def lightning_indexer(
    query,
    key,
    weights,
    q_ends,
    kv_lengths,
    block_table,
    topk,
    init_tokens,
    local_tokens,
):
    """Merge mandatory prefix/local tokens with the native causal candidates."""
    if key.shape[2] != 1 or key.shape[-1] != 128:
        raise ValueError(
            "Ascend 950 LongCat LSA expects one BF16 key head of width 128"
        )
    if key.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
        raise ValueError("Ascend 950 LongCat LSA requires BF16 query and key")
    q_ends = q_ends.to(device=query.device, dtype=torch.int32)
    kv_lengths = kv_lengths.to(device=query.device, dtype=torch.int32)
    weights = weights.reshape(-1, query.shape[1]).contiguous()
    candidates, _ = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights.to(query.dtype),
        actual_seq_lengths_query=q_ends,
        actual_seq_lengths_key=kv_lengths,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
    )
    rows = torch.arange(query.shape[0], dtype=torch.int32, device=query.device)
    sequence = torch.searchsorted(q_ends, rows, right=True)
    in_batch = sequence < q_ends.numel()
    sequence = sequence.clamp_max(q_ends.numel() - 1)
    visible = kv_lengths[sequence] - q_ends[sequence] + rows + 1
    visible = torch.where(in_batch, visible.clamp_min(0), 0)[:, None]
    prefix = visible.clamp_max(init_tokens)
    local_start = torch.maximum(prefix, visible - local_tokens)
    order = torch.arange(topk, dtype=torch.int32, device=query.device)[None, :]
    forced = torch.where(order < prefix, order, local_start + order - prefix)
    forced_valid = forced < visible
    chosen = candidates.squeeze(1)
    middle = (chosen >= prefix) & (chosen < local_start)

    scores_chunks = []
    for start in range(0, query.shape[0], 32):
        stop = start + 32
        indices = torch.where(middle[start:stop], chosen[start:stop], -1)
        keys = gather_candidate_keys_triton(
            key, block_table, sequence[start:stop], indices
        )
        logits = torch.bmm(query[start:stop].float(), keys.transpose(1, 2))
        scores_chunks.append(
            (logits.relu() * weights[start:stop, :, None].float()).sum(dim=1)
        )
    score_limit = torch.finfo(torch.float32).max
    native_scores = torch.cat(scores_chunks).clamp_max(score_limit / 4)
    native_scores = torch.where(middle, native_scores, -torch.inf)
    forced_scores = torch.where(forced_valid, score_limit / 2, -torch.inf)
    scores = torch.cat((forced_scores, native_scores), dim=-1)
    best_scores, best = scores.topk(topk, dim=-1, sorted=True)
    indices = torch.cat((forced, chosen), dim=-1).gather(-1, best)
    return torch.where(best_scores.isfinite(), indices, -1).to(torch.int32)
