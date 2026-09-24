import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dflash_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous out_tokens_out."
        )
    if not accept_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous accept_lens_out."
        )
    if not commit_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous commit_lens_out."
        )
    if not bonus_ids_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous bonus_ids_out."
        )
    if prefix_lens.ndim != 1:
        raise ValueError("DFLASH Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous new_seq_lens_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dflash_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    bonus_tokens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    bonus_tokens_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    bonus_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(bonus_tokens.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        bonus_tokens.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
):
    batch, slots, top_k = candidate_ids.shape
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


@triton.jit
def _draft_top1_pack_kernel(
    logits_ptr,
    out_ptr,
    row_stride,
    vocab_start,
    vocab_size,
    BLOCK: tl.constexpr,
):
    """Per-row argmax of one vocabulary shard, packed into a sortable int64.

    The high 32 bits hold the maximum as an order-preserving signed key, the low
    32 bits hold the global token id, so a plain integer comparison across ranks
    ranks the candidates by logit.
    """
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * row_stride

    best_value = tl.full((BLOCK,), float("-inf"), tl.float32)
    best_index = tl.zeros((BLOCK,), tl.int64) + vocab_size.to(tl.int64)
    for start in range(0, vocab_size, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_range = cols < vocab_size
        value = tl.load(row_ptr + cols, mask=in_range, other=float("-inf")).to(
            tl.float32
        )
        # Strict greater keeps the lowest index of a repeated maximum, which is
        # what torch.max reports.
        higher = value > best_value
        best_value = tl.where(higher, value, best_value)
        best_index = tl.where(higher, cols.to(tl.int64), best_index)

    top_value = tl.max(best_value, 0)
    top_index = tl.min(
        tl.where(best_value == top_value, best_index, vocab_size.to(tl.int64)), 0
    )

    bits = top_value.to(tl.int32, bitcast=True).to(tl.int64)
    # Float bits are not monotonic across the sign; remap so integer order
    # matches float order, still inside the signed 32-bit range.
    key = tl.where(bits >= 0, bits, -bits - tl.full((), 2147483649, tl.int64))
    packed = (key << 32) | (top_index + vocab_start.to(tl.int64))
    tl.store(out_ptr + row, packed)


@triton.jit
def _draft_top1_merge_kernel(
    gathered_ptr,
    out_ptr,
    rows,
    TP_SIZE: tl.constexpr,
):
    """Pick the winning rank per row and unpack its global token id."""
    row = tl.program_id(0).to(tl.int64)
    best = tl.load(gathered_ptr + row)
    # The int64 `offset` leads every addition, so the index stays 64-bit without
    # widening `rows`: Triton passes an int argument whose value is 1 as a plain
    # Python int, and `rows` is 1 whenever batch 1 meets a draft window of 2.
    offset = row
    for rank in tl.static_range(1, TP_SIZE):
        offset += rows
        candidate = tl.load(gathered_ptr + offset)
        # Strict greater on the key alone keeps the lowest rank on a tie, which
        # is what argmax over the gathered maxima reports.
        best = tl.where((candidate >> 32) > (best >> 32), candidate, best)
    tl.store(out_ptr + row, best & tl.full((), 0xFFFFFFFF, tl.int64))


def draft_top1_pack(
    logits: torch.Tensor, out: torch.Tensor, vocab_start: int
) -> None:
    """Write one packed (max logit, global token id) per row of `logits`."""
    rows, vocab_size = logits.shape
    block = min(2048, triton.next_power_of_2(vocab_size))
    _draft_top1_pack_kernel[(rows,)](
        logits,
        out,
        logits.stride(0),
        vocab_start,
        vocab_size,
        BLOCK=block,
        num_warps=16,
    )


def draft_top1_merge(gathered: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce the gathered per-rank candidates into one token id per row."""
    tp_size, rows = gathered.shape
    _draft_top1_merge_kernel[(rows,)](gathered, out, rows, TP_SIZE=tp_size, num_warps=1)
