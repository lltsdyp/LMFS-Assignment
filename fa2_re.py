import math

import torch
import triton
import triton.language as tl

@triton.jit
def _operand_product(lhs, rhs, mode: tl.constexpr, acc=None):
    """Convert the operands before issuing the requested dot-product variant."""
    if mode == "fp16":
        lhs16 = lhs.to(tl.float16)
        rhs16 = rhs.to(tl.float16)
        return tl.dot(lhs16, rhs16, acc)
    else:
        lhs32 = lhs.to(tl.float32)
        rhs32 = rhs.to(tl.float32)
        return tl.dot(lhs32, rhs32, acc, input_precision=mode)

@triton.jit
def _stage_padded_kernel(
    q_src, k_src, v_src, q_dst, k_dst, v_dst,
    sqh: tl.constexpr, sqr: tl.constexpr, sqd: tl.constexpr,
    skh: tl.constexpr, skr: tl.constexpr, skd: tl.constexpr,
    svh: tl.constexpr, svr: tl.constexpr, svd: tl.constexpr,
    seq: tl.constexpr, dim: tl.constexpr, pitch: tl.constexpr,
    col_block: tl.constexpr, row_block: tl.constexpr, copy_v: tl.constexpr,
):
    """Pack strided Q/K/V into half storage with explicit zero-filled padding."""
    col = tl.arange(0, col_block)
    row = tl.program_id(0) * row_block + tl.arange(0, row_block)
    head = tl.program_id(1).to(tl.int64)
    row_ok = row[:, None] < seq
    read_ok = row_ok & (col[None, :] < dim)
    write_ok = row_ok & (col[None, :] < pitch)
    dst_offset = (head * seq + row[:, None]) * pitch + col[None, :]

    q_ptrs = q_src + head * sqh + row[:, None] * sqr + col[None, :] * sqd
    q_data = tl.load(q_ptrs, read_ok, other=0)
    tl.store(q_dst + dst_offset, q_data.to(tl.float16), write_ok)

    k_ptrs = k_src + head * skh + row[:, None] * skr + col[None, :] * skd
    k_data = tl.load(k_ptrs, read_ok, other=0)
    tl.store(k_dst + dst_offset, k_data.to(tl.float16), write_ok)

    if copy_v:
        v_ptrs = v_src + head * svh + row[:, None] * svr + col[None, :] * svd
        v_data = tl.load(v_ptrs, read_ok, other=0)
        tl.store(v_dst + dst_offset, v_data.to(tl.float16), write_ok)

@triton.jit
def _generic_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    log_partition_ptr,
    output_row_sum_ptr,
    query_head_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_feature_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    key_feature_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    precision_mode: tl.constexpr,
):
    """Stream key/value tiles for the wide-head fallback."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    column_indices = tl.arange(0, key_block)
    feature_indices = tl.arange(0, feature_block)
    query = tl.load(
        query_ptr
        + head_index * query_head_stride
        + row_indices[:, None] * query_row_stride
        + feature_indices[None, :] * query_feature_stride,
        (row_indices[:, None] < sequence_length)
        & (feature_indices[None, :] < feature_count),
        other=0,
    )
    running_maximum = tl.full((query_block,), -float("inf"), tl.float32)
    normalizing_mass = tl.full((query_block,), 0, tl.float32)
    accumulator = tl.full((query_block, feature_block), 0, tl.float32)
    for tile_index in range(tl.cdiv(sequence_length, key_block)):
        key_indices = tile_index * key_block + column_indices
        key = tl.load(
            key_ptr
            + head_index * key_head_stride
            + feature_indices[:, None] * key_feature_stride
            + key_indices[None, :] * key_row_stride,
            (feature_indices[:, None] < feature_count)
            & (key_indices[None, :] < sequence_length),
            other=0,
        )
        value = tl.load(
            value_ptr
            + head_index * value_head_stride
            + key_indices[:, None] * value_row_stride
            + feature_indices[None, :] * value_feature_stride,
            (key_indices[:, None] < sequence_length)
            & (feature_indices[None, :] < feature_count),
            other=0,
        )
        score_tile = _operand_product(query, key, precision_mode) * score_scale
        score_tile = tl.where(
            key_indices[None, :] < sequence_length, score_tile, -float("inf")
        )
        updated_maximum = tl.maximum(running_maximum, tl.max(score_tile, 1))
        rescale_factor = tl.exp2(running_maximum - updated_maximum)
        probabilities = tl.exp2(score_tile - updated_maximum[:, None])
        accumulator = _operand_product(
            probabilities, value, precision_mode, accumulator * rescale_factor[:, None]
        )
        normalizing_mass = normalizing_mass * rescale_factor + tl.sum(probabilities, 1)
        running_maximum = updated_maximum
    stored_output = (accumulator / normalizing_mass[:, None]).to(
        output_ptr.dtype.element_ty
    )
    tl.store(
        output_row_sum_ptr + head_index * sequence_length + row_indices,
        tl.sum(stored_output.to(tl.float32), 1),
        row_indices < sequence_length,
    )
    tl.store(
        output_ptr
        + (head_index * sequence_length + row_indices[:, None]) * feature_count
        + feature_indices[None, :],
        stored_output,
        (row_indices[:, None] < sequence_length)
        & (feature_indices[None, :] < feature_count),
    )
    tl.store(
        log_partition_ptr + head_index * sequence_length + row_indices,
        running_maximum + tl.log2(normalizing_mass),
        row_indices < sequence_length,
    )

@triton.jit
def _prepare_gradient_rows(
    output_ptr,
    upstream_ptr,
    value_ptr,
    row_contraction_ptr,
    value_reduction_ptr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    upstream_head_stride: tl.constexpr,
    upstream_row_stride: tl.constexpr,
    upstream_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
):
    """Prepare row contractions and optional sequence-broadcast reductions."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    feature_indices = tl.arange(0, feature_block)
    valid_elements = (row_indices[:, None] < sequence_length) & (
        feature_indices[None, :] < feature_count
    )
    upstream_tile = tl.load(
        upstream_ptr
        + head_index * upstream_head_stride
        + row_indices[:, None] * upstream_row_stride
        + feature_indices[None, :] * upstream_feature_stride,
        valid_elements,
        other=0,
    ).to(tl.float32)
    output = tl.load(
        output_ptr
        + (head_index * sequence_length + row_indices[:, None]) * feature_count
        + feature_indices[None, :],
        valid_elements,
        other=0,
    )
    tl.store(
        row_contraction_ptr + head_index * sequence_length + row_indices,
        tl.sum(output.to(tl.float32) * upstream_tile, 1),
        row_indices < sequence_length,
    )

@triton.jit
def _generic_query_gradient(
    query_ptr,
    key_ptr,
    value_ptr,
    upstream_ptr,
    log_partition_ptr,
    row_contraction_ptr,
    value_reduction_ptr,
    query_gradient_ptr,
    query_head_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_feature_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    key_feature_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    upstream_head_stride: tl.constexpr,
    upstream_row_stride: tl.constexpr,
    upstream_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    attention_scale: tl.constexpr,
    precision_mode: tl.constexpr,
):
    """Compute query derivatives for a general upstream gradient."""
    head_index = tl.program_id(1).to(tl.int64)
    row_indices = tl.program_id(0) * query_block + tl.arange(0, query_block)
    column_indices = tl.arange(0, key_block)
    feature_indices = tl.arange(0, feature_block)
    valid_elements = (row_indices[:, None] < sequence_length) & (
        feature_indices[None, :] < feature_count
    )
    query = tl.load(
        query_ptr
        + head_index * query_head_stride
        + row_indices[:, None] * query_row_stride
        + feature_indices[None, :] * query_feature_stride,
        valid_elements,
        other=0,
    )
    upstream_tile = tl.load(
        upstream_ptr
        + head_index * upstream_head_stride
        + row_indices[:, None] * upstream_row_stride
        + feature_indices[None, :] * upstream_feature_stride,
        valid_elements,
        other=0,
    )
    log_partition = tl.load(
        log_partition_ptr + head_index * sequence_length + row_indices,
        row_indices < sequence_length,
        other=0,
    )
    row_contraction = tl.load(
        row_contraction_ptr + head_index * sequence_length + row_indices,
        row_indices < sequence_length,
        other=0,
    )
    accumulator = tl.full((query_block, feature_block), 0, tl.float32)
    probability_gain: tl.constexpr = 1.0
    for tile_index in range(tl.cdiv(sequence_length, key_block)):
        key_indices = tile_index * key_block + column_indices
        valid_transposed_elements = (feature_indices[:, None] < feature_count) & (
            key_indices[None, :] < sequence_length
        )
        transposed_keys = tl.load(
            key_ptr
            + head_index * key_head_stride
            + feature_indices[:, None] * key_feature_stride
            + key_indices[None, :] * key_row_stride,
            valid_transposed_elements,
            other=0,
        )
        score_tile = (
            _operand_product(query, transposed_keys, precision_mode) * score_scale
        )
        probabilities = tl.exp2(score_tile - log_partition[:, None])
        probabilities = tl.where(
            key_indices[None, :] < sequence_length, probabilities, 0
        )
        transposed_values = tl.load(
            value_ptr
            + head_index * value_head_stride
            + feature_indices[:, None] * value_feature_stride
            + key_indices[None, :] * value_row_stride,
            valid_transposed_elements,
            other=0,
        )
        probability_gradient = _operand_product(
            upstream_tile, transposed_values, precision_mode
        )
        score_gradient = (
            probabilities
            * (probability_gradient - row_contraction[:, None])
            * probability_gain
        )
        accumulator = _operand_product(
            score_gradient, tl.trans(transposed_keys), precision_mode, accumulator
        )
    tl.store(
        query_gradient_ptr
        + (head_index * sequence_length + row_indices[:, None]) * feature_count
        + feature_indices[None, :],
        accumulator * (attention_scale / probability_gain),
        valid_elements,
    )

@triton.jit
def _generic_key_value_gradient(
    query_ptr,
    key_ptr,
    value_ptr,
    upstream_ptr,
    log_partition_ptr,
    row_contraction_ptr,
    value_reduction_ptr,
    key_gradient_ptr,
    value_gradient_ptr,
    query_head_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_feature_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_row_stride: tl.constexpr,
    key_feature_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_row_stride: tl.constexpr,
    value_feature_stride: tl.constexpr,
    upstream_head_stride: tl.constexpr,
    upstream_row_stride: tl.constexpr,
    upstream_feature_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    feature_count: tl.constexpr,
    feature_block: tl.constexpr,
    query_block: tl.constexpr,
    key_block: tl.constexpr,
    score_scale: tl.constexpr,
    attention_scale: tl.constexpr,
    precision_mode: tl.constexpr,
):
    """Compute key/value derivatives without an attention-sized workspace."""
    head_index = tl.program_id(1).to(tl.int64)
    key_indices = tl.program_id(0) * key_block + tl.arange(0, key_block)
    row_indices = tl.arange(0, query_block)
    feature_indices = tl.arange(0, feature_block)
    valid_elements = (key_indices[:, None] < sequence_length) & (
        feature_indices[None, :] < feature_count
    )
    key = tl.load(
        key_ptr
        + head_index * key_head_stride
        + key_indices[:, None] * key_row_stride
        + feature_indices[None, :] * key_feature_stride,
        valid_elements,
        other=0,
    )
    key_accumulator = tl.full((key_block, feature_block), 0, tl.float32)
    value = tl.load(
        value_ptr
        + head_index * value_head_stride
        + key_indices[:, None] * value_row_stride
        + feature_indices[None, :] * value_feature_stride,
        valid_elements,
        other=0,
    )
    value_accumulator = tl.full((key_block, feature_block), 0, tl.float32)
    probability_gain: tl.constexpr = 1.0
    for tile_index in range(tl.cdiv(sequence_length, query_block)):
        query_indices = tile_index * query_block + row_indices
        valid_transposed_elements = (feature_indices[:, None] < feature_count) & (
            query_indices[None, :] < sequence_length
        )
        transposed_queries = tl.load(
            query_ptr
            + head_index * query_head_stride
            + feature_indices[:, None] * query_feature_stride
            + query_indices[None, :] * query_row_stride,
            valid_transposed_elements,
            other=0,
        )
        transposed_upstream = tl.load(
            upstream_ptr
            + head_index * upstream_head_stride
            + feature_indices[:, None] * upstream_feature_stride
            + query_indices[None, :] * upstream_row_stride,
            valid_transposed_elements,
            other=0,
        )
        log_partition = tl.load(
            log_partition_ptr + head_index * sequence_length + query_indices,
            query_indices < sequence_length,
            other=0,
        )
        row_contraction = tl.load(
            row_contraction_ptr + head_index * sequence_length + query_indices,
            query_indices < sequence_length,
            other=0,
        )
        score_tile = (
            _operand_product(key, transposed_queries, precision_mode) * score_scale
        )
        probabilities = tl.exp2(score_tile - log_partition[None, :])
        probabilities = tl.where(
            query_indices[None, :] < sequence_length, probabilities, 0
        )
        probability_gradient = _operand_product(
            value, transposed_upstream, precision_mode
        )
        score_gradient = (
            probabilities
            * (probability_gradient - row_contraction[None, :])
            * probability_gain
        )
        key_accumulator = _operand_product(
            score_gradient,
            tl.trans(transposed_queries),
            precision_mode,
            key_accumulator,
        )
        value_accumulator = _operand_product(
            probabilities * probability_gain,
            tl.trans(transposed_upstream),
            precision_mode,
            value_accumulator,
        )
    tl.store(
        key_gradient_ptr
        + (head_index * sequence_length + key_indices[:, None]) * feature_count
        + feature_indices[None, :],
        key_accumulator * (attention_scale / probability_gain),
        valid_elements,
    )
    value_gradient_tile = value_accumulator / probability_gain
    tl.store(
        value_gradient_ptr
        + (head_index * sequence_length + key_indices[:, None]) * feature_count
        + feature_indices[None, :],
        value_gradient_tile,
        valid_elements,
    )

class FlashAttention2(torch.autograd.Function):
    """Tiled attention for CUDA [H, N, D] tensors."""

    INPUT_DTYPE = torch.float32
    OUTPUT_DTYPE = torch.float16
    DOT_MODE = "fp16"

    @staticmethod
    def forward(ctx, query, key, value):
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Q, K and V must share shape")
        if query.device != key.device or query.device != value.device:
            raise ValueError("Q, K and V must share device")
        heads, seq, dim = query.shape
        if not (heads > 0 and seq > 0 and 1 <= dim <= 256):
            raise ValueError("Expected H/N > 0 and 1 <= D <= 256")
        output = torch.empty(query.shape, device=query.device, dtype=FlashAttention2.OUTPUT_DTYPE)
        partition = torch.empty((heads, seq), device=query.device, dtype=torch.float32)
        output_sum = torch.empty_like(partition)
        padded = max(16, triton.next_power_of_2(dim))
        staging = tuple(
            torch.empty((heads, seq, padded), device=query.device, dtype=torch.float16)
            for _ in range(3)
        )
        with torch.cuda.device(query.device):
            _stage_padded_kernel[triton.cdiv(seq, 32), heads](
                query, key, value, *staging,
                *query.stride(), *key.stride(), *value.stride(),
                seq, dim, padded, padded, 32, True, num_warps=4,
            )
        q, k, v = staging
        with torch.cuda.device(query.device):
            _generic_forward_kernel[triton.cdiv(seq, 32), heads](
                q, k, v, output, partition, output_sum,
                *q.stride(), *k.stride(), *v.stride(),
                seq, dim, max(16, triton.next_power_of_2(dim)), 32, 32,
                dim ** -0.5 * 1.4426950408889634, FlashAttention2.DOT_MODE,
                num_warps=4, num_stages=1,
            )
        ctx.save_for_backward(query, key, value, output, partition, output_sum)
        return output

    @staticmethod
    def backward(ctx, upstream):
        query, key, value, output, partition, output_sum = ctx.saved_tensors
        heads, seq, dim = query.shape
        # Start with the general-gradient kernels; specialize broadcasting later.
        upstream = upstream.contiguous()
        dq, dk, dv = (
            torch.empty(item.shape, device=item.device, dtype=item.dtype)
            for item in (query, key, value)
        )
        contraction = torch.empty_like(partition)
        reduction = torch.empty_like(partition)
        feature_block = max(16, triton.next_power_of_2(dim))
        shared = (
            *query.stride(), *key.stride(), *value.stride(), *upstream.stride(),
            seq, dim, feature_block, 32, 32,
            dim ** -0.5 * 1.4426950408889634, dim ** -0.5, FlashAttention2.DOT_MODE,
        )
        with torch.cuda.device(query.device):
            _prepare_gradient_rows[triton.cdiv(seq, 32), heads](
                output, upstream, value, contraction, reduction,
                *value.stride(), *upstream.stride(), seq, dim, feature_block, 32,
                num_warps=4,
            )
            _generic_query_gradient[triton.cdiv(seq, 32), heads](
                query, key, value, upstream, partition, contraction, reduction, dq,
                *shared, num_warps=4, num_stages=1,
            )
            _generic_key_value_gradient[triton.cdiv(seq, 32), heads](
                query, key, value, upstream, partition, contraction, reduction, dk, dv,
                *shared, num_warps=4, num_stages=1,
            )
        return dq, dk, dv
