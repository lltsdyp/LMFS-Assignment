import math

import torch
import triton
import triton.language as tl

def attention_reference(query, key, value):
    """Evaluate dense, non-causal attention for equal-shaped [H, N, D] inputs.

    This deliberately materializes the attention matrix and is intended for
    correctness checks, not large-input benchmarks. Accumulation uses FP32 for
    FP16/BF16 inputs and retains FP64 when checking derivatives numerically.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Q, K and V must share shape")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Q, K and V must share device")
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    scores = torch.matmul(query.to(dtype), key.to(dtype).transpose(-1, -2))
    scores = scores / math.sqrt(query.shape[-1])
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value.to(dtype))

def reference_gradients(query, key, value, upstream):
    """Recompute dense probabilities and return all three attention gradients."""
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    q, k, v = (item.to(dtype) for item in (query, key, value))
    scale = query.shape[-1] ** -0.5
    probabilities = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, -1)
    upstream = upstream.to(dtype)
    probability_gradient = torch.matmul(upstream, v.transpose(-1, -2))
    row_contraction = (probabilities * probability_gradient).sum(-1, keepdim=True)
    score_gradient = probabilities * (probability_gradient - row_contraction)
    return (
        (torch.matmul(score_gradient, k) * scale).to(query.dtype),
        (torch.matmul(score_gradient.transpose(-1, -2), q) * scale).to(key.dtype),
        torch.matmul(probabilities.transpose(-1, -2), upstream).to(value.dtype),
    )

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
        if seq % 32 or dim not in (16, 32, 64, 128):
            raise ValueError("This version requires N divisible by 32 and D=16/32/64/128")
        output = torch.empty(query.shape, device=query.device, dtype=FlashAttention2.OUTPUT_DTYPE)
        partition = torch.empty((heads, seq), device=query.device, dtype=torch.float32)
        output_sum = torch.empty_like(partition)
        q, k, v = query, key, value
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
        return reference_gradients(query, key, value, upstream)
