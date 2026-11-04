import math

import torch


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


class FlashAttention2(torch.autograd.Function):
    """Reference implementation with the evaluator's autograd entry point."""

    INPUT_DTYPE = torch.float32
    OUTPUT_DTYPE = torch.float16

    @staticmethod
    def forward(ctx, query, key, value):
        output = attention_reference(query, key, value)
        ctx.save_for_backward(query, key, value)
        return output.to(FlashAttention2.OUTPUT_DTYPE)

    @staticmethod
    def backward(ctx, upstream):
        return reference_gradients(*ctx.saved_tensors, upstream)


def check_reference():
    """Check the explicit backward against PyTorch autograd on the CPU."""
    torch.manual_seed(0)
    inputs = tuple(
        torch.randn(2, 17, 16, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    output = attention_reference(*inputs)
    upstream = torch.randn_like(output)
    expected = torch.autograd.grad(output, inputs, upstream)
    actual = reference_gradients(*inputs, upstream)
    for computed, reference in zip(actual, expected):
        torch.testing.assert_close(computed, reference, atol=1e-12, rtol=1e-12)
    # A summed loss supplies a zero-stride upstream tensor.
    broadcast = torch.ones((), dtype=output.dtype).expand_as(output)
    expected_output = attention_reference(*inputs)
    expected = torch.autograd.grad(expected_output, inputs, broadcast)
    actual = reference_gradients(*inputs, broadcast)
    for computed, reference in zip(actual, expected):
        torch.testing.assert_close(computed, reference, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    check_reference()
