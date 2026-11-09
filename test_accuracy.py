"""Accuracy regressions; enable the full A shape with FA2_FULL_A=1.

The default suite is deliberately small. The full configuration requires a
CUDA device with enough memory for the reference as well as the implementation.
These checks are not a performance benchmark or an official evaluator result.
"""

import os

import pytest
import torch

from fa2_re import FlashAttention2


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
CONFIGURATION_A = (24, 16384, 128)


def dense_reference(query, key, value):
    """Use FP32 softmax and products independently of the Triton implementation."""
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    return torch.matmul(torch.softmax(scores, -1), value.float())


def make_inputs(shape, dtype, strided=False):
    torch.manual_seed(17)
    heads, seq, dim = shape
    if strided:
        return tuple(
            torch.randn(heads, dim, seq, device="cuda", dtype=dtype)
            .transpose(1, 2)
            .requires_grad_()
            for _ in range(3)
        )
    return tuple(
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(3)
    )


def compare_attention(inputs, broadcast):
    actual = FlashAttention2.apply(*inputs)
    expected_inputs = tuple(item.detach().float().requires_grad_() for item in inputs)
    expected = dense_reference(*expected_inputs)
    if broadcast:
        upstream = torch.ones((), device="cuda", dtype=actual.dtype).expand_as(actual)
    else:
        upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, inputs, upstream)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, upstream.float())
    # Absolute tolerances account for FP16 operands/output and low-magnitude tails.
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-2)
    for computed, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(computed.float(), reference, atol=4e-3, rtol=3e-2)


@pytest.mark.parametrize("shape", [(2, 64, 64), (2, 129, 100), (1, 257, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("broadcast", [False, True])
def test_attention(shape, dtype, broadcast):
    compare_attention(make_inputs(shape, dtype), broadcast)


@pytest.mark.parametrize("broadcast", [False, True])
def test_noncontiguous_inputs(broadcast):
    compare_attention(make_inputs((2, 65, 100), torch.float32, strided=True), broadcast)


def test_shape_mismatch():
    query, key, value = make_inputs((1, 32, 64), torch.float32)
    with pytest.raises(ValueError, match="shape"):
        FlashAttention2.apply(query, key[:, :-1], value)


@pytest.mark.skipif(os.environ.get("FA2_FULL_A") != "1", reason="opt-in full A check")
def test_configuration_a():
    """Compare all heads and all gradients; keep only one reference head at a time."""
    inputs = make_inputs(CONFIGURATION_A, torch.float32)
    actual = FlashAttention2.apply(*inputs)
    gradients = torch.autograd.grad(actual.sum(), inputs)
    for head in range(CONFIGURATION_A[0]):
        reference_inputs = tuple(
            item[head:head + 1].detach().requires_grad_() for item in inputs
        )
        reference = dense_reference(*reference_inputs)
        reference_gradients = torch.autograd.grad(reference.sum(), reference_inputs)
        torch.testing.assert_close(
            actual[head:head + 1].float(), reference, atol=2e-3, rtol=2e-2
        )
        for computed, expected in zip(gradients, reference_gradients):
            torch.testing.assert_close(
                computed[head:head + 1].float(), expected, atol=4e-3, rtol=3e-2
            )
        del reference, reference_gradients, reference_inputs


@pytest.fixture(autouse=True)
def precise_torch_reference():
    """Do not let ambient TF32 settings weaken the independent reference."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
