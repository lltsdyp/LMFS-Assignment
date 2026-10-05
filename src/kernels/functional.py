"""Handwritten model-forward formulas using PyTorch tensor operations.

Feature and vocabulary dimensions are last. PyTorch selects each basic
operation's device implementation; inputs are never implicitly moved.
"""

import math

import torch
from torch import Tensor

from src.utils.tensors import check_same_device, compute_dtype


def linear(x: Tensor, weight: Tensor) -> Tensor:
    """Apply a bias-free projection with weight shaped (out, in)."""
    check_same_device(x, weight)
    return x @ weight.transpose(-1, -2)


def embedding(token_ids: Tensor, weight: Tensor) -> Tensor:
    """Look up rows of (vocab_size, d_model), rejecting negative IDs."""
    check_same_device(token_ids, weight)
    return weight.index_select(0, token_ids.reshape(-1)).reshape(
        *token_ids.shape, weight.shape[1]
    )


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize the last dimension and apply a learnable gain."""
    check_same_device(x, weight)
    # Nonfinite epsilon can produce NaNs or silently zero every activation.
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must match the last input dimension")
    work = x.to(compute_dtype(x))
    normalized = work * (work.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (normalized * weight.to(work.dtype)).to(x.dtype)


def silu(x: Tensor) -> Tensor:
    """Compute x * sigmoid(x) without a built-in activation or overflow."""
    work = x.to(compute_dtype(x))
    negative_exp = torch.exp(work.clamp_max(0))
    sigmoid = torch.where(
        work >= 0,
        1 / (1 + torch.exp(-work.clamp_min(0))),
        negative_exp / (1 + negative_exp),
    )
    return (work * sigmoid).to(x.dtype)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Stable softmax; entirely -inf masked rows produce zeros.

    Accumulation is in float32 for float16/bfloat16 inputs. Inputs should
    be finite scores or -inf masks, without NaN or positive infinity.
    """
    work = x.to(compute_dtype(x))
    maximum = work.amax(dim=dim, keepdim=True)
    fully_masked = torch.isneginf(maximum)
    maximum = torch.where(fully_masked, 0.0, maximum)
    numerator = torch.exp(work - maximum)
    denominator = numerator.sum(dim=dim, keepdim=True)
    denominator = torch.where(fully_masked, 1.0, denominator)
    return (numerator / denominator).to(x.dtype)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    is_causal: bool = False,
) -> Tensor:
    """Attention on (..., queries/keys, head_dim) with broadcastable masks.

    A boolean mask allows True positions; a floating mask adds score bias.
    The causal mask is upper-left aligned when sequence lengths differ.
    """
    check_same_device(q, k, v, *(() if mask is None else (mask,)))
    work_q, work_k = q.to(compute_dtype(q)), k.to(compute_dtype(k))
    scores = (work_q @ work_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if is_causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=q.device)
        allowed = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~allowed, float("-inf"))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        elif mask.is_floating_point():
            scores = scores + mask
        else:
            raise TypeError("mask must be boolean or floating point")
    probabilities = softmax(scores, dim=-1)
    return (probabilities @ v.to(probabilities.dtype)).to(v.dtype)


def rotary_embedding(
    x: Tensor, cos: Tensor, sin: Tensor, token_positions: Tensor
) -> Tensor:
    """Apply adjacent-pair rotations using caller-owned position caches."""
    check_same_device(x, cos, sin, token_positions)
    selected_cos, selected_sin = cos[token_positions], sin[token_positions]
    while selected_cos.ndim < x.ndim:
        selected_cos = selected_cos.unsqueeze(-3)
        selected_sin = selected_sin.unsqueeze(-3)
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    even, odd = x[..., 0::2].to(work_dtype), x[..., 1::2].to(work_dtype)
    rotated = torch.stack(
        (
            even * selected_cos - odd * selected_sin,
            even * selected_sin + odd * selected_cos,
        ),
        dim=-1,
    )
    return rotated.flatten(-2).to(x.dtype)
