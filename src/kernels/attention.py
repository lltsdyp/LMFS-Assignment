"""Rotary position encoding and causal multi-head self-attention."""

import torch
from torch import Tensor, nn

from src.kernels.functional import (
    rotary_embedding,
    scaled_dot_product_attention,
)
from src.kernels.layers import Linear


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent feature pairs of (..., seq_len, d_k) tensors.

    Positions may be (seq_len,), (batch, seq_len), or (..., seq_len).
    Missing intermediate dimensions (e.g. heads) are broadcast. Caches
    move with the module and are reconstructed rather than checkpointed.
    """

    def __init__(self, theta, d_k, max_seq_len, device=None, dtype=None):
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError("RoPE head dimension must be positive and even")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        cache_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        cos, sin = self._make_cache(device, cache_dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _make_cache(self, device, cache_dtype):
        features = torch.arange(
            0, self.d_k, 2, device=device, dtype=cache_dtype
        )
        frequencies = self.theta ** (-features / self.d_k)
        positions = torch.arange(
            self.max_seq_len, device=device, dtype=cache_dtype
        )
        angles = positions[:, None] * frequencies[None, :]
        return torch.cos(angles), torch.sin(angles)

    def _apply(self, fn, recurse=True):
        """Preserve accurate caches when model.to changes parameter dtype."""
        super()._apply(fn, recurse=recurse)
        cache_dtype = (
            torch.float64 if self.cos.dtype == torch.float64 else torch.float32
        )
        self.cos, self.sin = self._make_cache(self.cos.device, cache_dtype)
        return self

    def forward(
        self, x: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError("input's last dimension must match d_k")
        if token_positions is None:
            if x.shape[-2] > self.max_seq_len:
                raise ValueError("sequence exceeds RoPE cache length")
            token_positions = torch.arange(x.shape[-2], device=x.device)
        else:
            if token_positions.dtype not in (torch.int32, torch.int64):
                raise TypeError("token_positions must have integer dtype")
            if token_positions.shape[-1] != x.shape[-2]:
                raise ValueError("positions must match input sequence length")
            if torch.any(token_positions < 0) or torch.any(
                token_positions >= self.max_seq_len
            ):
                raise ValueError("token positions are outside the RoPE cache")
        return rotary_embedding(x, self.cos, self.sin, token_positions)


class MultiHeadSelfAttention(nn.Module):
    """Causal MHA with separate Q/K/V projections and RoPE on Q/K."""

    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len,
        theta=10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError(
                "d_model must be positive and divisible by num_heads"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            theta, self.head_dim, max_seq_len, device=device, dtype=dtype
        )

    def forward(
        self, x: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        def split_heads(projected: Tensor) -> Tensor:
            return projected.reshape(
                *x.shape[:-1], self.num_heads, self.head_dim
            ).transpose(-3, -2)

        q, k, v = (
            split_heads(self.q_proj(x)),
            split_heads(self.k_proj(x)),
            split_heads(self.v_proj(x)),
        )
        q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        attended = scaled_dot_product_attention(q, k, v, is_causal=True)
        merged = attended.transpose(-3, -2).reshape(*x.shape[:-1], self.d_model)
        return self.output_proj(merged)
