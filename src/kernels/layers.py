"""Basic learnable layers with CS336 parameter initializations."""

import math

import torch
from torch import Tensor, nn

from src.kernels import functional


class Linear(nn.Module):
    """Bias-free linear layer, storing weight as (out_features, in_features)."""

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return functional.linear(x, self.weight)


class Embedding(nn.Module):
    """Token lookup with truncated N(0, 1) initialization."""

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return functional.embedding(token_ids, self.weight)


class RMSNorm(nn.Module):
    """Last-dimension RMS normalization with gain initialized to one."""

    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        return functional.rms_norm(x, self.weight, self.eps)


class SiLU(nn.Module):
    """Parameter-free, manually computed SiLU activation."""

    def forward(self, x: Tensor) -> Tensor:
        return functional.silu(x)


def default_ffn_dim(d_model: int) -> int:
    """Nearest positive multiple of 64 to (8/3) * d_model."""
    return max(64, 64 * round((8 * d_model / 3) / 64))


class SwiGLU(nn.Module):
    """Gated FFN: w2(SiLU(w1(x)) * w3(x))."""

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = default_ffn_dim(d_model) if d_ff is None else d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(functional.silu(self.w1(x)) * self.w3(x))
