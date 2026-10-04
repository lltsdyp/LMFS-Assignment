"""Handwritten forward tensor operations, layers, and attention components.

Uses PyTorch tensor operations and autograd, not built-in neural layers,
functional implementations, optimizers, schedulers, or gradient clipping.
"""

from src.kernels import functional
from src.kernels.attention import (
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
)
from src.kernels.functional import (
    scaled_dot_product_attention,
    softmax,
)
from src.kernels.layers import Embedding, Linear, RMSNorm, SiLU, SwiGLU

__all__ = [
    "functional",
    "Linear",
    "Embedding",
    "RMSNorm",
    "SiLU",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "MultiHeadSelfAttention",
    "softmax",
    "scaled_dot_product_attention",
]
