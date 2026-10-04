"""Model assembly from handwritten layers: pre-norm blocks and a decoder LM."""

from dataclasses import dataclass

from torch import Tensor, nn

from src.kernels.attention import MultiHeadSelfAttention
from src.kernels.layers import (
    Embedding,
    Linear,
    RMSNorm,
    SwiGLU,
    default_ffn_dim,
)


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture metadata; serialize with dataclasses.asdict."""

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    norm_eps: float


class TransformerBlock(nn.Module):
    """Two pre-norm residual sublayers: causal attention and SwiGLU."""

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        context_length,
        rope_theta=10000.0,
        norm_eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model, norm_eps, device, dtype)
        self.attention = MultiHeadSelfAttention(
            d_model, num_heads, context_length, rope_theta, device, dtype
        )
        self.ffn_norm = RMSNorm(d_model, norm_eps, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self, x: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        x = x + self.attention(self.attention_norm(x), token_positions)
        return x + self.ffn(self.ffn_norm(x))


class TransformerLM(nn.Module):
    """Decoder-only LM returning all-position logits (batch, seq, vocab).

    Embedding and output weights are independent. No pretrained weights,
    padding, BOS/EOS insertion, loss, or tokenizer is hidden in forward.
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff=None,
        rope_theta=10000.0,
        norm_eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive")
        d_ff = default_ffn_dim(d_model) if d_ff is None else d_ff
        self.config = TransformerConfig(
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            norm_eps,
        )
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                norm_eps,
                device,
                dtype,
            )
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model, norm_eps, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(
        self, token_ids: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, seq_len)")
        if not 0 < token_ids.shape[1] <= self.config.context_length:
            raise ValueError("sequence length must be within context_length")
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.final_norm(x))
