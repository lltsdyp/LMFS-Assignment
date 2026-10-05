"""Handwritten stable loss calculations for training and validation."""

import torch
from torch import Tensor

from src.utils.tensors import check_same_device, compute_dtype


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> Tensor:
    """Next-token NLL for logits (..., vocab_size) and targets (...).

    Supports 'none', 'sum' and 'mean'. Ignored positions return zero and
    are excluded from the mean. An entirely ignored/empty batch returns
    differentiable zero. Use 'sum' and count targets for corpus PPL.
    """
    check_same_device(logits, targets)
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits' leading dimensions")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    work = logits.to(compute_dtype(logits))
    shifted = work - work.amax(dim=-1, keepdim=True)
    valid = targets != ignore_index
    safe_targets = torch.where(valid, targets, 0)
    target_scores = shifted.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    nll = torch.exp(shifted).sum(dim=-1).log() - target_scores
    nll = torch.where(valid, nll, 0.0)
    if reduction == "none":
        return nll
    total = nll.sum()
    return total if reduction == "sum" else total / valid.sum().clamp_min(1)
