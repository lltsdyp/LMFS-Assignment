"""Real-text validation using total target NLL and effective token count."""

import time

import torch

from src.training.lm.data import validation_batches
from src.utils.losses import cross_entropy
from src.utils.metrics import perplexity


@torch.no_grad()
def evaluate(model, tokens, batch_size, device) -> dict:
    """Evaluate every real next-token target and restore the caller's mode."""
    started = time.perf_counter()
    was_training = model.training
    total_nll, count = 0.0, 0
    model.eval()
    try:
        for inputs, targets in validation_batches(
            tokens, model.config.context_length, batch_size, device
        ):
            nll = cross_entropy(model(inputs), targets, reduction="none")
            total_nll += nll.to(torch.float64).sum().item()
            count += (targets != -100).sum().item()
    finally:
        model.train(was_training)
    if count != len(tokens) - 1:
        raise RuntimeError("validation target coverage is incomplete")
    return {
        "nll": total_nll,
        "target_tokens": count,
        "ppl": perplexity(total_nll, count),
        "seconds": time.perf_counter() - started,
    }
