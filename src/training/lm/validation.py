"""Real-text validation using total target NLL and effective token count."""

import time

import torch

from src.training.lm.data import validation_batches
from src.utils.losses import cross_entropy
from src.utils.metrics import perplexity


@torch.no_grad()
def evaluate(model, tokens, batch_size, device, *, autocast_dtype=None) -> dict:
    """Evaluate every real next-token target and restore the caller's mode."""
    started = time.perf_counter()
    was_training = model.training
    total_nll, count = 0.0, 0
    model.eval()
    try:
        for inputs, targets in validation_batches(
            tokens, model.config.context_length, batch_size, device
        ):
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits = model(inputs)
            nll = cross_entropy(logits, targets, reduction="none")
            total_nll += nll.to(torch.float64).sum().item()
            count += (targets != -100).sum().item()
    finally:
        model.train(was_training)
    return {
        "nll": total_nll,
        "target_tokens": count,
        "ppl": perplexity(total_nll, count),
        "seconds": time.perf_counter() - started,
    }
