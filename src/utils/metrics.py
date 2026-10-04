"""Corpus metrics computed from accumulated loss and effective counts."""

import math


def perplexity(total_nll: float, target_count: int) -> float:
    """Exponentiate token-weighted mean NLL, never a mean of batch PPLs."""
    if target_count <= 0:
        raise ValueError("perplexity requires at least one target token")
    if not math.isfinite(total_nll) or total_nll < 0:
        raise ValueError("total NLL must be finite and nonnegative")
    try:
        return math.exp(total_nll / target_count)
    except OverflowError:
        return float("inf")
