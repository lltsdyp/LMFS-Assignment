"""Formal BPE experiment scaffold.

The runner follows the handout's required timing boundary. It depends on the
student implementation in :mod:`bpe`; W&B table construction and logging are
left as TODOs so they can be connected to the course's ``wandb_record.py``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bpe import BPE


def run_bpe_experiment(config: Mapping[str, Any], run: Any) -> dict[str, Any]:
    """Train, reload, and fully validate BPE; return the required metrics.

    Expected config keys: ``train_file``, ``valid_file``, ``vocab_size``, and
    ``special_tokens``. The function is intended to run inside the W&B
    ``experiment("bpe", config)`` context described in the handout.
    """
    train_file = Path(config["train_file"])
    valid_file = Path(config["valid_file"])
    vocab_size = int(config["vocab_size"])
    special_tokens = list(config["special_tokens"])

    # TODO: Validate the paths and record the exact configuration in the run.
    bpe = BPE()
    bpe.train(train_file, vocab_size, special_tokens)

    # TODO: Use the required outputs/<run_id>/tokenizer prefix and ensure its
    # parent directory exists before saving.
    tokenizer_prefix = Path("outputs") / str(run.id) / "tokenizer"
    bpe.save(tokenizer_prefix)

    # Reload before timing so training or any earlier encoding cannot populate
    # caches used by the official validation measurement.
    bpe = BPE()
    bpe.load(tokenizer_prefix)
    with valid_file.open("r", encoding="utf-8", newline="") as stream:
        validation_text = stream.read()

    # The timed interval must include the full encode, full decode, and exact
    # round-trip comparison. File reading and tokenizer loading stay outside it.
    started = time.perf_counter()
    token_ids = bpe.encode(validation_text)
    decoded_text = bpe.decode(token_ids)
    roundtrip_passed = decoded_text == validation_text
    valid_seconds = time.perf_counter() - started

    # TODO: After timing, build the required first-200-token detail rows and
    # log them as a W&B Table. Do not include this work in valid_seconds.
    # TODO: Update W&B summary with the official metric names and print the
    # result as JSON, following the handout's train_bpe.py example.
    if len(token_ids) < 200 or not roundtrip_passed:
        raise ValueError("validation needs at least 200 tokens and an exact round trip")

    return {
        "bpe/valid_seconds": valid_seconds,
        "bpe/valid_token_count": len(token_ids),
        "bpe/roundtrip_passed": roundtrip_passed,
        "bpe/vocab_size": len(bpe.vocab),
    }
