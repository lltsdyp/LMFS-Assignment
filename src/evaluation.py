"""Measures full-text BPE validation and builds token inspection tables."""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src import bpe

TOKEN_COLUMNS = [
    "position",
    "token_id",
    "token_hex",
    "token_bytes",
    "decoded_token",
    "standalone_utf8_valid",
    "byte_start",
    "byte_end",
    "matches_input_bytes",
]


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Contains measured results and the first 200 encoded IDs.

    Attributes:
        seconds: Total encode, decode, and text comparison duration.
        token_count: Number of IDs in the complete encoded validation text.
        passed: Whether the entire decoded text exactly matches its input.
        first_ids: At most the first 200 IDs, sliced after timing ends.
    """

    seconds: float
    token_count: int
    passed: bool
    first_ids: list[int]


def validate(tokenizer: bpe.BPE, text: str) -> ValidationResult:
    """Times complete encode, complete decode, and exact text comparison.

    Args:
        tokenizer: Freshly loaded tokenizer with no validation cache warmup.
        text: Complete validation text read before timing starts.

    Returns:
        Timing and correctness results, excluding table generation and I/O.
    """
    started = time.perf_counter()
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    passed = decoded_text == text
    seconds = time.perf_counter() - started
    return ValidationResult(seconds, len(ids), passed, ids[:200])


def token_rows(
    tokenizer: bpe.BPE, text: str, ids: list[int]
) -> list[list[object]]:
    """Builds byte offsets and individual-token UTF-8 details after timing.

    Args:
        tokenizer: Model whose vocabulary supplied the IDs.
        text: Original full text used for matching raw byte slices.
        ids: The leading token IDs to inspect.

    Returns:
        Rows corresponding to TOKEN_COLUMNS, with half-open byte offsets.
    """
    raw = text.encode("utf-8")
    rows = []
    offset = 0
    for position, token_id in enumerate(ids):
        piece = tokenizer.vocab[token_id]
        end = offset + len(piece)
        try:
            piece.decode("utf-8", errors="strict")
            standalone_valid = True
        except UnicodeDecodeError:
            standalone_valid = False
        rows.append(
            [
                position,
                int(token_id),
                piece.hex(),
                repr(piece),
                repr(tokenizer.decode([token_id])),
                standalone_valid,
                offset,
                end,
                piece == raw[offset:end],
            ]
        )
        offset = end
    return rows
