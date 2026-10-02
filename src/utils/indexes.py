"""Builds encoding lookups from vocabulary, merges, and special tokens."""

from __future__ import annotations

import dataclasses

from src.utils import pretokenization


@dataclasses.dataclass(frozen=True, slots=True)
class EncodingIndexes:
    """Groups derived lookup tables without owning model data or caches.

    Attributes:
        token_to_id: Vocabulary byte strings mapped to token IDs.
        merge_ranks: Byte pairs mapped to their first learned merge rank.
        special_token_ids: Literal special-token strings mapped to token IDs.
    """

    token_to_id: dict[bytes, int]
    merge_ranks: dict[tuple[bytes, bytes], int]
    special_token_ids: dict[str, int]


def build_encoding_indexes(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> EncodingIndexes:
    """Builds lookup tables and appends missing special tokens to vocab.

    Args:
        vocab: Model vocabulary, updated in place when markers are missing.
            New IDs follow the greatest existing ID.
        merges: Learned byte pairs in merge order. Duplicate pairs retain
            their first rank.
        special_tokens: Literal strings reserved as indivisible tokens.

    Returns:
        Vocabulary, merge-rank, and special-token lookup tables.

    Raises:
        ValueError: If a special token is empty or is not a string.
    """
    pretokenization.validate_special_tokens(special_tokens)
    token_to_id = {token: token_id for token_id, token in vocab.items()}
    merge_ranks = {}
    for rank, pair in enumerate(merges):
        merge_ranks.setdefault(pair, rank)
    special_token_ids = {}
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in token_to_id:
            token_id = max(vocab, default=-1) + 1
            vocab[token_id] = token_bytes
            token_to_id[token_bytes] = token_id
        special_token_ids[special_token] = token_to_id[token_bytes]
    return EncodingIndexes(
        token_to_id=token_to_id,
        merge_ranks=merge_ranks,
        special_token_ids=special_token_ids,
    )
