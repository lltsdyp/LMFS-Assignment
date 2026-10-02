"""Provides ranked byte merging and cached pre-token encoding helpers.

Vocabulary lookups, merge ranks, and the cache belong to the caller. This
module only operates on explicitly supplied data and local merge buffers.
"""

from __future__ import annotations

from collections.abc import Mapping
import heapq

_MAX_CACHE_ENTRIES = 4096
_MAX_CACHE_WORD_BYTES = 1024


def encode_word(
    word: bytes,
    token_to_id: Mapping[bytes, int],
    merge_ranks: Mapping[tuple[bytes, bytes], int],
    encoding_cache: dict[bytes, tuple[int, ...]],
) -> tuple[int, ...]:
    """Encodes one byte pre-token using a bounded caller-owned cache.

    Args:
        word: UTF-8 bytes of one ordinary pre-token.
        token_to_id: Vocabulary bytes mapped to token IDs.
        merge_ranks: Learned pairs mapped to their training order.
        encoding_cache: Shared cache for eager and lazy encoding. Stores up to
            4096 pre-tokens of at most 1024 bytes and is cleared when full.

    Returns:
        Immutable token IDs for this pre-token.
    """
    ids = encoding_cache.get(word)
    if ids is None:
        ids = merge_word(word, token_to_id, merge_ranks)
        if len(word) <= _MAX_CACHE_WORD_BYTES:
            if len(encoding_cache) >= _MAX_CACHE_ENTRIES:
                encoding_cache.clear()
            encoding_cache[word] = ids
    return ids


def merge_word(
    word: bytes,
    token_to_id: Mapping[bytes, int],
    merge_ranks: Mapping[tuple[bytes, bytes], int],
) -> tuple[int, ...]:
    """Merges adjacent symbols by rank, choosing leftmost pairs on ties.

    Follows the ranked queue in tk-encode/src/models/bpe/
    merge_hot_cold_queue.rs in /data/jizimo/tokenizers. Parallel arrays preserve
    original positions while stale heap candidates are discarded.

    Args:
        word: Bytes of one ordinary pre-token.
        token_to_id: Vocabulary bytes mapped to token IDs.
        merge_ranks: Learned pairs mapped to their training order.

    Returns:
        Token IDs after applying all eligible ranked merges.
    """
    # Keep the word's temporary merge buffers local to this operation.
    # pylint: disable=too-many-locals
    symbols = [bytes((value,)) for value in word]
    previous = list(range(-1, len(symbols) - 1))
    following = list(range(1, len(symbols))) + [-1]
    heap: list[tuple[int, int, int]] = []
    for left in range(len(symbols) - 1):
        _push_ranked_pair(heap, symbols, following, left, merge_ranks)

    while heap:
        rank, left, right = heapq.heappop(heap)
        if (
            not symbols[left]
            or following[left] != right
            or merge_ranks.get((symbols[left], symbols[right])) != rank
        ):
            continue
        symbols[left] += symbols[right]
        symbols[right] = b""
        following[left] = following[right]
        if following[right] != -1:
            previous[following[right]] = left
        _push_ranked_pair(heap, symbols, following, previous[left], merge_ranks)
        _push_ranked_pair(heap, symbols, following, left, merge_ranks)

    return tuple(token_to_id[symbol] for symbol in symbols if symbol)


def _push_ranked_pair(
    heap: list[tuple[int, int, int]],
    symbols: list[bytes],
    following: list[int],
    left: int,
    merge_ranks: Mapping[tuple[bytes, bytes], int],
) -> None:
    """Queues a mergeable adjacent pair by rank and original left position.

    Args:
        heap: Inference candidate heap, updated in place.
        symbols: Current byte symbols at their original word positions.
        following: Linked-array indices of right neighbors, or -1 at the end.
        left: Position of the candidate's left symbol, or -1 for no candidate.
        merge_ranks: Learned pairs mapped to their training order.
    """
    if left == -1 or following[left] == -1:
        return
    right = following[left]
    rank = merge_ranks.get((symbols[left], symbols[right]))
    if rank is not None:
        heapq.heappush(heap, (rank, left, right))
