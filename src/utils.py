"""Provides stream, pair index, and heap helper functions.

Tokenizer rules are supplied by the caller. Each training data class lives in
its own module, and the BPE class owns the overall training algorithm.
"""

from __future__ import annotations

import collections
from collections.abc import Iterator
import heapq
from typing import TYPE_CHECKING, TextIO

import regex

if __package__:
    from src import heap_entry
    from src import pair_state
else:
    import heap_entry
    import pair_state

if TYPE_CHECKING:
    if __package__:
        from src import training_words
    else:
        import training_words


def _compile_special_pattern(special_tokens: list[str]) -> regex.Pattern | None:
    """Builds a literal pattern that prefers longer special tokens."""
    if not special_tokens:
        return None
    ordered_tokens = sorted(special_tokens, key=len, reverse=True)
    return regex.compile(
        "|".join(regex.escape(token) for token in ordered_tokens)
    )


def iter_pretokens(
    stream: TextIO,
    special_tokens: list[str],
    chunk_size: int,
    *,
    pretokenizer: regex.Pattern,
) -> Iterator[bytes]:
    """Yields pre-token bytes while preserving regex boundaries across reads.

    Two trailing matches are retained because completing a contraction such as
    "'l" can change both matches. Incomplete special tokens and whitespace
    lookahead also need input from the next read. A single long pre-token may
    exceed chunk_size; splitting it arbitrarily would change BPE training.

    Args:
        stream: A decoded text stream.
        special_tokens: Literal delimiters excluded from ordinary pre-tokens.
        chunk_size: The number of decoded characters to read at a time.
        pretokenizer: The compiled pattern supplied by the tokenizer.

    Yields:
        UTF-8 byte strings for ordinary pre-tokens.
    """
    special_pattern = _compile_special_pattern(special_tokens)
    special_width = max(map(len, special_tokens), default=0)
    pending = ""
    while True:
        chunk = stream.read(chunk_size)
        pending += chunk
        segment_start = 0
        if special_pattern is not None:
            for match in special_pattern.finditer(pending):
                # A short special token might still become a longer one.
                if chunk and match.start() > len(pending) - special_width:
                    break
                for word in pretokenizer.finditer(
                    pending, segment_start, match.start()
                ):
                    yield word.group().encode("utf-8")
                segment_start = match.end()

        if not chunk:
            for word in pretokenizer.finditer(pending, segment_start):
                yield word.group().encode("utf-8")
            return

        trailing_matches = collections.deque()
        consumed = segment_start
        safe_end = len(pending) - max(special_width - 1, 0)
        for word in pretokenizer.finditer(pending, segment_start):
            trailing_matches.append(word)
            if len(trailing_matches) <= 2:
                continue
            ready = trailing_matches.popleft()
            # Whitespace lookahead must precede an incomplete special token.
            if ready.end() + 1 >= safe_end:
                break
            yield ready.group().encode("utf-8")
            consumed = ready.end()
        pending = pending[consumed:]


def initialize_pair_states(
    words: training_words.TrainingWords,
) -> dict[int, pair_state.PairState]:
    """Builds weighted pair counts and candidate word postings."""
    states: dict[int, pair_state.PairState] = {}
    for word_id, frequency in enumerate(words.frequencies):
        for pair_id, occurrences in words.count_pairs(word_id).items():
            if pair_id in states:
                states[pair_id].count += occurrences * frequency
                states[pair_id].add(word_id)
            else:
                states[pair_id] = pair_state.PairState(
                    occurrences * frequency, word_id
                )
    return states


def build_pair_heap(
    states: dict[int, pair_state.PairState], vocab: dict[int, bytes]
) -> list[heap_entry.HeapEntry]:
    """Builds a heap with one current entry for each active pair."""
    heap = [
        heap_entry.HeapEntry(state.count, pair_id, vocab)
        for pair_id, state in states.items()
    ]
    heapq.heapify(heap)
    return heap


def pop_best_pair(
    heap: list[heap_entry.HeapEntry],
    states: dict[int, pair_state.PairState],
) -> heap_entry.HeapEntry:
    """Pops the best valid pair, discarding stale heap entries."""
    while heap:
        entry = heapq.heappop(heap)
        state = states.get(entry.pair_id)
        if state is not None and state.count == entry.count:
            return entry
    raise RuntimeError("pair heap lost an active pair")
