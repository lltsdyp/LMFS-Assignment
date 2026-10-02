"""Tracks BPE training pairs and orders merge candidates by frequency.

PairState owns weighted counts and word postings. HeapEntry defines heap tie
breaking. Helper functions build counts and discard stale merge candidates.
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.training import words

_TOKEN_ID_BITS = 32
_TOKEN_ID_MASK = (1 << _TOKEN_ID_BITS) - 1


@dataclasses.dataclass(slots=True)
class PairState:
    """Tracks a pair's weighted frequency and candidate word postings.

    Attributes:
        count: The global count, weighted by word frequency.
        word_ids: A single word ID or a set of candidate IDs. As in HF, stale
            postings are harmless because merging a word without the pair is
            a no-op. The postings are freed when the pair is retired.
    """

    count: int
    # Rare pairs avoid allocating a set for a single posting.
    word_ids: int | set[int]

    def add(self, word_id: int) -> None:
        """Registers a word containing this pair without duplicate postings."""
        if isinstance(self.word_ids, int):
            if self.word_ids != word_id:
                self.word_ids = {self.word_ids, word_id}
        else:
            self.word_ids.add(word_id)

    def __iter__(self) -> Iterator[int]:
        """Yields candidate IDs from singleton or set postings."""
        if isinstance(self.word_ids, int):
            yield self.word_ids
        else:
            yield from self.word_ids


class HeapEntry:
    """Orders pair snapshots by descending count and byte-pair lexicography.

    Attributes:
        count: The weighted frequency at the time the entry was inserted.
        pair_id: Two token IDs packed into one unsigned 64-bit integer.
    """

    __slots__ = ("count", "pair_id", "_vocab")

    def __init__(
        self, count: int, pair_id: int, vocab: dict[int, bytes]
    ) -> None:
        self.count = count
        self.pair_id = pair_id
        self._vocab = vocab

    def __lt__(self, other: HeapEntry) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair_bytes() > other.pair_bytes()

    def pair_bytes(self) -> tuple[bytes, bytes]:
        """Returns the byte strings of the two tokens in this pair."""
        return (
            self._vocab[self.pair_id >> _TOKEN_ID_BITS],
            self._vocab[self.pair_id & _TOKEN_ID_MASK],
        )


def initialize_pair_states(
    training_words: words.TrainingWords,
) -> dict[int, PairState]:
    """Builds weighted pair counts and candidate word postings."""
    states: dict[int, PairState] = {}
    for word_id, frequency in enumerate(training_words.frequencies):
        for pair_id, occurrences in training_words.count_pairs(word_id).items():
            if pair_id in states:
                states[pair_id].count += occurrences * frequency
                states[pair_id].add(word_id)
            else:
                states[pair_id] = PairState(occurrences * frequency, word_id)
    return states


def build_pair_heap(
    states: dict[int, PairState], vocab: dict[int, bytes]
) -> list[HeapEntry]:
    """Builds a heap with one current entry for each active pair."""
    heap = [
        HeapEntry(state.count, pair_id, vocab)
        for pair_id, state in states.items()
    ]
    heapq.heapify(heap)
    return heap


def pop_best_pair(
    heap: list[HeapEntry],
    states: dict[int, PairState],
) -> HeapEntry:
    """Pops the best valid pair, discarding stale heap entries."""
    while heap:
        entry = heapq.heappop(heap)
        state = states.get(entry.pair_id)
        if state is not None and state.count == entry.count:
            return entry
    raise RuntimeError("pair heap lost an active pair")
