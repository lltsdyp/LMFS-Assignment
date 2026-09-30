"""Tracks a BPE pair's weighted frequency and candidate word postings."""

from collections.abc import Iterator
import dataclasses


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
