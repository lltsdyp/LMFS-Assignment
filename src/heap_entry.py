"""Defines the BPE pair ordering used by the standard-library heapq module."""

from __future__ import annotations

_TOKEN_ID_BITS = 32
_TOKEN_ID_MASK = (1 << _TOKEN_ID_BITS) - 1


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
