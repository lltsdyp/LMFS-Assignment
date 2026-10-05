"""Shares GPT-2 pre-tokenization between training and text encoding.

PreTokenizer owns literal special-token matching and ordinary-text rules.
Complete input is matched directly; streams retain incomplete word boundaries.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator
from typing import TextIO

import regex

PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_PRETOKENIZER = regex.compile(PAT)
_END_OF_INPUT = object()


def pretokenize(text: str) -> Iterator[str]:
    """Splits ordinary text into GPT-2 pre-tokens without changing its contents.

    Args:
        text: Ordinary text with special-token boundaries already isolated.

    Returns:
        Pre-token strings in their original order.

    Raises:
        TypeError: If text is not a string.
    """
    return (match.group(0) for match in _PRETOKENIZER.finditer(text))


class PreTokenizer:
    """Shares literal marker recognition and ordinary pre-token boundaries."""

    def __init__(self, special_tokens: list[str]) -> None:
        # Empty markers would match every position and break text boundaries.
        if "" in special_tokens:
            raise ValueError("special tokens must be non-empty strings")
        ordered_tokens = sorted(special_tokens, key=len, reverse=True)
        self._special_pattern = (
            regex.compile("|".join(regex.escape(t) for t in ordered_tokens))
            if ordered_tokens
            else None
        )
        self._special_width = max(map(len, special_tokens), default=0)

    def iter_text(self, text: str) -> Iterator[tuple[bytes, bool]]:
        """Yields UTF-8 bytes and special-token flags from complete text.

        Args:
            text: Complete input whose whitespace and markers are retained.

        Yields:
            Pre-token bytes and whether they represent a literal special token.

        Raises:
            TypeError: If text is not a string.
        """
        start = 0
        for match in self._iter_special_matches(text):
            yield from self._iter_ordinary(text, start, match.start())
            yield match.group().encode("utf-8"), True
            start = match.end()
        yield from self._iter_ordinary(text, start)

    def iter_chunks(
        self, chunks: Iterable[str]
    ) -> Iterator[tuple[bytes, bool]]:
        """Splits concatenated chunks without changing pre-token boundaries.

        Args:
            chunks: Consecutive string chunks. Empty chunks do not end input.

        Yields:
            The same bytes and flags as iter_text() on concatenated input.
            Two trailing regex matches and possible special-token prefixes are
            retained for whitespace lookahead and incomplete contractions.
            Long pre-tokens remain buffered until their boundary is known.

        Raises:
            TypeError: If a chunk is not a string.
        """
        pending = ""
        source = iter(_nonempty_chunks(chunks))
        while True:
            chunk = next(source, _END_OF_INPUT)
            if chunk is _END_OF_INPUT:
                yield from self.iter_text(pending)
                return
            pending += chunk
            segment_start = 0
            for match in self._iter_special_matches(pending, complete=False):
                yield from self._iter_ordinary(
                    pending, segment_start, match.start()
                )
                yield match.group().encode("utf-8"), True
                segment_start = match.end()

            trailing_matches = collections.deque()
            consumed = segment_start
            safe_end = len(pending) - max(self._special_width - 1, 0)
            for word in _PRETOKENIZER.finditer(pending, segment_start):
                trailing_matches.append(word)
                if len(trailing_matches) <= 2:
                    continue
                ready = trailing_matches.popleft()
                # Whitespace lookahead must precede an incomplete marker.
                if ready.end() + 1 >= safe_end:
                    break
                yield ready.group().encode("utf-8"), False
                consumed = ready.end()
            pending = pending[consumed:]

    def iter_stream(self, stream: TextIO, chunk_size: int) -> Iterator[bytes]:
        """Yields ordinary pre-token bytes for training, excluding markers.

        Args:
            stream: Decoded text stream with original whitespace retained.
            chunk_size: Number of characters to read at a time.

        Yields:
            Ordinary UTF-8 pre-tokens, preserving boundaries across reads.

        Raises:
            ValueError: If chunk_size is not positive.
            TypeError: If the stream yields non-string chunks.
        """
        for token, is_special in self.iter_chunks(
            _read_chunks(stream, chunk_size)
        ):
            if not is_special:
                yield token

    def _iter_special_matches(
        self, text: str, *, complete: bool = True
    ) -> Iterator[regex.Match]:
        """Defers trailing markers that may extend when more input arrives."""
        if self._special_pattern is None:
            return
        for match in self._special_pattern.finditer(text):
            if not complete and match.start() > len(text) - self._special_width:
                break
            yield match

    @staticmethod
    def _iter_ordinary(
        text: str, start: int, end: int | None = None
    ) -> Iterator[tuple[bytes, bool]]:
        """Matches ordinary text between already isolated special boundaries."""
        for word in _PRETOKENIZER.finditer(text, start, end):
            yield word.group().encode("utf-8"), False


def _read_chunks(stream: TextIO, chunk_size: int) -> Iterator[str]:
    """Reads text lazily without treating arbitrary read boundaries as words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    while chunk := stream.read(chunk_size):
        yield chunk


def _nonempty_chunks(chunks: Iterable[str]) -> Iterator[str]:
    """Validates string chunks and skips empty pieces without ending input."""
    for chunk in chunks:
        if not isinstance(chunk, str):
            raise TypeError("text must be a string")
        if chunk:
            yield chunk
