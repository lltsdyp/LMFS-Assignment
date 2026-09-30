"""Stores and loads compact word sequences used by BPE training."""

from __future__ import annotations

import array
import collections
import logging
import os
import pathlib

import regex

if __package__:
    from src import utils
else:
    import utils

_TOKEN_ID_BITS = 32
_TOKEN_ID_MASK = (1 << _TOKEN_ID_BITS) - 1
_MIB = 1024 * 1024
_READ_CHUNK_SIZE = 1024 * 1024 * 32
logger = logging.getLogger(__name__)


class TrainingWords:
    """Stores word sequences and metadata in compact, shared integer arrays.

    Each word retains its initial capacity and is compacted in place on merge.

    Attributes:
        tokens: A flat array of unsigned 32-bit token IDs.
        offsets: Starting positions of words within tokens.
        lengths: Current token counts for individual words.
        frequencies: Corpus occurrence counts for individual words.
    """

    def __init__(self) -> None:
        self.tokens = array.array("I")
        self.offsets = array.array("Q")
        self.lengths = array.array("I")
        self.frequencies = array.array("Q")

    def append(self, word: bytes, frequency: int) -> None:
        """Appends a word as byte token IDs with its corpus frequency."""
        self.offsets.append(len(self.tokens))
        self.lengths.append(len(word))
        self.frequencies.append(frequency)
        self.tokens.extend(word)

    def count_pairs(self, word_id: int) -> collections.Counter[int]:
        """Counts adjacent packed pairs in a word, including overlaps."""
        start = self.offsets[word_id]
        end = start + self.lengths[word_id]
        return collections.Counter(
            (self.tokens[index] << _TOKEN_ID_BITS) | self.tokens[index + 1]
            for index in range(start, end - 1)
        )

    def merge(
        self, word_id: int, pair_id: int, token_id: int
    ) -> collections.Counter[int]:
        """Merges left to right and returns neighboring pair count changes.

        Like HF's Word.merge, records only the pairs beside each replacement.
        The selected pair itself is retired by the caller.
        """
        start = self.offsets[word_id]
        end = start + self.lengths[word_id]
        left_id = pair_id >> _TOKEN_ID_BITS
        right_id = pair_id & _TOKEN_ID_MASK
        changes: collections.Counter[int] = collections.Counter()
        tokens = self.tokens
        read_index = write_index = start
        while read_index < end:
            if (
                read_index + 1 < end
                and tokens[read_index] == left_id
                and tokens[read_index + 1] == right_id
            ):
                if write_index > start:
                    previous_id = tokens[write_index - 1]
                    changes[(previous_id << _TOKEN_ID_BITS) | left_id] -= 1
                    changes[(previous_id << _TOKEN_ID_BITS) | token_id] += 1
                if read_index + 2 < end:
                    next_id = tokens[read_index + 2]
                    changes[(right_id << _TOKEN_ID_BITS) | next_id] -= 1
                    changes[(token_id << _TOKEN_ID_BITS) | next_id] += 1
                tokens[write_index] = token_id
                read_index += 2
            else:
                tokens[write_index] = tokens[read_index]
                read_index += 1
            write_index += 1
        self.lengths[word_id] = write_index - start
        return changes

    def memory_usage_bytes(self) -> int:
        """Returns the logical size of token and metadata buffers in bytes."""
        buffers = (self.tokens, self.offsets, self.lengths, self.frequencies)
        return sum(len(buffer) * buffer.itemsize for buffer in buffers)

    @classmethod
    def from_file(
        cls,
        input_path: str | os.PathLike[str],
        special_tokens: list[str],
        *,
        pretokenizer: regex.Pattern,
    ) -> TrainingWords:
        """Counts byte strings in memory and packs each distinct word once.

        Args:
            input_path: The UTF-8 training corpus.
            special_tokens: Delimiters excluded from ordinary pre-tokens.
            pretokenizer: The compiled pattern supplied by the tokenizer.

        Returns:
            Packed word sequences with their global corpus frequencies.
        """
        words = cls()
        with pathlib.Path(input_path).open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            counts = collections.Counter(
                utils.iter_pretokens(
                    stream,
                    special_tokens,
                    _READ_CHUNK_SIZE,
                    pretokenizer=pretokenizer,
                )
            )
        logger.info("Counted %d distinct pre-tokens.", len(counts))

        # Consume the table so byte strings are freed as token arrays grow.
        while counts:
            word, frequency = counts.popitem()
            words.append(word, frequency)
        del counts
        logger.info(
            "Loaded %d distinct pre-tokens into "
            "%d MiB of packed training buffers.",
            len(words.lengths),
            words.memory_usage_bytes() // _MIB,
        )
        return words
