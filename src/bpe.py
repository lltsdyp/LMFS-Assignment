"""Byte-level BPE training and tokenizer API for Assignment 1.

GPT-2-style pre-tokenization and byte-level vocabulary training are implemented.
Encoding, decoding, and persistence are provided as scaffold interfaces.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Any

import regex
import logging

PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_PRETOKENIZER = regex.compile(PAT)
_BASE_VOCAB_SIZE = 256

PathInput = str | PathLike[str]
Merge = tuple[bytes, bytes]
logger = logging.getLogger(__name__)

def pretokenize(text: str) -> Iterator[str]:
    """Split text into GPT-2-style pre-tokens without changing its contents.

    Args:
      text: The text to split.

    Returns:
      An iterator over GPT-2-style pre-token strings in their original order.

    Raises:
      TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return (match.group(0) for match in _PRETOKENIZER.finditer(text))


class BPE:
    """Train, use, save, and load a byte-level BPE tokenizer.

    Vocabulary values are byte strings. Special tokens are also vocabulary
    entries and count toward ``vocab_size``.
    """

    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[Merge] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab) if vocab is not None else {}
        self.merges: list[Merge] = list(merges) if merges is not None else []
        self.special_tokens: list[str] = (
            list(special_tokens) if special_tokens is not None else []
        )

        # Rebuilt after training or loading.
        self._token_to_id: dict[bytes, int] = {}
        self._merge_ranks: dict[Merge, int] = {}
        self._special_token_ids: dict[str, int] = {}
        if self.vocab or self.merges or self.special_tokens:
            self._rebuild_indexes()

    def train(
        self,
        input_path: PathInput,
        vocab_size: int,
        special_tokens: list[str],
    ) -> None:
        """Train the vocabulary and ordered merge list from a UTF-8 text file.

        ``vocab_size`` includes the 256 byte tokens, special tokens, and learned
        merge tokens. Pair counts are weighted by pre-token frequency, and
        equal-frequency pairs are resolved by choosing the lexicographically
        greatest pair of byte strings.

        Args:
          input_path: UTF-8 text file used as the training corpus.
          vocab_size: Maximum total number of byte, special, and merged tokens.
          special_tokens: Strings reserved as indivisible special tokens.
          **kwargs: No optional keyword arguments are currently supported.

        Raises:
          ValueError: If the vocabulary is too small or a special token is
            invalid.
        """
        minimum_vocab_size = _BASE_VOCAB_SIZE + len(special_tokens)
        if vocab_size < minimum_vocab_size:
            raise ValueError("vocab_size must be at least 256 plus the number of special tokens")

        if len(special_tokens) != len(set(special_tokens)):
            logger.warning("duplicate special tokens were provided; keeping their first occurrence")
            special_tokens = list(dict.fromkeys(special_tokens))

        logging.info(f"Train corpus path: {input_path}.")
        with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
            corpus = stream.read()
        logging.info("Load corpus done.")

        # Remove special-token spans before counting ordinary byte pairs.
        if special_tokens:
            ordered_special_tokens = sorted(
                special_tokens, key=len, reverse=True
            )
            special_pattern = regex.compile(
                "|".join(
                    regex.escape(token) for token in ordered_special_tokens
                )
            )
            text_segments = regex.splititer(special_pattern, corpus)
        else:
            text_segments = iter((corpus,))

        # Duplicate pre-tokens share a count that weights pair frequencies.
        word_counts: Counter[tuple[bytes, ...]] = Counter(
            tuple(bytes((value,)) for value in pretoken.encode("utf-8"))
            for pretoken in chain.from_iterable(
                pretokenize(segment) for segment in text_segments
            )
        )

        vocab: dict[int, bytes] = {
            token_id: bytes((token_id,))
            for token_id in range(_BASE_VOCAB_SIZE)
        }
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")

        logger.info("byte-level splitting and special token loading done.")

        merges: list[Merge] = []
        while len(vocab) < vocab_size:
            # Show progress per 500 tokens
            if len(merges) % 500 == 0:
                logger.info(f"current vocab_size={len(vocab)}, max vocab_size={vocab_size}.")
            # Recount pairs inside each pre-token, never across its boundary.
            # TODO: Try optimizing training speed with priority queue.
            pair_counts: Counter[Merge] = Counter()
            for word, frequency in word_counts.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += frequency

            if not pair_counts:
                break

            pair = max(pair_counts, key=lambda item: (pair_counts[item], item))
            merged_token = pair[0] + pair[1]
            new_word_counts: Counter[tuple[bytes, ...]] = Counter()

            # Apply the selected merge left to right, without overlapping pairs.
            for word, frequency in word_counts.items():
                merged_word: list[bytes] = []
                index = 0
                while index < len(word):
                    if (
                        index + 1 < len(word)
                        and word[index] == pair[0]
                        and word[index + 1] == pair[1]
                    ):
                        merged_word.append(merged_token)
                        index += 2
                    else:
                        merged_word.append(word[index])
                        index += 1
                new_word_counts[tuple(merged_word)] += frequency

            word_counts = new_word_counts
            merges.append(pair)
            vocab[len(vocab)] = merged_token

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens)
        self._rebuild_indexes()

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("TODO: implement BPE encoding")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError("TODO: implement streaming BPE encoding")

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("TODO: implement BPE decoding")

    def save(self, path_prefix: PathInput) -> None:
        raise NotImplementedError("TODO: implement tokenizer serialization")

    def load(self, path_prefix: PathInput) -> None:
        raise NotImplementedError("TODO: implement tokenizer loading")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: PathInput,
        merges_filepath: PathInput,
        special_tokens: list[str] | None = None,
    ) -> BPE:
        raise NotImplementedError("TODO: implement from_files")

    def _rebuild_indexes(self) -> None:
        """Rebuild token-to-ID, merge-rank, and special-token lookup tables."""
        self._token_to_id = {
            token: token_id for token_id, token in self.vocab.items()
        }
        self._merge_ranks = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._special_token_ids = {}
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes in self._token_to_id:
                self._special_token_ids[special_token] = self._token_to_id[
                    token_bytes
                ]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[Merge],
    special_tokens: list[str] | None = None,
) -> BPE:
    """Adapter target used by ``tests/adapters.py`` for tokenizer tests."""
    return BPE(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: PathInput,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[Merge]]:
    """Adapter target used by ``tests/adapters.py`` for BPE training tests."""
    bpe = BPE()
    bpe.train(input_path, vocab_size, special_tokens, **kwargs)
    return bpe.vocab, bpe.merges
