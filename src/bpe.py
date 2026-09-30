"""Provides the BPE tokenizer interface, rules, and training algorithm.

BPE owns vocabulary construction and the incremental merge loop. TrainingWords,
PairState, and HeapEntry each have their own module; standalone helpers live in
utils. Encoding, decoding, and separate-file loading remain scaffold interfaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import heapq
import logging
import os
import pathlib
import pickle

import regex

if __package__:
    from src import heap_entry
    from src import pair_state
    from src import training_words
    from src import utils
else:
    import heap_entry
    import pair_state
    import training_words
    import utils

PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_PRETOKENIZER = regex.compile(PAT)
_BASE_VOCAB_SIZE = 256
_MIN_HEAP_COMPACTION_SIZE = 1024
_LOG_MERGE_INTERVAL = 50
logger = logging.getLogger(__name__)

PathInput = str | os.PathLike[str]
Merge = tuple[bytes, bytes]


def pretokenize(text: str) -> Iterator[str]:
    """Splits text into GPT-2-style pre-tokens without changing its contents.

    Args:
        text: The text to split.

    Returns:
        An iterator over pre-token strings in their original order.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return (match.group(0) for match in _PRETOKENIZER.finditer(text))


class BPE:
    """Trains, uses, saves, and loads a byte-level BPE tokenizer.

    Vocabulary values are byte strings. Special tokens are also vocabulary
    entries and count toward ``vocab_size``.

    Attributes:
        vocab: Token IDs mapped to their byte strings.
        merges: Learned byte pairs in merge order.
        special_tokens: Reserved strings excluded from ordinary training text.
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
        """Trains the vocabulary and ordered merge list from a UTF-8 text file.

        ``vocab_size`` includes the 256 byte tokens, special tokens, and learned
        merge tokens. Pair counts are weighted by pre-token frequency, and
        equal-frequency pairs are resolved by choosing the lexicographically
        greatest pair of byte strings.

        Args:
            input_path: UTF-8 text file used as the training corpus.
            vocab_size: Maximum total number of vocabulary entries.
            special_tokens: Strings reserved as indivisible special tokens.

        Raises:
            ValueError: If the vocabulary is too small or a special token is
                invalid.
        """
        # Keep vocabulary initialization and merging in one training flow.
        # pylint: disable=too-many-locals,too-many-branches
        if any(
            not isinstance(token, str) or not token for token in special_tokens
        ):
            raise ValueError("special tokens must be non-empty strings")
        if len(special_tokens) != len(set(special_tokens)):
            logger.warning(
                "Duplicate special tokens were provided; "
                "keeping their first occurrence."
            )
            special_tokens = list(dict.fromkeys(special_tokens))
        minimum_vocab_size = _BASE_VOCAB_SIZE + len(special_tokens)
        logger.info("Initial vocab size=%d", minimum_vocab_size)
        if vocab_size < minimum_vocab_size:
            raise ValueError(
                "vocab_size must be at least 256 plus the number "
                "of special tokens"
            )

        vocab: dict[int, bytes] = {
            token_id: bytes((token_id,)) for token_id in range(_BASE_VOCAB_SIZE)
        }
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")

        merges: list[Merge] = []
        logger.info("Train corpus path: %s.", input_path)

        # merge, optimized with inverted index and priority queue.
        if len(vocab) < vocab_size:
            words = training_words.TrainingWords.from_file(
                input_path,
                special_tokens,
                pretokenizer=_PRETOKENIZER,
            )
            logger.info("Read corpus file done.")
            states = utils.initialize_pair_states(words)
            heap = utils.build_pair_heap(states, vocab)
            while len(vocab) < vocab_size and states:
                if len(merges) % _LOG_MERGE_INTERVAL == 0:
                    logger.info(
                        "current vocab_size=%d, max vocab_size=%d.",
                        len(vocab),
                        vocab_size,
                    )
                entry = utils.pop_best_pair(heap, states)
                affected_words = states.pop(entry.pair_id)
                token_id = len(vocab)
                vocab[token_id] = b"".join(entry.pair_bytes())
                merges.append(entry.pair_bytes())
                changed: set[int] = set()
                for word_id in affected_words:
                    for pair_id, change in words.merge(
                        word_id, entry.pair_id, token_id
                    ).items():
                        if pair_id == entry.pair_id or not change:
                            continue
                        if pair_id not in states:
                            states[pair_id] = pair_state.PairState(0, word_id)
                        state = states[pair_id]
                        state.count += change * words.frequencies[word_id]
                        if change > 0:
                            state.add(word_id)
                        if not state.count:
                            del states[pair_id]
                        changed.add(pair_id)
                for pair_id in changed:
                    if pair_id in states:
                        heapq.heappush(
                            heap,
                            heap_entry.HeapEntry(
                                states[pair_id].count, pair_id, vocab
                            ),
                        )
                # Rebuild periodically to bound the number of stale entries.
                if len(heap) > max(_MIN_HEAP_COMPACTION_SIZE, 2 * len(states)):
                    heap = utils.build_pair_heap(states, vocab)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens)
        self._rebuild_indexes()

    def encode(self, text: str) -> list[int]:
        """Encodes text into token IDs (not implemented)."""
        raise NotImplementedError("TODO: implement BPE encoding")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encodes an iterable of text chunks (not implemented)."""
        raise NotImplementedError("TODO: implement streaming BPE encoding")

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs into text (not implemented)."""
        raise NotImplementedError("TODO: implement BPE decoding")

    def save(self, path: PathInput) -> None:
        """Saves tokenizer state to a single pickle file.

        Args:
            path: Output file path. An existing file is overwritten.
        """
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with pathlib.Path(path).open("wb") as stream:
            pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: PathInput) -> None:
        """Loads a trusted pickle file and rebuilds tokenizer indexes.

        Args:
            path: Input file previously written by save().
        """
        with pathlib.Path(path).open("rb") as stream:
            state = pickle.load(stream)
        self.vocab, self.merges, self.special_tokens = (
            state["vocab"],
            state["merges"],
            state["special_tokens"],
        )
        self._rebuild_indexes()

    @classmethod
    def from_files(
        cls,
        vocab_filepath: PathInput,
        merges_filepath: PathInput,
        special_tokens: list[str] | None = None,
    ) -> BPE:
        """Loads vocabulary and merge files (not implemented)."""
        raise NotImplementedError("TODO: implement from_files")

    def _rebuild_indexes(self) -> None:
        """Rebuilds token-to-ID, merge-rank, and special-token lookup tables."""
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
) -> tuple[dict[int, bytes], list[Merge]]:
    """Adapter target used by ``tests/adapters.py`` for BPE training tests."""
    bpe = BPE()
    bpe.train(input_path, vocab_size, special_tokens)
    return bpe.vocab, bpe.merges
