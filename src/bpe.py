"""Trains, encodes, and decodes text with a byte-level BPE tokenizer.

BPE owns vocabulary construction and the incremental merge loop. Training data
structures and pair helpers live in training. Text and encoding helpers
live in utils.pretokenization and utils.encoding. BPE owns the encoding flow,
model lookups, cache, byte decoding, and model persistence. utils.serializer
handles pickle file operations. utils.indexes builds encoding lookup tables
from model data.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import heapq
import logging
import os
import pathlib

from src.training import pairs
from src.training import words
from src.utils import encoding
from src.utils import indexes
from src.utils import pretokenization
from src.utils import serializer

_BASE_VOCAB_SIZE = 256
_MIN_HEAP_COMPACTION_SIZE = 1024
_LOG_MERGE_INTERVAL = 50
logger = logging.getLogger(__name__)

PathInput = str | os.PathLike[str]
Merge = tuple[bytes, bytes]


# Vocabulary, lookup tables, and the cache share the model's lifecycle.
# pylint: disable-next=too-many-instance-attributes
class BPE:
    """Trains, uses, saves, and loads a byte-level BPE tokenizer.

    Vocabulary values are byte strings. Special tokens are also vocabulary
    entries and count toward ``vocab_size``. When constructed with an existing
    vocabulary, missing special tokens are appended after its greatest ID.

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
        self.vocab: dict[int, bytes] = (
            dict(vocab)
            if vocab is not None
            else {
                token_id: bytes((token_id,))
                for token_id in range(_BASE_VOCAB_SIZE)
            }
        )
        self.merges: list[Merge] = list(merges) if merges is not None else []
        self.special_tokens: list[str] = (
            list(special_tokens) if special_tokens is not None else []
        )

        # Rebuilt after training or loading.
        self._token_to_id: dict[bytes, int] = {}
        self._merge_ranks: dict[Merge, int] = {}
        self._special_token_ids: dict[str, int] = {}
        self._pretokenizer: pretokenization.PreTokenizer
        self._encoding_cache: dict[bytes, tuple[int, ...]] = {}
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
        pretokenization.validate_special_tokens(special_tokens)
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
            training_words = words.TrainingWords.from_file(
                input_path,
                pretokenizer=pretokenization.PreTokenizer(special_tokens),
            )
            logger.info("Read corpus file done.")
            states = pairs.initialize_pair_states(training_words)
            heap = pairs.build_pair_heap(states, vocab)
            while len(vocab) < vocab_size and states:
                if len(merges) % _LOG_MERGE_INTERVAL == 0:
                    logger.info(
                        "current vocab_size=%d, max vocab_size=%d.",
                        len(vocab),
                        vocab_size,
                    )
                entry = pairs.pop_best_pair(heap, states)
                affected_words = states.pop(entry.pair_id)
                token_id = len(vocab)
                vocab[token_id] = b"".join(entry.pair_bytes())
                merges.append(entry.pair_bytes())
                changed: set[int] = set()
                for word_id in affected_words:
                    for pair_id, change in training_words.merge(
                        word_id, entry.pair_id, token_id
                    ).items():
                        if pair_id == entry.pair_id or not change:
                            continue
                        if pair_id not in states:
                            states[pair_id] = pairs.PairState(0, word_id)
                        state = states[pair_id]
                        state.count += (
                            change * training_words.frequencies[word_id]
                        )
                        if change > 0:
                            state.add(word_id)
                        if not state.count:
                            del states[pair_id]
                        changed.add(pair_id)
                for pair_id in changed:
                    if pair_id in states:
                        heapq.heappush(
                            heap,
                            pairs.HeapEntry(
                                states[pair_id].count, pair_id, vocab
                            ),
                        )
                # Rebuild periodically to bound the number of stale entries.
                if len(heap) > max(_MIN_HEAP_COMPACTION_SIZE, 2 * len(states)):
                    heap = pairs.build_pair_heap(states, vocab)

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens)
        self._rebuild_indexes()

    def encode(self, text: str) -> list[int]:
        """Encodes text into token IDs, retaining literal special tokens.

        Args:
            text: Text to pre-tokenize and encode as UTF-8 bytes.

        Returns:
            Token IDs in input order.

        Raises:
            TypeError: If text is not a string.
            KeyError: If the vocabulary lacks a required byte or merge token.
        """
        ids: list[int] = []
        for word, is_special in self._pretokenizer.iter_text(text):
            if is_special:
                ids.append(self._special_token_ids[word.decode("utf-8")])
            else:
                ids.extend(
                    encoding.encode_word(
                        word,
                        self._token_to_id,
                        self._merge_ranks,
                        self._encoding_cache,
                    )
                )
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yields IDs lazily, preserving boundaries across text chunks.

        Args:
            iterable: Consecutive string chunks of one text. Pre-tokens and
                special tokens may span multiple chunks.

        Yields:
            The same IDs as encode() on the concatenated input strings.
        """
        for word, is_special in self._pretokenizer.iter_chunks(iterable):
            if is_special:
                yield self._special_token_ids[word.decode("utf-8")]
            else:
                yield from encoding.encode_word(
                    word,
                    self._token_to_id,
                    self._merge_ranks,
                    self._encoding_cache,
                )

    def decode(self, ids: Iterable[int]) -> str:
        """Decodes joined token bytes as UTF-8, retaining special tokens.

        Args:
            ids: Token IDs in text order.

        Returns:
            Decoded text, with malformed UTF-8 replaced by U+FFFD.

        Raises:
            ValueError: If an ID is absent from the vocabulary.
        """
        try:
            token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID: {error.args[0]}") from error
        return token_bytes.decode("utf-8", errors="replace")

    def save(self, path: PathInput) -> None:
        """Saves vocabulary, merges, and special tokens in one pickle file.

        Args:
            path: Exact output file path, regardless of its extension.
        """
        output_path = pathlib.Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializer.write_pickle(
            output_path,
            {
                "vocab": self.vocab,
                "merges": self.merges,
                "special_tokens": self.special_tokens,
            },
        )

    def load(self, path: PathInput) -> None:
        """Loads a trusted model pickle and rebuilds all indexes.

        Args:
            path: Input file previously written by save().
        """
        state = serializer.read_pickle(pathlib.Path(path))
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
        """Loads a vocabulary dictionary and merge list from two pickles.

        Args:
            vocab_filepath: Pickle containing a dict[int, bytes] vocabulary.
            merges_filepath: Pickle containing ordered pairs of byte strings.
            special_tokens: Explicit marker list, or None for no markers.
                Use load() to restore the complete single-file model saved
                by save(), including its special-token settings.

        Returns:
            A tokenizer with restored bytes, merge order, and special tokens.
        """
        return cls(
            vocab=serializer.read_pickle(pathlib.Path(vocab_filepath)),
            merges=serializer.read_pickle(pathlib.Path(merges_filepath)),
            special_tokens=special_tokens,
        )

    def _rebuild_indexes(self) -> None:
        """Rebuilds inference lookups and clears the old encoding cache."""
        lookup_tables = indexes.build_encoding_indexes(
            self.vocab, self.merges, self.special_tokens
        )
        self._token_to_id = lookup_tables.token_to_id
        self._merge_ranks = lookup_tables.merge_ranks
        self._special_token_ids = lookup_tables.special_token_ids
        self._pretokenizer = pretokenization.PreTokenizer(self.special_tokens)
        self._encoding_cache.clear()


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
    **kwargs: object,
) -> tuple[dict[int, bytes], list[Merge]]:
    """Adapter target used by ``tests/adapters.py`` for BPE training tests."""
    del kwargs  # Compatibility with the official adapter's optional keywords.
    bpe = BPE()
    bpe.train(input_path, vocab_size, special_tokens)
    return bpe.vocab, bpe.merges
