"""Train and save a byte-level BPE tokenizer, then print a short summary."""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

import bpe

DEFAULT_TRAIN_FILE = pathlib.Path("/data/nlp_course/1-basic-data/owt_train.txt")
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_FILE = pathlib.Path("tokenizer.pkl")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command-line training options."""
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer"
    )
    parser.add_argument(
        "--train-file",
        type=pathlib.Path,
        default=DEFAULT_TRAIN_FILE,
        help=f"UTF-8 training corpus (default: {DEFAULT_TRAIN_FILE})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"target vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help=(
            "special token to reserve; may be supplied more than once "
            f"(default: {DEFAULT_SPECIAL_TOKENS})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Trains and saves a tokenizer, then logs training results."""
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s"
    )
    args = parse_args()
    if not args.train_file.is_file():
        raise FileNotFoundError(f"training corpus not found: {args.train_file}")

    special_tokens = (
        args.special_tokens
        if args.special_tokens is not None
        else DEFAULT_SPECIAL_TOKENS
    )

    tokenizer = bpe.BPE()
    started = time.perf_counter()
    tokenizer.train(args.train_file, args.vocab_size, special_tokens)
    elapsed = time.perf_counter() - started
    tokenizer.save(OUTPUT_FILE)

    logger.info("saved tokenizer: %s", OUTPUT_FILE.resolve())
    logger.info("training file: %s", args.train_file)
    logger.info("training time: %.2f seconds", elapsed)
    logger.info("vocabulary size: %d", len(tokenizer.vocab))
    logger.info("merge count: %d", len(tokenizer.merges))
    logger.info("first 10 merges:")
    for rank, (left, right) in enumerate(tokenizer.merges[:10]):
        print(f"  {rank}: {left!r} + {right!r} -> {left + right!r}")


if __name__ == "__main__":
    main()
