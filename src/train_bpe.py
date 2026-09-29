"""Train a byte-level BPE tokenizer and print a short summary."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from bpe import BPE
import logging

DEFAULT_TRAIN_FILE = Path("/data/nlp_course/1-basic-data/owt_train.txt")
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument(
        "--train-file",
        type=Path,
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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s: %(message)s"
    )
    args = parse_args()
    if not args.train_file.is_file():
        raise FileNotFoundError(f"training corpus not found: {args.train_file}")

    special_tokens = (
        args.special_tokens
        if args.special_tokens is not None
        else DEFAULT_SPECIAL_TOKENS
    )

    bpe = BPE()
    started = time.perf_counter()
    bpe.train(args.train_file, args.vocab_size, special_tokens)
    elapsed = time.perf_counter() - started

    logger.info(f"training file: {args.train_file}")
    logger.info(f"training time: {elapsed:.2f} seconds")
    logger.info(f"vocabulary size: {len(bpe.vocab)}")
    logger.info(f"merge count: {len(bpe.merges)}")
    logger.info("first 10 merges:")
    for rank, (left, right) in enumerate(bpe.merges[:10]):
        print(f"  {rank}: {left!r} + {right!r} -> {left + right!r}")


if __name__ == "__main__":
    main()
