"""Trains or loads BPE, validates the full corpus, and records W&B results."""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import wandb

from src import bpe
from src import evaluation
from src import wandb_record

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_COURSE_DATA = pathlib.Path("/data/nlp_course/1-basic-data")
_DEFAULT_DATA = (
    _COURSE_DATA if _COURSE_DATA.is_dir() else _PROJECT_ROOT / "data"
)
DEFAULT_TRAIN_FILE = _DEFAULT_DATA / "owt_train.txt"
DEFAULT_VALID_FILE = _DEFAULT_DATA / "owt_valid.txt"
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses training, full validation, persistence, and W&B options."""
    parser = argparse.ArgumentParser(
        description="Train or load BPE and record full-text validation in W&B"
    )
    parser.add_argument(
        "--train-file", type=pathlib.Path, default=DEFAULT_TRAIN_FILE
    )
    parser.add_argument(
        "--valid-file", type=pathlib.Path, default=DEFAULT_VALID_FILE
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        help="complete special-token list; repeat for each marker",
    )
    parser.add_argument(
        "--load-tokenizer",
        type=pathlib.Path,
        help="load an existing model pickle and skip training",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        help="output pickle file (default: outputs/<run-id>/tokenizer.pkl)",
    )
    parser.add_argument("--wandb-project", default="lmfs-assignment1")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"]
    )
    return parser.parse_args(argv)


def _record_validation(
    run: wandb.Run,
    tokenizer: bpe.BPE,
    text: str,
    result: evaluation.ValidationResult,
    model_path: pathlib.Path,
) -> None:
    """Records token details and the measured result after timing finishes."""
    rows = evaluation.token_rows(tokenizer, text, result.first_ids)
    print("前 200 个 encode ID：", result.first_ids, flush=True)
    print(
        "前 200 个 token 整体 decode：",
        repr(tokenizer.decode(result.first_ids)),
        flush=True,
    )
    print("逐 token 详情字段：", evaluation.TOKEN_COLUMNS, flush=True)
    for row in rows:
        print(row, flush=True)
    run.log(
        {
            "bpe/first_200_tokens": wandb.Table(
                columns=evaluation.TOKEN_COLUMNS, data=rows
            )
        }
    )
    metrics = {
        "bpe/valid_seconds": result.seconds,
        "bpe/valid_token_count": result.token_count,
        "bpe/roundtrip_passed": result.passed,
        "bpe/vocab_size": len(tokenizer.vocab),
    }
    run.summary.update(metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    print("任务二加载本人 BPE 的模型路径：", str(model_path), flush=True)
    wandb_record.log_sources(run, _PROJECT_ROOT)
    if result.token_count < 200 or not result.passed:
        raise ValueError("评估文本不足 200 个 token 或全文还原失败")


def main(argv: list[str] | None = None) -> None:
    """Runs a fresh-model full validation with the course timing boundaries."""
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s: %(message)s"
    )
    args = parse_args(argv)
    tokenizer = bpe.BPE()
    special_tokens = list(
        dict.fromkeys(
            args.special_tokens
            if args.special_tokens is not None
            else DEFAULT_SPECIAL_TOKENS
        )
    )
    if args.load_tokenizer is not None:
        tokenizer.load(args.load_tokenizer)
        if args.special_tokens is not None and (
            special_tokens != tokenizer.special_tokens
        ):
            raise ValueError(
                "--special-token must match the loaded model settings"
            )
        special_tokens = tokenizer.special_tokens

    config = {
        "train_file": str(args.train_file.resolve()),
        "valid_file": str(args.valid_file.resolve()),
        "vocab_size": (
            len(tokenizer.vocab)
            if args.load_tokenizer is not None
            else args.vocab_size
        ),
        "special_tokens": special_tokens,
        "load_tokenizer": (
            str(args.load_tokenizer.resolve())
            if args.load_tokenizer is not None
            else None
        ),
    }
    with wandb_record.experiment(
        "bpe",
        config,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    ) as run:
        if args.load_tokenizer is None:
            tokenizer.train(args.train_file, args.vocab_size, special_tokens)
        model_path = (
            args.output_file
            or _PROJECT_ROOT / "outputs" / run.id / "tokenizer.pkl"
        )
        tokenizer.save(model_path)
        tokenizer = bpe.BPE()
        tokenizer.load(model_path)
        with args.valid_file.open(encoding="utf-8", newline="") as stream:
            text = stream.read()
        logger.info("Validating the complete text: %d characters", len(text))
        result = evaluation.validate(tokenizer, text)
        _record_validation(run, tokenizer, text, result, model_path)


if __name__ == "__main__":
    main()
