"""Train a randomly initialized LM with own BPE and handwritten components."""

from __future__ import annotations

import time

# Start before importing NumPy/PyTorch: initialization is part of the budget.
_SCRIPT_STARTED = time.perf_counter()

import argparse
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import signal
import threading
import uuid

import torch

from src.bpe import BPE
from src.models import TransformerLM
from src.training.lm.checkpoint import (
    load_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
    write_json,
)
from src.training.lm.data import (
    prepare_tokens,
    sample_batch,
    tokenizer_identity,
)
from src.training.lm.optim import AdamW, clip_grad_norm_, cosine_learning_rate
from src.training.lm.validation import evaluate
from src.utils.losses import cross_entropy
from src.utils.metrics import perplexity
from src.utils.source import snapshot_sources

_ROOT = Path(__file__).resolve().parents[1]
_COURSE_DATA = Path("/data/nlp_course/1-basic-data")
_DATA = _COURSE_DATA if _COURSE_DATA.is_dir() else _ROOT / "data"
_MODEL_DEFAULTS = dict(
    context_length=256,
    d_model=512,
    d_ff=1344,
    rope_theta=10000.0,
    num_layers=4,
    num_heads=16,
    norm_eps=1e-5,
)
_TRAIN_DEFAULTS = dict(
    batch_size=1,
    max_lr=3e-4,
    min_lr=3e-5,
    betas=[0.9, 0.95],
    eps=1e-8,
    weight_decay=0.1,
    max_grad_norm=1.0,
    warmup_steps=100,
    cosine_steps=20000,
    seed=42,
    dtype="float32",
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="task-one BPE pickle; default tokenizer.pkl",
    )
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--valid-file", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=_ROOT / "data" / "lm_cache"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="checkpoint.pt; keep its .json timing sidecar",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="cache own-BPE token IDs without training",
    )
    modes.add_argument(
        "--eval-only",
        action="store_true",
        help="evaluate --resume on the full validation text",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    for name, default in _MODEL_DEFAULTS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(default),
            help=f"default {default}; restored on resume",
        )
    for name, default in _TRAIN_DEFAULTS.items():
        if name == "betas":
            parser.add_argument("--betas", nargs=2, type=float)
        elif name == "dtype":
            parser.add_argument("--dtype", choices=["float32", "bfloat16"])
        else:
            parser.add_argument(
                "--" + name.replace("_", "-"),
                type=type(default),
                help=f"default {default}; restored on resume",
            )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="absolute final update count, including resumed steps; default run until time limit",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="optional full validation interval; 0 means final only",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="overwrite local checkpoint every N updates; 0 means final only",
    )
    parser.add_argument("--time-budget-seconds", type=float, default=21600)
    parser.add_argument(
        "--finalize-reserve-seconds",
        type=float,
        default=600,
        help="reserve for saving and full final validation",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="lmfs-assignment1")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args(argv)
    if args.eval_only and args.resume is None:
        parser.error("--eval-only requires --resume")
    if args.prepare_only and args.resume is not None:
        parser.error("--prepare-only cannot be combined with --resume")
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    if args.eval_batch_size <= 0 or args.eval_every < 0 or args.save_every < 0:
        parser.error(
            "batch sizes must be positive; intervals must be nonnegative"
        )
    if (
        not math.isfinite(args.time_budget_seconds)
        or not math.isfinite(args.finalize_reserve_seconds)
        or not 0 <= args.finalize_reserve_seconds < args.time_budget_seconds
    ):
        parser.error(
            "require 0 <= finalize reserve < finite positive time budget"
        )
    return args


def _options(args, saved, identity):
    model = dict(_MODEL_DEFAULTS, vocab_size=identity["vocab_size"])
    training = dict(_TRAIN_DEFAULTS)
    if saved:
        model.update(saved["model_config"])
        training.update(saved["training_config"])
        if saved["tokenizer"] != identity:
            raise ValueError(
                "resume requires the identical BPE vocabulary, merges, and special tokens"
            )
    for key in _MODEL_DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            if saved and value != model[key]:
                raise ValueError(f"cannot change model {key} when resuming")
            model[key] = value
    for key in _TRAIN_DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            if saved and key in ("dtype", "seed") and value != training[key]:
                raise ValueError(f"cannot change {key} when resuming")
            training[key] = value
    if training["batch_size"] <= 0:
        raise ValueError("batch size must be positive")
    if any(
        model[key] <= 0
        for key in (
            "context_length",
            "d_model",
            "d_ff",
            "num_layers",
            "num_heads",
        )
    ):
        raise ValueError("model dimensions, layers, and heads must be positive")
    if (
        model["d_model"] % model["num_heads"]
        or (model["d_model"] // model["num_heads"]) % 2
    ):
        raise ValueError(
            "d_model must divide into even-dimensional attention heads"
        )
    if any(
        not math.isfinite(model[key]) or model[key] <= 0
        for key in ("rope_theta", "norm_eps")
    ):
        raise ValueError(
            "RoPE theta and norm epsilon must be finite and positive"
        )
    cosine_learning_rate(
        0,
        training["max_lr"],
        training["min_lr"],
        training["warmup_steps"],
        training["cosine_steps"],
    )
    AdamW._validate(
        training["max_lr"],
        training["betas"],
        training["eps"],
        training["weight_decay"],
    )
    if (
        not math.isfinite(training["max_grad_norm"])
        or training["max_grad_norm"] < 0
    ):
        raise ValueError("max_grad_norm must be finite and nonnegative")
    return model, training


def _start_wandb(args, config, source_dir, run_id):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="lm",
        config=config,
    )
    try:
        run.define_metric("step")
        run.define_metric("train/*", step_metric="step")
        run.define_metric("valid/*", step_metric="step")
        artifact = wandb.Artifact(f"lm-source-{run_id}", type="code")
        artifact.add_dir(str(source_dir))
        run.log_artifact(artifact)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


def main(argv=None, *, started_at=None):
    """Return final metrics; source printing, preparation, and validation are timed."""
    started = (
        started_at
        if started_at is not None
        else (
            _SCRIPT_STARTED if __name__ == "__main__" else time.perf_counter()
        )
    )
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(message)s"
    )
    args = parse_args(argv)
    device = torch.device(args.device)
    if (
        not args.prepare_only
        and device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable; check PyTorch/driver installation or pass --device cpu"
        )
    run_id = uuid.uuid4().hex[:12]
    output = args.output_dir or _ROOT / "outputs" / "lm" / run_id
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "sources" / run_id
    source_manifest = snapshot_sources(_ROOT, source_dir)
    saved = load_checkpoint(args.resume) if args.resume else None
    tokenizer_file = args.tokenizer
    if tokenizer_file is None:
        snapshot = (
            args.resume.parent / "tokenizer.pkl"
            if saved
            else _ROOT / "tokenizer.pkl"
        )
        tokenizer_file = (
            snapshot
            if snapshot.is_file()
            else Path(saved["tokenizer_file"]) if saved else snapshot
        )
    tokenizer = BPE()
    tokenizer.load(tokenizer_file)
    identity = tokenizer_identity(tokenizer)
    model_config, training = _options(args, saved, identity)
    train_file = args.train_file or (
        Path(saved["raw_paths"]["train"]) if saved else _DATA / "owt_train.txt"
    )
    valid_file = args.valid_file or (
        Path(saved["raw_paths"]["valid"]) if saved else _DATA / "owt_valid.txt"
    )
    if not args.eval_only and train_file.resolve() == valid_file.resolve():
        raise ValueError("training and validation files must be separate")
    logger.info("Encoding or checking caches with task-one BPE")
    valid_tokens, valid_meta = prepare_tokens(
        valid_file, tokenizer, args.cache_dir
    )
    train_tokens, train_meta = (
        (None, None)
        if args.eval_only
        else prepare_tokens(train_file, tokenizer, args.cache_dir)
    )
    if tokenizer_identity(tokenizer) != identity:
        raise RuntimeError("tokenizer changed while encoding")
    if saved:
        for key, metadata in (("valid", valid_meta), ("train", train_meta)):
            if metadata and any(
                metadata[field] != saved["data"][key][field]
                for field in ("text_sha256", "tokens_sha256", "token_count")
            ):
                raise ValueError(
                    f"resume requires the same {key} corpus and encoding"
                )
    tokenizer.save(output / "tokenizer.pkl")
    config = {
        "model": model_config,
        "training": training,
        "tokenizer": identity,
        "raw_paths": {
            "train": str(train_file.resolve()),
            "valid": str(valid_file.resolve()),
        },
        "data": {"train": train_meta, "valid": valid_meta},
        "device": args.device,
        "torch_version": str(torch.__version__),
        "time_budget_seconds": args.time_budget_seconds,
        "finalize_reserve_seconds": args.finalize_reserve_seconds,
        "max_steps": args.max_steps,
        "eval_batch_size": args.eval_batch_size,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "source_manifest": source_manifest,
        "validation_rule": "all N-1 next-token targets exactly once; nonoverlapping blocks; only right padding ignored; no inserted markers",
    }
    write_json(output / "config.json", config)
    if args.prepare_only:
        result = {
            "mode": "prepare",
            "train_tokens": len(train_tokens),
            "valid_tokens": len(valid_tokens),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output / "summary.json", result)
        print(json.dumps(result), flush=True)
        return result
    if (
        not args.eval_only
        and len(train_tokens) <= model_config["context_length"]
    ):
        raise ValueError(
            "training text needs at least context_length + 1 tokens"
        )
    torch.manual_seed(training["seed"])
    generator = torch.Generator().manual_seed(training["seed"])
    dtype = getattr(torch, training["dtype"])
    model = TransformerLM(**model_config, device=device, dtype=dtype)
    optimizer = AdamW(
        model.parameters(),
        lr=training["max_lr"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
        weight_decay=training["weight_decay"],
    )
    step = 0
    window = {"nll": 0.0, "tokens": 0, "updates": 0}
    prior_seconds = 0.0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.param_groups:
            group.update(
                betas=tuple(training["betas"]),
                eps=training["eps"],
                weight_decay=training["weight_decay"],
            )
        restore_rng(saved["rng"], generator)
        step, prior_seconds = saved["step"], saved["consumed_seconds"]
        window = dict(saved["train_window"])
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    config["resumed_from"] = str(args.resume.resolve()) if saved else None
    config["prior_seconds"] = prior_seconds
    # Model/optimizer state is copied by the loaders; release large CPU payloads.
    saved = None
    write_json(output / "config.json", config)
    logger.info(
        "%d parameters; step %d; device %s",
        config["parameter_count"],
        step,
        device,
    )
    run = _start_wandb(args, config, source_dir, run_id)
    checkpoint_path = output / "checkpoint.pt"
    checkpoint_timing = checkpoint_path.with_suffix(".pt.json")
    metrics_file = output / "metrics.jsonl"
    stop_requested = False
    previous_signal = None

    def interrupt(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        logger.info(
            "Stop requested; finish the update, save, and validate. A second interrupt aborts."
        )

    if threading.current_thread() is threading.main_thread():
        previous_signal = signal.signal(signal.SIGINT, interrupt)

    def elapsed():
        return time.perf_counter() - started

    def record(record):
        with metrics_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if run is not None:
            if record["kind"] == "train":
                run.log(
                    {
                        "step": record["step"],
                        "train/ppl": record["ppl"],
                        "train/mean_nll": record["nll"] / record["tokens"],
                        "train/lr": record["lr"],
                    },
                )
            else:
                run.log({"step": record["step"], "valid/ppl": record["ppl"]})

    def flush_window(lr):
        if window["tokens"]:
            record(
                dict(
                    window,
                    kind="train",
                    step=step,
                    lr=lr,
                    ppl=perplexity(window["nll"], window["tokens"]),
                    wall_seconds=elapsed(),
                )
            )
            window.update(nll=0.0, tokens=0, updates=0)

    def persist():
        payload = {
            "format_version": 1,
            "run_id": run_id,
            "model_config": asdict(model.config),
            "training_config": training,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "rng": rng_state(generator),
            "train_window": dict(window),
            "tokenizer": identity,
            "tokenizer_file": str((output / "tokenizer.pkl").resolve()),
            "raw_paths": config["raw_paths"],
            "data": config["data"],
            "consumed_seconds": prior_seconds + elapsed(),
        }
        save_checkpoint(checkpoint_path, payload)
        write_json(
            checkpoint_timing,
            {
                "run_id": run_id,
                "final_step": step,
                "cumulative_seconds": prior_seconds + elapsed(),
            },
        )

    try:
        model.train()
        stop_reason = "eval_only"
        lr = 0.0
        if not args.eval_only:
            while True:
                if stop_requested:
                    stop_reason = "interrupt"
                    break
                if args.max_steps is not None and step >= args.max_steps:
                    stop_reason = "max_steps"
                    break
                if (
                    prior_seconds + elapsed()
                    >= args.time_budget_seconds - args.finalize_reserve_seconds
                ):
                    stop_reason = "time_budget"
                    break
                # step 0 uses the handout warmup LR; each update increments step.
                lr = cosine_learning_rate(
                    step,
                    training["max_lr"],
                    training["min_lr"],
                    training["warmup_steps"],
                    training["cosine_steps"],
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                inputs, targets = sample_batch(
                    train_tokens,
                    model.config.context_length,
                    training["batch_size"],
                    generator,
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                nll = cross_entropy(model(inputs), targets, reduction="sum")
                value = nll.item()
                if not math.isfinite(value):
                    raise ValueError("training NLL is nonfinite")
                (nll / targets.numel()).backward()
                clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                step += 1
                window["nll"] += value
                window["tokens"] += targets.numel()
                window["updates"] += 1
                if step % 100 == 0:
                    flush_window(lr)
                if args.eval_every and step % args.eval_every == 0:
                    validation = evaluate(
                        model, valid_tokens, args.eval_batch_size, device
                    )
                    record(
                        dict(
                            validation,
                            kind="valid",
                            step=step,
                            wall_seconds=elapsed(),
                        )
                    )
                if args.save_every and step % args.save_every == 0:
                    persist()
            flush_window(lr)
            persist()
        # Validate the weights just saved; evaluation never updates them.
        final_validation = evaluate(
            model, valid_tokens, args.eval_batch_size, device
        )
        record(
            dict(
                final_validation,
                kind="valid",
                final=True,
                step=step,
                wall_seconds=elapsed(),
            )
        )
        final_seconds = elapsed()
        result = {
            "run_id": run_id,
            "final_step": step,
            "stop_reason": stop_reason,
            "parameter_count": config["parameter_count"],
            "validation": final_validation,
            "wall_seconds": final_seconds,
            "prior_seconds": prior_seconds,
            "cumulative_seconds": prior_seconds + final_seconds,
            "time_budget_seconds": args.time_budget_seconds,
            "budget_exceeded": prior_seconds + final_seconds
            > args.time_budget_seconds,
            "checkpoint": str(
                (args.resume if args.eval_only else checkpoint_path).resolve()
            ),
        }
        write_json(output / "summary.json", result)
        if not args.eval_only:
            write_json(checkpoint_timing, result)
        if run is not None:
            run.summary.update(
                {
                    "final_step": step,
                    "final_valid_ppl": final_validation["ppl"],
                    "wall_seconds": final_seconds,
                    "cumulative_seconds": result["cumulative_seconds"],
                    "budget_exceeded": result["budget_exceeded"],
                }
            )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        if previous_signal is not None:
            signal.signal(signal.SIGINT, previous_signal)
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
