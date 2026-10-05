"""In-memory BPE tokens, random windows, and full validation batches."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch

from src.bpe import BPE

logger = logging.getLogger(__name__)


def tokenizer_identity(tokenizer: BPE) -> dict:
    """Identify vocabulary IDs, ordered merges, and special-token settings."""
    # Model logits and embeddings index IDs directly from 0 to vocab_size - 1.
    if sorted(tokenizer.vocab) != list(range(len(tokenizer.vocab))):
        raise ValueError("BPE vocabulary IDs must be contiguous from zero")
    payload = {
        "vocab": [
            tokenizer.vocab[i].hex() for i in range(len(tokenizer.vocab))
        ],
        "merges": [[a.hex(), b.hex()] for a, b in tokenizer.merges],
        "special_tokens": tokenizer.special_tokens,
    }
    signature = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "sha256": signature,
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": list(tokenizer.special_tokens),
        "inserted_special_tokens": [],
    }


def prepare_tokens(
    text_path: Path,
    tokenizer: BPE,
    *,
    chunk_chars: int = 65536,
) -> tuple[np.ndarray, dict]:
    """Encode UTF-8 chunks into packed in-memory IDs without writing files.

    Rust preserves pre-token and special-token boundaries across reads.
    Digests identify the data when resuming; they are computed while encoding.
    """
    # read(0) loses the corpus; read(-1) loads it all instead of streaming.
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    started = time.perf_counter()
    dtype = np.dtype("<u2" if len(tokenizer.vocab) <= 65536 else "<u4")
    storage = bytearray()
    text_digest = hashlib.sha256()
    tokens_digest = hashlib.sha256()
    with text_path.open(encoding="utf-8", newline="") as stream:

        def chunks():
            while chunk := stream.read(chunk_chars):
                text_digest.update(chunk.encode("utf-8"))
                yield chunk

        for block in tokenizer.encode_blocks(chunks(), dtype.str):
            storage.extend(block)
            tokens_digest.update(block)
    # NumPy borrows the packed buffer instead of copying the entire corpus.
    tokens = np.frombuffer(storage, dtype=dtype)
    seconds = time.perf_counter() - started
    logger.info(
        "Encoded %s: %d tokens in %.2fs", text_path.name, len(tokens), seconds
    )
    return tokens, {
        "dtype": dtype.str,
        "text_sha256": text_digest.hexdigest(),
        "tokens_sha256": tokens_digest.hexdigest(),
        "token_count": len(tokens),
        "encoding_seconds": seconds,
    }


def sample_batch(tokens, context_length, batch_size, generator, device):
    """Uniform contiguous windows; each target is one token after its input."""
    if context_length <= 0 or batch_size <= 0:
        raise ValueError("context length and batch size must be positive")
    if len(tokens) <= context_length:
        raise ValueError("training text needs context_length + 1 tokens")
    starts = torch.randint(
        len(tokens) - context_length, (batch_size,), generator=generator
    ).tolist()
    windows = np.stack([tokens[s : s + context_length + 1] for s in starts])
    batch = torch.from_numpy(windows.astype(np.int64))
    return batch[:, :-1].to(device), batch[:, 1:].to(device)


def validation_batches(tokens, context_length, batch_size, device):
    """Visit all N-1 real targets once; mask only temporary right padding.

    Blocks reset positions to zero and carry at most context_length
    preceding input tokens. Padding uses existing ID 0, with target -100.
    Causality prevents suffix padding from affecting real targets.
    """
    if context_length <= 0 or batch_size <= 0 or len(tokens) < 2:
        raise ValueError("positive sizes and at least two tokens are required")
    for start in range(0, len(tokens) - 1, context_length * batch_size):
        offsets = list(
            range(
                start,
                min(start + context_length * batch_size, len(tokens) - 1),
                context_length,
            )
        )
        lengths = [min(context_length, len(tokens) - 1 - s) for s in offsets]
        width = max(lengths)
        inputs = torch.zeros(len(offsets), width, dtype=torch.long)
        targets = torch.full_like(inputs, -100)
        for row, (offset, length) in enumerate(zip(offsets, lengths)):
            inputs[row, :length] = torch.from_numpy(
                np.asarray(tokens[offset : offset + length], dtype=np.int64)
            )
            targets[row, :length] = torch.from_numpy(
                np.asarray(
                    tokens[offset + 1 : offset + length + 1], dtype=np.int64
                )
            )
        yield inputs.to(device), targets.to(device)
