"""Own-BPE text caching, random training windows, and full validation batches."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.bpe import BPE


def file_digest(path: Path) -> str:
    """Hash actual bytes; preserved raw text is the cache identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(tokenizer: BPE) -> dict:
    """Identify vocabulary IDs, ordered merges, and special-token settings."""
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
    cache_dir: Path,
    *,
    chunk_chars: int = 65536,
) -> tuple[np.memmap, dict]:
    """Encode unchanged UTF-8 text; reuse only content-verified caches.

    encode_iterable preserves pre-token and special-token boundaries
    across file reads. No marker or newline normalization is added.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    identity = tokenizer_identity(tokenizer)
    dtype = np.dtype("<u2" if len(tokenizer.vocab) <= 65536 else "<u4")
    metadata = {
        "format_version": 1,
        "text_sha256": file_digest(text_path),
        "tokenizer": identity,
        "dtype": dtype.str,
    }
    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / f"{key}.bin"
    manifest = cache_dir / f"{key}.json"
    reused = False
    if binary.is_file() and manifest.is_file():
        try:
            saved = json.loads(manifest.read_text())
            reused = (
                all(saved.get(k) == v for k, v in metadata.items())
                and saved.get("token_count", 0) >= 2
                and binary.stat().st_size
                == saved["token_count"] * dtype.itemsize
                and file_digest(binary) == saved["tokens_sha256"]
            )
        except (ValueError, KeyError, TypeError):
            reused = False
    if not reused:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            count = 0
            try:
                with text_path.open(encoding="utf-8", newline="") as stream:
                    chunks = iter(lambda: stream.read(chunk_chars), "")
                    ids = tokenizer.encode_iterable(chunks)
                    while values := list(itertools.islice(ids, 65536)):
                        if min(values) < 0 or max(values) >= len(
                            tokenizer.vocab
                        ):
                            raise ValueError(
                                "BPE returned an out-of-vocabulary ID"
                            )
                        block = np.asarray(values, dtype=dtype).tobytes()
                        output.write(block)
                        digest.update(block)
                        count += len(values)
                if count < 2:
                    raise ValueError("text must encode to at least two tokens")
                if file_digest(text_path) != metadata["text_sha256"]:
                    raise ValueError("raw text changed during tokenization")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        saved = dict(
            metadata, token_count=count, tokens_sha256=digest.hexdigest()
        )
        temporary.replace(binary)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_dir, delete=False
        ) as output:
            json.dump(saved, output, ensure_ascii=False, indent=2)
            temp_manifest = Path(output.name)
        temp_manifest.replace(manifest)
    tokens = np.memmap(
        binary, dtype=dtype, mode="r", shape=(saved["token_count"],)
    )
    return tokens, dict(
        saved, cache_file=str(binary.resolve()), cache_reused=reused
    )


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
