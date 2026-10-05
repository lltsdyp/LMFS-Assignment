"""Atomic local checkpoints including optimizer and sampling RNG state."""

import json
from pathlib import Path
import tempfile

import torch


def save_checkpoint(path: Path, payload: dict) -> None:
    """Replace a checkpoint only after its complete contents are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    """Load tensor/scalar states, with completed-run timing if available."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("unsupported language-model checkpoint format")
    timing_path = path.with_suffix(path.suffix + ".json")
    timing = json.loads(timing_path.read_text())
    if (
        timing["run_id"] != saved["run_id"]
        or timing["final_step"] != saved["step"]
    ):
        raise ValueError("checkpoint and timing metadata do not match")
    saved["consumed_seconds"] = timing["cumulative_seconds"]
    return saved


def write_json(path: Path, payload: dict) -> None:
    """Atomically persist human-readable run or timing metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def rng_state(generator) -> dict:
    """Tensor-only RNG serialization is compatible with weights_only=True."""
    return {
        "torch": torch.get_rng_state(),
        "sampling": generator.get_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng(state, generator) -> None:
    torch.set_rng_state(state["torch"])
    generator.set_state(state["sampling"])
    if torch.cuda.is_available() and state["cuda"]:
        for index, value in enumerate(
            state["cuda"][: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(value, device=index)
