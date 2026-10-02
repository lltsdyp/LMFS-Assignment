"""Provides pickle file I/O without managing tokenizer state or model paths."""

from __future__ import annotations

import pathlib
import pickle
from typing import Any


def write_pickle(path: pathlib.Path, data: object) -> None:
    """Writes a Python object to a pickle file."""
    with path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: pathlib.Path) -> Any:
    """Reads a Python object from a trusted pickle file."""
    with path.open("rb") as stream:
        return pickle.load(stream)
