"""Atomic persistence primitives shared by durable artifact owners."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import orjson
import torch


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Publish a complete file, then synchronize its containing directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)

    try:
        os.close(handle)
        yield temporary_path
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON file only after serialization succeeds."""
    encoded = orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    with atomic_output(path) as temporary:
        temporary.write_bytes(encoded + b"\n")


def atomic_torch_save(path: Path, value: Any) -> None:
    """Replace a checkpoint only after serialization succeeds."""
    with atomic_output(path) as temporary:
        torch.save(value, temporary)
