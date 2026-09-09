"""Atomic persistence primitives shared by durable artifact owners."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson
import torch


def atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write JSON data to disk using temporary file replacement."""
    encoded = orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)

    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
    """Atomically write torch checkpoint data to disk using temporary file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)

    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())

        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
