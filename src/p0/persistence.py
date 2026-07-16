"""Atomic persistence primitives shared by durable artifact owners."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def atomic_json_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_torch_save(path: Path, value: Any) -> None:
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
