"""Atomic persistence functions for league metadata."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from p0.training.league.state import LeagueState


def save_league_state(path: Path, state: LeagueState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = state.to_dict()
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(descriptor, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_league_state(path: Path) -> LeagueState | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed league state at {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Malformed league state at {path}")
    return LeagueState.from_mapping(value)


def save_torch_artifact(path: Path, value: Any) -> None:
    """Atomically replace a tensor-bearing league artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
