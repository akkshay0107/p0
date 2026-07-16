"""Atomic persistence functions for league metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from p0.persistence import atomic_json_save, atomic_torch_save
from p0.training.league.state import LeagueState


def save_league_state(path: Path, state: LeagueState) -> None:
    atomic_json_save(path, state.to_dict())


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
    atomic_torch_save(path, value)
