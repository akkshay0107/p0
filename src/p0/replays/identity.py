"""Pure Showdown identifier and replay-link normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import orjson

_HREF_PATTERN = re.compile(r"""href=["']([^"']+)["']""", flags=re.IGNORECASE)


class ReplaySide(StrEnum):
    """A stable side in a replay rather than a player-relative role."""

    P1 = "p1"
    P2 = "p2"

    @property
    def side_index(self) -> int:
        """Return the zero-based side index."""
        return 0 if self is ReplaySide.P1 else 1


@dataclass(frozen=True, slots=True, order=True)
class ReplayMemberId:
    """Stable identity of one member in an ordered open team sheet."""

    side: ReplaySide
    roster_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.side, ReplaySide):
            raise TypeError("ReplayMemberId.side must be a ReplaySide")
        if type(self.roster_index) is not int or not 0 <= self.roster_index < 6:
            raise ValueError("ReplayMemberId.roster_index must be in [0, 6)")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the member identity without losing its side."""
        return {"side": self.side.value, "roster_index": self.roster_index}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReplayMemberId:
        """Deserialize a strict member identity record."""
        if len(value) != 2 or "side" not in value or "roster_index" not in value:
            raise ValueError("Invalid ReplayMemberId fields")
        side = value["side"]
        roster_index = value["roster_index"]
        if not isinstance(side, str) or type(roster_index) is not int:
            raise ValueError("ReplayMemberId fields have invalid types")
        return cls(ReplaySide(side), roster_index)


def normalize_showdown_id(value: str) -> str:
    """Match Showdown's lowercase-alphanumeric identifier convention."""
    return "".join(
        character for character in value.casefold() if character.isascii() and character.isalnum()
    )


def canonical_format_id(
    value: Mapping[str, Any],
    *,
    expected: str | None = None,
) -> str | None:
    """Return a machine-format id from either API spelling or display text."""
    expected_id = None if expected is None else normalize_showdown_id(expected)
    for field in ("formatid", "format_id", "format"):
        candidate = value.get(field)
        if not isinstance(candidate, str) or not candidate:
            continue
        normalized = normalize_showdown_id(candidate)
        if expected_id is not None and normalized == expected_id:
            return expected_id
        if normalized:
            return normalized
    return expected_id


def replay_matches_format(item: object, expected: str) -> bool:
    """Accept search records using either machine ids or Showdown display names."""
    if not isinstance(item, Mapping):
        return isinstance(item, str) and _replay_id_matches(item, expected)
    replay_id = item.get("id", item.get("replay_id"))
    if isinstance(replay_id, str) and _replay_id_matches(replay_id, expected):
        return True
    return canonical_format_id(item, expected=expected) == normalize_showdown_id(expected)


def linked_replay_ids(payload: bytes, *, format_id: str) -> tuple[str, ...]:
    """Extract same-format sibling battle links without interpreting battle state."""
    try:
        value = orjson.loads(payload)
    except (UnicodeDecodeError, orjson.JSONDecodeError):
        return ()
    if not isinstance(value, Mapping):
        return ()
    log = value.get("log")
    if isinstance(log, str):
        lines = log.splitlines()
    elif isinstance(log, list) and all(isinstance(line, str) for line in log):
        lines = log
    else:
        return ()
    expected = normalize_showdown_id(format_id)
    links: set[str] = set()
    for line in lines:
        if not line.startswith(("|uhtml|", "|uhtmlchange|", "|html|")):
            continue
        for match in _HREF_PATTERN.finditer(line):
            path = urlparse(match.group(1)).path
            candidate = path.rsplit("/", 1)[-1].removeprefix("battle-")
            if _replay_id_matches(candidate, expected):
                links.add(candidate)
    return tuple(sorted(links))


def _replay_id_matches(replay_id: str, expected: str) -> bool:
    expected_id = normalize_showdown_id(expected)
    return replay_id.casefold().startswith(f"{expected_id}-")


__all__ = [
    "ReplayMemberId",
    "ReplaySide",
    "canonical_format_id",
    "linked_replay_ids",
    "normalize_showdown_id",
    "replay_matches_format",
]
