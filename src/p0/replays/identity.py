"""Pure Showdown identifier and replay-link normalization helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

_HREF_PATTERN = re.compile(r"""href=["']([^"']+)["']""", flags=re.IGNORECASE)


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
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
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
    "canonical_format_id",
    "linked_replay_ids",
    "normalize_showdown_id",
    "replay_matches_format",
]
