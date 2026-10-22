"""Structured diagnostics for event parsing and replay rejection."""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class ReplayRejectionCategory(StrEnum):
    """Stable top-level categories for replay rejection diagnostics."""

    UNSUPPORTED_EVENT = "UNSUPPORTED_EVENT"
    AMBIGUOUS_IDENTITY = "AMBIGUOUS_IDENTITY"
    INCONSISTENT_IDENTITY = "INCONSISTENT_IDENTITY"
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"
    INVALID_INPUT_CONTRACT = "INVALID_INPUT_CONTRACT"


class ReplayEventDiagnostic(NamedTuple):
    """Evidence explaining why one protocol line rejects a replay."""

    replay_id: str
    line_index: int
    tag: str
    normalized_effect: str
    normalized_cause: str
    raw_line: str
    reason: str
    category: ReplayRejectionCategory = ReplayRejectionCategory.INVALID_INPUT_CONTRACT


class ReplayEventParseError(ValueError):
    """Raised when a replay contains malformed or unsupported state events."""

    def __init__(self, diagnostics: tuple[ReplayEventDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("ReplayEventParseError requires at least one diagnostic")
        self.diagnostics = diagnostics
        self.category = diagnostics[0].category
        first = diagnostics[0]
        super().__init__(
            f"Replay {first.replay_id!r} rejected at line {first.line_index}: {first.reason}"
        )


__all__ = [
    "ReplayEventDiagnostic",
    "ReplayEventParseError",
    "ReplayRejectionCategory",
]
