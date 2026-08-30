"""Structured diagnostics for event parsing and replay rejection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReplayEventDiagnostic:
    """Evidence explaining why one protocol line rejects a replay."""

    replay_id: str
    line_index: int
    tag: str
    normalized_effect: str
    normalized_cause: str
    raw_line: str
    reason: str

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("ReplayEventDiagnostic.replay_id must not be empty")
        if type(self.line_index) is not int or self.line_index < 0:
            raise ValueError("ReplayEventDiagnostic.line_index must be nonnegative")
        if not self.raw_line.startswith("|"):
            raise ValueError("ReplayEventDiagnostic.raw_line must be a protocol line")
        if not self.reason:
            raise ValueError("ReplayEventDiagnostic.reason must not be empty")


class ReplayEventParseError(ValueError):
    """Raised when a replay contains malformed or unsupported state events."""

    def __init__(self, diagnostics: tuple[ReplayEventDiagnostic, ...]) -> None:
        if not diagnostics:
            raise ValueError("ReplayEventParseError requires at least one diagnostic")
        self.diagnostics = diagnostics
        first = diagnostics[0]
        super().__init__(
            f"Replay {first.replay_id!r} rejected at line {first.line_index}: {first.reason}"
        )


__all__ = ["ReplayEventDiagnostic", "ReplayEventParseError"]
