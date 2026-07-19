"""Small controlled-oracle harness for replay evidence regression fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from p0.replays.protocol import ReplayDocument, parse_replay_payload
from p0.replays.reconstruct import reconstruct_both


@dataclass(frozen=True, slots=True)
class OracleExpectation:
    perspective: int
    decision_index: int
    action: tuple[int, int] | None
    unsupported_tag: str = ""


@dataclass(frozen=True, slots=True)
class OracleCase:
    case_id: str
    payload: bytes | str | Mapping[str, Any]
    expectations: tuple[OracleExpectation, ...]


@dataclass(frozen=True, slots=True)
class OracleFailure:
    expectation: OracleExpectation
    reason: str


@dataclass(frozen=True, slots=True)
class OracleResult:
    case_id: str
    checked: int
    failures: tuple[OracleFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def validate_oracle(case: OracleCase) -> OracleResult:
    """Check exact-or-contained actions and explicit unsupported diagnostics."""
    document: ReplayDocument = parse_replay_payload(case.payload)
    perspectives = reconstruct_both(document)
    failures: list[OracleFailure] = []
    for expectation in case.expectations:
        if expectation.perspective not in (0, 1):
            failures.append(
                OracleFailure(expectation, "perspective is outside the two-player contract")
            )
            continue
        decisions = perspectives[expectation.perspective].decisions
        if expectation.decision_index >= len(decisions):
            failures.append(OracleFailure(expectation, "decision index is absent"))
            continue
        decision = decisions[expectation.decision_index]
        if decision.post_line_index > len(document.protocol_lines):
            failures.append(OracleFailure(expectation, "decision reads beyond the protocol cutoff"))
            continue
        if expectation.action is not None:
            if expectation.action not in decision.evidence.candidates:
                failures.append(
                    OracleFailure(expectation, "known action is not candidate-contained")
                )
        elif (
            expectation.unsupported_tag
            and expectation.unsupported_tag not in decision.evidence.tags
        ):
            failures.append(
                OracleFailure(
                    expectation, f"missing unsupported diagnostic {expectation.unsupported_tag!r}"
                )
            )
    return OracleResult(case.case_id, len(case.expectations), tuple(failures))


__all__ = [
    "OracleCase",
    "OracleExpectation",
    "OracleFailure",
    "OracleResult",
    "validate_oracle",
]
