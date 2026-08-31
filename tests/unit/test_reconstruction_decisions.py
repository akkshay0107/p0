"""Tests for v2 replay decision-window reconstruction."""

from __future__ import annotations

import pytest

from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import (
    BoundaryKind,
    infer_decision_windows,
    reconstruct_replay_decisions_both,
)
from p0.replays.reconstruction.events import parse_replay_events
from p0.replays.reconstruction.resolution import resolve_protocol_events
from tests.unit.replay_fixtures import decision_payload


def test_decision_reconstruction_shares_boundaries_across_perspectives() -> None:
    document = parse_replay_payload(decision_payload())

    first, second = reconstruct_replay_decisions_both(document)

    assert not first.diagnostics
    assert not second.diagnostics
    assert first.decisions
    assert second.decisions
    assert first.windows == second.windows
    assert [window.kind for window in first.windows] == [
        BoundaryKind.RESIDUAL_TERMINAL,
        BoundaryKind.TEAM_PREVIEW,
        BoundaryKind.NORMAL_TURN,
        BoundaryKind.RESIDUAL_TERMINAL,
    ]
    assert [decision.decision_type.name for decision in first.decisions] == [
        "TEAM_PREVIEW",
        "TURN",
    ]
    assert [decision.pre_line_index for decision in first.decisions] == [4, 10]
    assert [decision.post_line_index for decision in first.decisions] == [10, 15]
    assert all(decision.evidence.candidates for decision in first.decisions)


def test_decision_reconstruction_preserves_unknown_action_evidence() -> None:
    payload = decision_payload()
    payload["log"] = str(payload["log"])
    payload["log"] = payload["log"].replace(
        "|move|p1a: Pikachu|Protect|p1a: Pikachu", "|cant|p1a: Pikachu|par|"
    )
    payload["log"] = payload["log"].replace(
        "|move|p1b: Eevee|Tackle|p2b: Charmander", "|cant|p1b: Eevee|par|"
    )
    document = parse_replay_payload(payload)

    result = reconstruct_replay_decisions_both(document)[0]

    assert not result.diagnostics
    assert result.decisions[-1].evidence.label_kind.name == "UNKNOWN"
    assert "no_observed_order" in result.decisions[-1].evidence.tags


def test_actionful_replay_without_separators_is_unrecoverable() -> None:
    payload = decision_payload()
    payload["log"] = str(payload["log"]).replace("\n|\n", "\n")
    document = parse_replay_payload(payload)
    parsed = parse_replay_events(document)
    resolved = resolve_protocol_events(
        document.metadata.replay_id,
        document.ots,
        parsed.events,
    )

    with pytest.raises(ValueError, match="actions but no update separators"):
        infer_decision_windows(resolved.require_accepted())
