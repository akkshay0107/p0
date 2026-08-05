from __future__ import annotations

import pytest

from p0.battle.actions import (
    ActionKind,
    SlotAction,
    canonical_team_actions,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.battle.legality import DecisionView, SlotDecision, legal_actions, validate_joint_action
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.schema import LabelKind
from tests.stress._helpers import capture_showdown_decisions, stress_count


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_observed_showdown_orders_round_trip_to_recorded_actions(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=stress_count("P0_STRESS_ACTION_GAMES", 2),
    )
    assert decisions

    for decision in decisions:
        assert decision.chosen_action in decision.legal_joint_actions
        assert decision.chosen_order


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_showdown_order_sets_remain_nonempty_across_many_requests(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=stress_count("P0_STRESS_ACTION_GAMES", 2),
    )
    assert decisions
    assert all(decision.legal_joint_actions for decision in decisions)
    for decision in decisions:
        projected = tuple(
            sorted({action for pair in decision.legal_joint_actions for action in pair})
        )
        observed = tuple(sorted(set(decision.legal_actions[0] + decision.legal_actions[1])))
        assert set(projected) <= set(observed)
        # These are independently authored protocol invariants, not legality helpers.
        assert all(0 <= action < 49 for action in observed)
        assert any(action in observed for action in (7, 8, 9, 10, 11)) or any(
            action in observed for action in (1, 2, 3, 4, 5, 6, 48)
        )


@pytest.mark.stress
def test_action_ids_cover_all_boundary_categories() -> None:
    expected = {
        0: SlotAction(ActionKind.PASS),
        1: SlotAction(ActionKind.SWITCH, switch_slot=0),
        6: SlotAction(ActionKind.SWITCH, switch_slot=5),
        7: SlotAction(ActionKind.MOVE, move_slot=0, target=-2),
        11: SlotAction(ActionKind.MOVE, move_slot=0, target=2),
        26: SlotAction(ActionKind.MOVE, move_slot=3, target=2),
        27: SlotAction(ActionKind.MOVE, move_slot=0, target=-2, mega=True),
        46: SlotAction(ActionKind.MOVE, move_slot=3, target=2, mega=True),
        47: SlotAction(ActionKind.FORCED_MOVE, mega=True),
        48: SlotAction(ActionKind.FORCED_MOVE),
    }

    for action_id, semantic in expected.items():
        assert decode_action(action_id) == semantic
        assert encode_action(semantic) == action_id


@pytest.mark.stress
def test_team_preview_pairs_and_joint_constraints_preserve_uniqueness() -> None:
    pairs = tuple((first, second) for first in range(6) for second in range(6) if first != second)
    assert tuple(decode_team_pair(encode_team_pair(*pair)) for pair in pairs) == pairs

    selection = (5, 4, 3, 2, 0, 1)
    lead, back = canonical_team_actions(selection)
    assert team_selection(lead, back) == selection

    preview = DecisionView(
        slots=(SlotDecision(), SlotDecision()),
        team_preview=True,
    )
    assert legal_actions(preview, 0) == tuple(encode_team_pair(*pair) for pair in pairs)
    assert validate_joint_action(preview, 1, 15)
    assert not validate_joint_action(preview, 1, 14)

    regular = DecisionView(
        slots=(
            SlotDecision(switch_slots=(0, 1), move_targets=((-2, 2),), can_mega=True),
            SlotDecision(switch_slots=(0, 1), move_targets=((-2, 2),), can_mega=True),
        )
    )
    assert validate_joint_action(regular, 7, 11)
    assert not validate_joint_action(regular, 1, 1)
    assert not validate_joint_action(regular, 27, 31)

    forced = DecisionView(
        slots=(
            SlotDecision(forced_move=True, can_mega=True),
            SlotDecision(forced_move=True, can_mega=True),
        )
    )
    assert legal_actions(forced, 0) == (48, 47)


@pytest.mark.stress
def test_candidate_cap_degrades_to_explicit_unknown_evidence() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(move_targets=((-2, -1),)),
            SlotDecision(move_targets=((-2,),)),
        )
    )
    evidence = extract_action_evidence(
        EvidenceRequest(
            view=view,
            slots=(
                ObservedAction(alternatives=(7, 8), exact=False),
                ObservedAction(action=7),
            ),
            max_candidates=1,
        )
    )

    assert evidence.label_kind is LabelKind.UNKNOWN
    assert evidence.candidates == ()
    assert "candidate_cap_or_illegal" in evidence.tags
