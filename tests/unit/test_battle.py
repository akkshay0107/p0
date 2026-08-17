from __future__ import annotations

import numpy as np
import pytest

from p0.battle.actions import (
    ACT_SIZE,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.battle.events import (
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialTargetSlot,
    SpatialTurnRecorder,
    get_hp_fraction,
)
from p0.battle.legality import (
    DecisionView,
    SlotDecision,
    action_mask,
    apply_joint_constraints,
    legal_actions,
    second_action_mask,
    validate_joint_action,
)
from p0.battle.series import SeriesPerspectiveKey
from p0.format_config import ACTION_CONTRACT
from p0.model.tokenizer import tokenizer


def test_action_contract_round_trips_ids_and_describes_canonical_ranges() -> None:
    """Verify that action encoding/decoding round-trips all 49 discrete actions across all canonical ranges."""
    assert [encode_action(decode_action(action)) for action in range(ACT_SIZE)] == list(
        range(ACT_SIZE)
    )
    assert ACTION_CONTRACT["action_count"] == ACT_SIZE
    ranges = ACTION_CONTRACT["ranges"]
    assert [(entry["start"], entry["end"]) for entry in ranges] == [
        (0, 1),
        (1, 7),
        (7, 27),
        (27, 47),
        (47, 48),
        (48, 49),
    ]
    assert [decode_action(index).kind.name.lower() for index in (0, 1, 7, 27, 47, 48)] == [
        "pass",
        "switch",
        "move",
        "move",
        "forced_move",
        "forced_move",
    ]
    # Team preview lead pairs: 6 choose 2 permutations = 30 distinct ordered lead combinations
    actions = {
        encode_team_pair(first, second)
        for first in range(6)
        for second in range(6)
        if first != second
    }
    assert len(actions) == 30
    assert all(decode_team_pair(action)[0] != decode_team_pair(action)[1] for action in actions)
    assert team_selection(1, 8)[:4] == (0, 1, 2, 3)


def test_scalar_joint_constraints_match_policy_vectorization() -> None:
    """Verify that scalar joint validation (validate_joint_action) matches vectorized second_action_mask."""
    view = DecisionView(
        slots=(
            SlotDecision(
                switch_slots=(2, 3),
                move_targets=((-2, 1, 2), (0,), (), (1,)),
                can_mega=True,
            ),
            SlotDecision(
                switch_slots=(2, 4),
                move_targets=((-1, 1, 2), (), (0,), ()),
                can_mega=False,
            ),
        )
    )
    mask1 = action_mask(view)[0]
    for a1 in range(ACT_SIZE):
        if not mask1[a1]:
            continue
        mask2 = second_action_mask(view, a1)
        for a2 in range(ACT_SIZE):
            assert mask2[a2] == validate_joint_action(view, a1, a2)


def test_series_perspective_key_validation() -> None:
    """Verify format string serialization and parsing of SeriesPerspectiveKey."""
    key = SeriesPerspectiveKey(
        series_id="test-series-123",
        canonical_player=0,
    )
    assert key.series_id == "test-series-123"
    assert key.canonical_player == 0

    with pytest.raises(ValueError, match="must be non-empty"):
        SeriesPerspectiveKey(series_id="", canonical_player=0)

    with pytest.raises(ValueError, match="must be 0 or 1"):
        SeriesPerspectiveKey(series_id="test", canonical_player=2)


def test_action_encoding_and_decoding_boundary_errors() -> None:
    """Verify boundary checks on action encoding and decoding."""
    with pytest.raises(ValueError, match=r"Action must be in \[0, 49\)"):
        decode_action(-1)

    with pytest.raises(ValueError, match=r"Action must be in \[0, 49\)"):
        decode_action(ACT_SIZE)


def test_team_preview_bounds_and_validation_errors() -> None:
    """Verify team preview encoding and selection error handling."""
    with pytest.raises(ValueError, match="Team-preview indices are outside the roster"):
        encode_team_pair(-1, 0)

    with pytest.raises(ValueError, match="Team-preview indices are outside the roster"):
        encode_team_pair(0, 6)

    with pytest.raises(ValueError, match="Team-preview pairs must be distinct"):
        encode_team_pair(2, 2)

    with pytest.raises(ValueError, match="Invalid canonical team-preview action"):
        decode_team_pair(-1)

    with pytest.raises(ValueError, match="Invalid canonical team-preview action"):
        decode_team_pair(36)


def test_double_force_switch_with_single_available_switch() -> None:
    """Verify fallback behavior when both slots are forced to switch but only one replacement is available."""
    view = DecisionView(
        slots=(
            SlotDecision(switch_slots=(2,), force_switch=True),
            SlotDecision(switch_slots=(2,), force_switch=True),
        )
    )
    assert legal_actions(view, 0) == (3, 0)
    assert legal_actions(view, 1) == (3, 0)


def test_apply_joint_constraints_fallback_and_error_resilience() -> None:
    """Verify joint constraint filtering eliminates duplicate switch targets across both slots."""
    preview = DecisionView(slots=(SlotDecision(), SlotDecision()), team_preview=True)
    mask = np.ones(ACT_SIZE, dtype=np.bool_)
    apply_joint_constraints(mask, preview, first=-1)
    assert not mask.any()

    view = DecisionView(
        slots=(
            SlotDecision(switch_slots=(2,)),
            SlotDecision(switch_slots=(2,)),
        )
    )
    slot1_mask = np.zeros(ACT_SIZE, dtype=np.bool_)
    slot1_mask[3] = True
    # If slot 0 took switch to slot 2 (action 3), slot 1 cannot take switch 3 and falls back to pass (0)
    apply_joint_constraints(slot1_mask, view, first=3)
    assert slot1_mask[0]
    assert not slot1_mask[3]


def _struggle_battle(can_mega: bool) -> DecisionView:
    return DecisionView(
        slots=(
            SlotDecision(
                forced_move=True,
                can_mega=can_mega,
            ),
            SlotDecision(),
        )
    )


def test_struggle_env_roundtrip() -> None:
    """Verify forced moves map to action 48 (standard forced move) or 47 (mega-forced move)."""
    view = _struggle_battle(can_mega=False)
    assert list(legal_actions(view, 0)) == [48]

    mega_view = _struggle_battle(can_mega=True)
    mask = list(legal_actions(mega_view, 0))
    assert 48 in mask and 47 in mask


def test_action_ids_cover_all_boundary_categories() -> None:
    """Verify action decoding matches the defined action categories."""
    assert decode_action(0).kind.name == "PASS"
    assert decode_action(1).kind.name == "SWITCH"
    assert decode_action(6).kind.name == "SWITCH"
    assert decode_action(7).kind.name == "MOVE"
    assert decode_action(26).kind.name == "MOVE"
    assert decode_action(27).kind.name == "MOVE"
    assert decode_action(46).kind.name == "MOVE"
    assert decode_action(47).kind.name == "FORCED_MOVE"
    assert decode_action(48).kind.name == "FORCED_MOVE"


def test_team_preview_pairs_and_joint_constraints_preserve_uniqueness() -> None:
    """Verify team preview lead action pairs maintain valid 1-to-1 permutations."""
    pairs = [encode_team_pair(f, s) for f in range(6) for s in range(6) if f != s]
    assert len(pairs) == 30
    for action in pairs:
        f, s = decode_team_pair(action)
        assert f != s
        assert encode_team_pair(f, s) == action


def test_unknown_legality_masks_are_supersets_of_the_proven_mask() -> None:
    """Verify unknown legality generates superset action mask."""
    known_decision = SlotDecision(
        switch_slots=(2, 3),
        move_targets=((-2, 1, 2), (), (), ()),
        can_mega=False,
        legality_known=True,
    )
    unknown_decision = SlotDecision(
        switch_slots=(2, 3),
        move_targets=((-2, 1, 2), (), (), ()),
        can_mega=False,
        legality_known=False,
    )
    view_known = DecisionView(slots=(known_decision, SlotDecision()))
    view_unknown = DecisionView(slots=(unknown_decision, SlotDecision()))

    mask_known = action_mask(view_known)[0]
    mask_unknown = action_mask(view_unknown)[0]

    assert mask_unknown.sum() >= mask_known.sum()
    assert (mask_known & ~mask_unknown).sum() == 0


def test_hp_fraction_accepts_showdown_status_suffixes() -> None:
    """Verify get_hp_fraction correctly strips Showdown health color suffixes ('g', 'y') and handles fainted ('0 fnt')."""
    assert get_hp_fraction("50/100g") == 0.5
    assert get_hp_fraction("20/100y") == 0.2
    assert get_hp_fraction("0 fnt") == 0.0
    assert get_hp_fraction("100/100") == 1.0
    assert get_hp_fraction("invalid") == 0.0


def test_spatial_turn_recorder_turn_lifecycle() -> None:
    """Verify SpatialTurnRecorder initializes cleanly and resets on turn markers."""
    recorder = SpatialTurnRecorder(player_role="p1")
    records = recorder.to_records()
    assert len(records) == SPATIAL_SLOT_COUNT
    assert all(r.action_type == int(SpatialActionType.NONE) for r in records)

    recorder.apply_line(["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"], tokenizer)
    assert recorder.slots[0].action_type == int(SpatialActionType.MOVE)
    assert recorder.slots[0].order_rank == 0.25

    recorder.reset_turn()
    records_reset = recorder.to_records()
    assert all(r.action_type == int(SpatialActionType.NONE) for r in records_reset)


def test_spatial_turn_recorder_move_target_mapping() -> None:
    """Verify move target slot coordinates are mapped correctly from player perspective."""
    # From P1 perspective: p1a=0 (ALLY_LEFT), p1b=1 (ALLY_RIGHT), p2a=2 (OPP_LEFT), p2b=3 (OPP_RIGHT)
    recorder_p1 = SpatialTurnRecorder(player_role="p1")
    recorder_p1.apply_line(["", "move", "p1a: Pikachu", "Thunderbolt", "p2b: Blastoise"], tokenizer)
    assert recorder_p1.slots[0].target_slot == int(SpatialTargetSlot.OPP_RIGHT)

    # From P2 perspective: p2a=0 (ALLY_LEFT), p2b=1 (ALLY_RIGHT), p1a=2 (OPP_LEFT), p1b=3 (OPP_RIGHT)
    recorder_p2 = SpatialTurnRecorder(player_role="p2")
    recorder_p2.apply_line(["", "move", "p2b: Blastoise", "Surf", "p1a: Pikachu"], tokenizer)
    assert recorder_p2.slots[1].target_slot == int(SpatialTargetSlot.OPP_LEFT)


def test_spatial_turn_recorder_damage_and_crit_tracking() -> None:
    """Verify damage dealt accumulation and crit flags on attacker and defender."""
    recorder = SpatialTurnRecorder(player_role="p1")

    # P1A uses Thunderbolt on P2A
    recorder.apply_line(["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"], tokenizer)

    # P2A takes damage (from 1.0 to 0.40 -> delta = -0.60)
    hp_map = {"p2a: Charizard": 1.0}
    recorder.apply_line(
        ["", "-damage", "p2a: Charizard", "40/100"], tokenizer, hp_for=lambda k: hp_map[k]
    )

    # Crit lands on P2A
    recorder.apply_line(["", "-crit", "p2a: Charizard"], tokenizer)

    # Check P2A (slot 2)
    assert recorder.slots[2].hp_delta == pytest.approx(-0.60)
    assert recorder.slots[2].took_crit == 1.0

    # Check P1A (slot 0)
    assert recorder.slots[0].damage_dealt == pytest.approx(0.60)
    assert recorder.slots[0].landed_crit == 1.0


def test_spatial_turn_recorder_boosts_and_failures() -> None:
    """Verify stat boosts and move failure tracking."""
    recorder = SpatialTurnRecorder(player_role="p1")

    # P1B boosts Swords Dance
    recorder.apply_line(["", "-boost", "p1b: Ogerpon", "atk", "2"], tokenizer)
    assert recorder.slots[1].net_boost_delta == pytest.approx(2.0 / 6.0)

    # P2A unboosts
    recorder.apply_line(["", "-unboost", "p2a: Charizard", "spe", "1"], tokenizer)
    assert recorder.slots[2].net_boost_delta == pytest.approx(-1.0 / 6.0)

    # P1A uses move that fails
    recorder.apply_line(["", "move", "p1a: Pikachu", "Thunder Wave", "p2b: Landorus"], tokenizer)
    recorder.apply_line(["", "-immune", "p2b: Landorus"], tokenizer)
    assert recorder.slots[0].move_failed == 1.0


def test_spatial_turn_recorder_switch_faint_item() -> None:
    """Verify switch, faint, cant, and item consumption recording."""
    recorder = SpatialTurnRecorder(player_role="p1")

    recorder.apply_line(
        ["", "switch", "p1a: Incineroar", "Incineroar, L50, M", "100/100"], tokenizer
    )
    assert recorder.slots[0].action_type == int(SpatialActionType.SWITCH)

    recorder.apply_line(["", "faint", "p2a: Charizard"], tokenizer)
    assert recorder.slots[2].action_type == int(SpatialActionType.FAINT)

    recorder.apply_line(["", "-enditem", "p1a: Incineroar", "Sitrus Berry"], tokenizer)
    assert recorder.slots[0].item_consumed == 1.0
