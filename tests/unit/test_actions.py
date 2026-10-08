"""Tests for battle action encoding, decoding, and team selection."""

from __future__ import annotations

import pytest

from p0.battle.actions import (
    ACT_SIZE,
    TEAM_SIZE,
    ActionKind,
    SlotAction,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.format_config import FORMAT


class TestActions:
    def test_action_codec_matches_literal_contract_boundaries(self) -> None:
        """Pin literal action meanings at every category boundary."""
        assert ACT_SIZE == 49
        assert FORMAT.action_size == 49
        action_cases = (
            (0, SlotAction(ActionKind.PASS)),
            (1, SlotAction(ActionKind.SWITCH, switch_slot=0)),
            (6, SlotAction(ActionKind.SWITCH, switch_slot=5)),
            (7, SlotAction(ActionKind.MOVE, move_slot=0, target=-2)),
            (10, SlotAction(ActionKind.MOVE, move_slot=0, target=1)),
            (26, SlotAction(ActionKind.MOVE, move_slot=3, target=2)),
            (27, SlotAction(ActionKind.MOVE, move_slot=0, target=-2, mega=True)),
            (46, SlotAction(ActionKind.MOVE, move_slot=3, target=2, mega=True)),
            (47, SlotAction(ActionKind.FORCED_MOVE, mega=True)),
            (48, SlotAction(ActionKind.FORCED_MOVE)),
        )
        for action_id, expected_action in action_cases:
            assert decode_action(action_id) == expected_action
            assert encode_action(expected_action) == action_id
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

    def test_action_encoding_and_decoding_boundary_errors(self) -> None:
        """Verify boundary checks on action encoding and decoding."""
        with pytest.raises(ValueError, match=r"Action must be in \[0, 49\)"):
            decode_action(-1)

        with pytest.raises(ValueError, match=r"Action must be in \[0, 49\)"):
            decode_action(ACT_SIZE)

    def test_team_preview_bounds_and_validation_errors(self) -> None:
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

    def test_team_preview_pairs_round_trip_without_collisions(self) -> None:
        pairs = [
            (first, second)
            for first in range(TEAM_SIZE)
            for second in range(TEAM_SIZE)
            if first != second
        ]
        encoded = [encode_team_pair(first, second) for first, second in pairs]

        assert len(set(encoded)) == TEAM_SIZE * (TEAM_SIZE - 1)
        assert [decode_team_pair(action) for action in encoded] == pairs
