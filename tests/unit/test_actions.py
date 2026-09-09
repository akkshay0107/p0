"""Tests for battle action encoding, decoding, and team selection."""

from __future__ import annotations

import pytest

from p0.battle.actions import (
    ACT_SIZE,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.format_config import ACTION_CONTRACT, FORMAT


class TestActions:
    def test_action_contract_round_trips_ids_and_describes_canonical_ranges(self) -> None:
        """Verify that action encoding/decoding round-trips all 49 discrete actions across all canonical ranges."""
        assert [encode_action(decode_action(action)) for action in range(ACT_SIZE)] == list(
            range(ACT_SIZE)
        )
        assert ACTION_CONTRACT["action_count"] == ACT_SIZE
        assert FORMAT.action_size == ACT_SIZE
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

    def test_action_ids_cover_all_boundary_categories(self) -> None:
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

    def test_team_preview_pairs_and_joint_constraints_preserve_uniqueness(self) -> None:
        """Verify team preview lead action pairs maintain valid 1-to-1 permutations."""
        pairs = [encode_team_pair(f, s) for f in range(6) for s in range(6) if f != s]
        assert len(pairs) == 30
        for action in pairs:
            f, s = decode_team_pair(action)
            assert f != s
            assert encode_team_pair(f, s) == action
