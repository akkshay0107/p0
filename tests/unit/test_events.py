"""Tests for spatial battle event recording and health parsing."""

from __future__ import annotations

import pytest

from p0.battle.events import (
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialTargetSlot,
    SpatialTurnRecorder,
    get_hp_fraction,
)
from p0.model.tokenizer import tokenizer


class TestEvents:
    def test_hp_fraction_accepts_showdown_status_suffixes(self) -> None:
        """Verify get_hp_fraction correctly strips Showdown health color suffixes ('g', 'y') and handles fainted ('0 fnt')."""
        assert get_hp_fraction("50/100g") == 0.5
        assert get_hp_fraction("20/100y") == 0.2
        assert get_hp_fraction("0 fnt") == 0.0
        assert get_hp_fraction("100/100") == 1.0
        assert get_hp_fraction("invalid") == 0.0

    def test_spatial_turn_recorder_turn_lifecycle(self) -> None:
        """Verify SpatialTurnRecorder initializes cleanly and resets on turn markers."""
        recorder = SpatialTurnRecorder(player_role="p1")
        records = recorder.to_records()
        assert len(records) == SPATIAL_SLOT_COUNT
        assert all(r.action_type == int(SpatialActionType.NONE) for r in records)

        recorder.apply_line(
            ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"], tokenizer
        )
        assert recorder.slots[0].action_type == int(SpatialActionType.MOVE)
        assert recorder.slots[0].order_rank == 0.25

        recorder.reset_turn()
        records_reset = recorder.to_records()
        assert all(r.action_type == int(SpatialActionType.NONE) for r in records_reset)

    def test_spatial_turn_recorder_move_target_mapping(self) -> None:
        """Verify move target slot coordinates are mapped correctly from player perspective."""
        # From P1 perspective: p1a=0 (ALLY_LEFT), p1b=1 (ALLY_RIGHT), p2a=2 (OPP_LEFT), p2b=3 (OPP_RIGHT)
        recorder_p1 = SpatialTurnRecorder(player_role="p1")
        recorder_p1.apply_line(
            ["", "move", "p1a: Pikachu", "Thunderbolt", "p2b: Blastoise"], tokenizer
        )
        assert recorder_p1.slots[0].target_slot == int(SpatialTargetSlot.OPP_RIGHT)

        # From P2 perspective: p2a=0 (ALLY_LEFT), p2b=1 (ALLY_RIGHT), p1a=2 (OPP_LEFT), p1b=3 (OPP_RIGHT)
        recorder_p2 = SpatialTurnRecorder(player_role="p2")
        recorder_p2.apply_line(["", "move", "p2b: Blastoise", "Surf", "p1a: Pikachu"], tokenizer)
        assert recorder_p2.slots[1].target_slot == int(SpatialTargetSlot.OPP_LEFT)

    def test_spatial_turn_recorder_damage_and_crit_tracking(self) -> None:
        """Verify damage dealt accumulation and crit flags on attacker and defender."""
        recorder = SpatialTurnRecorder(player_role="p1")

        # P1A uses Thunderbolt on P2A
        recorder.apply_line(
            ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"], tokenizer
        )

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

    def test_spatial_turn_recorder_boosts_and_failures(self) -> None:
        """Verify stat boosts and move failure tracking."""
        recorder = SpatialTurnRecorder(player_role="p1")

        # P1B boosts Swords Dance
        recorder.apply_line(["", "-boost", "p1b: Ogerpon", "atk", "2"], tokenizer)
        assert recorder.slots[1].net_boost_delta == pytest.approx(2.0 / 6.0)

        # P2A unboosts
        recorder.apply_line(["", "-unboost", "p2a: Charizard", "spe", "1"], tokenizer)
        assert recorder.slots[2].net_boost_delta == pytest.approx(-1.0 / 6.0)

        # P1A uses move that fails
        recorder.apply_line(
            ["", "move", "p1a: Pikachu", "Thunder Wave", "p2b: Landorus"], tokenizer
        )
        recorder.apply_line(["", "-immune", "p2b: Landorus"], tokenizer)
        assert recorder.slots[0].move_failed == 1.0

    def test_spatial_turn_recorder_switch_faint_item(self) -> None:
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
