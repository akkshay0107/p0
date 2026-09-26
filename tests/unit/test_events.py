"""Tests for ordered spatial battle event recording and health parsing."""

from __future__ import annotations

import logging

import pytest

from p0.battle.events import (
    MAX_EVENT_RECORDS,
    EffectNamespace,
    EventDetail,
    EventKind,
    EventPosition,
    EventRecord,
    SpatialEventRecorder,
    get_hp_fraction,
)
from p0.model.tokenizer import tokenizer


def _no_hp(_identifier: str) -> float | None:
    return None


def _move_id(name: str) -> int:
    move_id, _ = tokenizer.resolve("moves", name)
    assert move_id != 0
    return move_id


class TestHpFraction:
    def test_accepts_showdown_status_suffixes(self) -> None:
        assert get_hp_fraction("50/100g") == 0.5
        assert get_hp_fraction("20/100y") == 0.2
        assert get_hp_fraction("0 fnt") == 0.0
        assert get_hp_fraction("100/100") == 1.0
        assert get_hp_fraction("invalid") == 0.0


class TestSpatialEventRecorder:
    def test_turn_marker_does_not_discard_the_previous_exchange(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")
        hp = {"p2a: Charizard": 1.0}

        recorder.apply_line(
            ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"], tokenizer, _no_hp
        )
        recorder.apply_line(["", "-damage", "p2a: Charizard", "40/100"], tokenizer, hp.get)
        recorder.apply_line(["", "upkeep"], tokenizer, _no_hp)
        recorder.apply_line(["", "turn", "2"], tokenizer, _no_hp)

        move, damage = recorder.consume()
        assert move == EventRecord(
            EventKind.MOVE,
            EventPosition.OWN_LEFT,
            EventPosition.OPPONENT_LEFT,
            _move_id("Thunderbolt"),
        )
        assert damage[:3] == (EventKind.DAMAGE, EventPosition.OWN_LEFT, EventPosition.OPPONENT_LEFT)
        assert (damage.amount, damage.amount_known) == (pytest.approx(-0.6), 1.0)
        recorder.apply_line(["", ""], tokenizer, _no_hp)
        assert recorder.pending() == ()

    def test_retried_decision_observes_the_same_interval(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")
        recorder.apply_line(
            ["", "switch", "p2a: Charizard", "Charizard, L50", "100/100"], tokenizer, _no_hp
        )

        first = recorder.consume()
        retried = recorder.consume()
        assert recorder.consumed is True
        recorder.apply_line(
            ["", "move", "p2a: Charizard", "Flamethrower", "p1a: Pikachu"], tokenizer, _no_hp
        )

        assert (
            first == retried == (EventRecord(EventKind.SWITCH, target=EventPosition.OPPONENT_LEFT),)
        )
        assert recorder.consumed is False
        assert [record.kind for record in recorder.pending()] == [EventKind.MOVE]

    def test_positions_follow_the_recording_player(self) -> None:
        line = ["", "move", "p2b: Blastoise", "Surf", "p1a: Pikachu"]
        p1_recorder = SpatialEventRecorder(player_role="p1")
        p2_recorder = SpatialEventRecorder(player_role="p2")

        p1_recorder.apply_line(line, tokenizer, _no_hp)
        p2_recorder.apply_line(line, tokenizer, _no_hp)

        assert p1_recorder.pending()[0][1:3] == (
            EventPosition.OPPONENT_RIGHT,
            EventPosition.OWN_LEFT,
        )
        assert p2_recorder.pending()[0][1:3] == (
            EventPosition.OWN_RIGHT,
            EventPosition.OPPONENT_LEFT,
        )

    def test_spread_damage_is_recorded_per_target_and_per_attacker(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")
        hp = {"p2a: Amoonguss": 1.0, "p2b: Incineroar": 0.8, "p1a: Garchomp": 1.0}

        recorder.apply_line(
            ["", "move", "p1a: Garchomp", "Rock Slide", "p2a: Amoonguss", "[spread] p2a,p2b"],
            tokenizer,
            _no_hp,
        )
        recorder.apply_line(["", "-damage", "p2a: Amoonguss", "70/100"], tokenizer, hp.get)
        recorder.apply_line(["", "-damage", "p2b: Incineroar", "50/100"], tokenizer, hp.get)
        recorder.apply_line(
            ["", "move", "p2b: Incineroar", "Flare Blitz", "p1a: Garchomp"], tokenizer, _no_hp
        )
        recorder.apply_line(["", "-damage", "p1a: Garchomp", "55/100"], tokenizer, hp.get)

        damage = [record for record in recorder.pending() if record.kind == EventKind.DAMAGE]
        assert [(record.source, record.target) for record in damage] == [
            (EventPosition.OWN_LEFT, EventPosition.OPPONENT_LEFT),
            (EventPosition.OWN_LEFT, EventPosition.OPPONENT_RIGHT),
            (EventPosition.OPPONENT_RIGHT, EventPosition.OWN_LEFT),
        ]
        assert [record.amount for record in damage] == pytest.approx([-0.3, -0.3, -0.45])

    def test_residual_damage_is_not_credited_to_the_last_attacker(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(
            ["", "move", "p2a: Flutter Mane", "Moonblast", "p1a: Garchomp"], tokenizer, _no_hp
        )
        recorder.apply_line(
            ["", "-damage", "p2a: Flutter Mane", "90/100", "[from] item: Life Orb"],
            tokenizer,
            _no_hp,
        )
        recorder.apply_line(
            [
                "",
                "-damage",
                "p2a: Flutter Mane",
                "80/100",
                "[from] item: Rocky Helmet",
                "[of] p1a: Garchomp",
            ],
            tokenizer,
            _no_hp,
        )

        residual = recorder.pending()[1:]
        assert [(record.source, record.target) for record in residual] == [
            (EventPosition.OPPONENT_LEFT, EventPosition.OPPONENT_LEFT),
            (EventPosition.OWN_LEFT, EventPosition.OPPONENT_LEFT),
        ]

    def test_unknown_previous_hp_is_not_recorded_as_zero_change(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(["", "-heal", "p1a: Incineroar", "75/100"], tokenizer, _no_hp)

        record = recorder.pending()[0]
        assert (record.kind, record.amount, record.amount_known) == (EventKind.HEAL, 0.0, 0.0)

    def test_ability_stat_drops_are_credited_to_the_ability_holder(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(
            ["", "switch", "p2a: Incineroar", "Incineroar, L50, M", "100/100"], tokenizer, _no_hp
        )
        recorder.apply_line(
            ["", "-ability", "p2a: Incineroar", "Intimidate", "boost"], tokenizer, _no_hp
        )
        recorder.apply_line(["", "-unboost", "p1a: Garchomp", "atk", "1"], tokenizer, _no_hp)
        recorder.apply_line(["", "-unboost", "p1b: Rillaboom", "atk", "1"], tokenizer, _no_hp)
        recorder.apply_line(
            ["", "move", "p1b: Rillaboom", "Snarl", "p2a: Incineroar"], tokenizer, _no_hp
        )
        recorder.apply_line(["", "-unboost", "p2a: Incineroar", "spa", "1"], tokenizer, _no_hp)

        assert recorder.pending()[1] == EventRecord(
            EventKind.ABILITY,
            EventPosition.OPPONENT_LEFT,
            detail=EventDetail.ABILITY_REPORTED,
            ability_id=tokenizer.id_for("abilities", "Intimidate"),
        )
        boosts = [record for record in recorder.pending() if record.kind == EventKind.BOOST]
        assert [r.ability_id for r in boosts] == [
            tokenizer.id_for("abilities", "Intimidate"),
            tokenizer.id_for("abilities", "Intimidate"),
            0,
        ]
        assert [(r.source, r.target, r.detail, r.amount) for r in boosts] == [
            (EventPosition.OPPONENT_LEFT, EventPosition.OWN_LEFT, EventDetail.ATK, -1 / 6),
            (EventPosition.OPPONENT_LEFT, EventPosition.OWN_RIGHT, EventDetail.ATK, -1 / 6),
            (EventPosition.OWN_RIGHT, EventPosition.OPPONENT_LEFT, EventDetail.SPA, -1 / 6),
        ]

    def test_protect_block_records_the_blocked_attacker(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(
            ["", "move", "p1a: Urshifu", "Close Combat", "p2b: Kingambit"], tokenizer, _no_hp
        )
        recorder.apply_line(["", "-activate", "p2b: Kingambit", "move: Protect"], tokenizer, _no_hp)

        assert recorder.pending()[1] == EventRecord(
            EventKind.ACTIVATE,
            EventPosition.OWN_LEFT,
            EventPosition.OPPONENT_RIGHT,
            _move_id("Protect"),
        )

    def test_attack_faint_and_replacement_in_one_slot_stay_ordered(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(
            ["", "move", "p2a: Chien-Pao", "Icicle Crash", "p1a: Garchomp"], tokenizer, _no_hp
        )
        recorder.apply_line(["", "faint", "p2a: Chien-Pao"], tokenizer, _no_hp)
        recorder.apply_line(
            ["", "switch", "p2a: Amoonguss", "Amoonguss, L50, F", "100/100"], tokenizer, _no_hp
        )

        assert [(r.kind, r.source, r.target) for r in recorder.pending()] == [
            (EventKind.MOVE, EventPosition.OPPONENT_LEFT, EventPosition.OWN_LEFT),
            (EventKind.FAINT, EventPosition.NONE, EventPosition.OPPONENT_LEFT),
            (EventKind.SWITCH, EventPosition.NONE, EventPosition.OPPONENT_LEFT),
        ]

    def test_ally_switch_records_both_positions(self) -> None:
        recorder = SpatialEventRecorder(player_role="p2")

        recorder.apply_line(["", "swap", "p2a: Indeedee", "1", ""], tokenizer, _no_hp)

        assert recorder.pending() == (
            EventRecord(EventKind.SWAP, EventPosition.OWN_LEFT, EventPosition.OWN_RIGHT),
        )

    def test_swap_with_invalid_or_out_of_range_position_defaults_target_to_none(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(["", "swap", "p1a: Indeedee", "3", ""], tokenizer, _no_hp)
        recorder.apply_line(["", "swap", "unknown_mon", "1", ""], tokenizer, _no_hp)

        assert recorder.pending() == (
            EventRecord(EventKind.SWAP, EventPosition.OWN_LEFT, EventPosition.NONE),
            EventRecord(EventKind.SWAP, EventPosition.NONE, EventPosition.NONE),
        )

    def test_cant_without_reason_records_cleanly(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(["", "cant", "p1a: Pikachu"], tokenizer, _no_hp)

        assert recorder.pending() == (
            EventRecord(
                EventKind.CANT, target=EventPosition.OWN_LEFT, detail=EventDetail.CANT_OTHER
            ),
        )

    def test_boost_with_invalid_stage_is_ignored(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(["", "-boost", "p1a: Pikachu", "atk", "invalid"], tokenizer, _no_hp)

        assert recorder.pending() == ()

    def test_flinch_and_blocked_move_reasons(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(["", "cant", "p1b: Ogerpon", "flinch"], tokenizer, _no_hp)
        recorder.apply_line(
            ["", "cant", "p2a: Farigiraf", "move: Taunt", "Trick Room"], tokenizer, _no_hp
        )

        assert recorder.pending() == (
            EventRecord(EventKind.CANT, target=EventPosition.OWN_RIGHT, detail=EventDetail.FLINCH),
            EventRecord(
                EventKind.CANT,
                target=EventPosition.OPPONENT_LEFT,
                move_id=_move_id("Trick Room"),
                detail=EventDetail.CANT_TAUNT,
            ),
        )

    def test_item_reveal_and_removal_are_distinct(self) -> None:
        recorder = SpatialEventRecorder(player_role="p1")

        recorder.apply_line(["", "-item", "p2a: Tornadus", "Covert Cloak"], tokenizer, _no_hp)
        recorder.apply_line(
            ["", "-enditem", "p1a: Incineroar", "Sitrus Berry", "[eat]"], tokenizer, _no_hp
        )

        assert [(r.target, r.detail) for r in recorder.pending()] == [
            (EventPosition.OPPONENT_LEFT, EventDetail.ITEM_PRESENT),
            (EventPosition.OWN_LEFT, EventDetail.ITEM_EATEN),
        ]

    def test_overflow_keeps_the_earliest_records_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = SpatialEventRecorder(player_role="p1")
        for _ in range(MAX_EVENT_RECORDS + 3):
            recorder.apply_line(["", "faint", "p2a: Charizard"], tokenizer, _no_hp)
        recorder.apply_line(["", "faint", "p1a: Pikachu"], tokenizer, _no_hp)

        with caplog.at_level(logging.WARNING, logger="p0.battle.events"):
            records = recorder.consume()

        assert len(records) == MAX_EVENT_RECORDS
        assert all(record.target == EventPosition.OPPONENT_LEFT for record in records)
        assert "Dropped 4 event records" in caplog.text
        recorder.apply_line(["", "faint", "p1a: Pikachu"], tokenizer, _no_hp)
        assert len(recorder.pending()) == 1

    @pytest.mark.parametrize(
        ("reason", "detail"),
        [
            ("par", EventDetail.CANT_PARALYSIS),
            ("slp", EventDetail.CANT_SLEEP),
            ("frz", EventDetail.CANT_FREEZE),
            ("recharge", EventDetail.CANT_RECHARGE),
            ("nopp", EventDetail.CANT_NO_PP),
            ("Attract", EventDetail.CANT_ATTRACT),
            ("Disable", EventDetail.CANT_DISABLE),
            ("Focus Punch", EventDetail.CANT_FOCUS_PUNCH),
            ("Shell Trap", EventDetail.CANT_SHELL_TRAP),
            ("move: Gravity", EventDetail.CANT_GRAVITY),
            ("move: Heal Block", EventDetail.CANT_HEAL_BLOCK),
            ("move: Imprison", EventDetail.CANT_IMPRISON),
            ("move: Throat Chop", EventDetail.CANT_THROAT_CHOP),
            ("unexpected", EventDetail.CANT_OTHER),
        ],
    )
    def test_cant_reasons(self, reason: str, detail: EventDetail) -> None:
        recorder = SpatialEventRecorder()
        recorder.apply_line(f"|cant|p1b: X|{reason}|Protect".split("|"), tokenizer, _no_hp)
        assert recorder.pending() == (
            EventRecord(
                EventKind.CANT,
                target=EventPosition.OWN_RIGHT,
                move_id=_move_id("Protect"),
                detail=detail,
            ),
        )

    @pytest.mark.parametrize(
        "ability", ["Armor Tail", "Damp", "Dazzling", "Queenly Majesty", "Truant"]
    )
    def test_cant_ability_participants(self, ability: str) -> None:
        recorder = SpatialEventRecorder()
        line = f"|cant|p2a: X|ability: {ability}"
        if ability != "Truant":
            line += "|Quick Attack|[of] p1b: Y"
        recorder.apply_line(line.split("|"), tokenizer, _no_hp)
        assert recorder.pending() == (
            EventRecord(
                EventKind.CANT,
                EventPosition.OPPONENT_LEFT,
                EventPosition.OPPONENT_LEFT if ability == "Truant" else EventPosition.OWN_RIGHT,
                0 if ability == "Truant" else _move_id("Quick Attack"),
                EventDetail.CANT_ABILITY,
                ability_id=tokenizer.id_for("abilities", ability),
            ),
        )

    @pytest.mark.parametrize(
        ("line", "detail", "source", "move", "ability", "item"),
        [
            (
                "|-item|p2a: X|Leftovers|[from] ability: Frisk|[of] p1b: Y",
                EventDetail.ITEM_PRESENT,
                EventPosition.OWN_RIGHT,
                "",
                "Frisk",
                "Leftovers",
            ),
            (
                "|-enditem|p2a: X|Leftovers|[from] move: Knock Off|[of] p1b: Y",
                EventDetail.ITEM_REMOVED_BY_MOVE,
                EventPosition.OWN_RIGHT,
                "Knock Off",
                "",
                "Leftovers",
            ),
            (
                "|-item|p2a: X|Leftovers|[from] move: Thief|[of] p1b: Y",
                EventDetail.ITEM_PRESENT,
                EventPosition.OPPONENT_LEFT,
                "Thief",
                "",
                "Leftovers",
            ),
            (
                "|-item|p2a: X|Leftovers|[from] ability: Pickpocket|[of] p1b: Y",
                EventDetail.ITEM_PRESENT,
                EventPosition.OPPONENT_LEFT,
                "",
                "Pickpocket",
                "Leftovers",
            ),
            (
                "|-enditem|p2a: X|Sitrus Berry|[eat]",
                EventDetail.ITEM_EATEN,
                EventPosition.OPPONENT_LEFT,
                "",
                "",
                "Sitrus Berry",
            ),
            (
                "|-enditem|p2a: X|Occa Berry|[weaken]",
                EventDetail.ITEM_USED,
                EventPosition.OPPONENT_LEFT,
                "",
                "",
                "Occa Berry",
            ),
            (
                "|-enditem|p2a: X|Normal Gem|[from] gem",
                EventDetail.ITEM_USED,
                EventPosition.OPPONENT_LEFT,
                "",
                "",
                "Normal Gem",
            ),
            (
                "|-enditem|p2a: X|Focus Sash",
                EventDetail.ITEM_ENDED,
                EventPosition.NONE,
                "",
                "",
                "Focus Sash",
            ),
            (
                "|-enditem|p2a: X|Sitrus Berry|[from] stealeat|[move] Bug Bite|[of] p1b: Y",
                EventDetail.ITEM_STOLEN_AND_EATEN,
                EventPosition.OWN_RIGHT,
                "Bug Bite",
                "",
                "Sitrus Berry",
            ),
        ],
    )
    def test_item_identity_and_cause(
        self,
        line: str,
        detail: EventDetail,
        source: EventPosition,
        move: str,
        ability: str,
        item: str,
    ) -> None:
        recorder = SpatialEventRecorder()
        recorder.apply_line(line.split("|"), tokenizer, _no_hp)
        assert recorder.pending() == (
            EventRecord(
                EventKind.ITEM,
                source,
                EventPosition.OPPONENT_LEFT,
                tokenizer.id_for("moves", move),
                detail,
                tokenizer.id_for("items", item),
                tokenizer.id_for("abilities", ability),
            ),
        )

    def test_trick_keeps_only_reported_item_results(self) -> None:
        recorder = SpatialEventRecorder()
        for line in (
            "|move|p1a: X|Trick|p2a: Y",
            "|-item|p2a: Y|Choice Scarf|[from] move: Trick",
            "|-item|p1a: X|Leftovers|[from] move: Trick",
        ):
            recorder.apply_line(line.split("|"), tokenizer, _no_hp)
        assert [r.kind for r in recorder.pending()] == [
            EventKind.MOVE,
            EventKind.ITEM,
            EventKind.ITEM,
        ]
        assert [(r.source, r.target, r.item_id) for r in recorder.pending()[1:]] == [
            (
                EventPosition.NONE,
                EventPosition.OPPONENT_LEFT,
                tokenizer.id_for("items", "Choice Scarf"),
            ),
            (EventPosition.NONE, EventPosition.OWN_LEFT, tokenizer.id_for("items", "Leftovers")),
        ]

    def test_ability_outcomes_preserve_identity_without_extra_announcements(self) -> None:
        recorder = SpatialEventRecorder()
        for line in (
            "|move|p1a: X|Tackle|p2a: Y",
            "|-damage|p2a: Y|80/100",
            "|-damage|p1a: X|90/100|[from] ability: Rough Skin|[of] p2a: Y",
            "|-heal|p1b: Z|90/100|[from] ability: Poison Heal",
            "|-boost|p1b: Z|spe|1|[from] ability: Speed Boost",
            "|-immune|p2b: W|[from] ability: Levitate",
            "|-activate|p2a: Y|ability: Storm Drain",
        ):
            recorder.apply_line(line.split("|"), tokenizer, _no_hp)
        records = recorder.pending()
        assert [r.ability_id for r in records] == [
            0,
            0,
            *[
                tokenizer.id_for("abilities", name)
                for name in ("Rough Skin", "Poison Heal", "Speed Boost", "Levitate", "Storm Drain")
            ],
        ]
        assert [(r.source, r.target) for r in records[2:]] == [
            (2, 0),
            (1, 1),
            (1, 1),
            (3, 3),
            (2, 4),
        ]
        assert all(r.move_id == 0 for r in records[1:])
        assert [r.detail for r in records[4:]] == [
            EventDetail.SPE,
            EventDetail.IMMUNE,
            EventDetail.ABILITY_ACTIVATED,
        ]

    def test_condition_transitions_keep_namespaces_and_explicit_sources(self) -> None:
        recorder = SpatialEventRecorder(player_role="p2")
        for line in (
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1b: X",
            "|-weather|SunnyDay|[upkeep]",
            "|-fieldstart|move: Trick Room|[of] p2a: Y",
            "|-sidestart|p1: Opponent|move: Tailwind",
            "|-sideend|p1: Opponent|move: Tailwind",
            "|-fieldend|move: Trick Room",
            "|-weather|none",
            "|-swapsideconditions",
        ):
            recorder.apply_line(line.split("|"), tokenizer, _no_hp)
        records = recorder.pending()
        assert [
            (r.kind, r.source, r.target, r.condition_namespace, r.condition_id) for r in records
        ] == [
            (EventKind.CONDITION_SET, 3, 7, EffectNamespace.WEATHER, 8),
            (EventKind.CONDITION_SET, 0, 7, EffectNamespace.FIELD, 12),
            (EventKind.CONDITION_SET, 4, 6, EffectNamespace.SIDE, 20),
            (EventKind.CONDITION_END, 4, 6, EffectNamespace.SIDE, 20),
            (EventKind.CONDITION_END, 4, 7, EffectNamespace.FIELD, 12),
            (EventKind.CONDITION_END, 4, 7, EffectNamespace.WEATHER, 0),
            (EventKind.SIDE_CONDITIONS_SWAPPED, 4, 8, EffectNamespace.NONE, 0),
        ]
        assert records[0].ability_id == tokenizer.id_for("abilities", "Drought")
        assert all(r.move_id == 0 for r in records)
        recorder.consume()
        recorder.apply_line("|-weather|SunnyDay|[upkeep]".split("|"), tokenizer, _no_hp)
        assert recorder.pending() == ()

    def test_unknown_identity_keeps_event_and_reports_diagnostic(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        recorder = SpatialEventRecorder()
        with caplog.at_level(logging.WARNING, logger="p0.battle.events"):
            recorder.apply_line("|-item|p1a: X|Unlisted Item".split("|"), tokenizer, _no_hp)
        assert recorder.pending()[0].kind == EventKind.ITEM
        assert recorder.pending()[0].item_id == 0
        assert "Unlisted Item" in caplog.text
