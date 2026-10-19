"""Tests for the owned replay reconstruction state reducer."""

from __future__ import annotations

from dataclasses import replace

import pytest

from p0.replays.identity import ReplayMemberId, ReplaySide
from p0.replays.reconstruction.decisions import build_decision_view
from p0.replays.reconstruction.diagnostics import (
    ReplayEventParseError,
    ReplayRejectionCategory,
)
from p0.replays.reconstruction.events import parse_protocol_events
from p0.replays.reconstruction.projection import project_battle_view
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.reconstruction.state import reduce_replay_state
from p0.replays.schema import OTSData, OTSMember, ProtocolLine

_P1_SPECIES = ("Lopunny", "Snorlax", "Furfrou", "Persian", "Watchog", "Castform")
_P2_SPECIES = ("Audino", "Kangaskhan", "Tauros", "Maushold", "Ditto", "Pikachu")
_ABILITIES = {
    "Lopunny": "Cute Charm",
    "Snorlax": "Immunity",
    "Furfrou": "Fur Coat",
    "Persian": "Limber",
    "Watchog": "Illuminate",
    "Castform": "Forecast",
    "Audino": "Healer",
    "Kangaskhan": "Early Bird",
    "Tauros": "Intimidate",
    "Maushold": "Friend Guard",
    "Ditto": "Limber",
    "Pikachu": "Static",
}


def _ots(
    side: ReplaySide,
    names: tuple[str, ...],
    *,
    illusion_member: str | None = None,
    moves: tuple[str, ...] | None = None,
) -> OTSData:
    default_moves = ("Protect", "Tackle") if side is ReplaySide.P2 else ("Protect", "Helping Hand")
    members = tuple(
        OTSMember(
            member_id=ReplayMemberId(side, index),
            nickname=name,
            species=name,
            item="Sitrus Berry",
            ability="Illusion" if name == illusion_member else _ABILITIES.get(name, "Pressure"),
            moves=default_moves if moves is None else moves,
            nature="Serious",
            gender="",
            level=50,
            evs="",
            raw_packed_set=name,
        )
        for index, name in enumerate(names)
    )
    return OTSData(side, "]".join(names), members)


def _complete_ots(*, illusion_member: str | None = None) -> tuple[OTSData, OTSData]:
    return (
        _ots(ReplaySide.P1, _P1_SPECIES, illusion_member=illusion_member),
        _ots(ReplaySide.P2, _P2_SPECIES),
    )


def _complete_delayed_ots() -> tuple[OTSData, OTSData]:
    delayed_moves = ("Protect", "Tackle", "Future Sight")
    return (
        _ots(ReplaySide.P1, _P1_SPECIES, moves=delayed_moves),
        _ots(ReplaySide.P2, _P2_SPECIES, moves=delayed_moves),
    )


def _complete_wish_ots() -> tuple[OTSData, OTSData]:
    return (
        _ots(
            ReplaySide.P1,
            _P1_SPECIES,
            moves=("Protect", "Tackle", "Wish", "Healing Wish"),
        ),
        _ots(ReplaySide.P2, _P2_SPECIES),
    )


def _resolved(
    *raw_lines: str,
    ots: tuple[OTSData, OTSData] | None = None,
):
    lines = tuple(
        ProtocolLine(index, raw, tuple(raw.split("|")), None) for index, raw in enumerate(raw_lines)
    )
    parsed = parse_protocol_events("state-test", lines)
    sheets = _complete_ots() if ots is None else ots
    return resolve_protocol_events("state-test", sheets, parsed.events).require_accepted()


class TestReconstructionState:
    def test_unmodelled_neutralizing_gas_pp_suppression_is_explicitly_unsupported(self) -> None:
        p1 = _ots(
            ReplaySide.P1,
            ("Pikachu", "Weezing", "Mew", "Charizard", "Garchomp", "Incineroar"),
            moves=("Thunderbolt", "Protect"),
        )
        p1 = replace(
            p1,
            members=tuple(
                replace(member, ability="Neutralizing Gas")
                if member.species == "Weezing"
                else member
                for member in p1.members
            ),
        )
        p2 = _ots(
            ReplaySide.P2, ("Corviknight", "Snorlax", "Tauros", "Maushold", "Ditto", "Audino")
        )
        p2 = replace(
            p2,
            members=tuple(
                replace(member, ability="Pressure") if member.species == "Corviknight" else member
                for member in p2.members
            ),
        )
        sheets = (p1, p2)
        events = _resolved(
            "|switch|p1a: Pikachu|Pikachu, L50|100/100",
            "|switch|p1b: Weezing|Weezing, L50|100/100",
            "|switch|p2a: Corviknight|Corviknight, L50|100/100",
            "|-ability|p1b: Weezing|Neutralizing Gas",
            "|move|p1a: Pikachu|Thunderbolt|p2a: Corviknight",
            ots=sheets,
        )

        result = reduce_replay_state("gas-pressure", sheets, events)

        assert result.snapshots == ()
        assert result.diagnostics[0].category is ReplayRejectionCategory.UNSUPPORTED_EVENT
        assert result.diagnostics[0].line_index == 4
        assert "Neutralizing Gas" in result.diagnostics[0].reason

    def test_requested_snapshots_still_apply_every_event(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-damage|p1a: Lopunny|50/100",
            "|-heal|p1a: Lopunny|75/100",
        )

        snapshots = reduce_replay_state(
            "state-test",
            _complete_ots(),
            events,
            snapshot_line_indices=(2,),
        ).require_accepted()

        assert tuple(snapshot.line_index for snapshot in snapshots) == (2,)
        assert snapshots[0].member(ReplayMemberId(ReplaySide.P1, 0)).hp_fraction == 0.75

    def test_clearallboost_clears_every_active_boost(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-unboost|p1b: Snorlax|def|1",
            "|-boost|p2a: Audino|spa|1",
            "|switch|p1a: Furfrou|Furfrou, L50|100/100",
            "|-clearallboost",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts).get("atk", 0) == 0
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 1)).boosts).get("def", 0) == 0
        assert dict(final.member(ReplayMemberId(ReplaySide.P2, 0)).boosts).get("spa", 0) == 0

    def test_clearboost_clears_the_named_member_boosts(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-unboost|p1a: Lopunny|def|1",
            "|-boost|p1b: Snorlax|spa|1",
            "|-clearboost|p1a: Lopunny",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        lopunny = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        snorlax = dict(final.member(ReplayMemberId(ReplaySide.P1, 1)).boosts)
        assert lopunny.get("atk", 0) == 0
        assert lopunny.get("def", 0) == 0
        assert snorlax["spa"] == 1

    def test_clearnegativeboost_preserves_positive_boosts(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-unboost|p1a: Lopunny|def|1",
            "|-clearnegativeboost|p1a: Lopunny",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(member.boosts)["atk"] == 2
        assert dict(member.boosts)["def"] == 0

    def test_copyboost_copies_donor_to_argument_zero(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Furfrou|Furfrou, L50|100/100",
            "|-boost|p1a: Lopunny|atk|1",
            "|-unboost|p1a: Lopunny|def|2",
            "|-boost|p1b: Furfrou|atk|2",
            "|-boost|p1b: Furfrou|def|1",
            "|-copyboost|p1a: Lopunny|p1b: Furfrou",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        copied = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        donor = dict(final.member(ReplayMemberId(ReplaySide.P1, 2)).boosts)
        assert copied["atk"] == 2
        assert copied["def"] == 1
        assert donor["atk"] == 2
        assert donor["def"] == 1

    def test_invertboost_inverts_each_member_boost(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-unboost|p1a: Lopunny|def|1",
            "|-invertboost|p1a: Lopunny",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        boosts = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        assert boosts["atk"] == -2
        assert boosts["def"] == 1
        assert boosts.get("spa", 0) == 0

    def test_setboost_sets_the_named_stat_absolute_value(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-unboost|p1a: Lopunny|def|1",
            "|-setboost|p1a: Lopunny|atk|3",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        boosts = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        assert boosts["atk"] == 3
        assert boosts["def"] == -1

    def test_unboost_decreases_the_named_stat(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-unboost|p1a: Lopunny|atk|1",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)["atk"] == 1

    def test_formechange_updates_the_transient_form(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-formechange|p1a: Lopunny|Lopunny-Mega, L50",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_form == "Lopunny-Mega"

    def test_faint_restores_transient_form(self) -> None:
        p1 = _ots(
            ReplaySide.P1,
            ("Morpeko", "Pikachu", "Conkeldurr", "Gliscor", "Talonflame", "Cofagrigus"),
        )
        p1 = replace(p1, members=(replace(p1.members[0], ability="Hunger Switch"), *p1.members[1:]))
        p2 = _ots(
            ReplaySide.P2,
            (
                "Incineroar",
                "Gholdengo",
                "Amoonguss",
                "Landorus-Therian",
                "Urshifu-Rapid-Strike",
                "Flutter Mane",
            ),
        )
        ots = (p1, p2)
        events = _resolved(
            "|switch|p1a: Morpeko|Morpeko, L50|100/100",
            "|-formechange|p1a: Morpeko|Morpeko-Hangry",
            "|-damage|p1a: Morpeko|0 fnt",
            "|faint|p1a: Morpeko",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()
        morpeko_id = ReplayMemberId(ReplaySide.P1, 0)
        fainted = snapshots[-1].member(morpeko_id)

        assert fainted.fainted
        assert fainted.current_form == "Morpeko"

    def test_faint_preserves_permanent_form(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|detailschange|p1a: Lopunny|Lopunny-Mega, L50",
            "|-damage|p1a: Lopunny|0 fnt",
            "|faint|p1a: Lopunny",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]

        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_form == "Lopunny-Mega"

    def test_mega_marks_the_side_as_having_mega_evolved(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-mega|p1a: Lopunny|Lopunny-Mega|Lopunnite",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert final.sides[0].used_mega is True

    def test_sidestart_and_sideend_update_side_conditions(self) -> None:
        events = _resolved("|-sidestart|p1: Player|move: Reflect", "|-sideend|p1: Player|Reflect")
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert dict(snapshots[0].sides[0].conditions) == {"reflect": 0}
        assert snapshots[1].sides[0].conditions == ()

    def test_singlemove_records_a_member_scoped_effect(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100", "|-singlemove|p1a: Lopunny|Destiny Bond"
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).effects) == {"destinybond": 0}

    def test_swapboost_swaps_only_the_named_stat(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Furfrou|Furfrou, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-boost|p1a: Lopunny|def|1",
            "|-boost|p1b: Furfrou|atk|1",
            "|-boost|p1b: Furfrou|def|3",
            "|-swapboost|p1a: Lopunny|p1b: Furfrou|atk",
        )
        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        lopunny = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        furfrou = dict(final.member(ReplayMemberId(ReplaySide.P1, 2)).boosts)
        assert lopunny["atk"] == 1
        assert lopunny["def"] == 1
        assert furfrou["atk"] == 2
        assert furfrou["def"] == 3

    def test_initialization_events_are_state_neutral(self) -> None:
        events = _resolved(
            "|clearpoke",
            "|player|p1|Alice|1",
            "|poke|p1|Lopunny|Lopunny, L50",
            "|showteam|p1|Lopunny",
            "|start",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        initial = replace(snapshots[0], line_index=0)
        assert initial.sides[0].active == (None, None)
        assert initial.turn == 0
        assert initial.weather == ()
        assert initial.fields == ()
        for snapshot in snapshots[1:]:
            assert replace(snapshot, line_index=0) == initial

    def test_source_shaped_state_neutral_effect_events(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-block|p1a: Lopunny|move: Protect",
            "|-immune|p1a: Lopunny",
            "|-hitcount|p1a: Lopunny|2",
            "|-zbroken|p1a: Lopunny",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        before = snapshots[1]
        for snapshot in snapshots[2:]:
            assert replace(snapshot, line_index=before.line_index) == before

    def test_source_shaped_weather_and_field_transitions(self) -> None:
        events = _resolved(
            "|-weather|SunnyDay|[from] move: Sunny Day",
            "|-fieldstart|move: Electric Terrain",
            "|-fieldactivate|move: Electric Terrain",
            "|-fieldend|move: Electric Terrain",
            "|-weather|none",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [dict(state.weather) for state in snapshots] == [{"sunnyday": 0}] * 4 + [{}]
        assert [dict(state.fields) for state in snapshots] == [
            {},
            {"electricterrain": 0},
            {"electricterrain": 0},
            {},
            {},
        ]

    def test_new_terrain_replaces_previous_terrain_without_fieldend(self) -> None:
        events = _resolved(
            "|-fieldstart|move: Gravity",
            "|-fieldstart|move: Misty Terrain",
            "|-fieldstart|move: Psychic Terrain",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert {name for name, _ in snapshots[-1].fields} == {"gravity", "psychicterrain"}

    def test_source_shaped_hp_transitions(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-damage|p1a: Lopunny|75/100|[from] item: Life Orb",
            "|-sethp|p1a: Lopunny|80/100|[silent]",
            "|-heal|p1a: Lopunny|100/100|[from] item: Leftovers",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [
            state.member(ReplayMemberId(ReplaySide.P1, 0)).hp_fraction for state in snapshots
        ] == [
            1.0,
            0.75,
            0.8,
            1.0,
        ]

    def test_hp_before_holds_the_hp_each_damage_and_heal_line_started_from(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-damage|p1a: Lopunny|75/100|[from] item: Life Orb",
            "|-sethp|p1a: Lopunny|80/100|[silent]",
            "|-heal|p1a: Lopunny|100/100|[from] item: Leftovers",
        )

        state = reduce_replay_state("state-test", _complete_ots(), events, snapshot_line_indices=())

        assert state.snapshots == ()
        assert state.hp_before == {1: 1.0, 3: 0.8}

    def test_status_is_applied_before_it_is_cured(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-status|p1a: Lopunny|brn",
            "|-curestatus|p1a: Lopunny|brn",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [state.member(ReplayMemberId(ReplaySide.P1, 0)).status for state in snapshots] == [
            None,
            "brn",
            None,
        ]

    def test_resist_berry_effect_line_keeps_the_item_symbiosis_passed(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100",
            "|-enditem|p1a: Lopunny|Sitrus Berry|[eat]",
            "|-activate|p1b: Snorlax|ability: Symbiosis|Leftovers|[of] p1a: Lopunny",
            "|-enditem|p1a: Lopunny|Sitrus Berry|[weaken]",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [state.member(ReplayMemberId(ReplaySide.P1, 0)).item for state in snapshots[2:]] == [
            None,
            "Leftovers",
            "Leftovers",
        ]

    def test_cud_chew_eats_an_earlier_berry_without_ending_the_held_item(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-enditem|p1a: Lopunny|Sitrus Berry|[eat]",
            "|-item|p1a: Lopunny|Leftovers|[from] move: Trick",
            "|turn|2",
            "|-activate|p1a: Lopunny|ability: Cud Chew",
            "|-enditem|p1a: Lopunny|Sitrus Berry|[eat]",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0)).item == "Leftovers"

    def test_item_used_on_receipt_ends_before_it_is_handed_over(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-enditem|p1a: Lopunny|White Herb",
            "|-item|p1a: Lopunny|White Herb|[from] move: Trick",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [state.member(ReplayMemberId(ReplaySide.P1, 0)).item for state in snapshots] == [
            "Sitrus Berry",
            "Sitrus Berry",
            None,
        ]

    def test_enditem_with_a_cause_must_match_the_held_item(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-enditem|p1a: Lopunny|White Herb|[from] move: Knock Off",
        )

        result = reduce_replay_state("state-test", _complete_ots(), events)

        assert result.diagnostics[0].reason == "enditem does not match the member's current item"

    def test_item_is_replaced_before_it_is_consumed(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-item|p1a: Lopunny|Leftovers",
            "|-enditem|p1a: Lopunny|Leftovers",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [state.member(ReplayMemberId(ReplaySide.P1, 0)).item for state in snapshots] == [
            "Sitrus Berry",
            "Leftovers",
            None,
        ]

    def test_roost_restores_flying_type_when_single_turn_effect_expires(self) -> None:
        sheets = (
            _ots(
                ReplaySide.P1,
                ("Pidgeot", "Snorlax", "Furfrou", "Persian", "Watchog", "Castform"),
                moves=("Roost",),
            ),
            _ots(ReplaySide.P2, _P2_SPECIES),
        )
        events = _resolved(
            "|switch|p1a: Pidgeot|Pidgeot, L50|100/100",
            "|-singleturn|p1a: Pidgeot|move: Roost",
            "|upkeep",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events).require_accepted()

        assert snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0)).current_types == ("Normal",)
        assert snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0)).current_types == (
            "Normal",
            "Flying",
        )

    def test_baton_pass_transfers_boosts_but_shed_tail_only_substitute(self) -> None:
        baton_events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100|[from] Baton Pass",
        )
        shed_events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-start|p1a: Lopunny|Substitute",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100|[from] Shed Tail",
        )
        baton = reduce_replay_state("state-test", _complete_ots(), baton_events).require_accepted()[
            -1
        ]
        shed = reduce_replay_state("state-test", _complete_ots(), shed_events).require_accepted()[
            -1
        ]
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)
        assert dict(baton.member(snorlax_id).boosts)["atk"] == 2
        assert dict(shed.member(snorlax_id).boosts)["atk"] == 0
        assert dict(shed.member(snorlax_id).effects) == {"substitute": 0}

    def test_baton_pass_preserves_copyable_source_and_dynamic_metadata(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-start|p1a: Lopunny|Leech Seed|[of] p2a: Audino",
            "|-start|p1a: Lopunny|perish3",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100|[from] Baton Pass",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        snorlax = final.member(ReplayMemberId(ReplaySide.P1, 1))

        assert dict(snorlax.effects) == {"leechseed": 0, "perishsong": 0}
        assert dict(snorlax.effect_sources) == {"leechseed": ReplayMemberId(ReplaySide.P2, 0)}
        assert snorlax.perish_count == 3

    def test_baton_pass_filters_condition_copy_set_from_pinned_source(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-start|p1a: Lopunny|Aqua Ring",
            "|-start|p1a: Lopunny|Ingrain",
            "|-start|p1a: Lopunny|Confusion",
            "|-start|p1a: Lopunny|Focus Energy",
            "|-start|p1a: Lopunny|Dragon Cheer",
            "|-start|p1a: Lopunny|stockpile2",
            "|-start|p1a: Lopunny|Trapped",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100|[from] Baton Pass",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        snorlax = final.member(ReplayMemberId(ReplaySide.P1, 1))

        assert dict(snorlax.effects) == {
            "aquaring": 0,
            "confusion": 0,
            "dragoncheer": 0,
            "focusenergy": 0,
            "ingrain": 0,
        }
        assert dict(snorlax.effect_variants) == {}

    def test_source_emitted_partial_trap_is_tracked_and_baton_passed(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-activate|p1a: Lopunny|move: Fire Spin|[of] p2a: Audino",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100|[from] Baton Pass",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        snorlax = final.member(ReplayMemberId(ReplaySide.P1, 1))

        assert dict(snorlax.effects) == {"firespin": 0}
        assert dict(snorlax.effect_sources) == {"firespin": ReplayMemberId(ReplaySide.P2, 0)}

    def test_copyboost_critical_effects_persist_across_turns(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Furfrou|Furfrou, L50|100/100",
            "|-start|p1b: Furfrou|Focus Energy",
            "|-copyboost|p1a: Lopunny|p1b: Furfrou",
            "|turn|1",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]

        assert "focusenergy" in dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).effects)

    def test_reflect_type_copies_the_annotated_targets_types(self) -> None:
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES),
            _ots(
                ReplaySide.P2,
                ("Tyranitar", "Kangaskhan", "Tauros", "Maushold", "Ditto", "Pikachu"),
            ),
        )
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Tyranitar|Tyranitar, L50|100/100",
            "|-start|p1a: Lopunny|typechange|[from] move: Reflect Type|[of] p2a: Tyranitar",
            ots=sheets,
        )

        final = reduce_replay_state("state-test", sheets, events).require_accepted()[-1]

        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_types == ("Rock", "Dark")

    def test_mimicry_end_restores_species_types(self) -> None:
        sheets = (
            _ots(
                ReplaySide.P1,
                ("Stunfisk-Galar", "Snorlax", "Furfrou", "Persian", "Watchog", "Castform"),
            ),
            _ots(ReplaySide.P2, _P2_SPECIES),
        )
        events = _resolved(
            "|switch|p1a: Stunfisk-Galar|Stunfisk-Galar, L50|100/100",
            "|-start|p1a: Stunfisk-Galar|typechange|Electric|[from] ability: Mimicry",
            "|-end|p1a: Stunfisk-Galar|typechange|[silent]",
            ots=sheets,
        )

        final = reduce_replay_state("state-test", sheets, events).require_accepted()[-1]

        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_types == (
            "Ground",
            "Steel",
        )

    def test_round_moved_up_by_round_spends_the_users_pp(self) -> None:
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES, moves=("Round",)),
            _ots(ReplaySide.P2, _P2_SPECIES, moves=("Round",)),
        )
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|move|p2a: Audino|Round|p1a: Lopunny",
            "|move|p1a: Lopunny|Round|p2a: Audino|[from] move: Round",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events).require_accepted()

        for member_id in (ReplayMemberId(ReplaySide.P1, 0), ReplayMemberId(ReplaySide.P2, 0)):
            (round_move,) = snapshots[-1].member(member_id).moves
            assert (round_move.current_pp, round_move.max_pp) == (15, 16)

    def test_spread_move_uses_champions_pp_formula_and_each_pressure_target(self) -> None:
        p1 = _ots(
            ReplaySide.P1,
            ("Charizard", "Pikachu", "Mew", "Garchomp", "Incineroar", "Gholdengo"),
            moves=("Heat Wave",),
        )
        p2 = _ots(
            ReplaySide.P2,
            ("Tyranitar", "Rotom-Wash", "Mewtwo", "Excadrill", "Gallade", "Basculegion"),
        )
        p2 = replace(
            p2,
            members=tuple(
                replace(member, ability="Pressure")
                if member.nickname in {"Tyranitar", "Rotom-Wash"}
                else member
                for member in p2.members
            ),
        )
        sheets = (p1, p2)
        events = _resolved(
            "|switch|p1a: Charizard|Charizard, L50|100/100",
            "|switch|p2a: Tyranitar|Tyranitar, L50|100/100",
            "|switch|p2b: Rotom-Wash|Rotom-Wash, L50|100/100",
            "|move|p1a: Charizard|Heat Wave|p2a: Tyranitar|[spread] p2a,p2b",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events).require_accepted()

        tackle = next(
            move
            for move in snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0)).moves
            if move.move_id == "heatwave"
        )
        assert tackle.max_pp == 12
        assert tackle.current_pp == 9

    def test_spread_move_does_not_charge_allied_pressure(self) -> None:
        p1 = _ots(
            ReplaySide.P1,
            ("Garchomp", "Mewtwo", "Mew", "Charizard", "Incineroar", "Gholdengo"),
            moves=("Earthquake",),
        )
        p1 = replace(
            p1,
            members=tuple(
                replace(member, ability="Pressure") if member.nickname == "Mewtwo" else member
                for member in p1.members
            ),
        )
        p2 = _ots(
            ReplaySide.P2,
            ("Pikachu", "Rotom-Wash", "Tyranitar", "Excadrill", "Gallade", "Basculegion"),
        )
        sheets = (p1, p2)
        events = _resolved(
            "|switch|p1a: Garchomp|Garchomp, L50|100/100",
            "|switch|p1b: Mewtwo|Mewtwo, L50|100/100",
            "|switch|p2a: Pikachu|Pikachu, L50|100/100",
            "|move|p1a: Garchomp|Earthquake|p2a: Pikachu|[spread] p1b,p2a",
            ots=sheets,
        )

        final = reduce_replay_state("state-test", sheets, events).require_accepted()[-1]
        earthquake = final.member(ReplayMemberId(ReplaySide.P1, 0)).moves[0]

        assert earthquake.current_pp == earthquake.max_pp - 1

    def test_single_target_move_does_not_charge_allied_pressure(self) -> None:
        p1 = _ots(
            ReplaySide.P1,
            ("Mew", "Mewtwo", "Pikachu", "Garchomp", "Incineroar", "Gholdengo"),
            moves=("Helping Hand",),
        )
        p1 = replace(
            p1,
            members=tuple(
                replace(member, ability="Pressure") if member.nickname == "Mewtwo" else member
                for member in p1.members
            ),
        )
        p2 = _ots(
            ReplaySide.P2,
            ("Charizard", "Rotom-Wash", "Tyranitar", "Excadrill", "Gallade", "Basculegion"),
        )
        sheets = (p1, p2)
        events = _resolved(
            "|switch|p1a: Mew|Mew, L50|100/100",
            "|switch|p1b: Mewtwo|Mewtwo, L50|100/100",
            "|switch|p2a: Charizard|Charizard, L50|100/100",
            "|move|p1a: Mew|Helping Hand|p1b: Mewtwo",
            ots=sheets,
        )

        final = reduce_replay_state("state-test", sheets, events).require_accepted()[-1]
        move = final.member(ReplayMemberId(ReplaySide.P1, 0)).moves[0]
        assert move.current_pp == move.max_pp - 1

    def test_named_pp_effects_follow_emitted_move_and_amount(self) -> None:
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES, moves=("Tackle",)),
            _ots(ReplaySide.P2, _P2_SPECIES, moves=("Spite",)),
        )
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|move|p1a: Lopunny|Tackle|p2a: Audino",
            "|-activate|p1a: Lopunny|move: Spite|Tackle|4",
            "|-activate|p1a: Lopunny|item: Leppa Berry|Tackle|[consumed]",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events).require_accepted()
        tackle = [state.member(ReplayMemberId(ReplaySide.P1, 0)).moves[0] for state in snapshots]
        # Tackle has 20 base PP in the Champions mod, so its maximum is (20 // 5 + 1) * 4 = 20.
        assert [move.current_pp for move in tackle] == [20, 20, 19, 15, 20]
        assert all(move.max_pp == 20 for move in tackle)

    def test_caused_spread_execution_charges_sleep_talk_owner_under_pressure(self) -> None:
        p1 = _ots(
            ReplaySide.P1,
            ("Charizard", "Pikachu", "Mew", "Garchomp", "Incineroar", "Gholdengo"),
            moves=("Sleep Talk",),
        )
        p2 = _ots(
            ReplaySide.P2,
            ("Tyranitar", "Rotom-Wash", "Mewtwo", "Excadrill", "Gallade", "Basculegion"),
        )
        p2 = replace(
            p2,
            members=tuple(
                replace(member, ability="Pressure")
                if member.nickname in {"Tyranitar", "Rotom-Wash"}
                else member
                for member in p2.members
            ),
        )
        sheets = (p1, p2)
        events = _resolved(
            "|switch|p1a: Charizard|Charizard, L50|100/100",
            "|switch|p2a: Tyranitar|Tyranitar, L50|100/100",
            "|switch|p2b: Rotom-Wash|Rotom-Wash, L50|100/100",
            "|move|p1a: Charizard|Sleep Talk|p1a: Charizard",
            "|move|p1a: Charizard|Heat Wave|p2a: Tyranitar|[from] move: Sleep Talk|[spread] p2a,p2b",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events).require_accepted()
        sleep_talk = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0)).moves[0]
        assert sleep_talk.max_pp == 12
        assert sleep_talk.current_pp == 9

    def test_switch_cleanup_preserves_persistent_state_and_old_snapshots(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|detailschange|p1a: Lopunny|Lopunny-Mega, L50",
            "|-damage|p1a: Lopunny|50/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|-ability|p1a: Lopunny|Borrowed Ability|[from] move: Skill Swap",
            "|switch|p1a: Furfrou|Furfrou, L50|100/100",
            "|switch|p1a: Lopunny|Lopunny-Mega, L50|50/100",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        furfrou_id = ReplayMemberId(ReplaySide.P1, 2)

        before_switch = snapshots[4].member(lopunny_id)
        after_switch = snapshots[5].member(lopunny_id)
        assert dict(before_switch.boosts)["atk"] == 2
        assert before_switch.ability.current == "Borrowed Ability"
        assert dict(after_switch.boosts)["atk"] == 0
        assert after_switch.ability.current == "Scrappy"
        assert after_switch.current_form == "Lopunny-Mega"
        assert after_switch.hp_fraction == 0.5
        assert snapshots[5].sides[0].active == (furfrou_id, None)
        assert snapshots[6].sides[0].active == (lopunny_id, None)

    def test_struggle_does_not_require_an_ots_move_slot(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|move|p1a: Lopunny|Struggle|p2a: Audino",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events)
        lopunny = final.require_accepted()[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert lopunny.last_move == "struggle"
        assert all(move.current_pp == move.max_pp for move in lopunny.moves)

    def test_recharge_cant_consumes_mustrecharge_state(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-mustrecharge|p1a: Lopunny",
            "|cant|p1a: Lopunny|recharge",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        before_cant = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        lopunny = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert dict(before_cant.effects)["mustrecharge"] == 0
        assert "mustrecharge" not in dict(lopunny.effects)

    def test_toxic_stage_resets_when_a_statused_member_reenters(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100 tox",
            "|turn|1",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100",
            "|switch|p1a: Lopunny|Lopunny, L50|100/100 tox",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        before_reentry = snapshots[0].member(ReplayMemberId(ReplaySide.P1, 0))
        lopunny = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert before_reentry.status == "tox"
        assert lopunny.status == "tox"
        assert lopunny.status_counter == 0

    def test_faint_cleanup_clears_status_counter(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100 tox",
            "|turn|1",
            "|-damage|p1a: Lopunny|0 fnt",
            "|faint|p1a: Lopunny",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        before_faint = snapshots[-2].member(ReplayMemberId(ReplaySide.P1, 0))
        lopunny = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert before_faint.status == "tox"
        assert lopunny.status is None
        assert lopunny.status_counter == 0

    def test_wish_follows_its_physical_slot_and_preserves_status(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Wish|p1a: Lopunny",
            "|upkeep",
            "|turn|2",
            "|switch|p1a: Snorlax|Snorlax, L50|50/100 brn",
            "|-heal|p1a: Snorlax|100/100 brn|[from] move: Wish|[wisher] p1a: Lopunny",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events
        ).require_accepted()
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[3].slot_conditions[0].move_id == "wish"
        assert snapshots[6].slot_conditions[0].move_id == "wish"
        assert snapshots[-1].slot_conditions == ()
        assert snapshots[-1].member(snorlax_id).hp_fraction == 1.0
        assert snapshots[-1].member(snorlax_id).status == "brn"

    def test_wish_survives_a_failed_effect_on_its_user(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|50/100 tox",
            "|turn|1",
            "|move|p1a: Lopunny|Wish|p1a: Lopunny",
            "|-fail|p1a: Lopunny|tox",
            "|upkeep",
            "|turn|2",
            "|-heal|p1a: Lopunny|100/100 tox|[from] move: Wish|[wisher] Lopunny",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events
        ).require_accepted()

        assert snapshots[3].slot_conditions[0].move_id == "wish"
        assert snapshots[-1].slot_conditions == ()

    def test_wish_expires_at_the_resolution_upkeep_without_healing(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Wish|p1a: Lopunny",
            "|upkeep",
            "|turn|2",
            "|upkeep",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events
        ).require_accepted()

        assert snapshots[2].slot_conditions[0].move_id == "wish"
        assert snapshots[-1].slot_conditions == ()

    def test_healing_wish_clears_replacement_status_and_consumes_slot_condition(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|move|p1a: Lopunny|Healing Wish|p1a: Lopunny",
            "|-damage|p1a: Lopunny|0 fnt",
            "|faint|p1a: Lopunny",
            "|switch|p1a: Snorlax|Snorlax, L50|50/100 brn",
            "|-heal|p1a: Snorlax|100/100|[from] move: Healing Wish",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events
        ).require_accepted()
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[1].slot_conditions[0].move_id == "healingwish"
        assert snapshots[-1].slot_conditions == ()
        assert snapshots[-1].member(snorlax_id).hp_fraction == 1.0
        assert snapshots[-1].member(snorlax_id).status is None

    def test_transform_is_an_immutable_overlay_with_its_own_pp(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-boost|p2a: Audino|atk|2",
            "|-transform|p1a: Lopunny|p2a: Audino",
            "|detailschange|p2a: Audino|Audino-Mega, L50",
            "|move|p1a: Lopunny|Tackle|p2a: Audino",
            "|-start|p1a: Lopunny|typechange|Water|[from] move: Soak",
            "|switch|p1a: Furfrou|Furfrou, L50|100/100",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        copied = snapshots[3].member(lopunny_id).transform
        after_target_form = snapshots[4].member(lopunny_id).transform
        after_move = snapshots[5].member(lopunny_id).transform

        assert copied is not None
        assert copied.species == "Audino"
        assert dict(copied.boosts)["atk"] == 2
        assert after_target_form == copied
        assert after_move is not None
        tackle = next(move for move in after_move.moves if move.move_id == "tackle")
        assert tackle.max_pp == 5
        assert tackle.current_pp == 4
        assert copied.source_member_id == ReplayMemberId(ReplaySide.P2, 0)
        assert snapshots[5].member(lopunny_id).base_stats[0][1] != copied.non_hp_base_stats[0][1]
        changed = snapshots[6]
        assert changed.member(lopunny_id).current_types == ("Water",)
        decision = build_decision_view(
            changed,
            _complete_ots()[0],
            0,
            (),
            preview=False,
        )
        view = project_battle_view(
            changed, _complete_ots(), perspective=0, decision=decision, view_cache={}
        )
        assert view.active_pokemon[0] is not None
        assert tuple(value.name for value in view.active_pokemon[0].types) == ("Water",)
        assert snapshots[7].member(lopunny_id).transform is None
        assert snapshots[7].member(lopunny_id).current_types == ("Normal",)
        assert tuple(move.move_id for move in snapshots[7].member(lopunny_id).moves) == (
            "protect",
            "helpinghand",
        )

    def test_type_added_after_transform_extends_the_copied_types(self) -> None:
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES),
            _ots(
                ReplaySide.P2,
                ("Pikachu", "Kangaskhan", "Tauros", "Maushold", "Ditto", "Audino"),
            ),
        )
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Pikachu|Pikachu, L50|100/100",
            "|-transform|p1a: Lopunny|p2a: Pikachu|[from] ability: Imposter",
            "|-start|p1a: Lopunny|typeadd|Ghost|[from] move: Trick-or-Treat",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events)

        lopunny = snapshots.require_accepted()[3].member(ReplayMemberId(ReplaySide.P1, 0))
        assert lopunny.current_types == ("Electric", "Ghost")

    def test_roost_ending_after_transform_restores_the_copied_types(self) -> None:
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES),
            _ots(
                ReplaySide.P2,
                ("Aerodactyl", "Kangaskhan", "Tauros", "Maushold", "Ditto", "Pikachu"),
            ),
        )
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Aerodactyl|Aerodactyl, L50|100/100",
            "|-transform|p1a: Lopunny|p2a: Aerodactyl|[from] ability: Imposter",
            "|-singleturn|p1a: Lopunny|move: Roost",
            "|upkeep",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events)

        accepted = snapshots.require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        assert accepted[3].member(lopunny_id).current_types == ("Rock",)
        assert accepted[4].member(lopunny_id).current_types == ("Rock", "Flying")

    def test_declared_team_size_excludes_unselected_open_sheet_reserves(self) -> None:
        events = _resolved(
            "|teamsize|p2|4",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|switch|p2b: Kangaskhan|Kangaskhan, L50|100/100",
            "|faint|p2a: Audino",
            "|switch|p2a: Tauros|Tauros, L50|100/100",
            "|faint|p2a: Tauros",
            "|switch|p2a: Maushold|Maushold, L50|100/100",
            "|faint|p2a: Maushold",
        )
        sheets = _complete_ots()
        snapshots = reduce_replay_state("state-test", sheets, events).require_accepted()
        final = snapshots[-1]

        view = build_decision_view(
            final,
            sheets[1],
            1,
            (),
            preview=False,
        )

        assert final.team_sizes == (4, 4)
        assert view.slots[0].switch_slots == ()
        assert not view.slots[0].force_switch

    def test_unnamed_skill_swap_uses_current_ability_components(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-activate|p1a: Lopunny|move: Skill Swap|||[of] p2a: Audino",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Healer"
        assert final.member(ReplayMemberId(ReplaySide.P2, 0)).ability.current == "Cute Charm"

    def test_ally_skill_swap_exchanges_an_ability_suppressed_by_gastro_acid(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100",
            "|-endability|p1b: Snorlax",
            "|-activate|p1b: Snorlax|Skill Swap|||[of] p1a: Lopunny",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        snorlax = final.member(ReplayMemberId(ReplaySide.P1, 1))
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Immunity"
        assert snorlax.ability.current == "Cute Charm"
        assert "gastroacid" in dict(snorlax.effects)

    def test_transform_copies_an_ability_suppressed_by_gastro_acid(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-endability|p2a: Audino",
            "|-transform|p1a: Lopunny|p2a: Audino|[from] ability: Imposter",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        transform = final.member(ReplayMemberId(ReplaySide.P1, 0)).transform
        assert transform is not None
        assert transform.ability == "Healer"

    def test_trace_source_reference_does_not_overwrite_the_target_ability(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-ability|p1a: Lopunny|Intimidate|[from] ability: Trace|[of] p2a: Audino",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Intimidate"
        assert final.member(ReplayMemberId(ReplaySide.P2, 0)).ability.current == "Healer"

    def test_endability_preserves_ability_and_records_gastro_acid(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-endability|p1a: Lopunny",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))
        assert member.ability.current == "Cute Charm"
        assert dict(member.effects) == {"gastroacid": 0}

    def test_perish_countdown_updates_one_effect_and_clears_on_switch(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-start|p1a: Lopunny|perish3",
            "|-start|p1a: Lopunny|perish2",
            "|-start|p1a: Lopunny|perish1",
            "|-start|p1a: Lopunny|perish0",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)

        assert tuple(snapshot.member(lopunny_id).perish_count for snapshot in snapshots[1:5]) == (
            3,
            2,
            1,
            0,
        )
        assert dict(snapshots[4].member(lopunny_id).effects) == {"perishsong": 0}
        assert snapshots[5].member(lopunny_id).perish_count is None
        assert dict(snapshots[5].member(lopunny_id).effects) == {}

    @pytest.mark.parametrize(
        "effect",
        (
            "perish4",
            "stockpile0",
            "stockpile4",
            "fallen0",
            "fallen6",
        ),
    )
    def test_invalid_dynamic_effect_variants_reject_without_partial_snapshots(
        self, effect: str
    ) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            f"|-start|p1a: Lopunny|{effect}",
        )

        result = reduce_replay_state("state-test", _complete_ots(), events)

        assert result.snapshots == ()
        assert result.diagnostics[0].reason == f"unsupported effect variant '{effect}'"

    @pytest.mark.parametrize(
        ("wire_effect", "canonical", "end_effect"),
        (
            ("stockpile1", "stockpile", "Stockpile"),
            ("stockpile2", "stockpile", "Stockpile"),
            ("stockpile3", "stockpile", "Stockpile"),
            ("fallen1", "supremeoverlord", "fallen1"),
            ("fallen2", "supremeoverlord", "fallen2"),
            ("fallen3", "supremeoverlord", "fallen3"),
            ("fallen4", "supremeoverlord", "fallen4"),
            ("fallen5", "supremeoverlord", "fallen5"),
        ),
    )
    def test_dynamic_effect_start_and_end_clear_metadata(
        self, wire_effect: str, canonical: str, end_effect: str
    ) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            f"|-start|p1a: Lopunny|{wire_effect}",
            f"|-end|p1a: Lopunny|{end_effect}",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        active = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        cleared = snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0))

        assert dict(active.effects) == {canonical: 0}
        assert dict(active.effect_variants) == {canonical: wire_effect}
        assert dict(cleared.effects) == {}
        assert dict(cleared.effect_variants) == {}

    def test_dynamic_effect_metadata_is_cleared_when_the_member_switches(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-start|p1a: Lopunny|stockpile2",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        active = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        lopunny = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(active.effects) == {"stockpile": 0}
        assert dict(active.effect_variants) == {"stockpile": "stockpile2"}
        assert dict(lopunny.effects) == {}
        assert dict(lopunny.effect_variants) == {}

    def test_repeated_stockpile_replaces_only_the_dynamic_layer_metadata(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-start|p1a: Lopunny|stockpile1",
            "|-start|p1a: Lopunny|stockpile2",
            "|-start|p1a: Lopunny|stockpile3",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        assert [
            dict(snapshot.member(ReplayMemberId(ReplaySide.P1, 0)).effect_variants)["stockpile"]
            for snapshot in snapshots[1:]
        ] == ["stockpile1", "stockpile2", "stockpile3"]

    @pytest.mark.parametrize(
        ("guard", "guard_id"),
        (
            ("Quick Guard", "quickguard"),
            ("Wide Guard", "wideguard"),
        ),
    )
    def test_side_guard_is_side_scoped_survives_actions_and_expires_at_upkeep(
        self, guard: str, guard_id: str
    ) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|turn|1",
            f"|-singleturn|p1a: Lopunny|{guard}",
            "|move|p2a: Audino|Tackle|p1a: Lopunny",
            f"|-activate|p1a: Lopunny|move: {guard}",
            "|upkeep",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        p1_id = ReplayMemberId(ReplaySide.P1, 0)
        during_guard = snapshots[4].sides[0]
        source = snapshots[4].member(p1_id)
        after_upkeep = snapshots[-1].sides[0]

        assert dict(during_guard.conditions) == {guard_id: 1}
        assert dict(source.effects) == {}
        assert source.protect_counter == 1
        assert dict(after_upkeep.conditions) == {}

    def test_breaking_move_removes_side_guard_and_protection_counter(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|turn|1",
            "|-singleturn|p1a: Lopunny|Wide Guard",
            "|-activate|p1a: Lopunny|move: Feint|[broken]",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        during_guard = snapshots[2]
        final = snapshots[-1]
        assert dict(during_guard.sides[0].conditions) == {"wideguard": 1}
        assert during_guard.member(ReplayMemberId(ReplaySide.P1, 0)).protect_counter == 1
        assert dict(final.sides[0].conditions) == {}
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).protect_counter == 0

    def test_breaking_move_removes_member_protect_effect(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|turn|1",
            "|-singleturn|p1a: Lopunny|Protect",
            "|-activate|p1a: Lopunny|move: Feint",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        during_protect = snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0))
        member = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(during_protect.effects) == {"protect": 1}
        assert during_protect.protect_counter == 1
        assert dict(member.effects) == {}
        assert member.protect_counter == 0

    def test_side_condition_swap_moves_lasting_conditions_and_keeps_guards(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|turn|1",
            "|-sidestart|p1: Player|move: Reflect",
            "|-sidestart|p2: Player|Spikes",
            "|-sidestart|p2: Player|Spikes",
            "|-singleturn|p1a: Lopunny|Wide Guard",
            "|-swapsideconditions",
            "|upkeep",
        )
        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()

        swapped = snapshots[7]
        assert dict(swapped.sides[0].conditions) == {"spikes": 2, "wideguard": 1}
        assert dict(swapped.sides[1].conditions) == {"reflect": 1}
        assert dict(snapshots[8].sides[0].conditions) == {"spikes": 2}

    def test_duplicate_side_guard_and_member_scoped_singleturn_have_distinct_lifecycles(
        self,
    ) -> None:
        duplicate_events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-singleturn|p1a: Lopunny|Wide Guard",
            "|-singleturn|p1a: Lopunny|Wide Guard",
        )
        duplicate = reduce_replay_state("state-test", _complete_ots(), duplicate_events)

        assert duplicate.snapshots == ()
        assert duplicate.diagnostics[0].line_index == 2
        assert duplicate.diagnostics[0].reason == "side guard 'wideguard' started twice"

    @pytest.mark.parametrize(
        "effect",
        (
            "Protect",
            "Endure",
            "Beak Blast",
            "Focus Punch",
            "Follow Me",
            "Helping Hand",
            "Roost",
            "Rage Powder",
        ),
    )
    def test_other_singleturn_effects_remain_member_scoped(self, effect: str) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            f"|-singleturn|p1a: Lopunny|{effect}",
            "|upkeep",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        during_effect = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        final = snapshots[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))

        assert dict(during_effect.effects) == {effect.lower().replace(" ", ""): 0}
        assert dict(final.sides[0].conditions) == {}
        assert dict(member.effects) == {}

    def test_protection_counter_resets_on_failure_and_switch(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|turn|1",
            "|-singleturn|p1a: Lopunny|Protect",
            "|-fail|p1a: Lopunny|move: Protect",
            "|-singleturn|p1a: Lopunny|Protect",
            "|switch|p1a: Snorlax|Snorlax, L50|100/100",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[2].member(lopunny_id).protect_counter == 1
        assert snapshots[3].member(lopunny_id).protect_counter == 0
        assert snapshots[4].member(lopunny_id).protect_counter == 1
        assert snapshots[5].member(snorlax_id).protect_counter == 0
        assert snapshots[5].member(lopunny_id).protect_counter == 0

    @pytest.mark.parametrize(
        "condition",
        (
            "craftyshield",
            "matblock",
            "firepledge",
            "grasspledge",
            "waterpledge",
            "gmaxcannonade",
            "gmaxsteelsurge",
            "gmaxvinelash",
            "gmaxvolcalith",
            "gmaxwildfire",
        ),
    )
    def test_side_conditions_outside_generated_active_allowlist_reject_structurally(
        self,
        condition: str,
    ) -> None:
        events = _resolved(
            "|-sidestart|p1: Player|move: " + condition,
        )
        result = reduce_replay_state("state-test", _complete_ots(), events)

        assert result.snapshots == ()
        assert result.diagnostics[0].reason == (
            f"unsupported side condition effect variant '{condition}'"
        )

    @pytest.mark.parametrize(
        (
            "source",
            "target",
            "move",
            "effect",
            "target_entry",
            "caster_entry",
            "impact_target",
            "target_id",
            "source_id",
        ),
        (
            (
                "p1a: Lopunny",
                "p2a: Audino",
                "Future Sight",
                "move: Future Sight",
                "p2a: Kangaskhan",
                "p1a: Snorlax",
                "p2a: Kangaskhan",
                ReplayMemberId(ReplaySide.P2, 1),
                ReplayMemberId(ReplaySide.P1, 0),
            ),
            (
                "p2a: Audino",
                "p1a: Lopunny",
                "Future Sight",
                "Future Sight",
                "p1a: Snorlax",
                "p2a: Kangaskhan",
                "p1a: Snorlax",
                ReplayMemberId(ReplaySide.P1, 1),
                ReplayMemberId(ReplaySide.P2, 0),
            ),
        ),
    )
    def test_delayed_move_tracks_target_slot_through_target_and_caster_replacement(
        self,
        source: str,
        target: str,
        move: str,
        effect: str,
        target_entry: str,
        caster_entry: str,
        impact_target: str,
        target_id: ReplayMemberId,
        source_id: ReplayMemberId,
    ) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|turn|1",
            f"|move|{source}|{move}|{target}",
            f"|-start|{source}|{effect}",
            f"|switch|{target_entry}|{target_entry.split(': ', 1)[1]}, L50|100/100",
            f"|switch|{caster_entry}|{caster_entry.split(': ', 1)[1]}, L50|100/100",
            "|turn|3",
            f"|-end|{impact_target}|{move}",
            f"|-damage|{impact_target}|50/100",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()
        scheduled = snapshots[4]
        delayed = scheduled.delayed_moves[0]
        finished = snapshots[-1]

        assert delayed.source_member_id == source_id
        assert delayed.target_slot == 0
        assert delayed.scheduled_turn == 3
        assert delayed.announced
        assert finished.delayed_moves == ()
        assert finished.member(target_id).hp_fraction == 0.5

    def test_delayed_move_stays_with_physical_slot_after_swap(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|switch|p2b: Kangaskhan|Kangaskhan, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Future Sight|p2a: Audino",
            "|-start|p1a: Lopunny|move: Future Sight",
            "|swap|p2a: Audino|1",
            "|turn|3",
            "|-end|p2a: Kangaskhan|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()

        post_swap = snapshots[6].delayed_moves[0]
        assert post_swap.target_side == ReplaySide.P2
        assert post_swap.target_slot == 0
        assert post_swap.source_member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert post_swap.scheduled_turn == 3
        assert snapshots[-1].delayed_moves == ()

    def test_failed_delayed_move_without_target_preserves_existing_condition(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Future Sight|p2a: Audino",
            "|-start|p1a: Lopunny|move: Future Sight",
            "|turn|2",
            "|move|p1a: Lopunny|Future Sight||[still]",
            "|-fail|p1a: Lopunny",
            "|turn|3",
            "|-end|p2a: Audino|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()

        post_fail = snapshots[7].delayed_moves[0]
        assert post_fail.source_member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert post_fail.target_side == ReplaySide.P2
        assert post_fail.target_slot == 0
        assert post_fail.scheduled_turn == 3
        assert snapshots[-1].delayed_moves == ()

    def test_delayed_move_to_fainted_target_stays_on_the_empty_slot_until_it_misses(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|switch|p2b: Kangaskhan|Kangaskhan, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Future Sight|p2a: Audino",
            "|-start|p1a: Lopunny|move: Future Sight",
            "|turn|2",
            "|-damage|p2b: Kangaskhan|0 fnt",
            "|faint|p2b: Kangaskhan",
            "|move|p1a: Lopunny|Future Sight|p2: Kangaskhan",
            "|-start|p1a: Lopunny|move: Future Sight",
            "|turn|3",
            "|-end|p2a: Audino|move: Future Sight",
            "|turn|4",
            "|-hint|Future Sight did not hit because the target is fainted.",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()

        on_audino, on_empty_slot = snapshots[10].delayed_moves
        assert (on_audino.target_slot, on_audino.scheduled_turn) == (0, 3)
        assert (on_empty_slot.target_slot, on_empty_slot.scheduled_turn) == (1, 4)
        assert snapshots[12].delayed_moves == (on_empty_slot,)
        assert snapshots[-1].delayed_moves == ()

    def test_delayed_move_hint_clears_condition_without_end_event(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Future Sight|p1b: Snorlax",
            "|-start|p1a: Lopunny|move: Future Sight",
            "|turn|2",
            "|switch|p1a: Furfrou|Furfrou, L50|100/100",
            "|switch|p1b: Lopunny|Lopunny, L50|100/100",
            "|turn|3",
            "|-hint|Future Sight did not hit because the target is the user.",
            "|turn|4",
            "|move|p1a: Furfrou|Future Sight|p1b: Lopunny",
            "|-start|p1a: Furfrou|move: Future Sight",
            "|turn|6",
            "|-end|p1b: Lopunny|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()

        assert len(snapshots[8].delayed_moves) == 1
        assert snapshots[8].delayed_moves[0].scheduled_turn == 3
        assert snapshots[9].delayed_moves == ()
        assert len(snapshots[12].delayed_moves) == 1
        assert snapshots[12].delayed_moves[0].scheduled_turn == 6
        assert snapshots[-1].delayed_moves == ()

    def test_delayed_move_hint_only_clears_the_fainted_target_slot(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|switch|p2b: Kangaskhan|Kangaskhan, L50|100/100",
            "|turn|1",
            "|move|p1a: Lopunny|Future Sight|p2a: Audino",
            "|-start|p1a: Lopunny|move: Future Sight",
            "|move|p1b: Snorlax|Future Sight|p2b: Kangaskhan",
            "|-start|p1b: Snorlax|move: Future Sight",
            "|turn|2",
            "|-damage|p2a: Audino|0 fnt",
            "|faint|p2a: Audino",
            "|turn|3",
            "|-hint|Future Sight did not hit because the target is fainted.",
            "|-end|p2b: Kangaskhan|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()

        assert len(snapshots[13].delayed_moves) == 1
        remaining = snapshots[13].delayed_moves[0]
        assert remaining.target_slot == 1
        assert remaining.source_member_id == ReplayMemberId(ReplaySide.P1, 1)
        assert snapshots[-1].delayed_moves == ()

    @pytest.mark.parametrize(
        ("lines", "line_index", "reason"),
        (
            (
                (
                    "|switch|p1a: Lopunny|Lopunny, L50|100/100",
                    "|-start|p1a: Lopunny|move: Future Sight",
                ),
                1,
                "futuresight start has an ambiguous pending move from the referenced caster",
            ),
            (
                (
                    "|switch|p1a: Lopunny|Lopunny, L50|100/100",
                    "|switch|p2a: Audino|Audino, L50|100/100",
                    "|turn|1",
                    "|move|p1a: Lopunny|Future Sight|p2a: Audino",
                    "|-start|p1a: Lopunny|move: Future Sight",
                    "|-end|p2a: Audino|move: Future Sight",
                ),
                5,
                "delayed move ended before its scheduled turn",
            ),
            (
                (
                    "|switch|p1a: Lopunny|Lopunny, L50|100/100",
                    "|switch|p2a: Audino|Audino, L50|100/100",
                    "|turn|1",
                    "|move|p1a: Lopunny|Future Sight|p2a: Audino",
                    "|-start|p1a: Lopunny|move: Future Sight",
                    "|move|p1a: Lopunny|Future Sight|p2a: Audino",
                ),
                5,
                "delayed move target slot already has a pending effect",
            ),
            (
                (
                    "|switch|p1a: Lopunny|Lopunny, L50|100/100",
                    "|switch|p2a: Audino|Audino, L50|100/100",
                    "|turn|1",
                    "|move|p1a: Lopunny|Future Sight|p2a: Audino",
                    "|-start|p1a: Lopunny|move: Future Sight",
                    "|turn|4",
                    "|-end|p2a: Audino|move: Future Sight",
                ),
                6,
                "delayed move ended after its scheduled turn",
            ),
        ),
    )
    def test_delayed_move_rejects_missing_pending_duplicate_early_or_late_hit(
        self,
        lines: tuple[str, ...],
        line_index: int,
        reason: str,
    ) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(*lines, ots=ots)

        result = reduce_replay_state("state-test", ots, events)

        assert result.snapshots == ()
        assert [(item.line_index, item.reason) for item in result.diagnostics] == [
            (line_index, reason)
        ]

    def test_delayed_move_source_only_context_is_rejected_during_event_parsing(self) -> None:
        line = ProtocolLine(
            0,
            "|move|p1a: Lopunny|Future Sight",
            ("", "move", "p1a: Lopunny", "Future Sight"),
            None,
        )

        parsed = parse_protocol_events("state-test", (line,))

        assert parsed.events[0].classification.rejects_replay
        assert "delayed move requires an explicit target" in parsed.diagnostics[0].reason

    def test_partial_trap_end_uses_the_protocol_effect_variant(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-activate|p1a: Lopunny|move: Fire Spin|[of] p2a: Audino",
            "|-end|p1a: Lopunny|Fire Spin|[partiallytrapped]|[silent]",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events).require_accepted()
        trapped = snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(trapped.effects) == {"firespin": 0}
        assert dict(trapped.effect_sources) == {"firespin": ReplayMemberId(ReplaySide.P2, 0)}

        ended = snapshots[3].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(ended.effects) == {}
        assert dict(ended.effect_sources) == {}

    def test_illusion_reveal_preserves_actual_state_and_causal_display_history(self) -> None:
        ots = _complete_ots(illusion_member="Snorlax")
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Furfrou|Furfrou, L50|100/100",
            "|-damage|p1a: Lopunny|25/100",
            "|-boost|p1a: Lopunny|atk|2",
            "|replace|p1a: Snorlax|Snorlax, L50",
            "|-end|p1a: Snorlax|Illusion",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)

        disguised = snapshots[0].member(snorlax_id)
        damaged = snapshots[2].member(snorlax_id)
        revealed = snapshots[4].member(snorlax_id)
        assert snapshots[0].sides[0].active[0] == snorlax_id
        public = snapshots[0].public_sides[0]
        assert public is not None and public.active[0] is not None
        assert public.active[0].current_form == "Lopunny"
        assert disguised.current_form == "Snorlax"
        assert not disguised.revealed
        assert dict(disguised.effects) == {"illusion": 0}
        assert damaged.hp_fraction == 0.25
        assert snapshots[2].member(lopunny_id).hp_fraction is None
        assert revealed.current_form == "Snorlax"
        assert revealed.revealed
        assert revealed.hp_fraction == 0.25
        assert dict(revealed.boosts)["atk"] == 2
        assert "illusion" not in dict(revealed.effects)
        assert public.active[0].current_form == "Lopunny"

    def test_duplicate_illusion_display_does_not_mutate_the_disguise_target(self) -> None:
        ots = _complete_ots(illusion_member="Snorlax")
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Lopunny|Lopunny, L50|100/100",
            "|-damage|p1b: Lopunny|50/100",
            "|replace|p1b: Snorlax|Snorlax, L50",
            "|-end|p1b: Snorlax|Illusion",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)
        final = snapshots[-1]

        assert final.sides[0].active == (lopunny_id, snorlax_id)
        assert final.member(lopunny_id).hp_fraction == 1.0
        assert final.member(snorlax_id).hp_fraction == 0.5
        assert final.member(lopunny_id).current_form == "Lopunny"
        assert final.member(snorlax_id).current_form == "Snorlax"

    def test_faint_before_illusion_reveal_updates_only_the_actual_member(self) -> None:
        ots = _complete_ots(illusion_member="Snorlax")
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-damage|p1a: Lopunny|0 fnt",
            "|faint|p1a: Lopunny",
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            ots=ots,
        )

        final = reduce_replay_state("state-test", ots, events).require_accepted()[-1]
        lopunny_id = ReplayMemberId(ReplaySide.P1, 0)
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)

        assert final.sides[0].active[0] == lopunny_id
        assert final.member(lopunny_id).hp_fraction == 1.0
        assert not final.member(lopunny_id).fainted
        assert final.member(snorlax_id).hp_fraction == 0.0
        assert final.member(snorlax_id).fainted

    def test_disguise_of_a_transformed_teammate_is_unsupported(self) -> None:
        ots = (
            _ots(
                ReplaySide.P1,
                ("Ditto", "Zoroark", "Furfrou", "Persian", "Watchog", "Castform"),
                illusion_member="Zoroark",
            ),
            _ots(ReplaySide.P2, _P2_SPECIES),
        )
        ots = (
            replace(
                ots[0],
                members=(replace(ots[0].members[0], ability="Imposter"), *ots[0].members[1:]),
            ),
            ots[1],
        )
        events = _resolved(
            "|switch|p1a: Ditto|Ditto, L50|100/100",
            "|switch|p1b: Furfrou|Furfrou, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            # A hit that does not reveal it shows the first Ditto is the real one.
            "|move|p2a: Audino|Tackle|p1a: Ditto",
            "|-damage|p1a: Ditto|90/100",
            "|-transform|p1a: Ditto|p2a: Audino|[from] ability: Imposter",
            # Zoroark copies the transformed Ditto, so it shows Ditto's name and Audino.
            "|switch|p1b: Ditto|Audino, L50|100/100",
            ots=ots,
        )

        result = reduce_replay_state("state-test", ots, events)

        assert result.snapshots == ()
        assert result.diagnostics[0].line_index == 6
        assert result.diagnostics[0].category is ReplayRejectionCategory.UNSUPPORTED_EVENT
        assert result.diagnostics[0].reason

    def test_disguise_of_a_ditto_copying_zoroark_is_unsupported(self) -> None:
        ots = (
            _ots(
                ReplaySide.P1,
                ("Ditto", "Zoroark", "Furfrou", "Persian", "Watchog", "Castform"),
                illusion_member="Zoroark",
            ),
            _ots(ReplaySide.P2, ("Zoroark", *_P2_SPECIES[1:])),
        )
        ots = (
            replace(
                ots[0],
                members=(replace(ots[0].members[0], ability="Imposter"), *ots[0].members[1:]),
            ),
            ots[1],
        )
        events = _resolved(
            "|switch|p1a: Ditto|Ditto, L50|100/100",
            "|switch|p1b: Furfrou|Furfrou, L50|100/100",
            "|switch|p2a: Zoroark|Zoroark, L50|100/100",
            "|move|p2a: Zoroark|Tackle|p1a: Ditto",
            "|-damage|p1a: Ditto|90/100",
            "|-transform|p1a: Ditto|p2a: Zoroark|[from] ability: Imposter",
            # p1's Zoroark copies its Ditto, which now looks like Zoroark.
            "|switch|p1b: Ditto|Zoroark, L50|100/100",
            ots=ots,
        )

        result = reduce_replay_state("state-test", ots, events)

        assert result.snapshots == ()
        assert result.diagnostics[0].line_index == 6
        assert result.diagnostics[0].category is ReplayRejectionCategory.UNSUPPORTED_EVENT
        assert result.diagnostics[0].reason

    def test_reflect_type_on_a_disguise_is_unsupported(self) -> None:
        ots = _complete_ots(illusion_member="Snorlax")
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|-start|p2a: Audino|typechange|[from] move: Reflect Type|[of] p1a: Lopunny",
            "|replace|p1a: Snorlax|Snorlax, L50",
            "|-end|p1a: Snorlax|Illusion",
            ots=ots,
        )

        result = reduce_replay_state("state-test", ots, events)

        assert result.snapshots == ()
        assert result.diagnostics[0].line_index == 2
        assert result.diagnostics[0].category is ReplayRejectionCategory.UNSUPPORTED_EVENT
        assert result.diagnostics[0].reason

    def test_invalid_reducer_transition_discards_all_snapshots(self) -> None:
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|-sideend|p1: Player|Reflect",
        )
        result = reduce_replay_state("state-test", _complete_ots(), events)

        assert result.snapshots == ()
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].line_index == 1
        assert result.diagnostics[0].reason == "side condition 'reflect' ended before it started"
        with pytest.raises(ReplayEventParseError, match="ended before it started"):
            result.require_accepted()


class TestReviveState:
    def test_revive_unfaints_the_benched_member_at_the_healed_hp(self) -> None:
        ots = (
            _ots(ReplaySide.P1, _P1_SPECIES, moves=("Protect", "Revival Blessing")),
            _ots(ReplaySide.P2, _P2_SPECIES),
        )
        events = _resolved(
            "|switch|p1a: Lopunny|Lopunny, L50|100/100",
            "|switch|p1b: Snorlax|Snorlax, L50|100/100 par",
            "|switch|p2a: Audino|Audino, L50|100/100",
            "|move|p2a: Audino|Tackle|p1b: Snorlax",
            "|-damage|p1b: Snorlax|0 fnt",
            "|faint|p1b: Snorlax",
            "|move|p1a: Lopunny|Revival Blessing|p1a: Lopunny",
            "|-heal|p1: Snorlax|49/100|[from] move: Revival Blessing",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events).require_accepted()
        snorlax_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[-2].member(snorlax_id).fainted
        revived = snapshots[-1].member(snorlax_id)
        assert not revived.fainted
        assert revived.hp_fraction == pytest.approx(0.49)
        # Fainting cleared the paralysis, and the revive does not bring it back.
        assert revived.status is None
        # The revive leaves Snorlax on the bench; its slot stays empty.
        assert snapshots[-1].sides[0].active == (ReplayMemberId(ReplaySide.P1, 0), None)
