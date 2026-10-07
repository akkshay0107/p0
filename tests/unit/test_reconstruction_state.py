"""Tests for the owned replay reconstruction state reducer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from p0.model.resources import default_runtime_resources
from p0.replays.identity import ReplayMemberId, ReplaySide
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import build_decision_view
from p0.replays.reconstruction.diagnostics import ReplayEventParseError
from p0.replays.reconstruction.events import parse_protocol_events
from p0.replays.reconstruction.projection import project_battle_view
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.reconstruction.state import (
    reconstruct_replay_state,
    reduce_replay_state,
)
from p0.replays.schema import OTSData, OTSMember, ProtocolLine

_GOLDEN_REPLAY_DIRECTORY = (
    Path(__file__).parents[2] / "src/p0/replays/reconstruction/golden_replays"
)
_GOLDEN_REPLAYS = tuple(sorted(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")))
_UNRESOLVED_ILLUSION_REPLAYS = frozenset(
    f"gen9championsvgc2026regmbbo3-{number}"
    for number in (
        "2670609846",
        "2670610791",
        "2670613319",
        "2670623194",
        "2670633916",
        "2670639242",
        "2670857037",
        "2670858949",
        "2670968279",
        "2670969833",
        "2671057223",
    )
)
_P1_SPECIES = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")
_P2_SPECIES = ("Golf", "Hotel", "India", "Juliet", "Kilo", "Lima")


def _dex() -> dict[str, object]:
    species: list[dict[str, object]] = []
    for index, name in enumerate((*_P1_SPECIES, *_P2_SPECIES, "Alpha-Mega", "Golf-Mega")):
        entry = {
            "id": name.lower().replace("-", ""),
            "name": name,
            "types": ["Normal"] if name not in {"Alpha-Mega", "Golf-Mega"} else ["Steel", "Fairy"],
            "baseStats": {
                "hp": 50 + index,
                "atk": 60 + index,
                "def": 70 + index,
                "spa": 80 + index,
                "spd": 90 + index,
                "spe": 100 + index,
            },
            "weightkg": 10 + index,
            "abilities": {"0": f"Ability {name}"},
        }
        if name.endswith("-Mega"):
            entry["baseSpecies"] = name.removesuffix("-Mega")
        species.append(entry)
    return {
        "species": species,
        "moves": [
            {
                "id": "protect",
                "name": "Protect",
                "type": "Normal",
                "category": "Status",
                "target": "self",
                "pp": 10,
            },
            {
                "id": "tackle",
                "name": "Tackle",
                "type": "Normal",
                "category": "Physical",
                "target": "normal",
                "pp": 35,
            },
            {
                "id": "mimic",
                "name": "Mimic",
                "type": "Normal",
                "category": "Status",
                "target": "normal",
                "pp": 10,
            },
            {
                "id": "futuresight",
                "name": "Future Sight",
                "type": "Psychic",
                "category": "Special",
                "target": "normal",
                "pp": 10,
                "basePower": 120,
            },
            {
                "id": "doomdesire",
                "name": "Doom Desire",
                "type": "Steel",
                "category": "Special",
                "target": "normal",
                "pp": 5,
                "basePower": 140,
            },
            {
                "id": "wish",
                "name": "Wish",
                "type": "Normal",
                "category": "Status",
                "target": "self",
                "pp": 10,
            },
            {
                "id": "healingwish",
                "name": "Healing Wish",
                "type": "Psychic",
                "category": "Status",
                "target": "self",
                "pp": 10,
            },
        ],
    }


def _ots(
    side: ReplaySide,
    names: tuple[str, ...],
    *,
    illusion_member: str | None = None,
    moves: tuple[str, ...] | None = None,
) -> OTSData:
    default_moves = ("Protect", "Tackle") if side is ReplaySide.P2 else ("Protect", "Mimic")
    members = tuple(
        OTSMember(
            member_id=ReplayMemberId(side, index),
            nickname=name,
            species=name,
            item=f"Item {name}",
            ability="Illusion" if name == illusion_member else f"Ability {name}",
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
    delayed_moves = ("Protect", "Tackle", "Future Sight", "Doom Desire")
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
    dex: dict[str, object] | None = None,
):
    lines = tuple(
        ProtocolLine(index, raw, tuple(raw.split("|")), None) for index, raw in enumerate(raw_lines)
    )
    parsed = parse_protocol_events("state-test", lines)
    sheets = _complete_ots() if ots is None else ots
    return resolve_protocol_events("state-test", sheets, parsed.events, dex=dex).require_accepted()


class TestReconstructionState:
    def test_requested_snapshots_still_apply_every_event(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-damage|p1a: Alpha|50/100",
            "|-heal|p1a: Alpha|75/100",
        )

        snapshots = reduce_replay_state(
            "state-test",
            _complete_ots(),
            events,
            dex=_dex(),
            snapshot_line_indices=(2,),
        ).require_accepted()

        assert tuple(snapshot.line_index for snapshot in snapshots) == (2,)
        assert snapshots[0].member(ReplayMemberId(ReplaySide.P1, 0)).hp_fraction == 0.75

    def test_snapshot_lines_must_belong_to_the_event_stream(self) -> None:
        events = _resolved("|switch|p1a: Alpha|Alpha, L50|100/100")

        with pytest.raises(ValueError, match="must belong"):
            reduce_replay_state(
                "state-test",
                _complete_ots(),
                events,
                dex=_dex(),
                snapshot_line_indices=(1,),
            )

    def test_clearallboost_clears_every_active_boost(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Bravo|Bravo, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1b: Bravo|def|1",
            "|-boost|p2a: Golf|spa|1",
            "|switch|p1a: Charlie|Charlie, L50|100/100",
            "|-clearallboost",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts).get("atk", 0) == 0
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 1)).boosts).get("def", 0) == 0
        assert dict(final.member(ReplayMemberId(ReplaySide.P2, 0)).boosts).get("spa", 0) == 0

    def test_clearboost_clears_the_named_member_boosts(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Bravo|Bravo, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1a: Alpha|def|1",
            "|-boost|p1b: Bravo|spa|1",
            "|-clearboost|p1a: Alpha",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        alpha = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        bravo = dict(final.member(ReplayMemberId(ReplaySide.P1, 1)).boosts)
        assert alpha.get("atk", 0) == 0
        assert alpha.get("def", 0) == 0
        assert bravo["spa"] == 1

    def test_clearnegativeboost_preserves_positive_boosts(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1a: Alpha|def|1",
            "|-clearnegativeboost|p1a: Alpha",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(member.boosts)["atk"] == 2
        assert dict(member.boosts)["def"] == 0

    def test_clearpositiveboost_preserves_negative_boosts(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1a: Alpha|def|1",
            "|-clearpositiveboost|p1a: Alpha|p2a: Golf|move: Psych Up",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(member.boosts)["atk"] == 0
        assert dict(member.boosts)["def"] == -1

    def test_copyboost_copies_donor_to_argument_zero(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Charlie|Charlie, L50|100/100",
            "|-boost|p1a: Alpha|atk|1",
            "|-unboost|p1a: Alpha|def|2",
            "|-boost|p1b: Charlie|atk|2",
            "|-boost|p1b: Charlie|def|1",
            "|-copyboost|p1a: Alpha|p1b: Charlie",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        copied = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        donor = dict(final.member(ReplayMemberId(ReplaySide.P1, 2)).boosts)
        assert copied["atk"] == 2
        assert copied["def"] == 1
        assert donor["atk"] == 2
        assert donor["def"] == 1

    def test_invertboost_inverts_each_member_boost(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1a: Alpha|def|1",
            "|-invertboost|p1a: Alpha",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        boosts = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        assert boosts["atk"] == -2
        assert boosts["def"] == 1
        assert boosts.get("spa", 0) == 0

    def test_setboost_sets_the_named_stat_absolute_value(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1a: Alpha|def|1",
            "|-setboost|p1a: Alpha|atk|3",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        boosts = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        assert boosts["atk"] == 3
        assert boosts["def"] == -1

    def test_unboost_decreases_the_named_stat(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-unboost|p1a: Alpha|atk|1",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)["atk"] == 1

    def test_formechange_updates_the_transient_form(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100", "|-formechange|p1a: Alpha|Alpha-Mega, L50"
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_form == "Alpha-Mega"

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
        dex = default_runtime_resources().dex
        events = _resolved(
            "|switch|p1a: Morpeko|Morpeko, L50|100/100",
            "|-formechange|p1a: Morpeko|Morpeko-Hangry",
            "|-damage|p1a: Morpeko|0 fnt",
            "|faint|p1a: Morpeko",
            ots=ots,
            dex=dex,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=dex).require_accepted()
        morpeko_id = ReplayMemberId(ReplaySide.P1, 0)
        fainted = snapshots[-1].member(morpeko_id)

        assert fainted.fainted
        assert fainted.current_form == "Morpeko"

    def test_faint_preserves_permanent_form(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|detailschange|p1a: Alpha|Alpha-Mega, L50",
            "|-damage|p1a: Alpha|0 fnt",
            "|faint|p1a: Alpha",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]

        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_form == "Alpha-Mega"

    def test_mega_marks_the_side_as_having_mega_evolved(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100", "|-mega|p1a: Alpha|Alpha-Mega|Alpha Stone"
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert final.sides[0].used_mega is True

    def test_sidestart_and_sideend_update_side_conditions(self) -> None:
        events = _resolved("|-sidestart|p1: Player|move: Reflect", "|-sideend|p1: Player|Reflect")
        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        assert dict(snapshots[0].sides[0].conditions) == {"reflect": 0}
        assert snapshots[1].sides[0].conditions == ()

    def test_singlemove_records_a_member_scoped_effect(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100", "|-singlemove|p1a: Alpha|Destiny Bond"
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).effects) == {"destinybond": 0}

    def test_swapboost_swaps_only_the_named_stat(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Charlie|Charlie, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-boost|p1a: Alpha|def|1",
            "|-boost|p1b: Charlie|atk|1",
            "|-boost|p1b: Charlie|def|3",
            "|-swapboost|p1a: Alpha|p1b: Charlie|atk",
        )
        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        alpha = dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).boosts)
        charlie = dict(final.member(ReplayMemberId(ReplaySide.P1, 2)).boosts)
        assert alpha["atk"] == 1
        assert alpha["def"] == 1
        assert charlie["atk"] == 2
        assert charlie["def"] == 3

    def test_initialization_events_are_state_neutral(self) -> None:
        events = _resolved(
            "|clearpoke",
            "|player|p1|Alice|1",
            "|poke|p1|Alpha|Alpha, L50",
            "|showteam|p1|Alpha",
            "|start",
        )
        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        initial = replace(snapshots[0], line_index=0)
        assert initial.sides[0].active == (None, None)
        assert initial.turn == 0
        assert initial.weather == ()
        assert initial.fields == ()
        for snapshot in snapshots[1:]:
            assert replace(snapshot, line_index=0) == initial

    def test_source_shaped_state_neutral_effect_events(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-block|p1a: Alpha|move: Protect",
            "|-immune|p1a: Alpha",
            "|-hitcount|p1a: Alpha|2",
            "|-zbroken|p1a: Alpha",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

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
        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

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

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        assert {name for name, _ in snapshots[-1].fields} == {"gravity", "psychicterrain"}

    def test_source_shaped_hp_transitions(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-damage|p1a: Alpha|75/100|[from] item: Life Orb",
            "|-sethp|p1a: Alpha|80/100|[silent]",
            "|-heal|p1a: Alpha|100/100|[from] item: Leftovers",
        )
        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        assert [
            state.member(ReplayMemberId(ReplaySide.P1, 0)).hp_fraction for state in snapshots
        ] == [
            1.0,
            0.75,
            0.8,
            1.0,
        ]

    def test_status_is_applied_before_it_is_cured(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-status|p1a: Alpha|brn",
            "|-curestatus|p1a: Alpha|brn",
        )
        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        assert [state.member(ReplayMemberId(ReplaySide.P1, 0)).status for state in snapshots] == [
            None,
            "brn",
            None,
        ]

    def test_item_is_replaced_before_it_is_consumed(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-item|p1a: Alpha|Leftovers",
            "|-enditem|p1a: Alpha|Leftovers",
        )
        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        assert [state.member(ReplayMemberId(ReplaySide.P1, 0)).item for state in snapshots] == [
            "Item Alpha",
            "Leftovers",
            None,
        ]

    def test_roost_restores_flying_type_when_single_turn_effect_expires(self) -> None:
        dex = _dex()
        cast(list[dict[str, object]], dex["species"])[0]["types"] = [
            "Flying",
            "Normal",
        ]
        cast(list[dict[str, object]], dex["moves"]).append(
            {
                "id": "roost",
                "name": "Roost",
                "type": "Flying",
                "category": "Status",
                "target": "self",
                "pp": 10,
            }
        )
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES, moves=("Roost",)),
            _ots(ReplaySide.P2, _P2_SPECIES),
        )
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-singleturn|p1a: Alpha|move: Roost",
            "|upkeep",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events, dex=dex).require_accepted()

        assert snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0)).current_types == ("Normal",)
        assert snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0)).current_types == (
            "Flying",
            "Normal",
        )

    def test_baton_pass_transfers_boosts_but_shed_tail_only_substitute(self) -> None:
        baton_events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|switch|p1a: Bravo|Bravo, L50|100/100|[from] Baton Pass",
        )
        shed_events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-start|p1a: Alpha|Substitute",
            "|switch|p1a: Bravo|Bravo, L50|100/100|[from] Shed Tail",
        )
        baton = reduce_replay_state(
            "state-test", _complete_ots(), baton_events, dex=_dex()
        ).require_accepted()[-1]
        shed = reduce_replay_state(
            "state-test", _complete_ots(), shed_events, dex=_dex()
        ).require_accepted()[-1]
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)
        assert dict(baton.member(bravo_id).boosts)["atk"] == 2
        assert dict(shed.member(bravo_id).boosts)["atk"] == 0
        assert dict(shed.member(bravo_id).effects) == {"substitute": 0}

    def test_baton_pass_preserves_copyable_source_and_dynamic_metadata(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-start|p1a: Alpha|Leech Seed|[of] p2a: Golf",
            "|-start|p1a: Alpha|perish3",
            "|switch|p1a: Bravo|Bravo, L50|100/100|[from] Baton Pass",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        bravo = final.member(ReplayMemberId(ReplaySide.P1, 1))

        assert dict(bravo.effects) == {"leechseed": 0, "perishsong": 0}
        assert dict(bravo.effect_sources) == {"leechseed": ReplayMemberId(ReplaySide.P2, 0)}
        assert bravo.perish_count == 3

    def test_baton_pass_filters_condition_copy_set_from_pinned_source(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-start|p1a: Alpha|Aqua Ring",
            "|-start|p1a: Alpha|Ingrain",
            "|-start|p1a: Alpha|Confusion",
            "|-start|p1a: Alpha|Focus Energy",
            "|-start|p1a: Alpha|Dragon Cheer",
            "|-start|p1a: Alpha|stockpile2",
            "|-start|p1a: Alpha|Trapped",
            "|-start|p1a: Alpha|Trapper",
            "|switch|p1a: Bravo|Bravo, L50|100/100|[from] Baton Pass",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        bravo = final.member(ReplayMemberId(ReplaySide.P1, 1))

        assert dict(bravo.effects) == {
            "aquaring": 0,
            "confusion": 0,
            "dragoncheer": 0,
            "focusenergy": 0,
            "ingrain": 0,
        }
        assert dict(bravo.effect_variants) == {}

    def test_source_emitted_partial_trap_is_tracked_and_baton_passed(self) -> None:
        dex = _dex()
        cast(list[dict[str, object]], dex["moves"]).append(
            {
                "id": "firespin",
                "name": "Fire Spin",
                "type": "Fire",
                "category": "Special",
                "target": "normal",
                "pp": 15,
                "volatileStatus": "partiallytrapped",
            }
        )
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-activate|p1a: Alpha|move: Fire Spin|[of] p2a: Golf",
            "|switch|p1a: Bravo|Bravo, L50|100/100|[from] Baton Pass",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=dex
        ).require_accepted()[-1]
        bravo = final.member(ReplayMemberId(ReplaySide.P1, 1))

        assert dict(bravo.effects) == {"firespin": 0}
        assert dict(bravo.effect_sources) == {"firespin": ReplayMemberId(ReplaySide.P2, 0)}

    def test_copyboost_critical_effects_persist_across_turns(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Charlie|Charlie, L50|100/100",
            "|-start|p1b: Charlie|Focus Energy",
            "|-copyboost|p1a: Alpha|p1b: Charlie",
            "|turn|1",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]

        assert "focusenergy" in dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).effects)

    def test_reflect_type_copies_the_annotated_targets_types(self) -> None:
        dex = _dex()
        species = cast(list[dict[str, object]], dex["species"])
        next(entry for entry in species if entry["name"] == "Golf")["types"] = ["Rock", "Dark"]
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-start|p1a: Alpha|typechange|[from] move: Reflect Type|[of] p2a: Golf",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=dex
        ).require_accepted()[-1]

        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_types == ("Rock", "Dark")

    def test_mimicry_end_restores_species_types(self) -> None:
        dex = _dex()
        species = cast(list[dict[str, object]], dex["species"])
        next(entry for entry in species if entry["name"] == "Alpha")["types"] = [
            "Ground",
            "Steel",
        ]
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-start|p1a: Alpha|typechange|Electric|[from] ability: Mimicry",
            "|-end|p1a: Alpha|typechange|[silent]",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=dex
        ).require_accepted()[-1]

        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).current_types == (
            "Ground",
            "Steel",
        )

    def test_spread_move_uses_champions_pp_formula_and_each_pressure_target(self) -> None:
        dex = default_runtime_resources().dex
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

        snapshots = reduce_replay_state("state-test", sheets, events, dex=dex).require_accepted()

        tackle = next(
            move
            for move in snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0)).moves
            if move.move_id == "heatwave"
        )
        assert tackle.max_pp == 12
        assert tackle.current_pp == 9

    def test_spread_move_does_not_charge_allied_pressure(self) -> None:
        dex = default_runtime_resources().dex
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

        final = reduce_replay_state("state-test", sheets, events, dex=dex).require_accepted()[-1]
        earthquake = final.member(ReplayMemberId(ReplaySide.P1, 0)).moves[0]

        assert earthquake.current_pp == earthquake.max_pp - 1

    def test_single_target_move_does_not_charge_allied_pressure(self) -> None:
        dex = default_runtime_resources().dex
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

        final = reduce_replay_state("state-test", sheets, events, dex=dex).require_accepted()[-1]
        move = final.member(ReplayMemberId(ReplaySide.P1, 0)).moves[0]
        assert move.current_pp == move.max_pp - 1

    def test_named_pp_effects_follow_emitted_move_and_amount(self) -> None:
        dex = _dex()
        moves = cast(list[dict[str, object]], dex["moves"])
        dex["moves"] = [
            *moves,
            {
                "id": "spite",
                "name": "Spite",
                "type": "Ghost",
                "category": "Status",
                "target": "normal",
                "pp": 10,
            },
        ]
        sheets = (
            _ots(ReplaySide.P1, _P1_SPECIES, moves=("Tackle",)),
            _ots(ReplaySide.P2, _P2_SPECIES, moves=("Spite",)),
        )
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|move|p1a: Alpha|Tackle|p2a: Golf",
            "|-activate|p1a: Alpha|move: Spite|Tackle|4",
            "|-activate|p1a: Alpha|item: Leppa Berry|Tackle|[consumed]",
            ots=sheets,
        )

        snapshots = reduce_replay_state("state-test", sheets, events, dex=dex).require_accepted()
        tackle = [state.member(ReplayMemberId(ReplaySide.P1, 0)).moves[0] for state in snapshots]
        assert [move.current_pp for move in tackle] == [32, 32, 31, 27, 32]
        assert all(move.max_pp == 32 for move in tackle)

    def test_caused_spread_execution_charges_sleep_talk_owner_under_pressure(self) -> None:
        dex = default_runtime_resources().dex
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

        snapshots = reduce_replay_state("state-test", sheets, events, dex=dex).require_accepted()
        sleep_talk = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0)).moves[0]
        assert sleep_talk.max_pp == 12
        assert sleep_talk.current_pp == 9

    def test_switch_cleanup_preserves_persistent_state_and_old_snapshots(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|detailschange|p1a: Alpha|Alpha-Mega, L50",
            "|-damage|p1a: Alpha|50/100",
            "|-boost|p1a: Alpha|atk|2",
            "|-ability|p1a: Alpha|Borrowed Ability|[from] move: Skill Swap",
            "|switch|p1a: Charlie|Charlie, L50|100/100",
            "|switch|p1a: Alpha|Alpha-Mega, L50|50/100",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        charlie_id = ReplayMemberId(ReplaySide.P1, 2)

        before_switch = snapshots[4].member(alpha_id)
        after_switch = snapshots[5].member(alpha_id)
        assert dict(before_switch.boosts)["atk"] == 2
        assert before_switch.ability.current == "Borrowed Ability"
        assert dict(after_switch.boosts)["atk"] == 0
        assert after_switch.ability.current == "Ability Alpha-Mega"
        assert after_switch.current_form == "Alpha-Mega"
        assert after_switch.hp_fraction == 0.5
        assert snapshots[5].sides[0].active == (charlie_id, None)
        assert snapshots[6].sides[0].active == (alpha_id, None)

    def test_implicit_action_moves_do_not_require_ots_move_slots(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|move|p1a: Alpha|Struggle|p2a: Golf",
            "|move|p1a: Alpha|Recharge|p1a: Alpha",
        )

        final = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())
        alpha = final.require_accepted()[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert alpha.last_move == "recharge"
        assert all(move.current_pp == move.max_pp for move in alpha.moves)

    def test_recharge_cant_consumes_mustrecharge_state(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-mustrecharge|p1a: Alpha",
            "|cant|p1a: Alpha|recharge",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        before_cant = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        alpha = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert dict(before_cant.effects)["mustrecharge"] == 0
        assert "mustrecharge" not in dict(alpha.effects)

    def test_toxic_stage_resets_when_a_statused_member_reenters(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100 tox",
            "|turn|1",
            "|switch|p1a: Bravo|Bravo, L50|100/100",
            "|switch|p1a: Alpha|Alpha, L50|100/100 tox",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        before_reentry = snapshots[0].member(ReplayMemberId(ReplaySide.P1, 0))
        alpha = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert before_reentry.status == "tox"
        assert alpha.status == "tox"
        assert alpha.status_counter == 0

    def test_faint_cleanup_clears_status_counter(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100 tox",
            "|turn|1",
            "|-damage|p1a: Alpha|0 fnt",
            "|faint|p1a: Alpha",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        before_faint = snapshots[-2].member(ReplayMemberId(ReplaySide.P1, 0))
        alpha = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))

        assert before_faint.status == "tox"
        assert alpha.status is None
        assert alpha.status_counter == 0

    def test_wish_follows_its_physical_slot_and_preserves_status(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Wish|p1a: Alpha",
            "|upkeep",
            "|turn|2",
            "|switch|p1a: Bravo|Bravo, L50|50/100 brn",
            "|-heal|p1a: Bravo|100/100 brn|[from] move: Wish|[wisher] p1a: Alpha",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events, dex=_dex()
        ).require_accepted()
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[3].slot_conditions[0].move_id == "wish"
        assert snapshots[6].slot_conditions[0].move_id == "wish"
        assert snapshots[-1].slot_conditions == ()
        assert snapshots[-1].member(bravo_id).hp_fraction == 1.0
        assert snapshots[-1].member(bravo_id).status == "brn"

    def test_wish_expires_at_the_resolution_upkeep_without_healing(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Wish|p1a: Alpha",
            "|upkeep",
            "|turn|2",
            "|upkeep",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events, dex=_dex()
        ).require_accepted()

        assert snapshots[2].slot_conditions[0].move_id == "wish"
        assert snapshots[-1].slot_conditions == ()

    def test_healing_wish_clears_replacement_status_and_consumes_slot_condition(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|move|p1a: Alpha|Healing Wish|p1a: Alpha",
            "|-damage|p1a: Alpha|0 fnt",
            "|faint|p1a: Alpha",
            "|switch|p1a: Bravo|Bravo, L50|50/100 brn",
            "|-heal|p1a: Bravo|100/100|[from] move: Healing Wish",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_wish_ots(), events, dex=_dex()
        ).require_accepted()
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[1].slot_conditions[0].move_id == "healingwish"
        assert snapshots[-1].slot_conditions == ()
        assert snapshots[-1].member(bravo_id).hp_fraction == 1.0
        assert snapshots[-1].member(bravo_id).status is None

    def test_transform_is_an_immutable_overlay_with_its_own_pp(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-boost|p2a: Golf|atk|2",
            "|-transform|p1a: Alpha|Golf",
            "|detailschange|p2a: Golf|Golf-Mega, L50",
            "|move|p1a: Alpha|Tackle|p2a: Golf",
            "|-start|p1a: Alpha|typechange|Water|[from] move: Soak",
            "|switch|p1a: Charlie|Charlie, L50|100/100",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        copied = snapshots[3].member(alpha_id).transform
        after_target_form = snapshots[4].member(alpha_id).transform
        after_move = snapshots[5].member(alpha_id).transform

        assert copied is not None
        assert copied.species == "Golf"
        assert dict(copied.boosts)["atk"] == 2
        assert after_target_form == copied
        assert after_move is not None
        tackle = next(move for move in after_move.moves if move.move_id == "tackle")
        assert tackle.max_pp == 5
        assert tackle.current_pp == 4
        assert copied.source_member_id == ReplayMemberId(ReplaySide.P2, 0)
        assert snapshots[5].member(alpha_id).base_stats[0][1] != copied.non_hp_base_stats[0][1]
        changed = snapshots[6]
        assert changed.member(alpha_id).current_types == ("Water",)
        decision = build_decision_view(
            changed,
            _complete_ots()[0],
            0,
            (),
            preview=False,
            dex=_dex(),
        )
        view = project_battle_view(
            changed, _complete_ots(), perspective=0, decision=decision, dex=_dex()
        )
        assert view.active_pokemon[0] is not None
        assert tuple(value.name for value in view.active_pokemon[0].types) == ("Water",)
        assert snapshots[7].member(alpha_id).transform is None
        assert snapshots[7].member(alpha_id).current_types == ("Normal",)
        assert tuple(move.move_id for move in snapshots[7].member(alpha_id).moves) == (
            "protect",
            "mimic",
        )

    def test_type_added_after_transform_extends_the_copied_types(self) -> None:
        dex = _dex()
        cast(list[dict[str, object]], dex["species"])[6]["types"] = ["Electric"]
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-transform|p1a: Alpha|p2a: Golf|[from] ability: Imposter",
            "|-start|p1a: Alpha|typeadd|Ghost|[from] move: Trick-or-Treat",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events, dex=dex)

        alpha = snapshots.require_accepted()[3].member(ReplayMemberId(ReplaySide.P1, 0))
        assert alpha.current_types == ("Electric", "Ghost")

    def test_roost_ending_after_transform_restores_the_copied_types(self) -> None:
        dex = _dex()
        cast(list[dict[str, object]], dex["species"])[6]["types"] = ["Rock", "Flying"]
        cast(list[dict[str, object]], dex["moves"]).append(
            {
                "id": "roost",
                "name": "Roost",
                "type": "Flying",
                "category": "Status",
                "target": "self",
                "pp": 10,
            }
        )
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-transform|p1a: Alpha|p2a: Golf|[from] ability: Imposter",
            "|-singleturn|p1a: Alpha|move: Roost",
            "|upkeep",
        )

        snapshots = reduce_replay_state("state-test", _complete_ots(), events, dex=dex)

        accepted = snapshots.require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        assert accepted[3].member(alpha_id).current_types == ("Rock",)
        assert accepted[4].member(alpha_id).current_types == ("Rock", "Flying")

    def test_declared_team_size_excludes_unselected_open_sheet_reserves(self) -> None:
        events = _resolved(
            "|teamsize|p2|4",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|switch|p2b: Hotel|Hotel, L50|100/100",
            "|faint|p2a: Golf",
            "|switch|p2a: India|India, L50|100/100",
            "|faint|p2a: India",
            "|switch|p2a: Juliet|Juliet, L50|100/100",
            "|faint|p2a: Juliet",
        )
        sheets = _complete_ots()
        snapshots = reduce_replay_state("state-test", sheets, events, dex=_dex()).require_accepted()
        final = snapshots[-1]

        view = build_decision_view(
            final,
            sheets[1],
            1,
            (),
            preview=False,
            dex=_dex(),
        )

        assert final.team_sizes == (4, 4)
        assert view.slots[0].switch_slots == ()
        assert not view.slots[0].force_switch

    def test_unnamed_skill_swap_uses_current_ability_components(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-activate|p1a: Alpha|move: Skill Swap|||[of] p2a: Golf",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Ability Golf"
        assert final.member(ReplayMemberId(ReplaySide.P2, 0)).ability.current == "Ability Alpha"

    def test_ally_skill_swap_exchanges_an_ability_suppressed_by_gastro_acid(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Bravo|Bravo, L50|100/100",
            "|-endability|p1b: Bravo",
            "|-activate|p1b: Bravo|Skill Swap|||[of] p1a: Alpha",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        bravo = final.member(ReplayMemberId(ReplaySide.P1, 1))
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Ability Bravo"
        assert bravo.ability.current == "Ability Alpha"
        assert "gastroacid" in dict(bravo.effects)

    def test_transform_copies_an_ability_suppressed_by_gastro_acid(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-endability|p2a: Golf",
            "|-transform|p1a: Alpha|p2a: Golf|[from] ability: Imposter",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        transform = final.member(ReplayMemberId(ReplaySide.P1, 0)).transform
        assert transform is not None
        assert transform.ability == "Ability Golf"

    def test_trace_source_reference_does_not_overwrite_the_target_ability(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-ability|p1a: Alpha|Intimidate|[from] ability: Trace|[of] p2a: Golf",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Intimidate"
        assert final.member(ReplayMemberId(ReplaySide.P2, 0)).ability.current == "Ability Golf"

    def test_endability_preserves_ability_and_records_gastro_acid(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-endability|p1a: Alpha",
        )

        final = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))
        assert member.ability.current == "Ability Alpha"
        assert dict(member.effects) == {"gastroacid": 0}

    def test_perish_countdown_updates_one_effect_and_clears_on_switch(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-start|p1a: Alpha|perish3",
            "|-start|p1a: Alpha|perish2",
            "|-start|p1a: Alpha|perish1",
            "|-start|p1a: Alpha|perish0",
            "|switch|p1a: Bravo|Bravo, L50|100/100",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)

        assert tuple(snapshot.member(alpha_id).perish_count for snapshot in snapshots[1:5]) == (
            3,
            2,
            1,
            0,
        )
        assert dict(snapshots[4].member(alpha_id).effects) == {"perishsong": 0}
        assert snapshots[5].member(alpha_id).perish_count is None
        assert dict(snapshots[5].member(alpha_id).effects) == {}

    @pytest.mark.parametrize(
        "effect",
        (
            "perish4",
            "stockpile0",
            "stockpile4",
            "protosynthesishp",
            "quarkdrivehp",
            "fallen0",
            "fallen6",
        ),
    )
    def test_invalid_dynamic_effect_variants_reject_without_partial_snapshots(
        self, effect: str
    ) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            f"|-start|p1a: Alpha|{effect}",
        )

        result = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())

        assert result.snapshots == ()
        assert result.diagnostics[0].reason == f"unsupported effect variant '{effect}'"

    @pytest.mark.parametrize(
        ("wire_effect", "canonical", "end_effect"),
        (
            ("stockpile1", "stockpile", "Stockpile"),
            ("stockpile2", "stockpile", "Stockpile"),
            ("stockpile3", "stockpile", "Stockpile"),
            ("protosynthesisatk", "protosynthesis", "Protosynthesis"),
            ("protosynthesisdef", "protosynthesis", "Protosynthesis"),
            ("protosynthesisspa", "protosynthesis", "Protosynthesis"),
            ("protosynthesisspd", "protosynthesis", "Protosynthesis"),
            ("protosynthesisspe", "protosynthesis", "Protosynthesis"),
            ("quarkdriveatk", "quarkdrive", "Quark Drive"),
            ("quarkdrivedef", "quarkdrive", "Quark Drive"),
            ("quarkdrivespa", "quarkdrive", "Quark Drive"),
            ("quarkdrivespd", "quarkdrive", "Quark Drive"),
            ("quarkdrivespe", "quarkdrive", "Quark Drive"),
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
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            f"|-start|p1a: Alpha|{wire_effect}",
            f"|-end|p1a: Alpha|{end_effect}",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        active = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        cleared = snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0))

        assert dict(active.effects) == {canonical: 0}
        assert dict(active.effect_variants) == {canonical: wire_effect}
        assert dict(cleared.effects) == {}
        assert dict(cleared.effect_variants) == {}

    def test_dynamic_effect_metadata_is_cleared_when_the_member_switches(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-start|p1a: Alpha|stockpile2",
            "|switch|p1a: Bravo|Bravo, L50|100/100",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        active = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        alpha = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(active.effects) == {"stockpile": 0}
        assert dict(active.effect_variants) == {"stockpile": "stockpile2"}
        assert dict(alpha.effects) == {}
        assert dict(alpha.effect_variants) == {}

    def test_repeated_stockpile_replaces_only_the_dynamic_layer_metadata(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-start|p1a: Alpha|stockpile1",
            "|-start|p1a: Alpha|stockpile2",
            "|-start|p1a: Alpha|stockpile3",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

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
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|turn|1",
            f"|-singleturn|p1a: Alpha|{guard}",
            "|move|p2a: Golf|Tackle|p1a: Alpha",
            f"|-activate|p1a: Alpha|move: {guard}",
            "|upkeep",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
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
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|turn|1",
            "|-singleturn|p1a: Alpha|Wide Guard",
            "|-activate|p1a: Alpha|move: Feint|[broken]",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        during_guard = snapshots[2]
        final = snapshots[-1]
        assert dict(during_guard.sides[0].conditions) == {"wideguard": 1}
        assert during_guard.member(ReplayMemberId(ReplaySide.P1, 0)).protect_counter == 1
        assert dict(final.sides[0].conditions) == {}
        assert final.member(ReplayMemberId(ReplaySide.P1, 0)).protect_counter == 0

    def test_breaking_move_removes_member_protect_effect(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|turn|1",
            "|-singleturn|p1a: Alpha|Protect",
            "|-activate|p1a: Alpha|move: Feint",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()

        during_protect = snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0))
        member = snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(during_protect.effects) == {"protect": 1}
        assert during_protect.protect_counter == 1
        assert dict(member.effects) == {}
        assert member.protect_counter == 0

    def test_side_condition_swap_is_rejected_as_unsupported(self) -> None:
        lines = tuple(
            ProtocolLine(index, raw, tuple(raw.split("|")), None)
            for index, raw in enumerate(
                (
                    "|switch|p1a: Alpha|Alpha, L50|100/100",
                    "|switch|p2a: Golf|Golf, L50|100/100",
                    "|turn|1",
                    "|-singleturn|p1a: Alpha|Quick Guard",
                    "|-singleturn|p2a: Golf|Mat Block",
                    "|-swapsideconditions",
                )
            )
        )
        parsed = parse_protocol_events("state-test", lines)
        result = resolve_protocol_events("state-test", _complete_ots(), parsed.events)
        assert result.events == ()
        assert "unsupported by the reconstruction contract" in result.diagnostics[0].reason

    def test_duplicate_side_guard_and_member_scoped_singleturn_have_distinct_lifecycles(
        self,
    ) -> None:
        duplicate_events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-singleturn|p1a: Alpha|Wide Guard",
            "|-singleturn|p1a: Alpha|Wide Guard",
        )
        duplicate = reduce_replay_state("state-test", _complete_ots(), duplicate_events, dex=_dex())

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
            "Instruct",
            "Magic Coat",
            "Roost",
            "Rage Powder",
            "Snatch",
            "Spotlight",
            "Shell Trap",
            "Powder",
        ),
    )
    def test_other_singleturn_effects_remain_member_scoped(self, effect: str) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            f"|-singleturn|p1a: Alpha|{effect}",
            "|upkeep",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        during_effect = snapshots[1].member(ReplayMemberId(ReplaySide.P1, 0))
        final = snapshots[-1]
        member = final.member(ReplayMemberId(ReplaySide.P1, 0))

        assert dict(during_effect.effects) == {effect.lower().replace(" ", ""): 0}
        assert dict(final.sides[0].conditions) == {}
        assert dict(member.effects) == {}

    def test_protection_counter_resets_on_failure_and_switch(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|turn|1",
            "|-singleturn|p1a: Alpha|Protect",
            "|-fail|p1a: Alpha|move: Protect",
            "|-singleturn|p1a: Alpha|Protect",
            "|switch|p1a: Bravo|Bravo, L50|100/100",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=_dex()
        ).require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)

        assert snapshots[2].member(alpha_id).protect_counter == 1
        assert snapshots[3].member(alpha_id).protect_counter == 0
        assert snapshots[4].member(alpha_id).protect_counter == 1
        assert snapshots[5].member(bravo_id).protect_counter == 0
        assert snapshots[5].member(alpha_id).protect_counter == 0

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
        dex = _dex()
        dex["legalProtocolEffects"] = {"side_condition": ["quickguard", "wideguard"]}

        result = reduce_replay_state("state-test", _complete_ots(), events, dex=dex)

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
                "p1a: Alpha",
                "p2a: Golf",
                "Future Sight",
                "move: Future Sight",
                "p2a: Hotel",
                "p1a: Bravo",
                "p2a: Hotel",
                ReplayMemberId(ReplaySide.P2, 1),
                ReplayMemberId(ReplaySide.P1, 0),
            ),
            (
                "p2a: Golf",
                "p1a: Alpha",
                "Doom Desire",
                "Doom Desire",
                "p1a: Bravo",
                "p2a: Hotel",
                "p1a: Bravo",
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
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
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

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()
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
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|switch|p2b: Hotel|Hotel, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|swap|p2a: Golf|1",
            "|turn|3",
            "|-end|p2a: Hotel|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()

        post_swap = snapshots[6].delayed_moves[0]
        assert post_swap.target_side == ReplaySide.P2
        assert post_swap.target_slot == 0
        assert post_swap.source_member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert post_swap.scheduled_turn == 3
        assert snapshots[-1].delayed_moves == ()

    def test_failed_delayed_move_without_target_preserves_existing_condition(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|turn|2",
            "|move|p1a: Alpha|Future Sight||[still]",
            "|-fail|p1a: Alpha",
            "|turn|3",
            "|-end|p2a: Golf|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()

        post_fail = snapshots[7].delayed_moves[0]
        assert post_fail.source_member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert post_fail.target_side == ReplaySide.P2
        assert post_fail.target_slot == 0
        assert post_fail.scheduled_turn == 3
        assert snapshots[-1].delayed_moves == ()

    def test_delayed_move_to_fainted_target_skips_untracked_condition(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|switch|p2b: Hotel|Hotel, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|turn|2",
            "|-damage|p2b: Hotel|0 fnt",
            "|faint|p2b: Hotel",
            "|move|p1a: Alpha|Future Sight|p2: Hotel",
            "|-start|p1a: Alpha|move: Future Sight",
            "|-hint|Future Sight did not hit because the target is fainted.",
            "|turn|3",
            "|-end|p2a: Golf|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()

        assert len(snapshots[10].delayed_moves) == 1
        assert snapshots[10].delayed_moves[0].target_slot == 0
        assert snapshots[10].delayed_moves[0].source_member_id == ReplayMemberId(ReplaySide.P1, 0)
        assert snapshots[11].delayed_moves == snapshots[10].delayed_moves
        assert snapshots[-1].delayed_moves == ()

    def test_delayed_move_hint_clears_condition_without_end_event(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Bravo|Bravo, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p1b: Bravo",
            "|-start|p1a: Alpha|move: Future Sight",
            "|turn|2",
            "|switch|p1a: Charlie|Charlie, L50|100/100",
            "|switch|p1b: Alpha|Alpha, L50|100/100",
            "|turn|3",
            "|-hint|Future Sight did not hit because the target is the user.",
            "|turn|4",
            "|move|p1a: Charlie|Future Sight|p1b: Alpha",
            "|-start|p1a: Charlie|move: Future Sight",
            "|turn|6",
            "|-end|p1b: Alpha|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()

        assert len(snapshots[8].delayed_moves) == 1
        assert snapshots[8].delayed_moves[0].scheduled_turn == 3
        assert snapshots[9].delayed_moves == ()
        assert len(snapshots[12].delayed_moves) == 1
        assert snapshots[12].delayed_moves[0].scheduled_turn == 6
        assert snapshots[-1].delayed_moves == ()

    def test_delayed_move_hint_only_clears_the_fainted_target_slot(self) -> None:
        ots = _complete_delayed_ots()
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Bravo|Bravo, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|switch|p2b: Hotel|Hotel, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|move|p1b: Bravo|Future Sight|p2b: Hotel",
            "|-start|p1b: Bravo|move: Future Sight",
            "|turn|2",
            "|-damage|p2a: Golf|0 fnt",
            "|faint|p2a: Golf",
            "|turn|3",
            "|-hint|Future Sight did not hit because the target is fainted.",
            "|-end|p2b: Hotel|move: Future Sight",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()

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
                    "|switch|p1a: Alpha|Alpha, L50|100/100",
                    "|-start|p1a: Alpha|move: Future Sight",
                ),
                1,
                "futuresight start has an ambiguous pending move from the referenced caster",
            ),
            (
                (
                    "|switch|p1a: Alpha|Alpha, L50|100/100",
                    "|switch|p2a: Golf|Golf, L50|100/100",
                    "|turn|1",
                    "|move|p1a: Alpha|Future Sight|p2a: Golf",
                    "|-start|p1a: Alpha|move: Future Sight",
                    "|-end|p2a: Golf|move: Future Sight",
                ),
                5,
                "delayed move ended before its scheduled turn",
            ),
            (
                (
                    "|switch|p1a: Alpha|Alpha, L50|100/100",
                    "|switch|p2a: Golf|Golf, L50|100/100",
                    "|turn|1",
                    "|move|p1a: Alpha|Future Sight|p2a: Golf",
                    "|-start|p1a: Alpha|move: Future Sight",
                    "|move|p1a: Alpha|Future Sight|p2a: Golf",
                ),
                5,
                "delayed move target slot already has a pending effect",
            ),
            (
                (
                    "|switch|p1a: Alpha|Alpha, L50|100/100",
                    "|switch|p2a: Golf|Golf, L50|100/100",
                    "|turn|1",
                    "|move|p1a: Alpha|Future Sight|p2a: Golf",
                    "|-start|p1a: Alpha|move: Future Sight",
                    "|turn|4",
                    "|-end|p2a: Golf|move: Future Sight",
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

        result = reduce_replay_state("state-test", ots, events, dex=_dex())

        assert result.snapshots == ()
        assert [(item.line_index, item.reason) for item in result.diagnostics] == [
            (line_index, reason)
        ]

    def test_delayed_move_source_only_context_is_rejected_during_event_parsing(self) -> None:
        line = ProtocolLine(
            0,
            "|move|p1a: Alpha|Future Sight",
            ("", "move", "p1a: Alpha", "Future Sight"),
            None,
        )

        parsed = parse_protocol_events("state-test", (line,))

        assert parsed.events[0].classification.rejects_replay
        assert "delayed move requires an explicit target" in parsed.diagnostics[0].reason

    def test_partial_trap_end_uses_the_protocol_effect_variant(self) -> None:
        dex = _dex()
        cast(list[dict[str, object]], dex["moves"]).append(
            {
                "id": "firespin",
                "name": "Fire Spin",
                "type": "Fire",
                "category": "Special",
                "target": "normal",
                "pp": 15,
                "volatileStatus": "partiallytrapped",
            }
        )
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|-activate|p1a: Alpha|move: Fire Spin|[of] p2a: Golf",
            "|-end|p1a: Alpha|Fire Spin|[partiallytrapped]|[silent]",
        )

        snapshots = reduce_replay_state(
            "state-test", _complete_ots(), events, dex=dex
        ).require_accepted()
        trapped = snapshots[2].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(trapped.effects) == {"firespin": 0}
        assert dict(trapped.effect_sources) == {"firespin": ReplayMemberId(ReplaySide.P2, 0)}

        ended = snapshots[3].member(ReplayMemberId(ReplaySide.P1, 0))
        assert dict(ended.effects) == {}
        assert dict(ended.effect_sources) == {}

    def test_illusion_reveal_preserves_actual_state_and_causal_display_history(self) -> None:
        ots = _complete_ots(illusion_member="Bravo")
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Charlie|Charlie, L50|100/100",
            "|-damage|p1a: Alpha|25/100",
            "|-boost|p1a: Alpha|atk|2",
            "|replace|p1a: Bravo|Bravo, L50",
            "|-end|p1a: Bravo|Illusion",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)

        disguised = snapshots[0].member(bravo_id)
        damaged = snapshots[2].member(bravo_id)
        revealed = snapshots[4].member(bravo_id)
        assert snapshots[0].sides[0].active[0] == bravo_id
        assert disguised.displayed_species == "Alpha"
        assert not disguised.revealed
        assert dict(disguised.effects) == {"illusion": 0}
        assert damaged.hp_fraction == 0.25
        assert snapshots[2].member(alpha_id).hp_fraction is None
        assert revealed.displayed_species == "Bravo"
        assert revealed.revealed
        assert revealed.hp_fraction == 0.25
        assert dict(revealed.boosts)["atk"] == 2
        assert "illusion" not in dict(revealed.effects)
        assert snapshots[0].member(bravo_id).displayed_species == "Alpha"

    def test_duplicate_illusion_display_does_not_mutate_the_disguise_target(self) -> None:
        ots = _complete_ots(illusion_member="Bravo")
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p1b: Alpha|Alpha, L50|100/100",
            "|-damage|p1b: Alpha|50/100",
            "|replace|p1b: Bravo|Bravo, L50",
            "|-end|p1b: Bravo|Illusion",
            ots=ots,
        )

        snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)
        final = snapshots[-1]

        assert final.sides[0].active == (alpha_id, bravo_id)
        assert final.member(alpha_id).hp_fraction == 1.0
        assert final.member(bravo_id).hp_fraction == 0.5
        assert final.member(alpha_id).displayed_species == "Alpha"
        assert final.member(bravo_id).displayed_species == "Bravo"

    def test_faint_before_illusion_reveal_updates_only_the_actual_member(self) -> None:
        ots = _complete_ots(illusion_member="Bravo")
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-damage|p1a: Alpha|0 fnt",
            "|faint|p1a: Alpha",
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            ots=ots,
        )

        final = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()[-1]
        alpha_id = ReplayMemberId(ReplaySide.P1, 0)
        bravo_id = ReplayMemberId(ReplaySide.P1, 1)

        assert final.sides[0].active[0] == alpha_id
        assert final.member(alpha_id).hp_fraction == 1.0
        assert not final.member(alpha_id).fainted
        assert final.member(bravo_id).hp_fraction == 0.0
        assert final.member(bravo_id).fainted

    def test_invalid_reducer_transition_discards_all_snapshots(self) -> None:
        events = _resolved(
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-sideend|p1: Player|Reflect",
        )
        result = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())

        assert result.snapshots == ()
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].line_index == 1
        assert result.diagnostics[0].reason == "side condition 'reflect' ended before it started"
        with pytest.raises(ReplayEventParseError, match="ended before it started"):
            result.require_accepted()

    @pytest.mark.skipif(not _GOLDEN_REPLAYS, reason="local golden replays are not present")
    @pytest.mark.parametrize("replay_path", _GOLDEN_REPLAYS, ids=lambda path: path.stem)
    def test_local_golden_replay_reduces_or_reports_known_rejection(
        self, replay_path: Path
    ) -> None:
        document = parse_replay_payload(replay_path.read_bytes())
        reconstruction = reconstruct_replay_state(document)

        if replay_path.stem in _UNRESOLVED_ILLUSION_REPLAYS:
            assert tuple(diagnostic.reason for diagnostic in reconstruction.diagnostics) == (
                "unresolved_illusion: active history has multiple valid assignments",
            )
            return

        snapshots = reconstruction.require_accepted()

        assert len(snapshots) == len(document.protocol_lines)
        assert tuple(snapshot.line_index for snapshot in snapshots) == tuple(
            range(len(document.protocol_lines))
        )
