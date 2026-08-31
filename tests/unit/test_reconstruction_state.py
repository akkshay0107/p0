"""Tests for the owned replay reconstruction state reducer."""

from __future__ import annotations

from pathlib import Path

import pytest

from p0.replays.identity import ReplayMemberId, ReplaySide
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.decisions import build_decision_view
from p0.replays.reconstruction.events import parse_protocol_events
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.reconstruction.state import (
    normalize_dynamic_effect,
    reconstruct_replay_state,
    reduce_replay_state,
)
from p0.replays.schema import OTSData, OTSMember, ProtocolLine

_GOLDEN_REPLAY_DIRECTORY = (
    Path(__file__).parents[2] / "src/p0/replays/reconstruction/golden_replays"
)
_GOLDEN_REPLAYS = tuple(sorted(_GOLDEN_REPLAY_DIRECTORY.glob("*.json")))
_P1_SPECIES = ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot")
_P2_SPECIES = ("Golf", "Hotel", "India", "Juliet", "Kilo", "Lima")


def _dex() -> dict[str, object]:
    species = []
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
):
    lines = tuple(
        ProtocolLine(index, raw, tuple(raw.split("|")), None) for index, raw in enumerate(raw_lines)
    )
    parsed = parse_protocol_events("state-test", lines)
    sheets = _complete_ots() if ots is None else ots
    return resolve_protocol_events("state-test", sheets, parsed.events).require_accepted()


def test_switch_cleanup_preserves_persistent_state_and_old_snapshots() -> None:
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


def test_implicit_action_moves_do_not_require_ots_move_slots() -> None:
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


def test_recharge_cant_consumes_mustrecharge_state() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-mustrecharge|p1a: Alpha",
        "|cant|p1a: Alpha|recharge",
    )

    final = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())
    alpha = final.require_accepted()[-1].member(ReplayMemberId(ReplaySide.P1, 0))

    assert "mustrecharge" not in dict(alpha.effects)


def test_toxic_stage_resets_when_a_statused_member_reenters() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100 tox",
        "|turn|1",
        "|switch|p1a: Bravo|Bravo, L50|100/100",
        "|switch|p1a: Alpha|Alpha, L50|100/100 tox",
    )

    final = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())
    alpha = final.require_accepted()[-1].member(ReplayMemberId(ReplaySide.P1, 0))

    assert alpha.status == "tox"
    assert alpha.status_counter == 0


def test_faint_cleanup_clears_status_counter() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100 tox",
        "|turn|1",
        "|-damage|p1a: Alpha|0 fnt",
        "|faint|p1a: Alpha",
    )

    final = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())
    alpha = final.require_accepted()[-1].member(ReplayMemberId(ReplaySide.P1, 0))

    assert alpha.status is None
    assert alpha.status_counter == 0


def test_wish_follows_its_physical_slot_and_preserves_status() -> None:
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


def test_wish_expires_at_the_resolution_upkeep_without_healing() -> None:
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


def test_healing_wish_clears_replacement_status_and_consumes_slot_condition() -> None:
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


def test_transform_is_an_immutable_overlay_with_its_own_pp() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|switch|p2a: Golf|Golf, L50|100/100",
        "|-boost|p2a: Golf|atk|2",
        "|-transform|p1a: Alpha|Golf",
        "|detailschange|p2a: Golf|Golf-Mega, L50",
        "|move|p1a: Alpha|Tackle|p2a: Golf",
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
    assert snapshots[6].member(alpha_id).transform is None
    assert tuple(move.move_id for move in snapshots[6].member(alpha_id).moves) == (
        "protect",
        "mimic",
    )


def test_declared_team_size_excludes_unselected_open_sheet_reserves() -> None:
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
        mega_items=frozenset(),
    )

    assert final.team_sizes == (4, 4)
    assert view.slots[0].switch_slots == ()
    assert not view.slots[0].force_switch


def test_unnamed_skill_swap_uses_current_ability_components() -> None:
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


def test_trace_source_reference_does_not_overwrite_the_target_ability() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|switch|p2a: Golf|Golf, L50|100/100",
        "|-ability|p1a: Alpha|Ability Golf|[from] ability: Trace|[of] p2a: Golf",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]
    assert final.member(ReplayMemberId(ReplaySide.P1, 0)).ability.current == "Ability Golf"
    assert final.member(ReplayMemberId(ReplaySide.P2, 0)).ability.current == "Ability Golf"


def test_endability_preserves_ability_and_records_gastro_acid() -> None:
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


def test_perish_countdown_updates_one_effect_and_clears_on_switch() -> None:
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


def test_invalid_perish_count_is_rejected_without_partial_snapshots() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-start|p1a: Alpha|perish4",
    )
    dex = _dex()
    dex["legalProtocolEffects"] = {"effect": ["perishsong"]}

    result = reduce_replay_state("state-test", _complete_ots(), events, dex=dex)

    assert result.snapshots == ()
    assert result.diagnostics[0].reason == "unsupported effect variant 'perish4'"


@pytest.mark.parametrize(
    ("effect", "canonical", "value"),
    (
        ("stockpile1", "stockpile", 1),
        ("stockpile2", "stockpile", 2),
        ("stockpile3", "stockpile", 3),
        ("protosynthesisatk", "protosynthesis", "atk"),
        ("protosynthesisdef", "protosynthesis", "def"),
        ("protosynthesisspa", "protosynthesis", "spa"),
        ("protosynthesisspd", "protosynthesis", "spd"),
        ("protosynthesisspe", "protosynthesis", "spe"),
        ("quarkdriveatk", "quarkdrive", "atk"),
        ("quarkdrivedef", "quarkdrive", "def"),
        ("quarkdrivespa", "quarkdrive", "spa"),
        ("quarkdrivespd", "quarkdrive", "spd"),
        ("quarkdrivespe", "quarkdrive", "spe"),
        ("fallen1", "supremeoverlord", 1),
        ("fallen2", "supremeoverlord", 2),
        ("fallen3", "supremeoverlord", 3),
        ("fallen4", "supremeoverlord", 4),
        ("fallen5", "supremeoverlord", 5),
    ),
)
def test_dynamic_effect_normalization_retains_canonical_data(
    effect: str, canonical: str, value: int | str
) -> None:
    variant = normalize_dynamic_effect(effect)

    assert variant is not None
    assert (variant.canonical_id, variant.wire_id, variant.value) == (canonical, effect, value)


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
def test_invalid_dynamic_effect_variants_reject_without_partial_snapshots(effect: str) -> None:
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
        ("protosynthesisatk", "protosynthesis", "Protosynthesis"),
        ("quarkdrivespe", "quarkdrive", "Quark Drive"),
        ("fallen3", "supremeoverlord", "fallen3"),
    ),
)
def test_dynamic_effect_start_and_end_clear_metadata(
    wire_effect: str, canonical: str, end_effect: str
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


def test_dynamic_effect_metadata_is_cleared_when_the_member_switches() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-start|p1a: Alpha|stockpile2",
        "|switch|p1a: Bravo|Bravo, L50|100/100",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]

    alpha = final.member(ReplayMemberId(ReplaySide.P1, 0))
    assert dict(alpha.effects) == {}
    assert dict(alpha.effect_variants) == {}


def test_repeated_stockpile_replaces_only_the_dynamic_layer_metadata() -> None:
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
        ("Crafty Shield", "craftyshield"),
        ("Mat Block", "matblock"),
        ("Quick Guard", "quickguard"),
        ("Wide Guard", "wideguard"),
    ),
)
def test_side_guard_is_side_scoped_survives_actions_and_expires_at_upkeep(
    guard: str, guard_id: str
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


def test_breaking_move_removes_side_guard_and_protection_counter() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|turn|1",
        "|-singleturn|p1a: Alpha|Wide Guard",
        "|-activate|p1a: Alpha|move: Feint|[broken]",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]

    assert dict(final.sides[0].conditions) == {}
    assert final.member(ReplayMemberId(ReplaySide.P1, 0)).protect_counter == 0


def test_breaking_move_removes_member_protect_effect() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|turn|1",
        "|-singleturn|p1a: Alpha|Protect",
        "|-activate|p1a: Alpha|move: Feint",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]

    member = final.member(ReplayMemberId(ReplaySide.P1, 0))
    assert dict(member.effects) == {}
    assert member.protect_counter == 0


def test_side_guard_follows_side_condition_swap_but_not_source_switch_or_faint() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|switch|p2a: Golf|Golf, L50|100/100",
        "|turn|1",
        "|-singleturn|p1a: Alpha|Quick Guard",
        "|-singleturn|p2a: Golf|Mat Block",
        "|-swapsideconditions",
        "|switch|p1a: Bravo|Bravo, L50|100/100",
        "|faint|p1a: Bravo",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]

    assert dict(final.sides[0].conditions) == {"matblock": 1}
    assert dict(final.sides[1].conditions) == {"quickguard": 1}


def test_duplicate_side_guard_and_member_scoped_singleturn_have_distinct_lifecycles() -> None:
    duplicate_events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-singleturn|p1a: Alpha|Wide Guard",
        "|-singleturn|p1a: Alpha|Wide Guard",
    )
    member_events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-singleturn|p1a: Alpha|Protect",
    )

    duplicate = reduce_replay_state("state-test", _complete_ots(), duplicate_events, dex=_dex())
    member = reduce_replay_state("state-test", _complete_ots(), member_events, dex=_dex())

    assert duplicate.snapshots == ()
    assert "started twice" in duplicate.diagnostics[0].reason
    assert dict(member.snapshots[-1].member(ReplayMemberId(ReplaySide.P1, 0)).effects) == {
        "protect": 0
    }


def test_member_singleturn_effect_expires_at_upkeep() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-singleturn|p1a: Alpha|Protect",
        "|upkeep",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]

    assert dict(final.member(ReplayMemberId(ReplaySide.P1, 0)).effects) == {}


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
def test_other_singleturn_effects_remain_member_scoped(effect: str) -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        f"|-singleturn|p1a: Alpha|{effect}",
    )

    final = reduce_replay_state(
        "state-test", _complete_ots(), events, dex=_dex()
    ).require_accepted()[-1]
    member = final.member(ReplayMemberId(ReplaySide.P1, 0))

    assert dict(final.sides[0].conditions) == {}
    assert dict(member.effects) == {effect.lower().replace(" ", ""): 0}


def test_protection_counter_resets_on_failure_and_switch() -> None:
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
        "|turn|2",
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
    assert delayed.announced
    assert finished.delayed_moves == ()
    assert finished.member(target_id).hp_fraction == 0.5


def test_delayed_move_stays_with_physical_slot_after_swap() -> None:
    ots = _complete_delayed_ots()
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|switch|p2a: Golf|Golf, L50|100/100",
        "|switch|p2b: Hotel|Hotel, L50|100/100",
        "|turn|1",
        "|move|p1a: Alpha|Future Sight|p2a: Golf",
        "|-start|p1a: Alpha|move: Future Sight",
        "|swap|p2a: Golf|1",
        "|turn|2",
        "|-end|p2a: Hotel|move: Future Sight",
        ots=ots,
    )

    snapshots = reduce_replay_state("state-test", ots, events, dex=_dex()).require_accepted()

    assert snapshots[5].delayed_moves[0].target_slot == 0
    assert snapshots[-1].delayed_moves == ()


@pytest.mark.parametrize(
    "lines",
    (
        (
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|-start|p1a: Alpha|move: Future Sight",
        ),
        (
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|-end|p2a: Golf|move: Future Sight",
        ),
        (
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
        ),
        (
            "|switch|p1a: Alpha|Alpha, L50|100/100",
            "|switch|p2a: Golf|Golf, L50|100/100",
            "|turn|1",
            "|move|p1a: Alpha|Future Sight|p2a: Golf",
            "|-start|p1a: Alpha|move: Future Sight",
            "|turn|3",
            "|-end|p2a: Golf|move: Future Sight",
        ),
    ),
)
def test_delayed_move_rejects_missing_pending_duplicate_early_or_late_hit(
    lines: tuple[str, ...],
) -> None:
    ots = _complete_delayed_ots()
    events = _resolved(*lines, ots=ots)

    result = reduce_replay_state("state-test", ots, events, dex=_dex())

    assert result.snapshots == ()
    assert result.diagnostics


def test_delayed_move_source_only_context_is_rejected_during_event_parsing() -> None:
    line = ProtocolLine(
        0,
        "|move|p1a: Alpha|Future Sight",
        ("", "move", "p1a: Alpha", "Future Sight"),
        None,
    )

    parsed = parse_protocol_events("state-test", (line,))

    assert parsed.events[0].classification.rejects_replay
    assert "delayed move requires an explicit target" in parsed.diagnostics[0].reason


def test_illusion_reveal_preserves_actual_state_and_causal_display_history() -> None:
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


def test_duplicate_illusion_display_does_not_mutate_the_disguise_target() -> None:
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


def test_faint_before_illusion_reveal_updates_only_the_actual_member() -> None:
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


def test_unsupported_reducer_transition_discards_all_snapshots() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|-center",
    )

    result = reduce_replay_state("state-test", _complete_ots(), events, dex=_dex())

    assert result.snapshots == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].line_index == 1
    assert "not implemented" in result.diagnostics[0].reason


@pytest.mark.skipif(not _GOLDEN_REPLAYS, reason="local golden replays are not present")
@pytest.mark.parametrize("replay_path", _GOLDEN_REPLAYS, ids=lambda path: path.stem)
def test_local_golden_replay_reduces_or_reports_known_rejection(replay_path: Path) -> None:
    document = parse_replay_payload(replay_path.read_bytes())
    reconstruction = reconstruct_replay_state(document)

    if reconstruction.diagnostics:
        assert tuple(diagnostic.reason for diagnostic in reconstruction.diagnostics) == (
            "unresolved_illusion: active history has multiple valid assignments",
        )
        return

    snapshots = reconstruction.require_accepted()

    assert len(snapshots) == len(document.protocol_lines)
    assert tuple(snapshot.line_index for snapshot in snapshots) == tuple(
        range(len(document.protocol_lines))
    )
