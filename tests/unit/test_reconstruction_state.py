"""Tests for the owned replay reconstruction state reducer."""

from __future__ import annotations

from pathlib import Path

import pytest

from p0.replays.identity import ReplayMemberId, ReplaySide
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruction.events import parse_protocol_events
from p0.replays.reconstruction.resolution import resolve_protocol_events
from p0.replays.reconstruction.state import reconstruct_replay_state, reduce_replay_state
from p0.replays.schema import OTSData, OTSMember, ProtocolLine

_GOLDEN_REPLAY = (
    Path(__file__).parents[2]
    / "src/p0/replays/reconstruction/golden_replays"
    / "gen9championsvgc2026regmbbo3-2641278886.json"
)
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
        ],
    }


def _ots(side: ReplaySide, names: tuple[str, ...]) -> OTSData:
    members = tuple(
        OTSMember(
            member_id=ReplayMemberId(side, index),
            nickname=name,
            species=name,
            item=f"Item {name}",
            ability=f"Ability {name}",
            moves=("Protect", "Tackle") if side is ReplaySide.P2 else ("Protect", "Mimic"),
            nature="Serious",
            gender="",
            level=50,
            evs="",
            raw_packed_set=name,
        )
        for index, name in enumerate(names)
    )
    return OTSData(side, "]".join(names), members)


def _complete_ots() -> tuple[OTSData, OTSData]:
    return _ots(ReplaySide.P1, _P1_SPECIES), _ots(ReplaySide.P2, _P2_SPECIES)


def _resolved(*raw_lines: str):
    lines = tuple(
        ProtocolLine(index, raw, tuple(raw.split("|")), None) for index, raw in enumerate(raw_lines)
    )
    parsed = parse_protocol_events("state-test", lines)
    return resolve_protocol_events("state-test", _complete_ots(), parsed.events).require_accepted()


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


def test_transform_is_an_immutable_overlay_with_its_own_pp() -> None:
    events = _resolved(
        "|switch|p1a: Alpha|Alpha, L50|100/100",
        "|switch|p2a: Golf|Golf, L50|100/100",
        "|-boost|p2a: Golf|atk|2",
        "|-transform|p1a: Alpha|p2a: Golf",
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


@pytest.mark.skipif(not _GOLDEN_REPLAY.is_file(), reason="local golden replay is not present")
def test_local_golden_replay_reduces_to_one_snapshot_per_line() -> None:
    document = parse_replay_payload(_GOLDEN_REPLAY.read_bytes())

    snapshots = reconstruct_replay_state(document).require_accepted()

    assert len(snapshots) == len(document.protocol_lines) == 186
    assert tuple(snapshot.line_index for snapshot in snapshots) == tuple(range(186))
