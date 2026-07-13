"""Temporary semantic characterization of the closed Workstream A-D contracts."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.weather import Weather

from p0.env import MegaEnv
from p0.format_config import RuntimeManifest, validate_artifact_manifest_reference
from p0.model.event_builder import BattleEvent, EventCollector, EventTypeId
from p0.model.observation_builder import from_battle
from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    Knownness,
    Provenance,
    SideId,
    TokenType,
)
from p0.team_data.stat_points import BaseStats, StatPoints, calculate_stats


def _action_battle() -> Any:
    team = [
        Pokemon(gen=9, species=species)
        for species in ("charizard", "venusaur", "pikachu", "gengar", "dragonite", "tyranitar")
    ]
    for move_id in ("protect", "tackle", "thunderbolt", "surf"):
        team[0]._moves._base_moves[move_id] = Move(move_id, 9)
    return SimpleNamespace(
        player_username="player",
        battle_tag="characterization",
        team={str(index): pokemon for index, pokemon in enumerate(team)},
        active_pokemon=[team[0], None],
        available_moves=[[Move("struggle", 9)], []],
    )


def test_all_49_individual_action_ids_retain_their_meaning() -> None:
    battle = _action_battle()
    assert "pass" in str(MegaEnv._action_to_order_individual(np.int64(0), battle, True, 0))
    for action in range(1, 7):
        order = MegaEnv._action_to_order_individual(np.int64(action), battle, True, 0)
        assert "switch" in str(order)
        assert order.order is list(battle.team.values())[action - 1]

    moves = list(battle.active_pokemon[0].moves.values())
    for action in range(7, 47):
        order = MegaEnv._action_to_order_individual(np.int64(action), battle, True, 0)
        assert order.order is moves[(action - 7) % 20 // 5]
        assert order.move_target == (action - 7) % 5 - 2
        assert bool(order.mega) is (action >= 27)

    for action, mega in ((47, True), (48, False)):
        order = MegaEnv._action_to_order_individual(np.int64(action), battle, True, 0)
        assert cast(Any, order.order).id == "struggle"
        assert bool(order.mega) is mega


def test_team_preview_pair_encoding_preserves_four_unique_members() -> None:
    battle = _action_battle()
    battle.teampreview = True
    order = MegaEnv.action_to_order(np.array([1, 8], dtype=np.int64), battle)
    assert order.message == "/team 123456"
    selected = order.message.removeprefix("/team ")
    assert len(selected[:4]) == len(set(selected[:4])) == 4


def _golden_battle() -> DoubleBattle:
    battle = DoubleBattle("golden", "characterization", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    ally = Pokemon(gen=9, species="charizard")
    ally._active = True
    ally._current_hp = 73
    ally._max_hp = 100
    ally._effects = {Effect.CONFUSION: 2}
    opponent = Pokemon(gen=9, species="venusaur")
    opponent._active = True
    battle._team = {"p1: Charizard": ally}
    battle._opponent_team = {"p2: Venusaur": opponent}
    battle._active_pokemon = {"p1a": ally}
    battle._opponent_active_pokemon = {"p2a": opponent}
    battle._weather = {Weather.SUNNYDAY: 1}
    battle._fields = {Field.TRICK_ROOM: 2}
    battle._turn = 3
    return battle


def test_golden_observation_shape_indices_and_enum_ids() -> None:
    observation = from_battle(_golden_battle())
    assert observation.token_type_ids.shape == (SEQUENCE_LENGTH,)
    assert observation.side_ids.shape == (SEQUENCE_LENGTH,)
    assert observation.slot_ids.shape == (SEQUENCE_LENGTH,)
    assert observation.categorical.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
    assert observation.numerical.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)
    assert observation.events_cat.shape == (EVENT_COUNT, EVENT_CATEGORICAL_WIDTH)
    assert observation.events_num.shape == (EVENT_COUNT, EVENT_NUMERICAL_WIDTH)
    assert observation.token_type_ids[[0, 1, 2, 25, 26]].tolist() == [
        TokenType.CLS,
        TokenType.POKEMON_SUPER,
        TokenType.POKEMON_NUMERIC,
        TokenType.FIELD_SUPER,
        TokenType.FIELD_NUMERIC,
    ]
    assert observation.side_ids[[0, 1, 13, 27, 29]].tolist() == [
        SideId.NONE,
        SideId.ALLY,
        SideId.OPPONENT,
        SideId.ALLY,
        SideId.OPPONENT,
    ]
    assert [member.value for member in Knownness] == [0, 1, 2, 3, 4]
    assert [member.value for member in Provenance] == [0, 1, 2, 3, 4, 5]
    assert observation.numerical[2, 5].item() == pytest.approx(0.73)


def test_event_priority_order_and_overflow_are_stable() -> None:
    events = [BattleEvent(EventTypeId.DAMAGE, "p1a: Charizard", order=i) for i in range(65)]
    events.append(BattleEvent(EventTypeId.MOVE, "p2a: Venusaur", order=65))
    selected = EventCollector.truncate_events(events, limit=EVENT_COUNT)
    assert len(selected) == EVENT_COUNT
    assert [event.order for event in selected] == [*range(63), 65]
    assert selected[-1].event_type == EventTypeId.MOVE


def test_stat_formula_and_manifest_rejection_are_stable(tmp_path) -> None:
    stats = calculate_stats(
        BaseStats(78, 84, 78, 109, 85, 100),
        StatPoints(hp=2, spa=32, spe=32),
        "modest",
    )
    assert stats == (155, 93, 98, 177, 105, 152)

    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(RuntimeManifest().to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_artifact_manifest_reference(
            {"runtime_manifest_sha256": "0" * 64}, manifest_path
        )
