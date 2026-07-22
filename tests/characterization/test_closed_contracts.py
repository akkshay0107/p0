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

from p0.battle.events import BattleEvent, EventTypeId, truncate_events
from p0.format_config import current_manifest, validate_artifact_runtime_contract
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_STATUS_COUNTER_KIND,
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    MAX_EFFECTS,
    NUM_EFFECT_START,
    NUM_IDX_STATUS_COUNTER,
    NUMERICAL_WIDTH,
    OBSERVATION_SCHEMA_VERSION,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
    TOKEN_IDX_GLOBAL_FIELD,
    TOKEN_IDX_OPPONENT_SIDE,
    CounterKind,
    Knownness,
    Provenance,
    SideId,
    TokenType,
)
from p0.runtime.poke_env_action_adapter import action_to_order, action_to_single_order
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.stat_points import BaseStats, StatPoints, calculate_stats


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
    assert "pass" in str(action_to_single_order(0, battle, True, 0))
    for action in range(1, 7):
        order = action_to_single_order(action, battle, True, 0)
        assert "switch" in str(order)
        assert order.order is list(battle.team.values())[action - 1]

    moves = list(battle.active_pokemon[0].moves.values())
    for action in range(7, 47):
        order = action_to_single_order(action, battle, True, 0)
        assert order.order is moves[(action - 7) % 20 // 5]
        assert order.move_target == (action - 7) % 5 - 2
        assert bool(order.mega) is (action >= 27)

    for action, mega in ((47, True), (48, False)):
        order = action_to_single_order(action, battle, True, 0)
        assert cast(Any, order.order).id == "struggle"
        assert bool(order.mega) is mega


def test_team_preview_pair_encoding_preserves_four_unique_members() -> None:
    battle = _action_battle()
    battle.teampreview = True
    order = action_to_order(np.array([1, 8], dtype=np.int64), battle)
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
    battle = _golden_battle()
    observation = ObservationBuilder(default_runtime_resources()).build(battle_view(battle))
    assert observation.token_type_ids.shape == (SEQUENCE_LENGTH,)
    assert observation.side_ids.shape == (SEQUENCE_LENGTH,)
    assert observation.slot_ids.shape == (SEQUENCE_LENGTH,)
    assert observation.categorical.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
    assert observation.numerical.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)
    assert observation.events_cat.shape == (EVENT_COUNT, EVENT_CATEGORICAL_WIDTH)
    assert observation.events_num.shape == (EVENT_COUNT, EVENT_NUMERICAL_WIDTH)
    assert observation.token_type_ids[[0, 1, 6, 12, 13, 14]].tolist() == [
        TokenType.POKEMON,
        TokenType.POKEMON,
        TokenType.POKEMON,
        TokenType.FIELD,
        TokenType.FIELD,
        TokenType.FIELD,
    ]
    assert observation.side_ids[[0, 1, 6, 12, 13, 14]].tolist() == [
        SideId.ALLY,
        SideId.ALLY,
        SideId.OPPONENT,
        SideId.NONE,
        SideId.ALLY,
        SideId.OPPONENT,
    ]
    assert [member.value for member in Knownness] == [0, 1, 2, 3, 4]
    assert [member.value for member in Provenance] == [0, 1, 2, 3, 4, 5]
    assert observation.numerical[0, 5].item() == pytest.approx(0.73)


def test_schema_v4_owned_entity_layout_is_pinned() -> None:
    """Pin the v4 owned-entity layout and fixed memory-channel dimensions."""
    assert OBSERVATION_SCHEMA_VERSION == 4
    assert SEQUENCE_LENGTH == 15
    assert MAX_EFFECTS == 12
    assert CATEGORICAL_WIDTH == 87
    assert NUMERICAL_WIDTH == 126
    assert CAT_IDX_STATUS_COUNTER_KIND == 50
    assert CAT_EFFECT_START == 51
    assert NUM_IDX_STATUS_COUNTER == 36
    assert NUM_EFFECT_START == 64
    assert (TOKEN_IDX_GLOBAL_FIELD, TOKEN_IDX_ALLY_SIDE, TOKEN_IDX_OPPONENT_SIDE) == (12, 13, 14)
    # the v2 super/numeric token-pair split is retired
    assert [member.name for member in TokenType] == ["POKEMON", "FIELD", "EVENT"]


def test_status_record_owns_counter_semantics() -> None:
    """StatusRecord (plan §3.1): SLP ages publicly, TOX stacks, others are presence-only."""
    from poke_env.battle.status import Status

    from p0.model.observation_builder import _status_counter_kind

    assert _status_counter_kind(Status.SLP) == CounterKind.TURN_AGE
    assert _status_counter_kind(Status.TOX) == CounterKind.STACK_COUNT
    assert _status_counter_kind(Status.BRN) == CounterKind.PRESENCE_ONLY
    assert _status_counter_kind(Status.PAR) == CounterKind.PRESENCE_ONLY
    assert _status_counter_kind(Status.FRZ) == CounterKind.PRESENCE_ONLY
    assert _status_counter_kind(None) == CounterKind.PRESENCE_ONLY

    battle = _golden_battle()
    ally = next(iter(battle.team.values()))
    ally._status = Status.TOX
    ally._status_counter = 2
    observation = ObservationBuilder(default_runtime_resources()).build(battle_view(battle))
    assert observation.categorical[0, CAT_IDX_STATUS_COUNTER_KIND] == CounterKind.STACK_COUNT
    assert observation.numerical[0, NUM_IDX_STATUS_COUNTER].item() == pytest.approx(2 / 5)


def test_event_priority_order_and_overflow_are_stable() -> None:
    events = [BattleEvent(EventTypeId.DAMAGE, "p1a: Charizard", order=i) for i in range(65)]
    events.append(BattleEvent(EventTypeId.MOVE, "p2a: Venusaur", order=65))
    selected = truncate_events(events, limit=EVENT_COUNT)
    assert len(selected) == EVENT_COUNT
    assert [event.order for event in selected] == [*range(EVENT_COUNT - 1), 65]
    assert selected[-1].event_type == EventTypeId.MOVE


def test_event_v4_contract_is_pinned() -> None:
    """Pin the event-v4 contract: new protocol coverage and re-tiered truncation."""
    from p0.battle.events import (
        EVENT_TYPE_COUNT,
        HIGH_PRIORITY_EVENTS,
        MEDIUM_PRIORITY_EVENTS,
    )

    assert EVENT_TYPE_COUNT == 44
    assert EventTypeId.CANT == 31
    assert EventTypeId.PREPARE == 32
    assert EventTypeId.SINGLEMOVE == 33
    assert EventTypeId.BOOST_SET == 34
    assert EventTypeId.BOOST_CLEAR == 35
    assert EventTypeId.BOOST_SWAP == 36
    assert EventTypeId.BOOST_INVERT == 37
    assert EventTypeId.BOOST_COPY == 38
    assert EventTypeId.TRANSFORM == 39
    assert EventTypeId.ABILITY_END == 40
    assert EventTypeId.ACTIVATE == 41
    assert EventTypeId.FIELD_ACTIVATE == 42
    assert EventTypeId.NO_TARGET == 43

    # structural/state transitions must outrank routine numeric events
    structural = {
        EventTypeId.MEGA,
        EventTypeId.WEATHER_START,
        EventTypeId.FIELD_START,
        EventTypeId.SIDE_START,
        EventTypeId.EFFECT_START,
        EventTypeId.ABILITY,
        EventTypeId.ITEM_TRANSFER,
        EventTypeId.DRAG,
        EventTypeId.CANT,
    }
    assert structural <= HIGH_PRIORITY_EVENTS
    assert {EventTypeId.DAMAGE, EventTypeId.HEAL, EventTypeId.BOOST} <= MEDIUM_PRIORITY_EVENTS
    assert HIGH_PRIORITY_EVENTS.isdisjoint(MEDIUM_PRIORITY_EVENTS)


def test_stat_formula_and_manifest_rejection_are_stable(tmp_path) -> None:
    stats = calculate_stats(
        BaseStats(78, 84, 78, 109, 85, 100),
        StatPoints(hp=2, spa=32, spe=32),
        "modest",
    )
    assert stats == (155, 93, 98, 177, 105, 152)

    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(current_manifest().to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_artifact_runtime_contract({"runtime_manifest_sha256": "0" * 64}, manifest_path)


def test_shared_contract_versions_and_enums_are_pinned() -> None:
    """Pin the cross-workstream schema constants frozen for the worktree split."""
    from p0.battle.series import MAX_PRIOR_GAMES, SERIES_SUMMARY_SCHEMA_VERSION
    from p0.replays.schema import (
        REPLAY_IR_SCHEMA_VERSION,
        DecisionType,
        GameEndReason,
        GroupingMethod,
        LabelKind,
        MaskProvenance,
    )
    from p0.replays.shards import SHARD_ARTIFACT_SCHEMA
    from p0.teams.corpus import CORPUS_MANIFEST_SCHEMA, CorpusSplit, SamplingPolicy

    assert REPLAY_IR_SCHEMA_VERSION == 1
    assert SERIES_SUMMARY_SCHEMA_VERSION == 1
    assert MAX_PRIOR_GAMES == 2
    assert SHARD_ARTIFACT_SCHEMA == "p0.replay_shard.v2"
    assert CORPUS_MANIFEST_SCHEMA == "p0.team_corpus.v1"

    pinned = {
        GroupingMethod: ["UNSPECIFIED", "PARENT_ROOM", "FALLBACK_SAME_PLAYERS"],
        GameEndReason: ["UNSPECIFIED", "NORMAL", "FORFEIT", "TIMEOUT"],
        DecisionType: [
            "UNSPECIFIED",
            "TEAM_PREVIEW",
            "TURN",
            "FORCED_SWITCH",
            "PIVOT_SWITCH",
            "FORCED_PASS",
        ],
        LabelKind: ["UNSPECIFIED", "EXACT", "PARTIAL", "UNKNOWN"],
        MaskProvenance: ["UNSPECIFIED", "CONSERVATIVE_RECONSTRUCTED", "ORACLE_REQUEST"],
        CorpusSplit: ["UNSPECIFIED", "TRAIN", "VALIDATION", "TEST", "HELD_OUT_ARCHETYPE"],
        SamplingPolicy: [
            "UNSPECIFIED",
            "USAGE_WEIGHTED",
            "UNIFORM_CANONICAL",
            "UNIFORM_ARCHETYPE",
            "RARE_COVERAGE",
            "MATCHUP_BALANCED",
        ],
    }
    for enum_cls, names in pinned.items():
        assert [member.name for member in enum_cls] == names
        assert [member.value for member in enum_cls] == list(range(len(names)))
