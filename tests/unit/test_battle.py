from __future__ import annotations

import logging
import random
import typing
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move, MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from p0.battle.actions import (
    ACT_SIZE,
    ActionKind,
    SlotAction,
    canonical_team_actions,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.battle.events import (
    EVENT_DIAGNOSTICS,
    BattleEvent,
    EventTypeId,
    RawBattleEvent,
    build_raw_event,
    get_hp_fraction,
    truncate_events,
)
from p0.battle.events import (
    parse_events as parse_protocol_events,
)
from p0.battle.legality import (
    DecisionView,
    SlotDecision,
    action_mask,
    legal_actions,
    second_action_mask,
    validate_joint_action,
)
from p0.battle.views import FixtureBattleView
from p0.format_config import ACTION_CONTRACT, FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import (
    ObservationBuilder,
    _ally_legality,
    _cached_imputed_stats,
    _get_ordered_pokemon,
    _get_pokemon_level_stats,
    _global_field_token_into,
    _iter_move_slots,
    _pokemon_categorical_into,
    _side_mega_available,
    _side_token_into,
    _slot_condition,
    _write_effects,
)
from p0.model.observation_builder import (
    _pokemon_numeric_into as _write_pokemon_numeric,
)
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    MAX_EFFECTS,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUM_IDX_TEAM_PREVIEW,
    NUMERICAL_WIDTH,
    UNKNOWN_LEGALITY,
    CounterKind,
    EffectNamespace,
    Provenance,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.model.tokenizer import tokenizer
from p0.replays.evidence import EvidenceRequest, ObservedAction, extract_action_evidence
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruct import reconstruct_both
from p0.replays.schema import LabelKind
from p0.runtime import poke_env_patches
from p0.runtime.env import MegaEnv, SimEnv
from p0.runtime.live_event_capture import consume_raw_events, set_raw_events
from p0.runtime.poke_env_action_adapter import (
    action_to_order,
    action_to_single_order,
    order_to_action,
    single_order_to_action,
)
from p0.runtime.poke_env_battle_adapter import battle_view, current_battle_view, decision_view
from p0.teams.source import ValidatedTeam
from p0.teams.stat_points import (
    BaseStats,
    Role,
    StatPoints,
    calculate_stats,
    classify_role,
    impute_candidates,
    imputed_stats,
    select_candidate,
)


def test_action_contract_round_trips_ids_and_describes_canonical_ranges() -> None:
    assert [encode_action(decode_action(action)) for action in range(ACT_SIZE)] == list(
        range(ACT_SIZE)
    )
    assert ACTION_CONTRACT["action_count"] == ACT_SIZE
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
    actions = {
        encode_team_pair(first, second)
        for first in range(6)
        for second in range(6)
        if first != second
    }
    assert len(actions) == 30
    assert all(decode_team_pair(action)[0] != decode_team_pair(action)[1] for action in actions)
    assert team_selection(1, 8)[:4] == (0, 1, 2, 3)


def test_scalar_joint_constraints_match_policy_vectorization() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(
                switch_slots=(2, 3),
                move_targets=((-2, 1, 2), (0,), (), (1,)),
                can_mega=True,
            ),
            SlotDecision(
                switch_slots=(2, 4),
                move_targets=((-1, 1), (2,), (0,), ()),
                can_mega=True,
            ),
        )
    )
    base = torch.from_numpy(action_mask(view)).unsqueeze(0)
    resources = default_runtime_resources()
    policy = build_policy(ModelConfig(32, 2, 1, 128), resources)
    for first in legal_actions(view, 0):
        logits = torch.zeros((1, 2, ACT_SIZE))
        masked = policy.actor._apply_sequential_masks(
            logits,
            torch.tensor([first]),
            base,
            torch.tensor([False]),
        )
        actual = torch.isfinite(masked[0, 1]).numpy()
        np.testing.assert_array_equal(actual, second_action_mask(view, first))


def test_event_parser_import_does_not_install_poke_env_patches() -> None:
    poke_env_patches.uninstall_for_tests()
    originals = (
        DoubleBattle.parse_message,
        Pokemon.switch_out,
        logging.Handler.handle,
    )
    __import__("p0.battle.events")
    assert originals == (
        DoubleBattle.parse_message,
        Pokemon.switch_out,
        logging.Handler.handle,
    )
    assert not poke_env_patches.is_installed()


def test_protocol_parser_accepts_an_injected_resource_resolver() -> None:
    class Resolver:
        def id_for(self, table: str, name: str | None) -> int:
            return 17 if table == "moves" and name == "Thunderbolt" else 0

        def effect_id_for(self, table: str, name: str | None) -> int:
            return 0

        def resolve(self, table: str, name: str | None) -> tuple[int, str]:
            resolved = self.id_for(table, name)
            return resolved, "known" if resolved else "oov"

    events = parse_protocol_events(
        [RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"))],
        Resolver(),
    )
    assert events[0].event_type is EventTypeId.MOVE
    assert events[0].move_id == 17


def test_unknown_replay_legality_uses_an_explicit_observation_sentinel() -> None:
    decision = DecisionView(
        slots=(
            SlotDecision(move_targets=((1,),), legality_known=False),
            SlotDecision(),
        )
    )
    battle = cast(Any, SimpleNamespace(decision=decision))

    moves, can_switch = _ally_legality(battle, 0, (None, None, None, None))

    assert moves == [UNKNOWN_LEGALITY] * 4
    assert can_switch == UNKNOWN_LEGALITY


def test_patch_installation_is_idempotent_reversible_and_logger_scoped() -> None:
    poke_env_patches.uninstall_for_tests()
    original = DoubleBattle.parse_message
    original_forme_change = Pokemon.forme_change
    poke_env_patches.install()
    installed = DoubleBattle.parse_message
    poke_env_patches.install()
    assert DoubleBattle.parse_message is installed
    assert installed is not original
    charizard = Pokemon(gen=9, species="charizard")
    charizard.forme_change("Charizard-Mega-Y, L50")
    assert charizard.species == "charizardmegay"
    poke_env_patches.uninstall_for_tests()
    assert DoubleBattle.parse_message is original
    assert Pokemon.forme_change is original_forme_change

    target = logging.getLogger("test.poke-env")
    other = logging.getLogger("test.other")
    poke_env_patches.install(target)
    record = logging.LogRecord("test", logging.WARNING, "", 0, "is active, but it's not", (), None)
    assert not target.filter(record)
    assert other.filter(record)
    poke_env_patches.uninstall_for_tests()


def test_live_adapter_and_pure_fixture_build_identical_observations() -> None:
    battle = DoubleBattle("view", "player", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    ally = Pokemon(gen=9, species="charizard")
    opponent = Pokemon(gen=9, species="venusaur")
    ally._active = True
    opponent._active = True
    battle._team = {"p1: Charizard": ally}
    battle._opponent_team = {"p2: Venusaur": opponent}
    battle._active_pokemon = {"p1a": ally}
    battle._opponent_active_pokemon = {"p2a": opponent}

    fixture = FixtureBattleView(
        team=battle.team,
        opponent_team=battle.opponent_team,
        active_pokemon=battle.active_pokemon,
        opponent_active_pokemon=battle.opponent_active_pokemon,
        available_moves=battle.available_moves,
        available_switches=battle.available_switches,
        can_mega_evolve=battle.can_mega_evolve,
        force_switch=battle.force_switch,
        trapped=battle.trapped,
        maybe_trapped=battle.maybe_trapped,
        teampreview=battle.teampreview,
        player_role=battle.player_role,
        wait=battle._wait,
        weather=battle.weather,
        fields=battle.fields,
        side_conditions=battle.side_conditions,
        opponent_side_conditions=battle.opponent_side_conditions,
        turn=battle.turn,
        used_mega_evolve=battle.used_mega_evolve,
        opponent_used_mega_evolve=battle.opponent_used_mega_evolve,
        decision=decision_view(battle),
    )
    builder = ObservationBuilder(default_runtime_resources())
    live = builder.build(battle_view(battle))
    pure = builder.build(fixture)
    for name in live._FIELD_NAMES:
        torch.testing.assert_close(getattr(live, name), getattr(pure, name))


def test_factory_shares_resources_and_preserves_state_dict_layout() -> None:
    resources = default_runtime_resources()
    config = ModelConfig(32, 2, 1, 128)
    direct = PolicyNet(config, resources)
    policy = build_policy(config, resources)
    builder = ObservationBuilder(resources=resources)
    assert policy.resources is policy.encoder.resources is builder.resources is resources
    assert policy.config == config == ModelConfig.from_dict(config.to_dict())
    assert direct.state_dict().keys() == policy.state_dict().keys()


def _local_parse_events(raw_events: list[RawBattleEvent]) -> list[BattleEvent]:
    return parse_protocol_events(raw_events, tokenizer)


def test_hp_fraction_accepts_showdown_status_suffixes() -> None:
    assert get_hp_fraction("50/100g") == 0.5
    assert get_hp_fraction("20/100y") == 0.2
    assert get_hp_fraction("0 fnt") == 0.0


def test_parse_events_returns_typed_events_in_protocol_order():
    raw_events = [
        RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
        RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100"), pre_hp=1.0),
        RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
        RawBattleEvent(("", "faint", "p2a: Charizard")),
    ]

    events = _local_parse_events(raw_events)

    assert all(isinstance(event, BattleEvent) for event in events)
    assert [event.event_type for event in events] == [
        EventTypeId.MOVE,
        EventTypeId.DAMAGE,
        EventTypeId.STATUS_SET,
        EventTypeId.FAINT,
    ]
    assert [event.order for event in events] == [0, 1, 2, 3]
    assert events[0].entity_id == "p1a: Pikachu"
    assert events[1].entity_id == "p2a: Charizard"
    assert events[1].value == -0.25


def test_parse_events_distinguishes_failed_and_blocked_moves():
    events = _local_parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
            RawBattleEvent(("", "-immune", "p2a: Charizard")),
            RawBattleEvent(("", "-fail", "p1a: Pikachu")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.MOVE,
        EventTypeId.BLOCKED,
        EventTypeId.FAILED,
    ]


def test_ability_field_and_move_evidence():
    events = _local_parse_events(
        [
            RawBattleEvent(
                ("", "move", "p1a: Mew", "Metronome", "p2a: Gengar", "[from] ability: Dancer")
            ),
            RawBattleEvent(("", "-ability", "p2a: Gengar", "Cursed Body")),
            RawBattleEvent(("", "-fieldstart", "move: Trick Room")),
            RawBattleEvent(("", "-fieldend", "move: Trick Room")),
        ]
    )

    assert events[0].target_id == "p2a: Gengar"
    assert events[0].flags & 4
    assert events[1].event_type == EventTypeId.ABILITY
    assert events[1].ability_id > 0
    assert [event.event_type for event in events[2:]] == [
        EventTypeId.FIELD_START,
        EventTypeId.FIELD_END,
    ]
    assert events[2].effect_id > 0


def test_status_codes_resolve_against_vocab():
    events = _local_parse_events(
        [
            RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
            RawBattleEvent(("", "-curestatus", "p2a: Charizard", "par")),
        ]
    )

    assert events[0].status_id == tokenizer.id_for("status", "par") > 0
    assert events[1].status_id == events[0].status_id


def test_cant_prepare_and_singlemove():
    events = _local_parse_events(
        [
            RawBattleEvent(("", "cant", "p1a: Pikachu", "flinch")),
            RawBattleEvent(("", "cant", "p1a: Pikachu", "slp", "Thunderbolt")),
            RawBattleEvent(("", "-prepare", "p1a: Charizard", "Fly", "p2a: Venusaur")),
            RawBattleEvent(("", "-singlemove", "p1a: Gengar", "Destiny Bond")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.CANT,
        EventTypeId.CANT,
        EventTypeId.PREPARE,
        EventTypeId.SINGLEMOVE,
    ]
    assert events[0].effect_id == tokenizer.id_for("volatiles", "flinch") > 0
    assert events[1].status_id == tokenizer.id_for("status", "slp") > 0
    assert events[1].move_id == tokenizer.id_for("moves", "thunderbolt") > 0
    assert events[2].move_id == tokenizer.id_for("moves", "fly") > 0
    assert events[2].target_id == "p2a: Venusaur"
    assert events[3].effect_id == tokenizer.id_for("volatiles", "destinybond") > 0


def test_boost_manipulation_family():
    events = _local_parse_events(
        [
            RawBattleEvent(
                ("", "-setboost", "p1a: Azumarill", "atk", "6", "[from] move: Belly Drum")
            ),
            RawBattleEvent(("", "-clearboost", "p1a: Azumarill")),
            RawBattleEvent(("", "-clearnegativeboost", "p1a: Azumarill")),
            RawBattleEvent(("", "-clearallboost")),
            RawBattleEvent(("", "-swapboost", "p1a: Malamar", "p2a: Incineroar", "atk, def")),
            RawBattleEvent(("", "-invertboost", "p2a: Incineroar")),
            RawBattleEvent(("", "-copyboost", "p1a: Ditto", "p2a: Dragapult")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.BOOST_SET,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_CLEAR,
        EventTypeId.BOOST_SWAP,
        EventTypeId.BOOST_INVERT,
        EventTypeId.BOOST_COPY,
    ]
    assert events[0].value == 1.0
    assert events[1].flags == 0
    assert events[2].flags == 1
    assert events[3].entity_id is None
    assert events[4].target_id == "p2a: Incineroar"
    assert events[6].target_id == "p2a: Dragapult"


def test_transform_endability_activate_notarget():
    events = _local_parse_events(
        [
            RawBattleEvent(("", "-transform", "p1a: Ditto", "p2a: Dragapult")),
            RawBattleEvent(("", "-endability", "p2a: Incineroar", "Intimidate")),
            RawBattleEvent(("", "-activate", "p1a: Dondozo", "move: Substitute", "[damage]")),
            RawBattleEvent(("", "-activate", "p2a: Incineroar", "ability: Intimidate")),
            RawBattleEvent(("", "-fieldactivate", "move: Perish Song")),
            RawBattleEvent(("", "-notarget", "p1a: Pikachu")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.TRANSFORM,
        EventTypeId.ABILITY_END,
        EventTypeId.ACTIVATE,
        EventTypeId.ACTIVATE,
        EventTypeId.FIELD_ACTIVATE,
        EventTypeId.NO_TARGET,
    ]
    assert events[0].target_id == "p2a: Dragapult"
    assert events[1].ability_id == tokenizer.id_for("abilities", "intimidate") > 0
    assert events[2].effect_id == tokenizer.id_for("volatiles", "substitute") > 0
    assert events[3].ability_id == tokenizer.id_for("abilities", "intimidate") > 0
    assert events[5].entity_id == "p1a: Pikachu"


def test_blocked_keeps_both_endpoints():
    events = _local_parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
            RawBattleEvent(("", "-activate", "p2a: Charizard", "move: Protect")),
            RawBattleEvent(("", "-immune", "p2a: Charizard")),
            RawBattleEvent(("", "-miss", "p1a: Pikachu", "p2a: Charizard")),
        ]
    )

    protect, immune, miss = events[1], events[2], events[3]
    assert protect.event_type == EventTypeId.BLOCKED
    assert (protect.entity_id, protect.target_id) == ("p1a: Pikachu", "p2a: Charizard")
    assert (immune.entity_id, immune.target_id, immune.flags) == (
        "p1a: Pikachu",
        "p2a: Charizard",
        1,
    )
    assert (miss.entity_id, miss.target_id, miss.flags) == (
        "p1a: Pikachu",
        "p2a: Charizard",
        2,
    )


def test_weather_upkeep_is_skipped():
    events = _local_parse_events(
        [
            RawBattleEvent(("", "-weather", "SunnyDay")),
            RawBattleEvent(("", "-weather", "SunnyDay", "[upkeep]")),
            RawBattleEvent(("", "-weather", "none")),
        ]
    )

    assert [event.event_type for event in events] == [
        EventTypeId.WEATHER_START,
        EventTypeId.WEATHER_END,
    ]
    assert events[0].effect_id == tokenizer.id_for("weathers", "sunnyday") > 0


def test_diagnostics_count_oov_and_missing_pre_hp():
    EVENT_DIAGNOSTICS.clear()

    events = _local_parse_events(
        [
            RawBattleEvent(("", "move", "p1a: Pikachu", "Not A Real Move", "p2a: Charizard")),
            RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100")),
        ]
    )

    assert EVENT_DIAGNOSTICS == {"oov_ids": 1, "missing_pre_hp": 1}
    assert events[1].value == 0.0  # missing pre-hp degrades to a zero delta
    EVENT_DIAGNOSTICS.clear()


def test_truncation_keeps_structural_events():
    events = [BattleEvent(EventTypeId.DAMAGE, "p1a: Pikachu", order=i) for i in range(64)]
    events.extend(
        [
            BattleEvent(EventTypeId.WEATHER_START, None, order=64),
            BattleEvent(EventTypeId.SIDE_START, "p2", order=65),
            BattleEvent(EventTypeId.DRAG, "p1a: Pikachu", order=66),
            BattleEvent(EventTypeId.MEGA, "p1a: Pikachu", order=67),
            BattleEvent(EventTypeId.EFFECT_START, "p2a: Gengar", order=68),
        ]
    )

    selected = truncate_events(events, limit=64)

    kept_types = {event.event_type for event in selected}
    assert {
        EventTypeId.WEATHER_START,
        EventTypeId.SIDE_START,
        EventTypeId.DRAG,
        EventTypeId.MEGA,
        EventTypeId.EFFECT_START,
    } <= kept_types
    orders = [event.order for event in selected]
    assert orders == sorted(orders)


def test_raw_event_pre_hp_snapshot():
    def pre_hp_for(identifier: str) -> float | None:
        assert identifier == "p2a: Charizard"
        return 0.75

    damage = build_raw_event(["", "-damage", "p2a: Charizard", "50/100"], pre_hp_for)
    move = build_raw_event(["", "move", "p1a: Pikachu", "Thunderbolt"], pre_hp_for)

    assert damage.pre_hp == 0.75
    assert damage.message == ("", "-damage", "p2a: Charizard", "50/100")
    assert move.pre_hp is None


def test_consume_events_clears_buffer_immediately():
    battle = DoubleBattle("events", "player", logging.getLogger(__name__), 9)
    set_raw_events(
        battle,
        [RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"))],
    )

    first = _local_parse_events(consume_raw_events(battle))
    second = _local_parse_events(consume_raw_events(battle))

    assert len(first) == 1
    assert second == []


_OBSERVATION_BUILDER = ObservationBuilder(default_runtime_resources())


def from_battle(battle, tok=tokenizer, stat_overrides=None):
    assert tok is _OBSERVATION_BUILDER.tokenizer
    return _OBSERVATION_BUILDER.build(battle_view(battle), stat_overrides)


def from_battle_into(battle, out, tok=tokenizer, stat_overrides=None):
    assert tok is _OBSERVATION_BUILDER.tokenizer
    _OBSERVATION_BUILDER.build_into(battle_view(battle), out, stat_overrides)


def _pokemon_numeric_into(pokemon, battle, *args, **kwargs):
    return _write_pokemon_numeric(pokemon, battle_view(battle), *args, **kwargs)


def make_real_pokemon(
    species: str = "charizard",
    ability: str = "blaze",
    item: str = "charizarditey",
    type_1: str | None = None,
    type_2: str | None = None,
    moves: dict[str, int] | None = None,
    effects: dict[Effect, int] | None = None,
    status: Status | None = None,
    current_hp: int = 100,
    max_hp: int = 100,
    boosts: dict[str, int] | None = None,
    protect_counter: int = 0,
    active_turns: int = 0,
    weightkg: float | None = None,
    status_counter: int = 0,
    preparing_move: str | None = None,
    last_move_id: str | None = None,
) -> Pokemon:
    """Helper to create a real Pokemon object and populate its slots."""
    p = Pokemon(gen=9, species=species)
    if ability:
        p._ability = ability
    if item:
        p._item = item
    if type_1:
        p._type_1 = PokemonType.from_name(type_1)
    if type_2:
        p._type_2 = PokemonType.from_name(type_2)
    if moves:
        for m_id, m_pp in moves.items():
            m = Move(m_id, 9)
            m._current_pp = m_pp
            p._moves._base_moves[m_id] = m
    if effects:
        p._effects = effects
    if status:
        p._status = status
    p._current_hp = current_hp
    p._max_hp = max_hp
    if boosts:
        p._boosts.update(boosts)
    p._protect_counter = protect_counter
    p._active_turns = active_turns
    if weightkg is not None:
        cast(Any, p)._weightkg = weightkg
    p._status_counter = status_counter
    if preparing_move:
        p._preparing_move = Move(preparing_move, 9)
    if last_move_id and last_move_id in p._moves._base_moves:
        p._moves._base_moves[last_move_id]._is_last_used = True
    return p


def make_real_battle(
    active_pokemon: list[Pokemon | None] | None = None,
    opponent_active_pokemon: list[Pokemon | None] | None = None,
    team: list[Pokemon] | None = None,
    opponent_team: list[Pokemon] | None = None,
    teampreview: bool = False,
    available_switches: list[list[Pokemon]] | None = None,
    weather: dict[Weather, int] | None = None,
    fields: dict[Field, int] | None = None,
    turn: int = 0,
    can_mega_evolve: list[bool] | None = None,
    side_conditions: dict[SideCondition, int] | None = None,
    opponent_side_conditions: dict[SideCondition, int] | None = None,
) -> DoubleBattle:
    """Helper to create a real DoubleBattle object and populate its slots/private dicts."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.ERROR)
    battle = DoubleBattle("tag", "user", logger, 9)
    battle._player_role = "p1"

    # Setup active pokemon
    active_dict = {}
    if active_pokemon:
        if len(active_pokemon) > 0 and active_pokemon[0] is not None:
            active_dict["p1a"] = active_pokemon[0]
            active_pokemon[0]._active = True
        if len(active_pokemon) > 1 and active_pokemon[1] is not None:
            active_dict["p1b"] = active_pokemon[1]
            active_pokemon[1]._active = True
    battle._active_pokemon = active_dict

    opponent_active_dict = {}
    if opponent_active_pokemon:
        if len(opponent_active_pokemon) > 0 and opponent_active_pokemon[0] is not None:
            opponent_active_dict["p2a"] = opponent_active_pokemon[0]
            opponent_active_pokemon[0]._active = True
        if len(opponent_active_pokemon) > 1 and opponent_active_pokemon[1] is not None:
            opponent_active_dict["p2b"] = opponent_active_pokemon[1]
            opponent_active_pokemon[1]._active = True
    battle._opponent_active_pokemon = opponent_active_dict

    # Setup teams
    if team:
        battle._team = {p.species: p for p in team}
    if opponent_team:
        battle._opponent_team = {p.species: p for p in opponent_team}

    battle._teampreview = teampreview
    battle._available_switches = available_switches or [[], []]
    battle._weather = weather or {}
    battle._fields = fields or {}
    battle._turn = turn
    battle._can_mega_evolve = can_mega_evolve or [False, False]
    battle._side_conditions = side_conditions or {}
    battle._opponent_side_conditions = opponent_side_conditions or {}
    return battle


# --- FIXTURES ---


@pytest.fixture(scope="module")
def battle_format():
    return FORMAT.battle_format


@pytest.fixture(scope="module")
def sample_team():
    team = """
Pikachu @ Light Ball
Ability: Static
Level: 50
Jolly Nature
- Fake Out
- Protect
- Thunderbolt
- Electroweb

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Modest Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Timid Nature
- Moonblast
- Tailwind
- Encore
- Protect

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Protect
- Low Kick

Glimmora @ Shuca Berry
Ability: Toxic Debris
Level: 50
Modest Nature
- Power Gem
- Sludge Bomb
- Earth Power
- Protect
"""
    return ValidatedTeam.from_showdown(team).packed


def test_pokemon_categorical_and_numeric_rows_real():
    # None Pokemon returns 24 zeros
    empty_cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    _pokemon_categorical_into(None, tokenizer, _iter_move_slots(None), empty_cat)
    assert not empty_cat.any()

    # Active Pokemon with custom moves and effects
    moves = {"closecombat": 10, "airslash": 15}
    effects = {Effect.CONFUSION: 1, Effect.DISABLE: 1}
    mon = make_real_pokemon(
        species="charizard",
        ability="blaze",
        item="charizarditey",
        type_1="Fire",
        type_2="Flying",
        moves=moves,
        effects=effects,
        status=Status.BRN,
    )
    mon._nature = "Jolly"

    cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    _pokemon_categorical_into(mon, tokenizer, _iter_move_slots(mon), cat)
    assert len(cat) == CATEGORICAL_WIDTH

    # Species, Ability, Item, Type 1, Type 2
    assert cat[0] == tokenizer.species_id(mon)
    assert cat[1] == tokenizer.ability_id(mon)
    assert cat[2] == tokenizer.item_id(mon)
    assert cat[3] == tokenizer.type_id(PokemonType.from_name("Fire"))
    assert cat[4] == tokenizer.type_id(PokemonType.from_name("Flying"))

    # 4 Moves (padded)
    assert cat[5] == tokenizer.move_id(Move("closecombat", 9))
    assert cat[6] == tokenizer.move_id(Move("airslash", 9))
    assert cat[7] == 0
    assert cat[8] == 0

    # 4 Move Types (padded)
    assert cat[9] == tokenizer.type_id(Move("closecombat", 9).type)
    assert cat[10] == tokenizer.type_id(Move("airslash", 9).type)
    assert cat[11] == 0
    assert cat[12] == 0

    # 4 Move Categories (padded)
    assert cat[13] == tokenizer.categories[MoveCategory.PHYSICAL]
    assert cat[14] == tokenizer.categories[MoveCategory.SPECIAL]
    assert cat[15] == 0
    assert cat[16] == 0

    # Status
    assert cat[17] == tokenizer.status_id(Status.BRN)

    assert not cat[18:24].any()

    # Nature
    assert cat[24] == tokenizer.nature_id(mon) > 0
    battle = make_real_battle()

    # None Pokemon returns mostly zeros except for condition flag (e.g. cond=1 -> row[2] = 1.0)
    none_row = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _pokemon_numeric_into(
        None, battle, cond=1, orig_idx=-1, move_slots=_iter_move_slots(None), row=none_row
    )
    assert len(none_row) == NUMERICAL_WIDTH
    assert none_row[2] == 1.0
    assert sum(none_row) == 1.0

    # closecombat max PP is 8; protect max PP is 16.
    moves = {"closecombat": 4, "protect": 8}
    effects = {Effect.CONFUSION: 2, Effect.DISABLE: 1}  # max duration for confusion=4, disable=4
    mon = make_real_pokemon(
        species="charizard",
        current_hp=80,
        max_hp=100,  # HP fraction = 0.8
        boosts={"atk": 3, "def": -1},
        moves=moves,
        protect_counter=2,
        active_turns=1,  # first_turn = True
        weightkg=75.0,  # Low kick category 0.6
        status_counter=3,
        effects=effects,
        preparing_move="closecombat",
    )

    # Test weight bounds low-kick categories
    weights_and_expected = [
        (5, 0.0),
        (15, 0.2),
        (35, 0.4),
        (75, 0.6),
        (150, 0.8),
        (250, 1.0),
    ]
    for w, val in weights_and_expected:
        mon._weightkg = w
        row = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
        _pokemon_numeric_into(
            mon, battle, cond=1, orig_idx=2, move_slots=_iter_move_slots(mon), row=row
        )
        assert abs(row[25] - val) < 1e-5

    mon._weightkg = 75
    row = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _pokemon_numeric_into(
        mon, battle, cond=1, orig_idx=2, move_slots=_iter_move_slots(mon), row=row
    )
    assert row[5] == pytest.approx(0.8)  # HP fraction
    assert abs(row[6] - 78.0 / 160.0) < 1e-5  # Charizard base HP is 78
    assert abs(row[7] - 84.0 / 160.0) < 1e-5  # Charizard base Atk is 84
    assert row[12] == 3.0 / 6.0  # Atk Boost
    assert row[13] == -1.0 / 6.0  # Def Boost
    assert row[19] == 4.0 / 8.0  # Move 1 PP ratio (4 / 8 max PP)
    assert row[20] == 8.0 / 16.0  # Move 2 PP ratio (8 / 16 max PP)
    assert row[21] == 0.0  # Move 3 (None)
    assert row[23] == 2.0 / 4.0  # Protect counter
    assert row[24] == 1.0  # First turn (since active_turns == 1)
    assert row[26] == (2 + 1) / 6.0  # Orig index ratio
    assert row[27] == 0.0  # Fainted (status is None)
    assert row[28] == 1.0  # cond == 1
    assert row[29] == 0.0  # cond == 2
    assert row[36] == 3.0 / 5.0  # Status counter
    assert row[42] == 1.0  # Preparing (preparing_move is not None)

    battle._can_mega_evolve = [True, False]
    row_mega_active = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _pokemon_numeric_into(
        mon,
        battle,
        cond=1,
        orig_idx=2,
        active_idx=0,
        move_slots=_iter_move_slots(mon),
        row=row_mega_active,
    )
    assert row_mega_active[30] == 1.0

    mon_mega = make_real_pokemon(species="charizardmegay")
    row_mega_form = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _pokemon_numeric_into(
        mon_mega,
        battle,
        cond=1,
        orig_idx=2,
        move_slots=_iter_move_slots(mon_mega),
        row=row_mega_form,
    )
    assert row_mega_form[31] == 1.0

    # Last move slot matching
    mon_last = make_real_pokemon(
        species="charizard",
        moves={"airslash": 10},
        last_move_id="airslash",
    )
    row_last_move = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _pokemon_numeric_into(
        mon_last,
        battle,
        cond=1,
        orig_idx=2,
        move_slots=_iter_move_slots(mon_last),
        row=row_last_move,
    )
    assert row_last_move[32] == 1.0  # First move slot matched last_move


def test_ordered_pokemon_and_slot_conditions_real():
    p1 = make_real_pokemon(species="aerodactyl")
    p2 = make_real_pokemon(species="archaludon")
    p3 = make_real_pokemon(species="azumarill")
    p4 = make_real_pokemon(species="basculegion")
    p5 = make_real_pokemon(species="camerupt", status=Status.FNT)
    p6 = make_real_pokemon(species="dragonite")

    team = [p1, p2, p3, p4, p5, p6]

    battle_tp = make_real_battle(team=team, teampreview=True)
    ordered_tp = _get_ordered_pokemon(battle_tp, is_opponent=False)
    assert len(ordered_tp) == 6
    assert ordered_tp[0][0] == p1
    assert ordered_tp[5][0] == p6
    # orig_idx mapping
    assert ordered_tp[0][1] == 0
    assert ordered_tp[5][1] == 5

    # Active: p1, p2
    # Bench switches: p3, p4
    battle_reg = make_real_battle(
        active_pokemon=[p1, p2],
        team=team,
        teampreview=False,
        available_switches=[[p3, p4], [p3, p4]],
    )
    ordered_reg = _get_ordered_pokemon(battle_reg, is_opponent=False)
    assert len(ordered_reg) == 6

    # Active slots come first; inactive members retain original roster order.
    assert ordered_reg[0][0] == p1  # active slot 0
    assert ordered_reg[0][2] == 0  # active_idx
    assert ordered_reg[1][0] == p2  # active slot 1
    assert ordered_reg[1][2] == 1  # active_idx

    bench_mons = [ordered_reg[i][0] for i in range(2, 5)]
    assert p3 in bench_mons
    assert p4 in bench_mons
    assert p5 in bench_mons

    assert ordered_reg[5][0] == p6  # dropped

    # Request-backed selection is persistent even when trapping makes all
    # available-switch lists temporarily empty.
    for mon in (p1, p2, p3, p4):
        mon._selected_in_teampreview = True
    trapped_battle = make_real_battle(
        active_pokemon=[p1, p2],
        team=team,
        teampreview=False,
        available_switches=[[], []],
    )
    trapped_battle._trapped = [True, True]
    ordered_trapped = _get_ordered_pokemon(trapped_battle, is_opponent=False)
    assert [entry[0] for entry in ordered_trapped[:4]] == [p1, p2, p3, p4]
    assert (
        _slot_condition(
            trapped_battle,
            p3,
            2,
            is_opponent=False,
            selected_allies={p1, p2, p3, p4},
        )
        == 2
    )

    # Request metadata can be partial. A currently available switch must still
    # be treated as selected so an empty active slot cannot trim its token.
    for mon in team:
        mon._selected_in_teampreview = False
        mon._last_request = None
    p2._selected_in_teampreview = True
    partial_request_battle = make_real_battle(
        active_pokemon=[None, p2],
        team=team,
        teampreview=False,
        available_switches=[[p3, p6], [p3, p6]],
    )
    ordered_partial = _get_ordered_pokemon(partial_request_battle, is_opponent=False)
    partial_mons = [entry[0] for entry in ordered_partial]
    assert p6 in partial_mons

    # Empty left active slot: right active must stay at index 1 with a None
    # placeholder at index 0, so seq positions match env action positions.
    battle_left_empty = make_real_battle(
        active_pokemon=[None, p2],
        team=team,
        teampreview=False,
        available_switches=[[p3, p4], [p3, p4]],
    )
    ordered_le = _get_ordered_pokemon(battle_left_empty, is_opponent=False)
    assert len(ordered_le) == 6
    assert ordered_le[0] == (None, -1, None)
    assert ordered_le[1][0] == p2
    assert ordered_le[1][2] == 1  # active_idx preserved
    # The placeholder overflows the row budget; eviction is selection-independent
    # and removes an inactive fainted token first.
    le_mons = [entry[0] for entry in ordered_le]
    assert p5 not in le_mons
    assert p6 in le_mons

    # Same invariant for the opponent side
    battle_opp_left_empty = make_real_battle(
        opponent_active_pokemon=[None, p2],
        opponent_team=team,
        teampreview=False,
    )
    ordered_opp_le = _get_ordered_pokemon(battle_opp_left_empty, is_opponent=True)
    assert ordered_opp_le[0] == (None, -1, None)
    assert ordered_opp_le[1][0] == p2
    p1 = make_real_pokemon(species="aerodactyl")
    p_fainted = make_real_pokemon(species="camerupt", status=Status.FNT)

    battle = make_real_battle()
    assert _slot_condition(battle, None, 0, is_opponent=False) == 0

    battle_tp = make_real_battle(teampreview=True)
    assert _slot_condition(battle_tp, p1, 0, is_opponent=False) == 2

    battle_reg = make_real_battle(teampreview=False)
    assert _slot_condition(battle_reg, p1, 1, is_opponent=False) == 1

    assert _slot_condition(battle_reg, p_fainted, 2, is_opponent=False) == 3
    # fainted takes precedence over the active-slot index
    assert _slot_condition(battle_reg, p_fainted, 0, is_opponent=False) == 3

    assert _slot_condition(battle_reg, p1, 2, is_opponent=True) == 2

    battle_sw = make_real_battle(available_switches=[[p1]])
    assert _slot_condition(battle_sw, p1, 2, is_opponent=False) == 2
    p2 = make_real_pokemon(species="dragonite")
    assert _slot_condition(battle_sw, p2, 3, is_opponent=False) == -1


def test_global_and_side_field_tokens_include_mega_availability():
    battle = make_real_battle(turn=3)

    # Rain duration: Rain started at turn 1. Duration = 5. Left: max(0, 5 - (3 - 1)) / 5 = 3 / 5 = 0.6
    battle._weather = {Weather.RAINDANCE: 1}
    battle._fields = {
        Field.TRICK_ROOM: 2
    }  # started at turn 2. Duration = 5. Left: (5 - (3 - 2)) / 5 = 0.8
    battle._teampreview = False

    cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    num = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _global_field_token_into(battle, tokenizer, cat, num)
    assert len(cat) == CATEGORICAL_WIDTH
    assert len(num) == NUMERICAL_WIDTH
    assert not cat[:2].any()
    assert num[2] == 0.0  # teampreview
    assert num[3] == 3.0 / 24.0  # turn scaling
    battle = make_real_battle(turn=4)
    conditions = {
        SideCondition.TAILWIND: 2,  # duration=4. Left: max(0, 4 - (4 - 2)) / 4 = 2 / 4 = 0.5
        SideCondition.AURORA_VEIL: 1,  # duration=5. Left: max(0, 5 - (4 - 1)) / 5 = 2 / 5 = 0.4
        SideCondition.TOXIC_SPIKES: 2,  # layers = 2. Value: 2 / 2 = 1.0
    }

    cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    num = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _side_token_into(
        battle,
        conditions,
        tokenizer,
        fainted_count=3,
        mega_available=True,
        cat=cat,
        num=num,
    )
    assert len(cat) == CATEGORICAL_WIDTH
    assert len(num) == NUMERICAL_WIDTH
    assert not cat[:3].any()
    assert abs(num[3] - 0.5) < 1e-5  # 3 fainted out of 6
    assert num[4] == 1.0  # mega still available

    cat_used = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    num_used = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    _side_token_into(
        battle,
        conditions,
        tokenizer,
        fainted_count=3,
        mega_available=False,
        cat=cat_used,
        num=num_used,
    )
    assert num_used[4] == 0.0
    mega = make_real_pokemon(species="charizard", item="charizarditey")
    regular = make_real_pokemon(species="dragonite", item="choicescarf")
    unused_mega = make_real_pokemon(species="aerodactyl", item="aerodactylite")
    team = [mega, regular, unused_mega]
    battle = make_real_battle(active_pokemon=[mega, regular], team=team)

    mega._selected_in_teampreview = True
    regular._selected_in_teampreview = True
    assert _side_mega_available(
        battle,
        is_opponent=False,
        selected_allies={mega, regular},
    )

    assert not _side_mega_available(
        battle,
        is_opponent=False,
        selected_allies={regular},
    )

    battle._used_mega_evolve = True
    assert not _side_mega_available(
        battle,
        is_opponent=False,
        selected_allies={mega, regular},
    )


def test_effect_overflow_is_counted_and_enforced():
    cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    num = np.zeros(NUMERICAL_WIDTH, dtype=np.float32)
    entries = [
        (
            EffectNamespace.POKEMON,
            effect_id,
            CounterKind.ACTION_COUNT,
            float(effect_id),
            0.0,
            False,
            0.0,
        )
        for effect_id in range(MAX_EFFECTS + 3, 0, -1)
    ]

    _write_effects(entries, cat, num)

    assert num[NUM_IDX_EFFECT_COUNT] == MAX_EFFECTS + 3
    assert num[NUM_IDX_EFFECT_OVERFLOW] == 3
    assert cat[CAT_EFFECT_START] == 1

    # unmarked truncation (count over capacity with no overflow flag) is rejected
    obs = StructuredObservation.empty_batch(1)
    obs.numerical[0, 1, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS + 1
    with pytest.raises(ValueError, match="overflow"):
        obs.validate_overflow_contract()


def test_events_ground_to_slots_and_are_idempotent():
    switched_out = make_real_pokemon(species="charizard")
    switched_in = make_real_pokemon(species="venusaur")
    opponent = make_real_pokemon(species="tyranitar")
    battle = make_real_battle(
        active_pokemon=[switched_in, None],
        opponent_active_pokemon=[opponent, None],
        team=[switched_out, switched_in],
        opponent_team=[opponent],
    )
    battle._team = {
        "p1: Charizard": switched_out,
        "p1: Venusaur": switched_in,
    }
    battle._opponent_team = {"p2: Tyranitar": opponent}
    set_raw_events(
        battle,
        [
            RawBattleEvent(("", "switch", "p1a: Venusaur", "Venusaur, L50", "100/100")),
            RawBattleEvent(("", "move", "p2a: Tyranitar", "Rock Slide", "p1a: Venusaur")),
        ],
    )

    obs = from_battle(battle, tokenizer)

    assert obs.events_cat[:2, 0].tolist() == [
        EventTypeId.SWITCH_IN,
        EventTypeId.MOVE,
    ]
    assert obs.events_cat[:2, 4].tolist() == [1, 2]
    assert obs.events_side_ids[:2].tolist() == [SideId.ALLY, SideId.OPPONENT]
    assert obs.events_slot_ids[:2].tolist() == [1, 1]

    # rebuilding the same decision yields the identical event window
    rebuilt_obs = from_battle(battle, tokenizer)
    assert torch.equal(rebuilt_obs.events_cat, obs.events_cat)
    assert torch.equal(rebuilt_obs.events_num, obs.events_num)

    # a new decision (new request) drains a fresh, now-empty window
    battle._last_request = {"turn": "next"}
    next_obs = from_battle(battle, tokenizer)
    assert torch.count_nonzero(next_obs.events_cat) == 0
    assert torch.count_nonzero(next_obs.events_num) == 0


def test_side_events_ground_to_owning_side():
    ally = make_real_pokemon(species="charizard")
    opponent = make_real_pokemon(species="venusaur")
    battle = make_real_battle(
        active_pokemon=[ally, None],
        opponent_active_pokemon=[opponent, None],
        team=[ally],
        opponent_team=[opponent],
    )
    # Showdown side identifiers carry the username: "p1: Username", not "p1".
    set_raw_events(
        battle,
        [
            RawBattleEvent(("", "-sidestart", "p1: SomeUser", "move: Tailwind")),
            RawBattleEvent(("", "-sidestart", "p2: OtherUser", "move: Light Screen")),
        ],
    )

    obs = from_battle(battle, tokenizer)

    assert obs.events_cat[:2, 0].tolist() == [EventTypeId.SIDE_START, EventTypeId.SIDE_START]
    assert obs.events_side_ids[:2].tolist() == [SideId.ALLY, SideId.OPPONENT]
    assert obs.events_slot_ids[:2].tolist() == [0, 0]
    assert obs.events_cat[0, 5].item() > 0  # tailwind resolved in side_conditions


def test_event_order_recompacts():
    ally = make_real_pokemon(species="charizard")
    opponent = make_real_pokemon(species="venusaur")
    battle = make_real_battle(
        active_pokemon=[ally, None],
        opponent_active_pokemon=[opponent, None],
        team=[ally],
        opponent_team=[opponent],
    )
    battle._team = {"p1: Charizard": ally}
    overflow = 6
    # low-priority flood followed by high-priority moves: survivors keep gapped
    # original orders, which must re-compact into a dense positional range
    raw_events = [
        RawBattleEvent(("", "-boost", "p1a: Charizard", "atk", "1"))
        for _ in range(EVENT_COUNT + overflow - 10)
    ]
    raw_events.extend(
        RawBattleEvent(("", "move", "p1a: Charizard", "Tackle", "p2a: Venusaur")) for _ in range(10)
    )
    set_raw_events(battle, raw_events)

    obs = from_battle(battle, tokenizer)

    # positional ids stay dense in [1, EVENT_COUNT]; the order scalar stays in [0, 1)
    assert obs.events_cat[:, 4].tolist() == list(range(1, EVENT_COUNT + 1))
    assert obs.events_num[:, 1].max().item() < 1.0
    assert obs.events_metadata[1].item() == float(overflow)


def test_from_battle_into_overwrites_and_validates_output_buffer():
    ally = make_real_pokemon(
        species="charizard",
        moves={"airslash": 10, "protect": 8},
        effects={Effect.CONFUSION: 2},
        current_hp=73,
        max_hp=100,
    )
    opponent = make_real_pokemon(
        species="venusaur",
        moves={"gigadrain": 6},
        status=Status.BRN,
    )
    battle = make_real_battle(
        active_pokemon=[ally, None],
        opponent_active_pokemon=[opponent, None],
        team=[ally],
        opponent_team=[opponent],
        weather={Weather.SUNNYDAY: 1},
        fields={Field.TRICK_ROOM: 2},
        turn=3,
    )

    expected = from_battle(battle, tokenizer)
    out = StructuredObservation.empty_batch(1)[0]
    out.token_type_ids.fill_(99)
    out.side_ids.fill_(99)
    out.slot_ids.fill_(99)
    out.categorical.fill_(99)
    out.numerical.fill_(99.0)

    assert out.events_cat is not None
    out.events_cat.fill_(99)

    assert out.events_num is not None
    out.events_num.fill_(99.0)

    assert out.events_side_ids is not None
    out.events_side_ids.fill_(99)

    assert out.events_slot_ids is not None
    out.events_slot_ids.fill_(99)

    from_battle_into(battle, out, tokenizer)

    assert torch.equal(out.token_type_ids, expected.token_type_ids)
    assert torch.equal(out.side_ids, expected.side_ids)
    assert torch.equal(out.slot_ids, expected.slot_ids)
    assert torch.equal(out.categorical, expected.categorical)
    assert torch.equal(out.numerical, expected.numerical)
    assert expected.events_cat is not None and torch.equal(out.events_cat, expected.events_cat)
    assert expected.events_num is not None and torch.equal(out.events_num, expected.events_num)
    assert expected.events_side_ids is not None and torch.equal(
        out.events_side_ids, expected.events_side_ids
    )
    assert expected.events_slot_ids is not None and torch.equal(
        out.events_slot_ids, expected.events_slot_ids
    )
    assert not torch.any(out.categorical == 99)
    assert not torch.any(out.numerical == 99)
    battle = make_real_battle()
    invalid = StructuredObservation.empty_batch(1)[0]
    invalid.numerical = invalid.numerical.to(torch.float64)

    with pytest.raises(ValueError, match="Invalid numerical"):
        from_battle_into(battle, invalid)


def test_stat_resolution_provenance_and_cache_behavior():
    pokemon = make_real_pokemon(species="charizard")
    pokemon._nature = None
    values, provenance = _get_pokemon_level_stats(pokemon, True, None)
    assert values == (0.0,) * 6
    assert provenance == Provenance.UNKNOWN

    import typing

    expected = typing.cast(tuple[int, int, int, int, int, int], tuple((155, 93, 98, 177, 105, 152)))
    values, provenance = _get_pokemon_level_stats(pokemon, True, expected)
    assert values == tuple(float(value) for value in expected)
    assert provenance == Provenance.IMPUTED
    pokemon = make_real_pokemon(
        species="charizard",
        moves={"heatwave": 10, "solarbeam": 10, "protect": 10, "weatherball": 10},
    )
    pokemon._nature = "modest"
    cache = {}
    first = _cached_imputed_stats(pokemon, cache)
    second = _cached_imputed_stats(pokemon, cache)
    assert first is second
    assert len(cache) == 1


def test_sim_env_embed_and_mask_share_one_decision_view(monkeypatch):
    battle = make_real_battle()
    from p0.runtime import poke_env_battle_adapter

    original_decision_view = poke_env_battle_adapter.decision_view
    decision_builds = 0

    def counted_decision_view(current_battle):
        nonlocal decision_builds
        decision_builds += 1
        return original_decision_view(current_battle)

    monkeypatch.setattr(poke_env_battle_adapter, "decision_view", counted_decision_view)
    env = SimEnv.__new__(SimEnv)
    cast(Any, env).agent1 = SimpleNamespace(username=battle.player_username)
    cast(Any, env).agent2 = SimpleNamespace(username="other-player")
    env._observation_builder = _OBSERVATION_BUILDER
    env._battle_view_factory = battle_view
    out1 = StructuredObservation.empty_batch(1)[0]
    out2 = StructuredObservation.empty_batch(1)[0]
    env.set_observation_targets(out1, out2)

    result = env.embed_battle(battle)
    mask = env.get_action_mask(battle)

    assert result is out1
    assert result.token_type_ids[0] == TokenType.POKEMON
    assert len(mask) == FORMAT.action_size * 2
    assert decision_builds == 1


def test_sim_env_training_state_restores_teams_and_preserves_game_boundary(monkeypatch):
    class TeamBuilder:
        def __init__(self, packed: str):
            self.packed = packed

        def yield_team(self) -> str:
            return self.packed

    class Player:
        def __init__(self, packed: str):
            self._team = TeamBuilder(packed)

        def update_team(self, packed: str) -> None:
            self._team = TeamBuilder(packed)

    env = SimEnv.__new__(SimEnv)
    env._agent_rng = random.Random(10)
    env._opponent_rng = random.Random(11)
    env._series_scores = [1, 0]
    env._series_games_played = 2
    env._decision_steps = 17
    env._resume_reset_pending = False
    env.series_id = "series-1"
    cast(Any, env).agent1 = Player("agent-team")
    cast(Any, env).agent2 = Player("opponent-team")

    state = env.training_state()
    cast(Any, env).agent1.update_team("wrong-agent-team")
    cast(Any, env).agent2.update_team("wrong-opponent-team")
    env._series_scores = [0, 0]
    env._series_games_played = 0

    env.restore_training_state(state)
    monkeypatch.setattr(MegaEnv, "reset", lambda self, seed=None, options=None: "reset")

    assert env.reset() == "reset"
    assert cast(Any, env).agent1._team.yield_team() == "agent-team"
    assert cast(Any, env).agent2._team.yield_team() == "opponent-team"
    assert env.series_scores == [1, 0]
    assert env.series_games_played == 2

    env.reset()
    assert env.series_games_played == 3


def test_concurrent_universal_effect_stress_state():
    mon = make_real_pokemon(
        effects={
            Effect.TAUNT: 1,
            Effect.LEECH_SEED: 1,
            Effect.SUBSTITUTE: 1,
            Effect.SALT_CURE: 1,
            Effect.ENCORE: 1,
            Effect.DISABLE: 1,
            Effect.YAWN: 2,
            Effect.PERISH3: 3,
            Effect.TRAPPED: 1,
        }
    )
    battle = make_real_battle(active_pokemon=[mon, None], team=[mon], turn=3)
    battle._side_conditions = {
        SideCondition.REFLECT: 1,
        SideCondition.LIGHT_SCREEN: 1,
        SideCondition.AURORA_VEIL: 1,
        SideCondition.TAILWIND: 1,
        SideCondition.SAFEGUARD: 1,
        SideCondition.SPIKES: 3,
        SideCondition.TOXIC_SPIKES: 2,
    }
    battle._weather = {Weather.RAINDANCE: 1}
    battle._fields = {
        Field.GRASSY_TERRAIN: 1,
        Field.TRICK_ROOM: 1,
        Field.WONDER_ROOM: 1,
        Field.MAGIC_ROOM: 1,
        Field.GRAVITY: 1,
    }

    obs = from_battle(battle, tokenizer)

    assert obs.numerical[0, NUM_IDX_EFFECT_COUNT] == 9
    assert obs.numerical[13, NUM_IDX_EFFECT_COUNT] == 7
    assert obs.numerical[12, NUM_IDX_EFFECT_COUNT] == 6
    assert obs.numerical[:, NUM_IDX_EFFECT_OVERFLOW].sum() == 0
    pokemon_effects = obs.categorical[0, CAT_EFFECT_START::EFFECT_CATEGORICAL_WIDTH]
    assert torch.count_nonzero(pokemon_effects) == 9
    namespaces = obs.categorical[12, CAT_EFFECT_START + 2 :: EFFECT_CATEGORICAL_WIDTH]
    assert EffectNamespace.FIELD in namespaces
    assert EffectNamespace.WEATHER in namespaces


CHARIZARD = BaseStats(78, 84, 78, 109, 85, 100)


def test_first_point_and_nature_truncation_match_showdown():
    base = BaseStats(100, 100, 100, 100, 100, 100)
    zero = calculate_stats(base, StatPoints(), "adamant")
    one = calculate_stats(base, StatPoints(atk=1), "adamant")
    assert zero == (175, 132, 120, 108, 120, 120)
    assert one[1] == zero[1] + 1


@pytest.mark.parametrize(
    "points, error",
    [
        (dict(hp=33), r"\[0, 32\]"),
        (dict(hp=32, atk=32, defense=3), "at most 66"),
    ],
)
def test_stat_point_validation(points, error):
    with pytest.raises(ValueError, match=error):
        StatPoints(**points)


def _input(*, moves=("Heat Wave", "Protect"), categories=("special", "status"), nature="modest"):
    return typing.cast(
        dict[str, typing.Any],
        dict(
            nature=nature,
            item="charizarditey",
            ability="blaze",
            moves=moves,
            move_categories=categories,
            base_stats=CHARIZARD,
        ),
    )


def test_imputer_is_legal_deterministic_and_role_sensitive():
    value = _input()
    assert classify_role(value["nature"], value["moves"], value["move_categories"]) == Role.SPECIAL
    assert impute_candidates(**value) == impute_candidates(**value)
    assert impute_candidates(**value)[0].points == StatPoints(hp=2, spa=32, spe=32)
    assert all(sum(candidate.points.as_tuple()) <= 66 for candidate in impute_candidates(**value))
    assert select_candidate(seed=7, **value) == select_candidate(seed=7, **value)


def test_imputed_level_stats_are_cached_by_static_team_facts():
    value = _input()
    imputed_stats.cache_clear()
    first = imputed_stats(**value)
    second = imputed_stats(**value)
    assert first == second
    assert imputed_stats.cache_info().hits == 1


def test_trick_room_and_support_shapes_do_not_assume_fast_offense():
    trick_room = _input(
        moves=("Trick Room", "Heat Wave"),
        categories=("status", "special"),
        nature="quiet",
    )
    support = _input(moves=("Protect", "Follow Me"), categories=("status", "status"), nature="calm")
    assert (
        classify_role(trick_room["nature"], trick_room["moves"], trick_room["move_categories"])
        == Role.TRICK_ROOM
    )
    assert impute_candidates(**trick_room)[0].points.spe == 0
    assert (
        classify_role(support["nature"], support["moves"], support["move_categories"])
        == Role.SUPPORT
    )
    assert impute_candidates(**support)[0].points == StatPoints(hp=32, defense=17, spd=17)


def _struggle_battle(move_id: str, can_mega: bool) -> DoubleBattle:
    mock_active = SimpleNamespace(moves={"tackle": SimpleNamespace(id="tackle")}, fainted=False)
    return cast(
        DoubleBattle,
        SimpleNamespace(
            player_username="player",
            battle_tag="battle",
            teampreview=False,
            _wait=False,
            force_switch=[False, False],
            trapped=[False, False],
            maybe_trapped=[False, False],
            active_pokemon=[mock_active, None],
            available_moves=[[SimpleNamespace(id=move_id)], []],
            available_switches=[[], []],
            team={},
            can_mega_evolve=[can_mega, False],
            valid_orders=[[], []],
            get_possible_showdown_targets=lambda move, mon: [0],
        ),
    )


def test_struggle_env_roundtrip():
    battle = _struggle_battle("struggle", can_mega=False)
    assert list(legal_actions(decision_view(battle), 0)) == [48]

    order = action_to_single_order(48, battle, fake=True, position=0)
    assert cast(Any, order.order).id == "struggle"
    assert not order.mega
    assert single_order_to_action(order, battle, fake=True, position=0) == 48

    mega_battle = _struggle_battle("recharge", can_mega=True)
    mask = list(legal_actions(decision_view(mega_battle), 0))
    assert 48 in mask
    assert 47 in mask

    order = action_to_single_order(47, mega_battle, fake=True, position=0)
    assert cast(Any, order.order).id == "recharge"
    assert order.mega
    assert single_order_to_action(order, mega_battle, fake=True, position=0) == 47


def test_struggle_policy_logits():
    B = 2
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())

    obs = StructuredObservation.empty_batch(B)
    obs.numerical[:, :, -1] = 0.5  # fake ratios so orig_ids aren't all 0

    action_mask = torch.zeros((B, 2, ACT_SIZE), dtype=torch.bool)
    action_mask[:, 0, 48] = True
    action_mask[:, 1, 47] = True

    enc = policy.encode(obs, action_mask)
    memory = policy.empty_memory(B)

    reduced = policy.actor.reducer(enc.tokens, *memory)
    z = reduced.cls
    k_entity_extended = policy.actor._compute_keys(reduced.pokemon)
    logits, keys = policy.actor._compute_pointer_logits(
        z, k_entity_extended, enc.aux[:, 0], enc.numerical, head_idx=0
    )

    torch.testing.assert_close(keys[:, 48], policy.actor.struggle_key.unsqueeze(0).expand(B, -1))

    torch.testing.assert_close(
        keys[:, 47], (policy.actor.struggle_key + policy.actor.mega_emb).unsqueeze(0).expand(B, -1)
    )

    actions = torch.tensor([[48, 47], [48, 47]])
    out = policy.evaluate(enc, action_mask, actions, *memory)

    assert torch.isfinite(out.logits[:, 0, 48]).all()

    logits2 = policy.actor._apply_sequential_masks(
        out.logits, torch.tensor([47, 47]), action_mask, torch.tensor([False, False])
    )
    assert (logits2[:, 1, 47] == float("-inf")).all()


def test_action_ids_cover_all_boundary_categories() -> None:
    expected = {
        0: SlotAction(ActionKind.PASS),
        1: SlotAction(ActionKind.SWITCH, switch_slot=0),
        6: SlotAction(ActionKind.SWITCH, switch_slot=5),
        7: SlotAction(ActionKind.MOVE, move_slot=0, target=-2),
        11: SlotAction(ActionKind.MOVE, move_slot=0, target=2),
        26: SlotAction(ActionKind.MOVE, move_slot=3, target=2),
        27: SlotAction(ActionKind.MOVE, move_slot=0, target=-2, mega=True),
        46: SlotAction(ActionKind.MOVE, move_slot=3, target=2, mega=True),
        47: SlotAction(ActionKind.FORCED_MOVE, mega=True),
        48: SlotAction(ActionKind.FORCED_MOVE),
    }
    for action_id, semantic in expected.items():
        assert decode_action(action_id) == semantic
        assert encode_action(semantic) == action_id


def test_team_preview_pairs_and_joint_constraints_preserve_uniqueness() -> None:
    pairs = tuple((first, second) for first in range(6) for second in range(6) if first != second)
    assert tuple(decode_team_pair(encode_team_pair(*pair)) for pair in pairs) == pairs
    selection = (5, 4, 3, 2, 0, 1)
    lead, back = canonical_team_actions(selection)
    assert team_selection(lead, back) == selection

    preview = DecisionView(slots=(SlotDecision(), SlotDecision()), team_preview=True)
    assert legal_actions(preview, 0) == tuple(encode_team_pair(*pair) for pair in pairs)
    assert validate_joint_action(preview, 1, 15)
    assert not validate_joint_action(preview, 1, 14)

    regular = DecisionView(
        slots=(
            SlotDecision(switch_slots=(0, 1), move_targets=((-2, 2),), can_mega=True),
            SlotDecision(switch_slots=(0, 1), move_targets=((-2, 2),), can_mega=True),
        )
    )
    assert validate_joint_action(regular, 7, 11)
    assert not validate_joint_action(regular, 1, 1)
    assert not validate_joint_action(regular, 27, 31)

    forced = DecisionView(
        slots=(
            SlotDecision(forced_move=True, can_mega=True),
            SlotDecision(forced_move=True, can_mega=True),
        )
    )
    assert legal_actions(forced, 0) == (48, 47)


def test_candidate_cap_degrades_to_explicit_unknown_evidence() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(move_targets=((-2, -1),)),
            SlotDecision(move_targets=((-2,),)),
        )
    )
    evidence = extract_action_evidence(
        EvidenceRequest(
            view=view,
            slots=(
                ObservedAction(alternatives=(7, 8), exact=False),
                ObservedAction(action=7),
            ),
            max_candidates=1,
        )
    )
    assert evidence.label_kind is LabelKind.UNKNOWN
    assert evidence.candidates == ()
    assert "candidate_cap_or_illegal" in evidence.tags


def test_protocol_event_stream_matches_showdown_golden_types() -> None:
    from tests.stress.replay_fixtures import GOLDEN_EVENT_TYPES, golden_raw_events

    EVENT_DIAGNOSTICS.clear()
    events = parse_protocol_events(list(golden_raw_events()), tokenizer)
    assert tuple(event.event_type for event in events) == GOLDEN_EVENT_TYPES
    assert events[0].entity_id == "p1a: Pikachu"
    damage = next(event for event in events if event.event_type is EventTypeId.DAMAGE)
    heal = next(event for event in events if event.event_type is EventTypeId.HEAL)
    unboost = next(event for event in events if event.event_type is EventTypeId.UNBOOST)
    assert damage.value == pytest.approx(-0.25)
    assert heal.value == pytest.approx(0.15)
    assert unboost.value == pytest.approx(-1 / 6)
    assert any(event.event_type is EventTypeId.MEGA for event in events)
    assert EVENT_DIAGNOSTICS["oov_ids"] >= 1


def test_event_truncation_keeps_priority_events_and_protocol_order() -> None:
    from tests.stress.replay_fixtures import golden_raw_events

    EVENT_DIAGNOSTICS.clear()
    events = parse_protocol_events(list(golden_raw_events()), tokenizer)
    truncated = truncate_events(events, limit=12)
    assert len(truncated) == 12
    assert [event.order for event in truncated] == sorted(event.order for event in truncated)
    assert any(event.event_type is EventTypeId.MOVE for event in truncated)
    assert any(event.event_type is EventTypeId.SWITCH_IN for event in truncated)
    assert any(event.event_type is EventTypeId.FAINT for event in truncated)


def test_malformed_and_incomplete_protocol_lines_are_diagnosed_without_fabrication() -> None:
    from tests.stress.replay_fixtures import GOLDEN_EVENT_TYPES, golden_raw_events

    EVENT_DIAGNOSTICS.clear()
    raw_events = list(golden_raw_events())
    raw_events.extend(
        (
            RawBattleEvent(("",)),
            RawBattleEvent(("", "chat", "ignored")),
            RawBattleEvent(("", "switch", "p1a: Pikachu")),
            RawBattleEvent(("", "-damage", "p2a: Charizard", "50/100")),
            RawBattleEvent(("", "-status", "p2a: Charizard")),
            RawBattleEvent(("", "move", "p1a: Pikachu")),
        )
    )
    events = parse_protocol_events(raw_events, tokenizer)
    assert len(events) == len(GOLDEN_EVENT_TYPES) + 1
    assert EVENT_DIAGNOSTICS["oov_ids"] >= 1
    assert EVENT_DIAGNOSTICS["missing_pre_hp"] == 1
    damage = events[-1]
    assert damage.event_type is EventTypeId.DAMAGE
    assert damage.value == 0.0


def test_event_truncation_handles_below_equal_and_above_capacity_limits() -> None:
    from tests.stress.replay_fixtures import golden_raw_events

    events = parse_protocol_events(list(golden_raw_events()) * 2, tokenizer)
    assert len(events) > EVENT_COUNT
    for limit in (EVENT_COUNT - 1, EVENT_COUNT, EVENT_COUNT + 1):
        truncated = truncate_events(events, limit=limit)
        assert len(truncated) == limit
        assert [event.order for event in truncated] == sorted(event.order for event in truncated)


def test_observation_overflow_contract_holds_at_capacity_boundaries() -> None:
    observation = StructuredObservation.empty_batch(3)
    observation.numerical[0, 0, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS - 1
    observation.numerical[1, 0, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS
    observation.numerical[2, 0, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS + 2
    observation.numerical[2, 0, NUM_IDX_EFFECT_OVERFLOW] = 2
    observation.events_cat[0, : EVENT_COUNT - 1, 0] = 1
    observation.events_cat[1, :, 0] = 1
    observation.events_cat[2, :, 0] = 1
    observation.events_metadata[:] = torch.tensor(
        ((EVENT_COUNT - 1, 0), (EVENT_COUNT, 0), (EVENT_COUNT + 3, 3)),
        dtype=torch.float32,
    )
    observation.validate_overflow_contract()
    assert observation.overflow_totals() == (2, 3)


def test_reconstructed_observations_clear_reused_buffer_state() -> None:
    from tests.stress.replay_fixtures import golden_replay_payload

    document = parse_replay_payload(golden_replay_payload("buffer-reuse"))
    perspective = reconstruct_both(document)[0]
    assert len(perspective.snapshots) >= 2
    builder = ObservationBuilder(default_runtime_resources())
    output = StructuredObservation.empty_batch(1)[0]
    for snapshot in perspective.snapshots:
        snapshot.view.events = list(snapshot.events)
        output.token_type_ids.fill_(99)
        output.categorical.fill_(99)
        output.numerical.fill_(99.0)
        output.events_cat.fill_(99)
        output.events_num.fill_(99.0)
        output.events_side_ids.fill_(99)
        output.events_slot_ids.fill_(99)
        output.events_metadata.fill_(99.0)
        builder.build_into(snapshot.view, output)
        output.validate(batch_rank=0)
        output.validate_overflow_contract()
        assert all(torch.isfinite(tensor).all() for tensor in output.tensors())
        assert output.events_metadata[0].item() == len(snapshot.events)
        assert not torch.any(output.events_cat[len(snapshot.events) :, :])
        assert not torch.any(output.events_num[len(snapshot.events) :, :])
        assert output.token_type_ids[0].item() == int(TokenType.POKEMON)
        assert output.side_ids[0].item() == int(SideId.ALLY)
        assert tuple(output.slot_ids[:6].tolist()) == (1, 2, 3, 4, 5, 6)
        assert tuple(output.slot_ids[6:12].tolist()) == (1, 2, 3, 4, 5, 6)
        assert output.numerical[12, NUM_IDX_TEAM_PREVIEW].item() == float(snapshot.view.teampreview)
        assert output.numerical[12, 3].item() == pytest.approx(snapshot.view.turn / 24.0)


class _AdapterBattleState:
    pass


def _adapter_battle(*, teampreview: bool = False, forced: bool = False) -> DoubleBattle:
    active = SimpleNamespace(
        moves={"tackle": SimpleNamespace(id="tackle")},
        fainted=False,
        base_species="Pikachu",
    )
    team = {
        f"p1: Species-{index}": SimpleNamespace(base_species=f"Species-{index}")
        for index in range(6)
    }
    battle = _AdapterBattleState()
    for name, value in {
        "player_username": "player",
        "battle_tag": "stress-battle",
        "teampreview": teampreview,
        "team": team,
        "active_pokemon": [active, None],
        "opponent_active_pokemon": [None, None],
        "available_moves": [[SimpleNamespace(id="struggle" if forced else "tackle")], []],
        "available_switches": [[], []],
        "valid_orders": [[], []],
        "can_mega_evolve": [False, False],
        "force_switch": [False, False],
        "trapped": [False, False],
        "maybe_trapped": [False, False],
        "_wait": False,
        "player_role": "p1",
        "opponent_team": {},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "turn": 1,
        "used_mega_evolve": False,
        "opponent_used_mega_evolve": False,
        "get_possible_showdown_targets": lambda move, pokemon: [0],
    }.items():
        setattr(battle, name, value)
    return cast(DoubleBattle, battle)


def test_runtime_action_adapters_round_trip_control_move_and_preview_orders() -> None:
    battle = _adapter_battle()
    for action in (0, 7):
        order = action_to_single_order(action, battle, fake=True, position=0)
        assert int(single_order_to_action(order, battle, fake=True, position=0)) == action
    forced_battle = _adapter_battle(forced=True)
    forced_order = action_to_single_order(48, forced_battle, fake=True, position=0)
    assert int(single_order_to_action(forced_order, forced_battle, fake=True, position=0)) == 48
    assert np.array_equal(
        order_to_action(action_to_order(np.array([-2, -2]), battle), battle), [-2, -2]
    )
    assert np.array_equal(
        order_to_action(action_to_order(np.array([-1, -1]), battle), battle), [-1, -1]
    )
    preview = _adapter_battle(teampreview=True)
    selected = np.array([1, 15], dtype=np.int64)
    preview_order = action_to_order(selected, preview)
    assert np.array_equal(order_to_action(preview_order, preview), selected)


def test_battle_view_cache_refreshes_decisions_without_replacing_facade() -> None:
    battle = _adapter_battle()
    first = current_battle_view(battle)
    first_decision = first.decision
    assert first is current_battle_view(battle)
    assert first_decision is first.decision
    battle._wait = True
    refreshed = battle_view(battle)
    assert refreshed is first
    assert refreshed.decision is not first_decision
    assert refreshed.decision.wait is True
    assert decision_view(battle) == refreshed.decision


def test_runtime_action_adapters_reject_invalid_orders_in_strict_mode() -> None:
    battle = _adapter_battle()
    with pytest.raises(ValueError):
        action_to_single_order(26, battle, fake=False, position=0)
    with pytest.raises((TypeError, ValueError)):
        order_to_action(cast(Any, SimpleNamespace()), battle, strict=True)


def test_recharge_is_encoded_as_forced_move() -> None:
    battle = _adapter_battle()
    cast(Any, battle).available_moves = [[SimpleNamespace(id="recharge")], []]
    order = action_to_single_order(48, battle, fake=True, position=0)
    assert int(single_order_to_action(order, battle, fake=True, position=0)) == 48
