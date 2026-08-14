from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from p0.battle.events import (
    EventTypeId,
    RawBattleEvent,
)
from p0.battle.legality import DecisionView, SlotDecision
from p0.battle.views import FixtureBattleView
from p0.model.observation_builder import (
    ObservationBuilder,
    _cached_imputed_stats,
    _get_ordered_pokemon,
    _get_pokemon_level_stats,
    _slot_condition,
)
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_NATURE,
    EFFECT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    MAX_EFFECTS,
    NUM_IDX_CAN_MEGA,
    NUM_IDX_CAN_SWITCH_OUT,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_PREPARING,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_TEAM_PREVIEW,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
    TOKEN_IDX_GLOBAL_FIELD,
    TOKEN_IDX_OPPONENT_SIDE,
    EffectNamespace,
    Provenance,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.model.tokenizer import tokenizer
from p0.runtime.live_event_capture import set_raw_events
from p0.runtime.poke_env_battle_adapter import battle_view, decision_view


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


_OBSERVATION_BUILDER = ObservationBuilder(default_runtime_resources())


def from_battle(battle, tok=tokenizer, stat_overrides=None):
    assert tok is _OBSERVATION_BUILDER.tokenizer
    return _OBSERVATION_BUILDER.build(battle_view(battle), stat_overrides)


def from_battle_into(battle, out, tok=tokenizer, stat_overrides=None):
    assert tok is _OBSERVATION_BUILDER.tokenizer
    _OBSERVATION_BUILDER.build_into(battle_view(battle), out, stat_overrides)


def test_observation_builder_serializes_pokemon_features() -> None:
    """Verify ObservationBuilder converts Pokemon attributes into correctly scaled and indexed categorical/numerical tensors.
    
    Verifies:
    - Categorical columns: species, ability, item, typing (type 1 & 2), move slots (0..3), status condition, nature.
    - Numerical columns: HP fraction, base stats (scaled by 160), stat boosts (scaled by 6), move PP ratios, protect counter,
      weight bracket, status duration counter, move preparation flag, mega evolution availability.
    """
    builder = _OBSERVATION_BUILDER

    mon = make_real_pokemon(
        species="charizard",
        ability="blaze",
        item="charizarditey",
        type_1="Fire",
        type_2="Flying",
        moves={"closecombat": 4, "protect": 8},
        effects={Effect.CONFUSION: 2, Effect.DISABLE: 1},
        status=Status.BRN,
        current_hp=80,
        max_hp=100,
        boosts={"atk": 3, "def": -1},
        protect_counter=2,
        active_turns=1,
        weightkg=75.0,
        status_counter=3,
        preparing_move="closecombat",
    )
    mon._nature = "Jolly"
    battle = make_real_battle(active_pokemon=[mon, None], team=[mon], can_mega_evolve=[True, False])

    obs = builder.build(battle_view(battle))

    # Validate Categoricals
    cat = obs.categorical[0]
    assert cat[0] == tokenizer.species_id(mon)
    assert cat[1] == tokenizer.ability_id(mon)
    assert cat[2] == tokenizer.item_id(mon)
    assert cat[3] == tokenizer.type_id(PokemonType.from_name("Fire"))
    assert cat[4] == tokenizer.type_id(PokemonType.from_name("Flying"))
    assert cat[5] == tokenizer.move_id(Move("closecombat", 9))
    assert cat[6] == tokenizer.move_id(Move("protect", 9))
    assert cat[7] == 0 and cat[8] == 0  # padded move slots
    assert cat[17] == tokenizer.status_id(Status.BRN)
    assert cat[CAT_IDX_NATURE] == tokenizer.nature_id(mon)

    # Validate Numericals
    num = obs.numerical[0]
    assert num[5] == pytest.approx(0.8)  # HP fraction (80 / 100)
    assert abs(num[6] - 78.0 / 160.0) < 1e-5  # Base HP normalized by 160
    assert abs(num[7] - 84.0 / 160.0) < 1e-5  # Base Atk normalized by 160
    assert num[12] == 3.0 / 6.0  # Atk Boost (+3 / 6)
    assert num[13] == -1.0 / 6.0  # Def Boost (-1 / 6)
    assert num[19] == 4.0 / 8.0  # Move 1 PP ratio (4 / 8 max PP)
    assert num[20] == 8.0 / 16.0  # Move 2 PP ratio (8 / 16 max PP)
    assert num[23] == 2.0 / 4.0  # Protect counter normalized by 4
    assert num[24] == 1.0  # First turn active
    assert num[25] == pytest.approx(0.6)  # Low kick weight bracket for 75kg
    assert num[36] == 3.0 / 5.0  # Status counter normalized by 5
    assert num[NUM_IDX_PREPARING] == 1.0  # Preparing move flag
    assert num[NUM_IDX_CAN_MEGA] == 1.0  # Can mega evolve flag


def test_ordered_pokemon_and_slot_conditions_real() -> None:
    """Verify Pokemon slot ordering across Team Preview (original roster order) and active battle (active slots first, then bench)."""
    p1 = make_real_pokemon(species="aerodactyl")
    p2 = make_real_pokemon(species="archaludon")
    p3 = make_real_pokemon(species="azumarill")
    p4 = make_real_pokemon(species="basculegion")
    p5 = make_real_pokemon(species="camerupt", status=Status.FNT)
    p6 = make_real_pokemon(species="dragonite")
    team = [p1, p2, p3, p4, p5, p6]

    # Teampreview ordering
    battle_tp = make_real_battle(team=team, teampreview=True)
    ordered_tp = _get_ordered_pokemon(battle_tp, is_opponent=False)
    assert len(ordered_tp) == 6
    assert ordered_tp[0][0] == p1
    assert ordered_tp[5][0] == p6

    # Regular battle ordering (active slots first, then bench)
    battle_reg = make_real_battle(
        active_pokemon=[p1, p2],
        team=team,
        teampreview=False,
        available_switches=[[p3, p4], [p3, p4]],
    )
    ordered_reg = _get_ordered_pokemon(battle_reg, is_opponent=False)
    assert len(ordered_reg) == 6
    assert ordered_reg[0][0] == p1
    assert ordered_reg[1][0] == p2
    assert [entry[0] for entry in ordered_reg[2:]] == [p3, p4, p5, p6]

    # Empty left active slot placeholder
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

    # Slot condition checks
    assert _slot_condition(battle_reg, None, 0, is_opponent=False) == 0
    assert _slot_condition(battle_tp, p1, 0, is_opponent=False) == 2
    assert _slot_condition(battle_reg, p1, 1, is_opponent=False) == 1
    assert _slot_condition(battle_reg, p5, 2, is_opponent=False) == 3


def test_global_and_side_field_tokens_include_mega_availability() -> None:
    """Verify global field and side tokens include turn fraction scaling, team preview indicator, and mega availability."""
    ally_mega = make_real_pokemon(species="charizard", item="charizarditey")
    battle = make_real_battle(
        active_pokemon=[ally_mega, None],
        team=[ally_mega],
        weather={Weather.SUNNYDAY: 3},
        fields={Field.ELECTRIC_TERRAIN: 4},
        side_conditions={SideCondition.REFLECT: 2},
        opponent_side_conditions={SideCondition.LIGHT_SCREEN: 1},
        turn=5,
        teampreview=False,
        can_mega_evolve=[True, False],
    )

    obs = _OBSERVATION_BUILDER.build(battle_view(battle))

    # Global Field Token (index 12)
    assert obs.token_type_ids[TOKEN_IDX_GLOBAL_FIELD] == TokenType.FIELD
    assert obs.numerical[TOKEN_IDX_GLOBAL_FIELD, 3] == pytest.approx(5 / 24.0)  # turn / 24
    assert obs.numerical[TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] == 0.0

    # Ally Side Token (index 13)
    assert obs.token_type_ids[TOKEN_IDX_ALLY_SIDE] == TokenType.FIELD
    assert obs.numerical[TOKEN_IDX_ALLY_SIDE, 4] == 1.0  # Ally mega available

    # Opponent Side Token (index 14)
    assert obs.token_type_ids[TOKEN_IDX_OPPONENT_SIDE] == TokenType.FIELD
    assert obs.numerical[TOKEN_IDX_OPPONENT_SIDE, 4] == 0.0


def test_effect_overflow_is_counted_and_enforced() -> None:
    """Verify effect capacity capping: slots record exact effect count, overflow count, and truncate to MAX_EFFECTS."""
    effects = {
        Effect.TAUNT: 1,
        Effect.LEECH_SEED: 1,
        Effect.SUBSTITUTE: 1,
        Effect.SALT_CURE: 1,
        Effect.ENCORE: 1,
        Effect.DISABLE: 1,
        Effect.YAWN: 2,
        Effect.PERISH3: 3,
        Effect.TRAPPED: 1,
        Effect.CONFUSION: 1,
        Effect.HEAL_BLOCK: 1,
        Effect.TORMENT: 1,
        Effect.INFESTATION: 1,
        Effect.OCTOLOCK: 1,
    }
    mon = make_real_pokemon(effects=effects)
    battle = make_real_battle(active_pokemon=[mon, None], team=[mon])

    obs = _OBSERVATION_BUILDER.build(battle_view(battle))
    # Total effects: 14, overflow over MAX_EFFECTS (12) is 2
    assert obs.numerical[0, NUM_IDX_EFFECT_COUNT] == 14
    assert obs.numerical[0, NUM_IDX_EFFECT_OVERFLOW] == 2.0
    pokemon_effects = obs.categorical[0, CAT_EFFECT_START::EFFECT_CATEGORICAL_WIDTH]
    assert torch.count_nonzero(pokemon_effects) == MAX_EFFECTS


def test_concurrent_universal_effect_stress_state() -> None:
    """Verify simultaneous active effects across Pokémon, side conditions, weather, and terrain tokens."""
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

    obs = _OBSERVATION_BUILDER.build(battle_view(battle))

    assert obs.numerical[0, NUM_IDX_EFFECT_COUNT] == 9
    assert obs.numerical[13, NUM_IDX_EFFECT_COUNT] == 7
    assert obs.numerical[12, NUM_IDX_EFFECT_COUNT] == 6
    assert obs.numerical[:, NUM_IDX_EFFECT_OVERFLOW].sum() == 0
    pokemon_effects = obs.categorical[0, CAT_EFFECT_START::EFFECT_CATEGORICAL_WIDTH]
    assert torch.count_nonzero(pokemon_effects) == 9
    namespaces = obs.categorical[12, CAT_EFFECT_START + 2 :: EFFECT_CATEGORICAL_WIDTH]
    assert EffectNamespace.FIELD in namespaces
    assert EffectNamespace.WEATHER in namespaces


def test_events_ground_to_slots_and_are_idempotent() -> None:
    """Verify raw battle events ground to specific slot/side IDs and subsequent calls on identical turns are idempotent."""
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

    # Rebuilding without advancing turn produces identical event tensors
    rebuilt_obs = from_battle(battle, tokenizer)
    assert torch.equal(rebuilt_obs.events_cat, obs.events_cat)
    assert torch.equal(rebuilt_obs.events_num, obs.events_num)

    # Advancing request clears event queue
    battle._last_request = {"turn": "next"}
    next_obs = from_battle(battle, tokenizer)
    assert torch.count_nonzero(next_obs.events_cat) == 0
    assert torch.count_nonzero(next_obs.events_num) == 0


def test_side_events_ground_to_owning_side() -> None:
    """Verify side-level condition events (-sidestart) ground to the appropriate owning side token."""
    ally = make_real_pokemon(species="charizard")
    opponent = make_real_pokemon(species="venusaur")
    battle = make_real_battle(
        active_pokemon=[ally, None],
        opponent_active_pokemon=[opponent, None],
        team=[ally],
        opponent_team=[opponent],
    )
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
    assert obs.events_cat[0, 5].item() > 0


def test_event_order_recompacts() -> None:
    """Verify that event order tagging re-compacts to 1..EVENT_COUNT when raw events exceed EVENT_COUNT capacity."""
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
    raw_events = [
        RawBattleEvent(("", "-boost", "p1a: Charizard", "atk", "1"))
        for _ in range(EVENT_COUNT + overflow - 10)
    ]
    raw_events.extend(
        RawBattleEvent(("", "move", "p1a: Charizard", "Tackle", "p2a: Venusaur")) for _ in range(10)
    )
    set_raw_events(battle, raw_events)

    obs = from_battle(battle, tokenizer)

    assert obs.events_cat[:, 4].tolist() == list(range(1, EVENT_COUNT + 1))
    assert obs.events_num[:, 1].max().item() < 1.0
    assert obs.events_metadata[1].item() == float(overflow)


def test_from_battle_into_overwrites_and_validates_output_buffer() -> None:
    """Verify from_battle_into performs in-place tensor writing into pre-allocated memory buffers without stale artifact leakage."""
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
    # Poison buffer with sentinel 99 values to ensure everything is cleanly overwritten
    out.token_type_ids.fill_(99)
    out.side_ids.fill_(99)
    out.slot_ids.fill_(99)
    out.categorical.fill_(99)
    out.numerical.fill_(99.0)
    out.events_cat.fill_(99)
    out.events_num.fill_(99.0)
    out.events_side_ids.fill_(99)
    out.events_slot_ids.fill_(99)

    from_battle_into(battle, out, tokenizer)

    assert torch.equal(out.token_type_ids, expected.token_type_ids)
    assert torch.equal(out.side_ids, expected.side_ids)
    assert torch.equal(out.slot_ids, expected.slot_ids)
    assert torch.equal(out.categorical, expected.categorical)
    assert torch.equal(out.numerical, expected.numerical)
    assert torch.equal(out.events_cat, expected.events_cat)
    assert torch.equal(out.events_num, expected.events_num)
    assert torch.equal(out.events_side_ids, expected.events_side_ids)
    assert torch.equal(out.events_slot_ids, expected.events_slot_ids)
    assert not torch.any(out.categorical == 99)
    assert not torch.any(out.numerical == 99)

    # Validate dtype validation rejection
    invalid = StructuredObservation.empty_batch(1)[0]
    invalid.numerical = invalid.numerical.to(torch.float64)
    with pytest.raises(ValueError, match="Invalid numerical"):
        from_battle_into(battle, invalid)


def test_stat_resolution_provenance_and_cache_behavior() -> None:
    """Verify stat resolution provenance tracking (UNKNOWN, IMPUTED from OTS spreads, SELF_KNOWN) and caching."""
    pokemon = make_real_pokemon(species="charizard")
    pokemon._nature = None
    values, provenance = _get_pokemon_level_stats(pokemon, True, None)
    assert values == (0.0,) * 6
    assert provenance == Provenance.UNKNOWN

    # Test OTS-imputed stats
    expected = cast(tuple[int, int, int, int, int, int], tuple((155, 93, 98, 177, 105, 152)))
    values, provenance = _get_pokemon_level_stats(pokemon, True, expected)
    assert values == tuple(float(value) for value in expected)
    assert provenance == Provenance.IMPUTED

    # Test our own known Pokemon stats (Provenance.SELF_KNOWN)
    pokemon.stats = {"hp": 153, "atk": 104, "def": 98, "spa": 177, "spd": 105, "spe": 152}
    values_self, provenance_self = _get_pokemon_level_stats(pokemon, False, None)
    assert values_self == (153.0, 104.0, 98.0, 177.0, 105.0, 152.0)
    assert provenance_self == Provenance.SELF_KNOWN

    # Verify stat imputation caching
    pokemon = make_real_pokemon(
        species="charizard",
        moves={"heatwave": 10, "solarbeam": 10, "protect": 10, "weatherball": 10},
    )
    pokemon._nature = "modest"
    cache: dict[tuple[str, str | None, tuple[str, ...]], Any] = {}
    first = _cached_imputed_stats(pokemon, cache)
    second = _cached_imputed_stats(pokemon, cache)
    assert first is second
    assert len(cache) == 1


def test_observation_overflow_contract_holds_at_capacity_boundaries() -> None:
    """Verify validate_overflow_contract verifies effect and event overflow totals against metadata counters."""
    observation = StructuredObservation.empty_batch(1)[0]
    observation.numerical[:, NUM_IDX_EFFECT_COUNT] = torch.tensor(
        (0,) * 12 + (MAX_EFFECTS, MAX_EFFECTS + 2, 0),
        dtype=torch.float32,
    )
    observation.numerical[:, NUM_IDX_EFFECT_OVERFLOW] = torch.tensor(
        (0.0,) * 12 + (0.0, 2.0, 0.0),
        dtype=torch.float32,
    )
    observation.events_metadata = torch.tensor(
        ((EVENT_COUNT - 1, 0), (EVENT_COUNT, 0), (EVENT_COUNT + 3, 3)),
        dtype=torch.float32,
    )
    observation.validate_overflow_contract()
    assert observation.overflow_totals() == (2, 3)


def _legality_fixture_view(decision: DecisionView) -> FixtureBattleView:
    battle = DoubleBattle("legality", "player", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    allies = [Pokemon(gen=9, species=species) for species in ("charizard", "blastoise")]
    bench = Pokemon(gen=9, species="pikachu")
    opponent = Pokemon(gen=9, species="venusaur")
    for mon in (*allies, opponent):
        mon._active = mon is not bench
    battle._team = {"p1: Charizard": allies[0], "p1: Blastoise": allies[1], "p1: Pikachu": bench}
    battle._opponent_team = {"p2: Venusaur": opponent}
    battle._active_pokemon = {"p1a": allies[0], "p1b": allies[1]}
    battle._opponent_active_pokemon = {"p2a": opponent}

    return FixtureBattleView(
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
        teampreview=False,
        player_role=battle.player_role,
        wait=False,
        weather=battle.weather,
        fields=battle.fields,
        side_conditions=battle.side_conditions,
        opponent_side_conditions=battle.opponent_side_conditions,
        turn=battle.turn,
        used_mega_evolve=battle.used_mega_evolve,
        opponent_used_mega_evolve=battle.opponent_used_mega_evolve,
        decision=decision,
    )


def test_unknown_legality_is_gated_rather_than_written_as_illegal() -> None:
    """Verify that when decision legality is unknown, legality columns are 0 and unknown legality gate flags are raised (1.0)."""
    builder = ObservationBuilder(default_runtime_resources())
    slot = SlotDecision(switch_slots=(2,), move_targets=((-2, 1), (), (), ()), can_mega=True)
    proven = builder.build(_legality_fixture_view(DecisionView(slots=(slot, slot))))

    unknown_slot = replace(slot, legality_known=False)
    unknown = builder.build(
        _legality_fixture_view(DecisionView(slots=(unknown_slot, unknown_slot)))
    )

    legality_columns = slice(NUM_IDX_MOVE_LEGAL, NUM_IDX_CAN_SWITCH_OUT + 1)
    assert proven.numerical[0, legality_columns].any()
    assert not unknown.numerical[:, legality_columns].any()
    assert unknown.numerical[0, NUM_IDX_CAN_MEGA] == 0.0

    # Gate columns must indicate unknown status on active slots
    assert proven.numerical[:, NUM_IDX_LEGALITY_UNKNOWN].tolist() == [0.0] * SEQUENCE_LENGTH
    assert unknown.numerical[:2, NUM_IDX_LEGALITY_UNKNOWN].tolist() == [1.0, 1.0]
    assert unknown.numerical[2:, NUM_IDX_LEGALITY_UNKNOWN].sum() == 0.0

    gates = slice(NUM_IDX_SLOT_LEGALITY_UNKNOWN, NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2)
    assert unknown.numerical[TOKEN_IDX_ALLY_SIDE, gates].tolist() == [1.0, 1.0]
    assert proven.numerical[TOKEN_IDX_ALLY_SIDE, gates].tolist() == [0.0, 0.0]
    assert unknown.numerical[TOKEN_IDX_OPPONENT_SIDE, gates].tolist() == [0.0, 0.0]


def test_switch_slots_identify_roster_members_not_shared_base_species() -> None:
    """Verify decision_view distinguishes duplicate base species on team by their specific roster index."""
    active = SimpleNamespace(
        moves={"tackle": SimpleNamespace(id="tackle")},
        fainted=False,
        base_species="Pikachu",
    )
    team = {
        f"p1: Species-{index}": SimpleNamespace(base_species=f"Species-{index}")
        for index in range(6)
    }
    twin = SimpleNamespace(base_species="Urshifu")
    other = SimpleNamespace(base_species="Urshifu")
    team["p1: Urshifu-A"] = twin
    team["p1: Urshifu-B"] = other

    battle = SimpleNamespace(
        player_username="player",
        battle_tag="test-roster",
        teampreview=False,
        team=team,
        active_pokemon=[active, None],
        opponent_active_pokemon=[None, None],
        available_moves=[[SimpleNamespace(id="tackle")], []],
        available_switches=[[twin], []],
        valid_orders=[[], []],
        can_mega_evolve=[False, False],
        force_switch=[False, False],
        trapped=[False, False],
        maybe_trapped=[False, False],
        _wait=False,
        player_role="p1",
        opponent_team={},
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=1,
        used_mega_evolve=False,
        opponent_used_mega_evolve=False,
        get_possible_showdown_targets=lambda move, pokemon: [0],
    )
    assert decision_view(cast(Any, battle)).slots[0].switch_slots == (6,)


def test_reconstructed_observations_clear_reused_buffer_state() -> None:
    """Verify build_into on reconstructed replay snapshots clears and overwrites observation buffer tensors across turns."""
    from p0.replays.protocol import parse_replay_payload
    from p0.replays.reconstruct import reconstruct_both
    from tests.unit.replay_fixtures import golden_replay_payload

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


def test_empty_slot_and_fainted_pokemon_zero_padding() -> None:
    """Verify empty/unrevealed bench slots write empty slot condition (1.0) and zero out stat/move numerical columns."""
    mon = make_real_pokemon(species="pikachu")
    battle = make_real_battle(active_pokemon=[mon, None], team=[mon])

    obs = _OBSERVATION_BUILDER.build(battle_view(battle))

    # Empty ally bench slot (slot index 2): condition 0 (empty) is encoded at numerical[1]
    assert obs.token_type_ids[2] == TokenType.POKEMON
    assert obs.categorical[2, 0].item() == 0  # No species
    assert obs.numerical[2, 1].item() == 1.0  # Slot condition empty
    assert not obs.numerical[2, 5:].any()  # All Pokemon stats/HP/boosts are zero

    # Opponent slots (slots 6..11) all empty
    for slot_idx in range(6, 12):
        assert obs.categorical[slot_idx, 0].item() == 0
        assert obs.numerical[slot_idx, 1].item() == 1.0
        assert not obs.numerical[slot_idx, 5:].any()


def test_field_and_weather_turn_fraction_scaling() -> None:
    """Verify turn numbers are scaled by 1/24 on the global field token and all output tensors contain finite numbers."""
    mon = make_real_pokemon(species="charizard")
    battle = make_real_battle(
        active_pokemon=[mon, None],
        team=[mon],
        weather={Weather.SUNNYDAY: 3},
        fields={Field.ELECTRIC_TERRAIN: 5},
        turn=12,
    )
    obs = _OBSERVATION_BUILDER.build(battle_view(battle))

    # Global field token turn count is normalized by 24 (12 / 24 = 0.5)
    assert obs.numerical[TOKEN_IDX_GLOBAL_FIELD, 3].item() == pytest.approx(12.0 / 24.0)
    assert all(torch.isfinite(t).all() for t in obs.tensors())
