import asyncio
import logging
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
from poke_env.player import RandomPlayer

from p0.battle.events import EventTypeId, RawBattleEvent
from p0.format_config import FORMAT
from p0.model.observation_builder import (
    ObservationBuilder,
    _cached_imputed_stats,
    _get_ordered_pokemon,
    _get_pokemon_level_stats,
    _global_field_token_into,
    _iter_move_slots,
    _pokemon_categorical_into,
    _side_mega_available,
    _side_token_into,
    _slot_condition,
)
from p0.model.observation_builder import (
    _pokemon_numeric_into as _write_pokemon_numeric,
)
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    EffectNamespace,
    Provenance,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.model.tokenizer import tokenizer
from p0.runtime.env import SimEnv
from p0.runtime.live_event_capture import set_raw_events
from p0.runtime.poke_env_battle_adapter import battle_view
from p0.teams.source import ValidatedTeam
from p0.teams.stat_points import PrecomputedStats

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

    # Order must be: Active (p1, p2) -> Switch/Fainted bench (p3, p4, p5) -> Dropped bench (p6)
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
    assert partial_mons.index(p6) < 5

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
    # placeholder overflows the 6-row budget; the trimmed mon must be an
    # unrevealed, unfainted one (likely unbrought) — fainted p5 must survive
    le_mons = [entry[0] for entry in ordered_le]
    assert p5 in le_mons
    assert p6 not in le_mons

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


def test_from_battle_real_end_to_end():
    """Verify that from_battle generates structured observation tensors with correct dimensions using real objects."""
    p1 = make_real_pokemon(species="charizard")
    team = [p1]
    battle = make_real_battle(
        active_pokemon=[p1, None],
        opponent_active_pokemon=[None, None],
        team=team,
        opponent_team=[],
        teampreview=False,
    )

    obs = from_battle(battle, tokenizer)
    assert obs.token_type_ids.shape == (SEQUENCE_LENGTH,)
    assert obs.side_ids.shape == (SEQUENCE_LENGTH,)
    assert obs.slot_ids.shape == (SEQUENCE_LENGTH,)
    assert obs.categorical.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
    assert obs.numerical.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)

    assert obs.events_cat.shape == (EVENT_COUNT, EVENT_CATEGORICAL_WIDTH)
    assert obs.events_num.shape == (EVENT_COUNT, EVENT_NUMERICAL_WIDTH)
    assert obs.events_side_ids.shape == (EVENT_COUNT,)
    assert obs.events_slot_ids.shape == (EVENT_COUNT,)

    assert obs.token_type_ids[0] == TokenType.CLS
    assert obs.token_type_ids[1] == TokenType.POKEMON
    assert obs.token_type_ids[12] == TokenType.POKEMON
    assert obs.token_type_ids[13] == TokenType.FIELD
    assert obs.token_type_ids[14] == TokenType.FIELD
    assert obs.token_type_ids[15] == TokenType.FIELD

    assert obs.side_ids[0] == SideId.NONE
    assert obs.side_ids[1] == SideId.ALLY
    assert obs.side_ids[7] == SideId.OPPONENT
    assert obs.side_ids[13] == SideId.NONE
    assert obs.side_ids[14] == SideId.ALLY
    assert obs.side_ids[15] == SideId.OPPONENT


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
    assert obs.events_num[:, 2].max().item() == float(overflow)


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
    assert torch.count_nonzero(out.categorical[0]) == 0
    assert torch.count_nonzero(out.numerical[0]) == 0
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

    expected = PrecomputedStats((155, 93, 98, 177, 105, 152))
    values, provenance = _get_pokemon_level_stats(pokemon, True, expected)
    assert values == tuple(float(value) for value in expected.values)
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
    assert result.token_type_ids[0] == TokenType.CLS
    assert len(mask) == FORMAT.action_size * 2
    assert decision_builds == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_observation_builder_live(showdown_server, battle_format, sample_team):
    """Exercise observation construction against the real local Showdown protocol."""
    captured_battles = []
    captured_errors = []

    class CapturePlayer(RandomPlayer):
        def teampreview(self, battle):
            try:
                captured_battles.append((battle.teampreview, from_battle(battle, tokenizer)))
            except Exception as exc:
                captured_errors.append(f"teampreview: {exc}")
            return super().teampreview(battle)

        def choose_move(self, battle):
            try:
                captured_battles.append((battle.teampreview, from_battle(battle, tokenizer)))
            except Exception as exc:
                captured_errors.append(f"choose_move: {exc}")
            return super().choose_move(battle)

    p1 = CapturePlayer(
        battle_format=battle_format,
        server_configuration=showdown_server,
        team=sample_team,
        max_concurrent_battles=1,
    )
    p2 = RandomPlayer(
        battle_format=battle_format,
        server_configuration=showdown_server,
        team=sample_team,
        max_concurrent_battles=1,
    )

    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=15.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Battle timed out. Internal errors: {captured_errors}")
    except Exception as exc:
        pytest.fail(f"Battle failed with exception: {exc}. Internal errors: {captured_errors}")

    assert not captured_errors
    assert captured_battles
    assert any(is_teampreview for is_teampreview, _ in captured_battles)
    assert any(not is_teampreview for is_teampreview, _ in captured_battles)
    for _, obs in captured_battles:
        assert isinstance(obs, StructuredObservation)
        assert obs.categorical.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
        assert obs.numerical.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)


def test_pokemon_nature_in_categorical():
    mon = make_real_pokemon(species="charizard")
    mon._nature = "Jolly"

    cat = np.zeros(CATEGORICAL_WIDTH, dtype=np.int64)
    _pokemon_categorical_into(mon, tokenizer, _iter_move_slots(mon), cat)
    assert len(cat) == CATEGORICAL_WIDTH
    jolly_id = tokenizer.nature_id(mon)
    assert jolly_id > 0
    assert cat[24] == jolly_id


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

    assert obs.numerical[1, NUM_IDX_EFFECT_COUNT] == 9
    assert obs.numerical[14, NUM_IDX_EFFECT_COUNT] == 7
    assert obs.numerical[13, NUM_IDX_EFFECT_COUNT] == 6
    assert obs.numerical[:, NUM_IDX_EFFECT_OVERFLOW].sum() == 0
    pokemon_effects = obs.categorical[1, CAT_EFFECT_START::EFFECT_CATEGORICAL_WIDTH]
    assert torch.count_nonzero(pokemon_effects) == 9
    namespaces = obs.categorical[13, CAT_EFFECT_START + 2 :: EFFECT_CATEGORICAL_WIDTH]
    assert EffectNamespace.FIELD in namespaces
    assert EffectNamespace.WEATHER in namespaces
