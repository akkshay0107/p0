import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from poke_env import LocalhostServerConfiguration
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.player import RandomPlayer

from src.model.observation_builder import (
    _get_ordered_pokemon,
    _global_field_token,
    _pokemon_categorical,
    _pokemon_numeric,
    _side_token,
    _slot_condition,
    from_battle,
)
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    SideId,
    TokenType,
)
from src.model.tokenizer import tokenizer
from src.team_picker import RandomTeamFromPool

# --- HELPERS TO INSTANTIATE REAL POKE_ENV OBJECTS ---


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
    weightkg: int | None = None,
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
        p._weightkg = weightkg
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
    return "gen9championsvgc2026regma"


@pytest.fixture(scope="module")
def sample_team():
    root_dir = Path(__file__).resolve().parent.parent
    teams_dir = root_dir / "teams"
    if not teams_dir.exists():
        pytest.skip("No teams directory found.")
    team_files = [
        path.read_text(encoding="utf-8")
        for path in teams_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    return RandomTeamFromPool(team_files)


def test_pokemon_categorical_real():
    """Verify that _pokemon_categorical maps all features correctly to vocabulary IDs using real Pokemon."""
    # None Pokemon returns 24 zeros
    assert _pokemon_categorical(None, tokenizer) == [0] * CATEGORICAL_WIDTH

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

    cat = _pokemon_categorical(mon, tokenizer)
    assert len(cat) == CATEGORICAL_WIDTH

    # Species, Ability, Item, Type 1, Type 2
    assert cat[0] == tokenizer.species_id(mon)
    assert cat[1] == tokenizer.ability_id(mon)
    assert cat[2] == tokenizer.item_id(mon)
    assert cat[3] == tokenizer.type_id("Fire")
    assert cat[4] == tokenizer.type_id("Flying")

    # 4 Moves (padded)
    assert cat[5] == tokenizer.move_id("closecombat")
    assert cat[6] == tokenizer.move_id("airslash")
    assert cat[7] == 0
    assert cat[8] == 0

    # 4 Move Types (padded)
    assert cat[9] == tokenizer.type_id("Fighting")
    assert cat[10] == tokenizer.type_id("Flying")
    assert cat[11] == 0
    assert cat[12] == 0

    # 4 Move Categories (padded)
    assert cat[13] == tokenizer.categories.get("physical")
    assert cat[14] == tokenizer.categories.get("special")
    assert cat[15] == 0
    assert cat[16] == 0

    # Status
    assert cat[17] == tokenizer.status_id(Status.BRN)

    # 6 Volatiles
    vol_ids = cat[18:24]
    assert sum(1 for v in vol_ids if v > 0) == 2


def test_pokemon_numeric_real():
    """Verify that _pokemon_numeric computes properly scaled attributes using real Pokemon."""
    battle = make_real_battle()

    # None Pokemon returns mostly zeros except for condition flag (e.g. cond=1 -> row[2] = 1.0)
    none_row = _pokemon_numeric(None, battle, cond=1, orig_idx=-1)
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
        row = _pokemon_numeric(mon, battle, cond=1, orig_idx=2)
        assert abs(row[25] - val) < 1e-5

    mon._weightkg = 75
    row = _pokemon_numeric(mon, battle, cond=1, orig_idx=2)
    assert row[5] == 0.8  # HP fraction
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
    assert row[37] == 2.0 / 4.0  # Confusion duration (max 4)
    assert row[38] == 1.0 / 4.0  # Disable duration (max 4)
    assert row[42] == 1.0  # Preparing (preparing_move is not None)

    battle._can_mega_evolve = [True, False]
    row_mega_active = _pokemon_numeric(mon, battle, cond=1, orig_idx=2, active_idx=0)
    assert row_mega_active[30] == 1.0

    mon_mega = make_real_pokemon(species="charizardmegay")
    row_mega_form = _pokemon_numeric(mon_mega, battle, cond=1, orig_idx=2)
    assert row_mega_form[31] == 1.0

    # Last move slot matching
    mon_last = make_real_pokemon(
        species="charizard",
        moves={"airslash": 10},
        last_move_id="airslash",
    )
    row_last_move = _pokemon_numeric(mon_last, battle, cond=1, orig_idx=2)
    assert row_last_move[32] == 1.0  # First move slot matched last_move


def test_get_ordered_pokemon_real():
    """Verify that _get_ordered_pokemon correctly sequences active, switches, fainted, and drops using real Pokemon."""
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


def test_slot_condition_real():
    """Verify condition values assigned based on slot index, fainted, switches, or opponent using real Pokemon."""
    p1 = make_real_pokemon(species="aerodactyl")
    p_fainted = make_real_pokemon(species="camerupt", status=Status.FNT)

    battle = make_real_battle()
    assert _slot_condition(battle, None, 0, is_opponent=False) == 0

    battle_tp = make_real_battle(teampreview=True)
    assert _slot_condition(battle_tp, p1, 0, is_opponent=False) == 2

    battle_reg = make_real_battle(teampreview=False)
    assert _slot_condition(battle_reg, p1, 1, is_opponent=False) == 1

    assert _slot_condition(battle_reg, p_fainted, 2, is_opponent=False) == 3

    assert _slot_condition(battle_reg, p1, 2, is_opponent=True) == 2

    battle_sw = make_real_battle(available_switches=[[p1]])
    assert _slot_condition(battle_sw, p1, 2, is_opponent=False) == 2
    p2 = make_real_pokemon(species="dragonite")
    assert _slot_condition(battle_sw, p2, 3, is_opponent=False) == -1


def test_global_field_token_real():
    """Verify that weather and Trick Room durations scale correctly in global fields using real battles."""
    battle = make_real_battle(turn=3)

    # Rain duration: Rain started at turn 1. Duration = 5. Left: max(0, 5 - (3 - 1)) / 5 = 3 / 5 = 0.6
    battle._weather = {Weather.RAINDANCE: 1}
    battle._fields = {
        Field.TRICK_ROOM: 2
    }  # started at turn 2. Duration = 5. Left: (5 - (3 - 2)) / 5 = 0.8
    battle._teampreview = False

    cat, num = _global_field_token(battle, tokenizer)
    assert len(cat) == CATEGORICAL_WIDTH
    assert len(num) == 4

    assert cat[0] == tokenizer.weathers.get(Weather.RAINDANCE)
    assert cat[1] == tokenizer.id_for("trickroom", "trickroom")

    assert abs(num[0] - 0.6) < 1e-5
    assert abs(num[1] - 0.8) < 1e-5
    assert num[2] == 0.0  # teampreview
    assert num[3] == 3.0 / 16.0  # turn scaling


def test_side_token_real():
    """Verify side condition turns and fainted counts mapping using real battles."""
    battle = make_real_battle(turn=4)
    conditions = {
        SideCondition.TAILWIND: 2,  # duration=4. Left: max(0, 4 - (4 - 2)) / 4 = 2 / 4 = 0.5
        SideCondition.AURORA_VEIL: 1,  # duration=5. Left: max(0, 5 - (4 - 1)) / 5 = 2 / 5 = 0.4
        SideCondition.TOXIC_SPIKES: 2,  # layers = 2. Value: 2 / 2 = 1.0
    }

    cat, num = _side_token(battle, conditions, tokenizer, fainted_count=3)
    assert len(cat) == CATEGORICAL_WIDTH
    assert len(num) == 4

    assert cat[0] == tokenizer.side_conditions.get(SideCondition.AURORA_VEIL)
    assert cat[1] == tokenizer.side_conditions.get(SideCondition.TAILWIND)
    assert cat[2] == tokenizer.side_conditions.get(SideCondition.TOXIC_SPIKES).get(2)

    assert abs(num[0] - 0.4) < 1e-5
    assert abs(num[1] - 0.5) < 1e-5
    assert abs(num[2] - 1.0) < 1e-5
    assert abs(num[3] - 0.5) < 1e-5  # 3 fainted out of 6


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

    assert obs.token_type_ids[0] == TokenType.CLS
    assert obs.token_type_ids[1] == TokenType.POKEMON_SUPER
    assert obs.token_type_ids[2] == TokenType.POKEMON_NUMERIC
    assert obs.token_type_ids[25] == TokenType.GLOBAL_FIELD
    assert obs.token_type_ids[26] == TokenType.ALLY_SIDE
    assert obs.token_type_ids[27] == TokenType.OPPONENT_SIDE

    assert obs.side_ids[0] == SideId.NONE
    assert obs.side_ids[1] == SideId.ALLY
    assert obs.side_ids[13] == SideId.OPPONENT
    assert obs.side_ids[25] == SideId.NONE
    assert obs.side_ids[26] == SideId.ALLY
    assert obs.side_ids[27] == SideId.OPPONENT

    obs_dict = from_battle(battle, tokenizer, as_dict=True)
    assert isinstance(obs_dict, dict)
    assert "categorical" in obs_dict
    assert isinstance(obs_dict["categorical"], torch.Tensor)


@pytest.mark.asyncio
async def test_observation_builder_live(showdown_server, battle_format, sample_team):
    """
    Spins up two RandomPlayers against the live local Showdown server.
    Hooks into `teampreview` and `choose_move` to extract live DoubleBattle states,
    and runs `from_battle` to verify the resulting observation arrays.
    """
    captured_battles = []
    captured_errors = []

    class CapturePlayer(RandomPlayer):
        def teampreview(self, battle):
            try:
                obs = from_battle(battle, tokenizer, as_dict=True)
                captured_battles.append((battle.turn, battle.teampreview, obs))
            except Exception as e:
                captured_errors.append(f"teampreview: {e}")
            return super().teampreview(battle)

        def choose_move(self, battle):
            try:
                obs = from_battle(battle, tokenizer, as_dict=True)
                captured_battles.append((battle.turn, battle.teampreview, obs))
            except Exception as e:
                captured_errors.append(f"choose_move: {e}")
            return super().choose_move(battle)

    p1 = CapturePlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=sample_team,
        max_concurrent_battles=1,
    )
    p2 = RandomPlayer(
        battle_format=battle_format,
        server_configuration=LocalhostServerConfiguration,
        team=sample_team,
        max_concurrent_battles=1,
    )

    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=15.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Battle timed out. Internal errors: {captured_errors}")
    except Exception as e:
        pytest.fail(f"Battle failed with exception: {e}. Internal errors: {captured_errors}")

    if captured_errors:
        pytest.fail(f"Errors occurred during observation building: {captured_errors}")

    assert len(captured_battles) > 0, "No battle states were captured."

    seen_teampreview = False
    seen_normal_turn = False

    for _, is_teampreview, obs in captured_battles:
        assert isinstance(obs, dict)
        assert "categorical" in obs
        assert "numerical" in obs

        cat = obs["categorical"]
        num = obs["numerical"]

        assert cat.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
        assert num.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)

        global_field_num = num[25]
        if is_teampreview:
            seen_teampreview = True
            # teampreview flag is at index 2 in Global Field numerical array
            assert global_field_num[2].item() == 1.0
        else:
            seen_normal_turn = True
            assert global_field_num[2].item() == 0.0

    assert seen_teampreview, "Did not capture a teampreview state."
    assert seen_normal_turn, "Did not capture a normal turn state."
