import re
from functools import lru_cache
from pathlib import Path

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from transformers import BertModel, BertTokenizerFast

from lookups import (
    ACT_SIZE,
    EFFECT_DESCRIPTION,
    EXTRA_SZ,
    ITEM_DESCRIPTION,
    MOVES,
    POKEMON,
    POKEMON_DESCRIPTION,
    STATUS_DESCRIPTION,
    TINYBERT_SZ,
)


# Pre-compiled constants and regexes for optimization
SLOT_STATUS_DESC = {
    -1: "This Pokemon is DROPPED. It is not part of the battle.",
    0: "This pokemon MAY or MAY NOT be in the back as a switch.",
    1: "This pokemon IS ACTIVE. It is currently on the field.",
    2: "This pokemon is IN THE BACK. It is able to switch in.",
    3: "This pokemon has FAINTED. It no longer participates in the battle.",
    4: "This pokemon CANNOT BE SWITCHED IN. May or may not be in team.",
}
DEFAULT_SLOT_STATUS_DESC = "We do not know about this pokemon."

SLOT_STATUS_IDX = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

# Regex patterns for _get_turn_summary
MOVE_RE = re.compile(r"\|move\|([^|]+)\|([^|]+)")
DAMAGE_RE = re.compile(r"\|-damage\|([^|]+)\|([^|]+)")
FAINT_RE = re.compile(r"\|faint\|([^|]+)")
STATUS_RE = re.compile(r"\|-status\|([^|]+)\|([^|]+)")
BOOST_RE = re.compile(r"\|-(boost|unboost)\|([^|]+)\|([^|]+)\|([^|]+)")
ABILITY_RE = re.compile(r"\|-ability\|([^|]+)\|([^|]+)")
TERA_RE = re.compile(r"\|-terastallize\|([^|]+)\|([^|]+)")
CLEAN_ID_RE = re.compile(r"[^a-z0-9]")


def _to_id_str(s: str) -> str:
    return CLEAN_ID_RE.sub("", s.lower())


def _get_last_move(battle: DoubleBattle, pokemon: Pokemon) -> str | None:
    # Check current turn's events first
    for event in reversed(battle.current_observation.events):
        if event[1] == "move":
            try:
                event_mon = battle.get_pokemon(event[2])
                if event_mon == pokemon:
                    move_name = event[3]
                    return _to_id_str(move_name)
            except Exception:
                continue

    # Check observations from previous turns
    for turn in range(battle.turn, 0, -1):
        if turn not in battle.observations:
            continue
        obs = battle.observations[turn]
        for event in reversed(obs.events):
            if event[1] == "move":
                try:
                    event_mon = battle.get_pokemon(event[2])
                    if event_mon == pokemon:
                        move_name = event[3]
                        return _to_id_str(move_name)
                except Exception:
                    continue
    return None


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    # normalized turns left
    if start_turn < 0:
        return 0
    val = max(0, duration - (battle.turn - start_turn))
    return val / float(duration)


def _get_pokemon_obs(
    pokemon: Pokemon | None, battle: DoubleBattle, cond: int, orig_idx: int
) -> tuple[tuple[str, str], list[float]]:
    """
    cond indicates whether we know if pokemon is active, benched, dropped, fainted or unknown
    -1 = dropped
    0 = unknown
    1 = active
    2 = benched
    3 = fainted
    4 = stuck out (dropped from own team / pokemon inside is trapped)
    """
    last_move_id = _get_last_move(battle, pokemon) if pokemon and cond == 1 else None

    # Text input for each pokemon
    pokemon_str = _get_pokemon_text(pokemon, cond, last_move_id)

    # Extra inputs for each pokemon (roughly normalized to [0,1])
    pokemon_row = [0.0] * EXTRA_SZ
    if pokemon is None:
        # If no pokemon, we still set the slot status to unknown (index 1)
        role_idx = SLOT_STATUS_IDX.get(cond, 1)
        pokemon_row[55 + role_idx] = 1.0
        return pokemon_str, pokemon_row

    # Types One-Hot (0-53)
    # Type 1 (0-17)
    if pokemon.type_1:
        pokemon_row[pokemon.type_1.value - 1] = 1.0
    # Type 2 (18-35)
    if pokemon.type_2:
        pokemon_row[18 + pokemon.type_2.value - 1] = 1.0
    # Tera Type (36-53)
    if pokemon.is_terastallized:
        pokemon_row[36 + pokemon.tera_type.value - 1] = 1.0

    # Tera Flag (54)
    pokemon_row[54] = 1.0 if pokemon.is_terastallized else 0.0

    # Slot Status One-Hot (55-60)
    role_idx = SLOT_STATUS_IDX.get(cond, 1)
    pokemon_row[55 + role_idx] = 1.0

    # Numerical Stats (61-82)
    # HP (61)
    pokemon_row[61] = pokemon.current_hp_fraction if pokemon.current_hp is not None else 0.0

    # Base Stats (62-67)
    stats = ["hp", "atk", "def", "spa", "spd", "spe"]
    for i, stat in enumerate(stats):
        pokemon_row[62 + i] = pokemon.base_stats[stat] / 200.0

    # Boosts (68-74)
    boosts = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    for i, boost in enumerate(boosts):
        pokemon_row[68 + i] = pokemon.boosts[boost] / 6.0

    # PP (75-78)
    for i, move in enumerate(pokemon.moves):
        if i < 4:
            pokemon_row[75 + i] = pokemon.moves[move].current_pp / pokemon.moves[move].max_pp

    # Misc Stats (79-82)
    pokemon_row[79] = min(pokemon.protect_counter, 4) / 4.0
    pokemon_row[80] = float(pokemon.first_turn)
    pokemon_row[81] = pokemon.weight / 300.0
    pokemon_row[82] = (orig_idx + 1) / 6.0

    # 5. Last Move One-Hot (83-87)
    if last_move_id and pokemon:
        move_ids = list(pokemon.moves.keys())
        if last_move_id in move_ids:
            move_idx = move_ids.index(last_move_id)
            if move_idx < 4:
                pokemon_row[83 + move_idx] = 1.0
            else:
                pokemon_row[87] = 1.0
        else:
            pokemon_row[87] = 1.0
    else:
        pokemon_row[87] = 1.0

    # 6. Status one-hot and counter (88-97)
    statuses = [Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP]
    for i, s in enumerate(statuses):
        if pokemon.status == s:
            pokemon_row[88 + i] = 1.0
            pokemon_row[93 + i] = min(getattr(pokemon, "status_counter", 0), 5) / 5.0

    # 7. Effects one-hot and counter (98-103)
    curr_effects = pokemon.effects
    effects = [Effect.CONFUSION, Effect.TAUNT, Effect.ENCORE]
    for i, e in enumerate(effects):
        if e in curr_effects:
            pokemon_row[98 + i] = 1.0
            pokemon_row[101 + i] = curr_effects[e] / 5.0

    return pokemon_str, pokemon_row


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool
) -> list[tuple[Pokemon | None, int]]:
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    def get_orig_idx(mon):
        if mon is None or is_opponent:
            return -1
        for i, m in enumerate(battle.team.values()):
            if m == mon:
                return i
        return -1

    if battle.teampreview:
        res = [(m, get_orig_idx(m)) for m in team.values()]
        return (res + [(None, -1)] * 6)[:6]

    # Pack actives first, then the rest of the team to avoid None slots if mon exists
    res = []
    for m in active:
        if m is not None:
            res.append((m, get_orig_idx(m)))

    assigned = {m for m, i in res}
    others_list = [m for m in team.values() if m not in assigned]

    if is_opponent:
        res += [(m, -1) for m in others_list]
    else:
        # My team: prioritize bench (fainted or switchable) over dropped
        possible_switches = {mon for switches in battle.available_switches for mon in switches}
        bench, dropped = [], []
        for mon in others_list:
            idx = get_orig_idx(mon)
            if mon.fainted or mon in possible_switches:
                bench.append((mon, idx))
            else:
                dropped.append((mon, idx))
        res += bench + dropped

    return (res + [(None, -1)] * 6)[:6]


def _get_locals(battle: DoubleBattle):
    """
    Returns turn remain counts for various field and side effects.
    """
    # Global effects
    trick_room_turns = _get_turns_left(battle, battle.fields.get(Field.TRICK_ROOM, -1))
    grassy_terrain_turns = _get_turns_left(battle, battle.fields.get(Field.GRASSY_TERRAIN, -1))
    psychic_terrain_turns = _get_turns_left(battle, battle.fields.get(Field.PSYCHIC_TERRAIN, -1))

    rain_turns = _get_turns_left(battle, battle._weather.get(Weather.RAINDANCE, -1))
    sun_turns = _get_turns_left(battle, battle._weather.get(Weather.SUNNYDAY, -1))
    snow_turns = _get_turns_left(battle, battle._weather.get(Weather.SNOW, -1))

    global_effects = [
        trick_room_turns,
        grassy_terrain_turns,
        psychic_terrain_turns,
        sun_turns,
        rain_turns,
        snow_turns,
    ]

    p1_row = global_effects + [
        _get_turns_left(battle, battle.side_conditions.get(SideCondition.TAILWIND, -1), duration=4),
        _get_turns_left(battle, battle.side_conditions.get(SideCondition.AURORA_VEIL, -1)),
    ]

    p2_row = global_effects + [
        _get_turns_left(
            battle,
            battle.opponent_side_conditions.get(SideCondition.TAILWIND, -1),
            duration=4,
        ),
        _get_turns_left(battle, battle.opponent_side_conditions.get(SideCondition.AURORA_VEIL, -1)),
    ]

    return p1_row, p2_row
