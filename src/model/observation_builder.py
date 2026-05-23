from __future__ import annotations

from typing import Any, Iterable

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather

from src.lookups import ACT_SIZE
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    MAX_VOLATILES,
    MOVE_SLOTS,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TEAM_SIZE,
    SideId,
    StructuredObservation,
    TokenType,
)
from src.model.tokenizer import PokemonTokenizer, tokenizer

SLOT_STATUS_IDX = {-1: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

WEATHER_TO_ID = {
    Weather.RAINDANCE: "rain",
    Weather.SUNNYDAY: "sun",
    Weather.SANDSTORM: "sand",
    Weather.SNOW: "snow",
}

FIELD_TO_ID = {
    Field.TRICK_ROOM: "trickroom",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
}

SIDE_CONDITION_TO_ID = {
    SideCondition.TAILWIND: "tailwind",
    SideCondition.AURORA_VEIL: "auroraveil",
    SideCondition.REFLECT: "reflect",
    SideCondition.LIGHT_SCREEN: "lightscreen",
    SideCondition.SAFEGUARD: "safeguard",
}

GLOBAL_CONDITION_IDS = {
    "none": 0,
    "rain": 1,
    "sun": 2,
    "sand": 3,
    "snow": 4,
    "trickroom": 5,
    "grassyterrain": 6,
    "psychicterrain": 7,
    "electricterrain": 8,
    "mistyterrain": 9,
}

SIDE_CONDITION_IDS = {
    "none": 0,
    "tailwind": 1,
    "auroraveil": 2,
    "reflect": 3,
    "lightscreen": 4,
    "safeguard": 5,
}


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    if start_turn < 0:
        return 0.0
    return max(0.0, duration - (battle.turn - start_turn)) / float(duration)


def _get_last_move(battle: DoubleBattle, pokemon: Pokemon) -> str | None:
    observations = [getattr(battle, "current_observation", None)]
    observations.extend(battle.observations.get(turn) for turn in range(battle.turn, 0, -1))
    for obs in observations:
        if obs is None:
            continue
        for event in reversed(obs.events):
            if len(event) > 3 and event[1] == "move":
                try:
                    if battle.get_pokemon(event[2]) == pokemon:
                        return PokemonTokenizer.normalize_id(event[3])
                except Exception:
                    continue
    return None


def _safe_fraction(num: float | int | None, den: float | int | None) -> float:
    if not den:
        return 0.0
    return float(num or 0.0) / float(den)


def _iter_move_slots(pokemon: Pokemon | None) -> list[Any | None]:
    if pokemon is None:
        return [None] * MOVE_SLOTS
    moves = list(pokemon.moves.values())[:MOVE_SLOTS]
    return moves + [None] * (MOVE_SLOTS - len(moves))


def _pokemon_categorical(
    pokemon: Pokemon | None,
    tok: PokemonTokenizer,
) -> list[int]:
    if pokemon is None:
        return [0] * CATEGORICAL_WIDTH

    move_slots = _iter_move_slots(pokemon)
    move_ids = [tok.move_id(move) if move is not None else 0 for move in move_slots]
    move_type_ids = [tok.move_type_id(move) if move is not None else 0 for move in move_slots]
    volatile_ids = tok.volatile_ids(getattr(pokemon, "effects", None))

    return [
        tok.species_id(pokemon),
        tok.ability_id(pokemon),
        tok.item_id(pokemon),
        tok.type_id(getattr(pokemon, "type_1", None)),
        tok.type_id(getattr(pokemon, "type_2", None)),
        *move_ids,
        *move_type_ids,
        tok.status_id(getattr(pokemon, "status", None)),
        *volatile_ids,
    ]


def _pokemon_numeric(
    pokemon: Pokemon | None,
    battle: DoubleBattle,
    cond: int,
    orig_idx: int,
) -> list[float]:
    row = [0.0] * NUMERICAL_WIDTH
    row[SLOT_STATUS_IDX.get(cond, 1)] = 1.0

    if pokemon is None:
        return row

    row[6] = float(getattr(pokemon, "current_hp_fraction", 0.0) or 0.0)

    base_stats = getattr(pokemon, "base_stats", {}) or {}
    for i, stat in enumerate(["hp", "atk", "def", "spa", "spd", "spe"]):
        row[7 + i] = float(base_stats.get(stat, 0.0)) / 200.0

    boosts = getattr(pokemon, "boosts", {}) or {}
    for i, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
        row[13 + i] = float(boosts.get(stat, 0.0)) / 6.0

    for i, move in enumerate(_iter_move_slots(pokemon)):
        if move is not None:
            row[20 + i] = _safe_fraction(getattr(move, "current_pp", 0), getattr(move, "max_pp", 0))

    row[24] = min(float(getattr(pokemon, "protect_counter", 0.0) or 0.0), 4.0) / 4.0
    row[25] = float(bool(getattr(pokemon, "first_turn", False)))
    row[26] = min(float(getattr(pokemon, "weight", 0.0) or 0.0), 300.0) / 300.0
    row[27] = 0.0 if orig_idx < 0 else (orig_idx + 1) / float(TEAM_SIZE)
    row[28] = float(bool(getattr(pokemon, "fainted", False)))
    row[29] = float(cond == 1)
    row[30] = float(cond == 2)
    row[31] = float(_can_mega(pokemon))
    row[32] = float(_is_mega_form(pokemon))

    last_move_id = _get_last_move(battle, pokemon) if cond == 1 else None
    if last_move_id:
        move_ids = [PokemonTokenizer.normalize_id(move_id) for move_id in pokemon.moves.keys()]
        if last_move_id in move_ids:
            row[33] = (move_ids.index(last_move_id) + 1) / float(MOVE_SLOTS)

    row[34] = min(float(getattr(pokemon, "status_counter", 0.0) or 0.0), 5.0) / 5.0
    row[35] = min(max(len(getattr(pokemon, "effects", {}) or {}), 0), MAX_VOLATILES) / float(
        MAX_VOLATILES
    )
    return row


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool
) -> list[tuple[Pokemon | None, int]]:
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    def get_orig_idx(mon: Pokemon | None) -> int:
        if mon is None or is_opponent:
            return -1
        for i, team_mon in enumerate(battle.team.values()):
            if team_mon == mon:
                return i
        return -1

    if battle.teampreview:
        res = [(mon, get_orig_idx(mon)) for mon in team.values()]
        return (res + [(None, -1)] * TEAM_SIZE)[:TEAM_SIZE]

    res: list[tuple[Pokemon | None, int]] = []
    for mon in active:
        if mon is not None:
            res.append((mon, get_orig_idx(mon)))

    assigned = {mon for mon, _ in res}
    others = [mon for mon in team.values() if mon not in assigned]

    if is_opponent:
        res += [(mon, -1) for mon in others]
    else:
        possible_switches = {mon for switches in battle.available_switches for mon in switches}
        bench, dropped = [], []
        for mon in others:
            idx = get_orig_idx(mon)
            if mon.fainted or mon in possible_switches:
                bench.append((mon, idx))
            else:
                dropped.append((mon, idx))
        res += bench + dropped

    return (res + [(None, -1)] * TEAM_SIZE)[:TEAM_SIZE]


def _slot_condition(
    battle: DoubleBattle, mon: Pokemon | None, seq_idx: int, is_opponent: bool
) -> int:
    if mon is None:
        return 0
    if battle.teampreview:
        return 2
    if seq_idx < 2:
        return 1
    if mon.fainted:
        return 3
    if is_opponent:
        return 2
    possible_switches = {switch for switches in battle.available_switches for switch in switches}
    return 2 if mon in possible_switches else -1


def _global_field_token(battle: DoubleBattle) -> tuple[list[int], list[float]]:
    condition_ids = []
    durations = []

    for weather, start_turn in getattr(battle, "_weather", {}).items():
        name = WEATHER_TO_ID.get(weather)
        if name:
            condition_ids.append(GLOBAL_CONDITION_IDS[name])
            durations.append(_get_turns_left(battle, start_turn))

    for field, start_turn in battle.fields.items():
        name = FIELD_TO_ID.get(field)
        if name:
            condition_ids.append(GLOBAL_CONDITION_IDS[name])
            durations.append(_get_turns_left(battle, start_turn))

    condition_ids = condition_ids[:6] + [0] * max(0, 6 - len(condition_ids))
    durations = durations[:6] + [0.0] * max(0, 6 - len(durations))
    numerical = durations + [float(battle.teampreview), battle.turn / 16.0]
    return condition_ids + [0] * (CATEGORICAL_WIDTH - len(condition_ids)), numerical + [0.0] * (
        NUMERICAL_WIDTH - len(numerical)
    )


def _side_token(
    battle: DoubleBattle,
    conditions: dict[SideCondition, int],
) -> tuple[list[int], list[float]]:
    condition_ids = []
    durations = []
    for condition, start_turn in conditions.items():
        name = SIDE_CONDITION_TO_ID.get(condition)
        if name:
            condition_ids.append(SIDE_CONDITION_IDS[name])
            duration = 4 if condition == SideCondition.TAILWIND else 5
            durations.append(_get_turns_left(battle, start_turn, duration=duration))

    condition_ids = condition_ids[:6] + [0] * max(0, 6 - len(condition_ids))
    durations = durations[:6] + [0.0] * max(0, 6 - len(durations))
    return condition_ids + [0] * (CATEGORICAL_WIDTH - len(condition_ids)), durations + [0.0] * (
        NUMERICAL_WIDTH - len(durations)
    )


def from_battle(
    battle: AbstractBattle,
    tok: PokemonTokenizer | None = None,
    *,
    as_dict: bool = False,
) -> StructuredObservation | dict[str, torch.Tensor]:
    assert isinstance(battle, DoubleBattle)
    tok = tok or tokenizer

    token_types = [TokenType.CLS]
    sides = [SideId.NONE]
    slots = [0]
    categorical = [[0] * CATEGORICAL_WIDTH]
    numerical = [[0.0] * NUMERICAL_WIDTH]

    for side, is_opponent in ((SideId.ALLY, False), (SideId.OPPONENT, True)):
        for idx, (mon, orig_idx) in enumerate(_get_ordered_pokemon(battle, is_opponent)):
            cond = _slot_condition(battle, mon, idx, is_opponent)
            slot_id = idx + 1

            token_types.append(TokenType.POKEMON_SUPER)
            sides.append(side)
            slots.append(slot_id)
            categorical.append(_pokemon_categorical(mon, tok))
            numerical.append([0.0] * NUMERICAL_WIDTH)

            token_types.append(TokenType.POKEMON_NUMERIC)
            sides.append(side)
            slots.append(slot_id)
            categorical.append([0] * CATEGORICAL_WIDTH)
            numerical.append(_pokemon_numeric(mon, battle, cond, orig_idx))

    global_cat, global_num = _global_field_token(battle)
    token_types.append(TokenType.GLOBAL_FIELD)
    sides.append(SideId.NONE)
    slots.append(0)
    categorical.append(global_cat)
    numerical.append(global_num)

    ally_cat, ally_num = _side_token(battle, battle.side_conditions)
    token_types.append(TokenType.ALLY_SIDE)
    sides.append(SideId.ALLY)
    slots.append(0)
    categorical.append(ally_cat)
    numerical.append(ally_num)

    opp_cat, opp_num = _side_token(battle, battle.opponent_side_conditions)
    token_types.append(TokenType.OPPONENT_SIDE)
    sides.append(SideId.OPPONENT)
    slots.append(0)
    categorical.append(opp_cat)
    numerical.append(opp_num)

    obs = StructuredObservation(
        token_type_ids=torch.tensor(token_types, dtype=torch.long),
        side_ids=torch.tensor(sides, dtype=torch.long),
        slot_ids=torch.tensor(slots, dtype=torch.long),
        categorical=torch.tensor(categorical, dtype=torch.long),
        numerical=torch.tensor(numerical, dtype=torch.float32),
    )

    if obs.token_type_ids.numel() != SEQUENCE_LENGTH:
        raise RuntimeError(f"Structured observation length drifted to {obs.token_type_ids.numel()}")

    return obs.as_dict() if as_dict else obs


def _is_mega_form(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    species = PokemonTokenizer.normalize_id(getattr(pokemon, "species", ""))
    return "mega" in species or species.endswith("primal")


def _can_mega(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    for attr in ("can_mega_evolve", "can_mega", "can_mega_evo"):
        value = getattr(pokemon, attr, None)
        if value is not None:
            return bool(value)
    item = PokemonTokenizer.normalize_id(getattr(pokemon, "item", ""))
    return (item in {"redorb", "blueorb"} or "ite" in item) and not _is_mega_form(pokemon)


def _team_has_mega(team: Iterable[Pokemon]) -> bool:
    return any(_is_mega_form(mon) for mon in team if mon is not None)


def _battle_can_mega(battle: DoubleBattle, pos: int) -> bool:
    attrs = ("can_mega_evolve", "can_mega", "can_mega_evo")
    for attr in attrs:
        value = getattr(battle, attr, None)
        if isinstance(value, (list, tuple)) and pos < len(value):
            return bool(value[pos])
        if isinstance(value, dict):
            return bool(value.get(pos, False))
    active = battle.active_pokemon[pos] if pos < len(battle.active_pokemon) else None
    return _can_mega(active) and not _team_has_mega(battle.team.values())


def get_action_mask(battle: AbstractBattle) -> torch.Tensor:
    assert isinstance(battle, DoubleBattle)
    if battle.teampreview:
        mask = [0] * ACT_SIZE
        for action in range(36):
            p1 = action // 6 + 1
            p2 = action % 6 + 1
            if p1 < p2 and p1 <= len(battle.team) and p2 <= len(battle.team):
                mask[action] = 1
        return torch.tensor([mask, mask], dtype=torch.uint8)

    def single_action_mask(pos: int) -> list[int]:
        switch_space = [
            i + 1
            for i, pokemon in enumerate(battle.team.values())
            if not battle.trapped[pos]
            and pokemon.base_species in [p.base_species for p in battle.available_switches[pos]]
        ]
        active_mon = battle.active_pokemon[pos]
        if battle._wait or (any(battle.force_switch) and not battle.force_switch[pos]):
            actions = [0]
        elif all(battle.force_switch) and len(battle.available_switches[pos]) == 1:
            actions = switch_space + [0]
        elif active_mon is None:
            actions = switch_space
        else:
            available_move_ids = {move.id for move in battle.available_moves[pos]}
            move_spaces = [
                [
                    7 + 5 * i + target + 2
                    for target in battle.get_possible_showdown_targets(move, active_mon)
                ]
                for i, move in enumerate(active_mon.moves.values())
                if move.id in available_move_ids
            ]
            move_space = [action for target_actions in move_spaces for action in target_actions]
            mega_space = [action + 20 for action in move_space if _battle_can_mega(battle, pos)]
            if (
                not move_space
                and len(battle.available_moves[pos]) == 1
                and battle.available_moves[pos][0].id in {"struggle", "recharge"}
            ):
                move_space = [9]
            actions = switch_space + move_space + mega_space
        actions = actions or [0]
        return [int(action in actions) for action in range(ACT_SIZE)]

    return torch.tensor([single_action_mask(0), single_action_mask(1)], dtype=torch.uint8)
