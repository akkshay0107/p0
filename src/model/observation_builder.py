from __future__ import annotations

from typing import Any, Iterable

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition

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

# TODO: probably the biggest bottleneck in throughput trajectory
# try to profile a battle -> action forward pass and then
# optimize this implementation entirely


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    if start_turn < 0:
        return 0.0
    return max(0.0, duration - (battle.turn - start_turn)) / float(duration)


def _get_last_move(battle: DoubleBattle, pokemon: Pokemon) -> str | None:
    for event in reversed(battle._replay_data):
        if len(event) > 2:
            ev_type = event[1]
            if ev_type in ("move", "switch", "drag", "replace"):
                try:
                    if battle.get_pokemon(event[2]) == pokemon:
                        if ev_type == "move" and len(event) > 3:
                            return PokemonTokenizer.normalize_id(event[3])
                        else:
                            # pokemon hasn't used a move since switching in
                            return None
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
    move_category_ids = [
        tok.move_category_id(move) if move is not None else 0 for move in move_slots
    ]
    volatile_ids = tok.volatile_ids(getattr(pokemon, "effects", None))

    return [
        tok.species_id(pokemon),
        tok.ability_id(pokemon),
        tok.item_id(pokemon),
        tok.type_id(getattr(pokemon, "type_1", None)),
        tok.type_id(getattr(pokemon, "type_2", None)),
        *move_ids,
        *move_type_ids,
        *move_category_ids,
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
    row[cond + 1] = 1.0

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


def _global_field_token(
    battle: DoubleBattle, tok: PokemonTokenizer
) -> tuple[list[int], list[float]]:
    # categorical slots:
    # slot 0: weather ID
    # slot 1: Trick Room ID
    # terrain and gravity to be added later
    weather_id = 0
    weather_duration = 0.0
    for weather, start_turn in getattr(battle, "_weather", {}).items():
        idx = tok.weathers.get(weather, 0)
        if idx:
            weather_id = idx
            weather_duration = _get_turns_left(battle, start_turn)
            break  # Only one weather can be active at a time

    trickroom_id = 0
    trickroom_duration = 0.0
    if Field.TRICK_ROOM in battle.fields:
        start_turn = battle.fields[Field.TRICK_ROOM]
        trickroom_id = tok.id_for("trickroom", "trickroom")
        trickroom_duration = _get_turns_left(battle, start_turn)

    categorical = [weather_id, trickroom_id] + [0] * (CATEGORICAL_WIDTH - 2)
    numerical = [
        weather_duration,
        trickroom_duration,
        float(battle.teampreview),
        battle.turn / 16.0,
    ]
    numerical = numerical + [0.0] * (NUMERICAL_WIDTH - len(numerical))
    return categorical, numerical


def _side_token(
    battle: DoubleBattle,
    conditions: dict[SideCondition, int],
    tok: PokemonTokenizer,
) -> tuple[list[int], list[float]]:
    active_conds = []
    for condition, start_turn in conditions.items():
        idx = tok.side_conditions.get(condition, 0)
        if idx:
            duration = 4 if condition == SideCondition.TAILWIND else 5
            turns_left = _get_turns_left(battle, start_turn, duration=duration)
            active_conds.append((idx, turns_left))

    # similar processing to volatiles set
    active_conds.sort(key=lambda x: x[0])
    active_conds = active_conds[:2]

    condition_ids = [c[0] for c in active_conds] + [0] * (2 - len(active_conds))
    durations = [c[1] for c in active_conds] + [0.0] * (2 - len(active_conds))

    categorical = condition_ids + [0] * (CATEGORICAL_WIDTH - len(condition_ids))
    numerical = durations + [0.0] * (NUMERICAL_WIDTH - len(durations))
    return categorical, numerical


def from_battle(
    battle: AbstractBattle,
    tok: PokemonTokenizer | None = None,
    *,
    as_dict: bool = False,
) -> StructuredObservation | dict[str, torch.Tensor]:
    assert isinstance(battle, DoubleBattle)
    tok = tok or tokenizer

    token_types = [TokenType.CLS] + [0] * (SEQUENCE_LENGTH - 1)
    sides = [SideId.NONE] + [0] * (SEQUENCE_LENGTH - 1)
    slots = [0] * SEQUENCE_LENGTH
    categorical = [[0] * CATEGORICAL_WIDTH for _ in range(SEQUENCE_LENGTH)]
    numerical = [[0.0] * NUMERICAL_WIDTH for _ in range(SEQUENCE_LENGTH)]

    idx = 1
    for side, is_opponent in ((SideId.ALLY, False), (SideId.OPPONENT, True)):
        for slot_idx, (mon, orig_idx) in enumerate(_get_ordered_pokemon(battle, is_opponent)):
            cond = _slot_condition(battle, mon, slot_idx, is_opponent)
            slot_id = slot_idx + 1

            token_types[idx] = TokenType.POKEMON_SUPER
            sides[idx] = side
            slots[idx] = slot_id
            categorical[idx] = _pokemon_categorical(mon, tok)
            # numerical is already 0.0 initialized
            idx += 1

            token_types[idx] = TokenType.POKEMON_NUMERIC
            sides[idx] = side
            slots[idx] = slot_id
            # categorical is already 0 initialized
            numerical[idx] = _pokemon_numeric(mon, battle, cond, orig_idx)
            idx += 1

    global_cat, global_num = _global_field_token(battle, tok)
    token_types[idx] = TokenType.GLOBAL_FIELD
    sides[idx] = SideId.NONE
    slots[idx] = 0
    categorical[idx] = global_cat
    numerical[idx] = global_num
    idx += 1

    ally_cat, ally_num = _side_token(battle, battle.side_conditions, tok)
    token_types[idx] = TokenType.ALLY_SIDE
    sides[idx] = SideId.ALLY
    slots[idx] = 0
    categorical[idx] = ally_cat
    numerical[idx] = ally_num
    idx += 1

    opp_cat, opp_num = _side_token(battle, battle.opponent_side_conditions, tok)
    token_types[idx] = TokenType.OPPONENT_SIDE
    sides[idx] = SideId.OPPONENT
    slots[idx] = 0
    categorical[idx] = opp_cat
    numerical[idx] = opp_num
    idx += 1

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
        available_base_species = {p.base_species for p in battle.available_switches[pos]}
        switch_space = [
            i + 1
            for i, pokemon in enumerate(battle.team.values())
            if not battle.trapped[pos] and pokemon.base_species in available_base_species
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

        mask = [0] * ACT_SIZE
        for a in actions:
            mask[a] = 1
        return mask

    return torch.tensor([single_action_mask(0), single_action_mask(1)], dtype=torch.uint8)
