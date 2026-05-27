from __future__ import annotations

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition

from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    MOVE_SLOTS,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TEAM_SIZE,
    SideId,
    StructuredObservation,
    TokenType,
)
from src.model.tokenizer import PokemonTokenizer, tokenizer


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    if start_turn < 0:
        return 0.0
    return max(0.0, duration - (battle.turn - start_turn)) / float(duration)


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
    move_ids = []
    move_type_ids = []
    move_category_ids = []
    for move in move_slots:
        if move is not None:
            move_ids.append(tok.move_id(move))
            move_type_ids.append(tok.move_type_id(move))
            move_category_ids.append(tok.move_category_id(move))
        else:
            move_ids.append(0)
            move_type_ids.append(0)
            move_category_ids.append(0)

    volatile_ids = tok.volatile_ids(pokemon.effects)

    return [
        tok.species_id(pokemon),
        tok.ability_id(pokemon),
        tok.item_id(pokemon),
        tok.type_id(pokemon.type_1),
        tok.type_id(pokemon.type_2),
        *move_ids,
        *move_type_ids,
        *move_category_ids,
        tok.status_id(pokemon.status),
        *volatile_ids,
    ]


def _pokemon_numeric(
    pokemon: Pokemon | None,
    battle: DoubleBattle,
    cond: int,
    orig_idx: int,
    active_idx: int | None = None,
) -> list[float]:
    row = [0.0] * NUMERICAL_WIDTH
    row[cond + 1] = 1.0

    if pokemon is None:
        return row

    row[5] = float(pokemon.current_hp_fraction)

    base_stats = pokemon.base_stats
    for i, stat in enumerate(["hp", "atk", "def", "spa", "spd", "spe"]):
        row[6 + i] = float(base_stats[stat]) / 160.0

    boosts = pokemon.boosts
    for i, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
        row[12 + i] = float(boosts[stat]) / 6.0

    for i, move in enumerate(_iter_move_slots(pokemon)):
        if move is not None:
            row[19 + i] = _safe_fraction(move.current_pp, move.max_pp)

    row[23] = min(float(pokemon.protect_counter), 4.0) / 4.0
    row[24] = float(pokemon.first_turn)

    # embedding based on low kick tables (since that is what matters)
    weight = pokemon.weight
    if weight < 10.0:
        row[25] = 0.0
    elif weight < 25.0:
        row[25] = 0.2
    elif weight < 50.0:
        row[25] = 0.4
    elif weight < 100.0:
        row[25] = 0.6
    elif weight < 200.0:
        row[25] = 0.8
    else:
        row[25] = 1.0

    row[26] = 0.0 if orig_idx < 0 else (orig_idx + 1) / float(TEAM_SIZE)
    row[27] = float(pokemon.fainted)
    row[28] = float(cond == 1)
    row[29] = float(cond == 2)
    row[30] = float(_can_mega(pokemon, battle, active_idx))
    row[31] = float(_is_mega_form(pokemon))

    last_move = pokemon.last_move if (pokemon is not None and cond == 1) else None
    if last_move:
        last_move_id = PokemonTokenizer.normalize_id(last_move.id)
        move_ids = [PokemonTokenizer.normalize_id(move_id) for move_id in pokemon.moves.keys()]
        if last_move_id in move_ids:
            move_idx = move_ids.index(last_move_id)
            if move_idx < 4:
                row[32 + move_idx] = 1.0

    row[36] = min(float(pokemon.status_counter), 5.0) / 5.0

    volatiles = [
        Effect.CONFUSION,
        Effect.DISABLE,
        Effect.ENCORE,
        Effect.LEECH_SEED,
        Effect.THROAT_CHOP,
    ]
    max_durations = {
        Effect.CONFUSION: 4.0,
        Effect.DISABLE: 4.0,
        Effect.ENCORE: 3.0,
        Effect.LEECH_SEED: 1.0,
        Effect.THROAT_CHOP: 2.0,
    }
    for i, effect in enumerate(volatiles):
        val = pokemon.effects.get(effect, 0)
        max_dur = max_durations.get(effect, 1.0)
        row[37 + i] = min(float(val), max_dur) / max_dur

    row[42] = float(pokemon.preparing)
    return row


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool
) -> list[tuple[Pokemon | None, int, int | None]]:
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
        res = [(mon, get_orig_idx(mon), None) for mon in team.values()]
        return (res + [(None, -1, None)] * TEAM_SIZE)[:TEAM_SIZE]

    res: list[tuple[Pokemon | None, int, int | None]] = []
    for active_idx, mon in enumerate(active):
        if mon is not None:
            res.append((mon, get_orig_idx(mon), None if is_opponent else active_idx))

    assigned = {mon for mon, _, _ in res}
    others = [mon for mon in team.values() if mon not in assigned]

    if is_opponent:
        res += [(mon, -1, None) for mon in others]
    else:
        possible_switches = {mon for switches in battle.available_switches for mon in switches}
        bench, dropped = [], []
        for mon in others:
            idx = get_orig_idx(mon)
            if mon.fainted or mon in possible_switches:
                bench.append((mon, idx, None))
            else:
                dropped.append((mon, idx, None))
        res += bench + dropped

    return (res + [(None, -1, None)] * TEAM_SIZE)[:TEAM_SIZE]


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
    for weather, start_turn in battle.weather.items():
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
    return categorical, numerical


def _side_token(
    battle: DoubleBattle,
    conditions: dict[SideCondition, int],
    tok: PokemonTokenizer,
    fainted_count: int,
) -> tuple[list[int], list[float]]:
    auroraveil_id = tok.side_conditions.get(SideCondition.AURORA_VEIL, 0)
    if isinstance(auroraveil_id, dict):
        auroraveil_id = 0
    tailwind_id = tok.side_conditions.get(SideCondition.TAILWIND, 0)
    if isinstance(tailwind_id, dict):
        tailwind_id = 0

    toxic_spikes_dict = tok.side_conditions.get(SideCondition.TOXIC_SPIKES, {})
    if isinstance(toxic_spikes_dict, int):
        toxic_spikes_dict = {}

    cat = [0] * CATEGORICAL_WIDTH
    num = [0.0] * 4

    if SideCondition.AURORA_VEIL in conditions:
        cat[0] = auroraveil_id
        num[0] = _get_turns_left(battle, conditions[SideCondition.AURORA_VEIL], duration=5)

    if SideCondition.TAILWIND in conditions:
        cat[1] = tailwind_id
        num[1] = _get_turns_left(battle, conditions[SideCondition.TAILWIND], duration=4)

    if SideCondition.TOXIC_SPIKES in conditions:
        layers = conditions[SideCondition.TOXIC_SPIKES]
        cat[2] = toxic_spikes_dict.get(layers, 0)
        num[2] = float(layers) / 2.0

    num[3] = float(fainted_count) / float(TEAM_SIZE)

    return cat, num


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
        for slot_idx, (mon, orig_idx, active_idx) in enumerate(
            _get_ordered_pokemon(battle, is_opponent)
        ):
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
            numerical[idx] = _pokemon_numeric(mon, battle, cond, orig_idx, active_idx)
            idx += 1

    global_cat, global_num = _global_field_token(battle, tok)
    token_types[idx] = TokenType.GLOBAL_FIELD
    sides[idx] = SideId.NONE
    slots[idx] = 0
    categorical[idx] = global_cat
    numerical[idx][: len(global_num)] = global_num
    idx += 1

    ally_fainted = sum(1 for mon in battle.team.values() if mon.fainted)
    ally_cat, ally_num = _side_token(battle, battle.side_conditions, tok, ally_fainted)
    token_types[idx] = TokenType.ALLY_SIDE
    sides[idx] = SideId.ALLY
    slots[idx] = 0
    categorical[idx] = ally_cat
    numerical[idx][: len(ally_num)] = ally_num
    idx += 1

    opp_fainted = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
    opp_cat, opp_num = _side_token(battle, battle.opponent_side_conditions, tok, opp_fainted)
    token_types[idx] = TokenType.OPPONENT_SIDE
    sides[idx] = SideId.OPPONENT
    slots[idx] = 0
    categorical[idx] = opp_cat
    numerical[idx][: len(opp_num)] = opp_num
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
    species = PokemonTokenizer.normalize_id(pokemon.species)
    return "mega" in species or species.endswith("primal")


def _can_mega(pokemon: Pokemon | None, battle: DoubleBattle, active_idx: int | None = None) -> bool:
    if pokemon is None:
        return False
    if active_idx is not None:
        return battle.can_mega_evolve[active_idx]
    # fallback if the above doesnt work
    item = PokemonTokenizer.normalize_id(pokemon.item)
    return (item in {"redorb", "blueorb"} or "ite" in item) and not _is_mega_form(pokemon)
