from __future__ import annotations

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
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

# avoid re-allocating this dict on every _pokemon_numeric call.
_VOLATILE_MAX_DURATIONS: dict[Effect, float] = {
    Effect.CONFUSION: 4.0,
    Effect.DISABLE: 4.0,
    Effect.ENCORE: 3.0,
    Effect.LEECH_SEED: 1.0,
    Effect.THROAT_CHOP: 2.0,
}
_VOLATILE_ORDER = list(_VOLATILE_MAX_DURATIONS.keys())

_ZERO_CATEGORICAL = [0] * CATEGORICAL_WIDTH
_ZERO_NUMERICAL = [0.0] * NUMERICAL_WIDTH


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    if start_turn < 0:
        return 0.0
    return max(0.0, duration - (battle.turn - start_turn)) / float(duration)


def _safe_fraction(num: float | int | None, den: float | int | None) -> float:
    if not den:
        return 0.0
    return float(num or 0.0) / float(den)


def _iter_move_slots(pokemon: Pokemon | None) -> list[Move | None]:
    if pokemon is None:
        return [None] * MOVE_SLOTS
    moves = list(pokemon.moves.values())[:MOVE_SLOTS]
    return moves + [None] * (MOVE_SLOTS - len(moves))


def _pokemon_categorical(
    pokemon: Pokemon | None,
    tok: PokemonTokenizer,
    move_slots: list[Move | None],
) -> list[int]:
    if pokemon is None:
        return [0] * CATEGORICAL_WIDTH

    move_ids = [tok.move_id(m) if m else 0 for m in move_slots]
    move_type_ids = [tok.move_type_id(m) if m else 0 for m in move_slots]
    move_category_ids = [tok.move_category_id(m) if m else 0 for m in move_slots]

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
    move_slots: list[Move | None],
    active_idx: int | None = None,
) -> list[float]:
    row = [0.0] * NUMERICAL_WIDTH
    row[cond + 1] = 1.0

    if pokemon is None:
        return row

    row[5] = float(pokemon.current_hp_fraction)

    base_stats = pokemon.base_stats
    row[6] = float(base_stats["hp"]) / 160.0
    row[7] = float(base_stats["atk"]) / 160.0
    row[8] = float(base_stats["def"]) / 160.0
    row[9] = float(base_stats["spa"]) / 160.0
    row[10] = float(base_stats["spd"]) / 160.0
    row[11] = float(base_stats["spe"]) / 160.0

    boosts = pokemon.boosts
    row[12] = float(boosts["atk"]) / 6.0
    row[13] = float(boosts["def"]) / 6.0
    row[14] = float(boosts["spa"]) / 6.0
    row[15] = float(boosts["spd"]) / 6.0
    row[16] = float(boosts["spe"]) / 6.0
    row[17] = float(boosts["accuracy"]) / 6.0
    row[18] = float(boosts["evasion"]) / 6.0

    for i, move in enumerate(move_slots):
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

    if cond == 1 and pokemon.last_move:
        last_move_id = pokemon.last_move.id
        for move_idx, move in enumerate(move_slots):
            if move is not None and move.id == last_move_id:
                row[32 + move_idx] = 1.0
                break

    row[36] = min(float(pokemon.status_counter), 5.0) / 5.0

    for i, effect in enumerate(_VOLATILE_ORDER):
        val = pokemon.effects.get(effect, 0)
        max_dur = _VOLATILE_MAX_DURATIONS[effect]
        row[37 + i] = min(float(val), max_dur) / max_dur

    row[42] = float(pokemon.preparing)
    return row


def _pad_team(
    res: list[tuple[Pokemon | None, int, int | None]],
) -> list[tuple[Pokemon | None, int, int | None]]:
    pad_len = TEAM_SIZE - len(res)
    if pad_len > 0:
        res.extend([(None, -1, None)] * pad_len)
    elif pad_len < 0:
        res = res[:TEAM_SIZE]
    return res


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool, possible_switches: set[Pokemon] | None = None
) -> list[tuple[Pokemon | None, int, int | None]]:
    # returns list of (pokemon, orig_id, active_id)
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    if is_opponent:
        # opponent: orig_idx is always -1, no switch set needed.
        if battle.teampreview:
            res = [(mon, -1, None) for mon in team.values()]
            return _pad_team(res)

        res: list[tuple[Pokemon | None, int, int | None]] = []
        assigned: set[Pokemon] = set()
        for mon in active:
            if mon is not None:
                res.append((mon, -1, None))
                assigned.add(mon)
        res += [(mon, -1, None) for mon in team.values() if mon not in assigned]
        return _pad_team(res)

    # ally side: build the orig_idx map once instead of scanning per pokemon.
    orig_idx_map = {mon: i for i, mon in enumerate(battle.team.values())}

    if battle.teampreview:
        res = [(mon, orig_idx_map.get(mon, -1), None) for mon in team.values()]
        return _pad_team(res)

    if possible_switches is None:
        possible_switches = {mon for switches in battle.available_switches for mon in switches}

    res = []
    assigned = set()
    for active_idx, mon in enumerate(active):
        if mon is not None:
            res.append((mon, orig_idx_map.get(mon, -1), active_idx))
            assigned.add(mon)

    bench, dropped = [], []
    for mon in team.values():
        if mon in assigned:
            continue
        idx = orig_idx_map.get(mon, -1)
        if mon.fainted or mon in possible_switches:
            bench.append((mon, idx, None))
        else:
            dropped.append((mon, idx, None))
    res += bench + dropped

    return _pad_team(res)


def _slot_condition(
    battle: DoubleBattle,
    mon: Pokemon | None,
    seq_idx: int,
    is_opponent: bool,
    possible_switches: set[Pokemon] | None = None,
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
    if possible_switches is None:
        possible_switches = {s for switches in battle.available_switches for s in switches}
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
        trickroom_id = tok.trickroom_id
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
    cat = [0] * CATEGORICAL_WIDTH
    num = [0.0] * 4

    auroraveil_turn = conditions.get(SideCondition.AURORA_VEIL)
    if auroraveil_turn is not None:
        cat[0] = tok.side_conditions.get(SideCondition.AURORA_VEIL, 0)
        num[0] = _get_turns_left(battle, auroraveil_turn, duration=5)

    tailwind_turn = conditions.get(SideCondition.TAILWIND)
    if tailwind_turn is not None:
        cat[1] = tok.side_conditions.get(SideCondition.TAILWIND, 0)
        num[1] = _get_turns_left(battle, tailwind_turn, duration=4)

    toxic_spikes_layers = conditions.get(SideCondition.TOXIC_SPIKES)
    if toxic_spikes_layers is not None:
        toxic_spikes_dict = tok.side_conditions.get(SideCondition.TOXIC_SPIKES, {})
        cat[2] = toxic_spikes_dict.get(toxic_spikes_layers, 0)
        num[2] = float(toxic_spikes_layers) / 2.0

    num[3] = float(fainted_count) / float(TEAM_SIZE)

    return cat, num


def from_battle(
    battle: AbstractBattle,
    tok: PokemonTokenizer | None = None,
) -> StructuredObservation:
    assert isinstance(battle, DoubleBattle)
    tok = tok or tokenizer

    token_types = [TokenType.CLS] + [0] * (SEQUENCE_LENGTH - 1)
    sides = [SideId.NONE] + [0] * (SEQUENCE_LENGTH - 1)
    slots = [0] * SEQUENCE_LENGTH
    categorical = [_ZERO_CATEGORICAL] * SEQUENCE_LENGTH
    numerical = [_ZERO_NUMERICAL] * SEQUENCE_LENGTH

    possible_switches: set[Pokemon] = {
        mon for switches in battle.available_switches for mon in switches
    }

    idx = 1
    for side, is_opponent in ((SideId.ALLY, False), (SideId.OPPONENT, True)):
        for slot_idx, (mon, orig_idx, active_idx) in enumerate(
            _get_ordered_pokemon(
                battle, is_opponent, possible_switches if not is_opponent else None
            )
        ):
            cond = _slot_condition(
                battle, mon, slot_idx, is_opponent, possible_switches if not is_opponent else None
            )
            slot_id = slot_idx + 1
            move_slots = _iter_move_slots(mon)

            token_types[idx] = TokenType.POKEMON_SUPER
            sides[idx] = side
            slots[idx] = slot_id
            categorical[idx] = _pokemon_categorical(mon, tok, move_slots)
            # numerical is already 0.0 initialized
            idx += 1

            token_types[idx] = TokenType.POKEMON_NUMERIC
            sides[idx] = side
            slots[idx] = slot_id
            # categorical is already 0 initialized
            numerical[idx] = _pokemon_numeric(mon, battle, cond, orig_idx, move_slots, active_idx)
            idx += 1

    global_cat, global_num = _global_field_token(battle, tok)
    token_types[idx] = TokenType.GLOBAL_FIELD
    sides[idx] = SideId.NONE
    slots[idx] = 0
    categorical[idx] = global_cat
    numerical[idx] = global_num + [0.0] * (NUMERICAL_WIDTH - len(global_num))
    idx += 1

    ally_fainted = sum(mon.fainted for mon in battle.team.values())
    ally_cat, ally_num = _side_token(battle, battle.side_conditions, tok, ally_fainted)
    token_types[idx] = TokenType.ALLY_SIDE
    sides[idx] = SideId.ALLY
    slots[idx] = 0
    categorical[idx] = ally_cat
    numerical[idx] = ally_num + [0.0] * (NUMERICAL_WIDTH - len(ally_num))
    idx += 1

    opp_fainted = sum(mon.fainted for mon in battle.opponent_team.values())
    opp_cat, opp_num = _side_token(battle, battle.opponent_side_conditions, tok, opp_fainted)
    token_types[idx] = TokenType.OPPONENT_SIDE
    sides[idx] = SideId.OPPONENT
    slots[idx] = 0
    categorical[idx] = opp_cat
    numerical[idx] = opp_num + [0.0] * (NUMERICAL_WIDTH - len(opp_num))
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

    return obs


def _is_mega_form(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    species = pokemon.species
    if not species:
        return False
    species_lower = species.lower()
    return "mega" in species_lower or species_lower.endswith("primal")


def _can_mega(pokemon: Pokemon | None, battle: DoubleBattle, active_idx: int | None = None) -> bool:
    if pokemon is None:
        return False
    if active_idx is not None:
        return battle.can_mega_evolve[active_idx]
    # fallback if the above doesnt work
    item = pokemon.item
    if not item:
        return False
    item_lower = item.lower()
    return (item_lower in {"redorb", "blueorb"} or "ite" in item_lower) and not _is_mega_form(
        pokemon
    )
