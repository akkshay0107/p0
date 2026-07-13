from __future__ import annotations

import json
from typing import Mapping
from weakref import WeakKeyDictionary

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition

from p0.model.event_builder import BattleEvent, EventCollector
from p0.model.structured_observation import (
    CAT_KNOWNNESS_START,
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    MAX_EFFECTS,
    MOVE_SLOTS,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUM_PROVENANCE_START,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TEAM_SIZE,
    CounterKind,
    EffectNamespace,
    Knownness,
    Provenance,
    SideId,
    StructuredObservation,
    TokenType,
    effect_cat_slice,
    effect_num_slice,
)
from p0.model.tokenizer import PokemonTokenizer, tokenizer
from p0.paths import DEFAULT_PATHS
from p0.team_data.stat_points import BaseStats, ImputationInput, PrecomputedStats, imputed_stats

_DEX_PATH = DEFAULT_PATHS.data_root / "champions_dex.json"
with _DEX_PATH.open("r", encoding="utf-8") as _stream:
    _TRANSFORMATIONS = json.load(_stream)["transformations"]
_MEGA_ITEMS = frozenset(
    PokemonTokenizer.normalize_id(item)
    for entry in _TRANSFORMATIONS
    if entry.get("isMega")
    for item in entry.get("requiredItems", [])
)
_MEGA_FORMS = frozenset(
    PokemonTokenizer.normalize_id(entry["id"]) for entry in _TRANSFORMATIONS if entry.get("isMega")
)

_STACKABLE_SIDE_EFFECTS = frozenset({SideCondition.SPIKES, SideCondition.TOXIC_SPIKES})
_BATTLE_STAT_CACHE: WeakKeyDictionary[DoubleBattle, dict[Pokemon, PrecomputedStats]] = (
    WeakKeyDictionary()
)


def _knownness(value: object | None, resolved_id: int) -> Knownness:
    if value is None or value == "":
        return Knownness.KNOWN_NONE
    return Knownness.KNOWN if resolved_id else Knownness.OOV


def _write_effects(
    entries: list[tuple[EffectNamespace, int, CounterKind, float, float, bool, float]],
    categorical: np.ndarray,
    numerical: np.ndarray,
) -> None:
    entries.sort(key=lambda entry: (int(entry[0]), entry[1]))
    if numerical.shape[0] <= NUM_IDX_EFFECT_OVERFLOW:
        return
    numerical[NUM_IDX_EFFECT_COUNT] = float(len(entries))
    numerical[NUM_IDX_EFFECT_OVERFLOW] = float(max(0, len(entries) - MAX_EFFECTS))
    for index, (namespace, effect_id, kind, value, stacks, remaining_known, remaining) in enumerate(
        entries[:MAX_EFFECTS]
    ):
        categorical[effect_cat_slice(index)] = (effect_id, int(kind), int(namespace))
        numerical[effect_num_slice(index)] = (
            1.0,
            float(value),
            float(stacks),
            float(remaining_known),
            float(remaining),
        )


def _safe_fraction(num: float | int | None, den: float | int | None) -> float:
    if not den:
        return 0.0
    return (num or 0) / den


def _iter_move_slots(pokemon: Pokemon | None) -> list[Move | None]:
    if pokemon is None:
        return [None] * MOVE_SLOTS
    moves = list(pokemon.moves.values())[:MOVE_SLOTS]
    return moves + [None] * (MOVE_SLOTS - len(moves))


def _pokemon_categorical_into(
    pokemon: Pokemon | None,
    tok: PokemonTokenizer,
    move_slots: list[Move | None],
    row: np.ndarray,
) -> None:
    if pokemon is None:
        return

    row[0] = tok.species_id(pokemon)
    row[1] = tok.ability_id(pokemon)
    row[2] = tok.item_id(pokemon)
    row[3] = tok.type_id(pokemon.type_1)
    row[4] = tok.type_id(pokemon.type_2)
    for i, move in enumerate(move_slots):
        if move is not None:
            row[5 + i] = tok.move_id(move)
            row[9 + i] = tok.move_type_id(move)
            row[13 + i] = tok.move_category_id(move)
    row[17] = tok.status_id(pokemon.status)
    row[24] = tok.nature_id(pokemon)
    values = (
        pokemon.species or pokemon.base_species,
        pokemon.ability,
        pokemon.item,
        pokemon.type_1,
        pokemon.type_2,
        *move_slots,
        *(move.type if move is not None else None for move in move_slots),
        *(move.category if move is not None else None for move in move_slots),
        pokemon.status,
    )
    for index, value in enumerate(values):
        row[CAT_KNOWNNESS_START + index] = _knownness(value, int(row[index]))
    row[CAT_KNOWNNESS_START + 24] = _knownness(pokemon.nature, int(row[24]))


def _imputation_input(pokemon: Pokemon) -> ImputationInput | None:
    if not pokemon.species or not pokemon.nature or len(pokemon.moves) != MOVE_SLOTS:
        return None
    moves = tuple(pokemon.moves.values())
    return ImputationInput(
        species=PokemonTokenizer.normalize_id(pokemon.species),
        nature=str(pokemon.nature).lower(),
        item=PokemonTokenizer.normalize_id(pokemon.item or ""),
        ability=PokemonTokenizer.normalize_id(pokemon.ability or ""),
        moves=tuple(move.id for move in moves),
        move_categories=tuple(move.category.name.lower() for move in moves),
        base_stats=BaseStats.from_mapping(pokemon.base_stats),
        level=int(pokemon.level or 50),
    )


def _cached_imputed_stats(
    pokemon: Pokemon, cache: dict[Pokemon, PrecomputedStats]
) -> PrecomputedStats | None:
    result = cache.get(pokemon)
    if result is not None:
        return result
    value = _imputation_input(pokemon)
    if value is None:
        return None
    result = imputed_stats(value)
    cache[pokemon] = result
    return result


def _get_pokemon_level_stats(
    pokemon: Pokemon,
    is_opponent: bool,
    precomputed: PrecomputedStats | None,
) -> tuple[tuple[float, ...], Provenance]:
    stats = pokemon.stats
    if not is_opponent and stats is not None:
        values = [stats.get(key) for key in ("hp", "atk", "def", "spa", "spd", "spe")]
        if all(value is not None for value in values):
            return tuple(float(value) for value in values), Provenance.SELF_KNOWN  # type: ignore

    if precomputed is not None:
        return tuple(float(value) for value in precomputed.values), Provenance.IMPUTED
    return (0.0,) * 6, Provenance.UNKNOWN


def _has_exact_stats(pokemon: Pokemon) -> bool:
    stats = pokemon.stats
    return stats is not None and all(
        stats.get(key) is not None for key in ("hp", "atk", "def", "spa", "spd", "spe")
    )


def _pokemon_numeric_into(
    pokemon: Pokemon | None,
    battle: DoubleBattle,
    cond: int,
    orig_idx: int,
    move_slots: list[Move | None],
    row: np.ndarray,
    active_idx: int | None = None,
    is_opponent: bool = False,
    precomputed_stats: PrecomputedStats | None = None,
) -> None:
    row[cond + 1] = 1.0

    if pokemon is None:
        return

    row[5] = float(pokemon.current_hp_fraction)

    base_stats = pokemon.base_stats
    row[6] = base_stats["hp"] / 160.0
    row[7] = base_stats["atk"] / 160.0
    row[8] = base_stats["def"] / 160.0
    row[9] = base_stats["spa"] / 160.0
    row[10] = base_stats["spd"] / 160.0
    row[11] = base_stats["spe"] / 160.0

    boosts = pokemon.boosts
    row[12] = boosts["atk"] / 6.0
    row[13] = boosts["def"] / 6.0
    row[14] = boosts["spa"] / 6.0
    row[15] = boosts["spd"] / 6.0
    row[16] = boosts["spe"] / 6.0
    row[17] = boosts["accuracy"] / 6.0
    row[18] = boosts["evasion"] / 6.0

    for i, move in enumerate(move_slots):
        if move is not None:
            row[19 + i] = _safe_fraction(move.current_pp, move.max_pp)

    row[23] = min(pokemon.protect_counter, 4) / 4.0
    row[24] = pokemon.first_turn

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
    row[27] = pokemon.fainted
    row[28] = cond == 1
    row[29] = cond == 2
    row[30] = _can_mega(pokemon, battle, active_idx)
    row[31] = _is_mega_form(pokemon)

    last_move_id = None
    custom_last_move = EventCollector.last_move(pokemon)
    if custom_last_move:
        last_move_id = tokenizer.normalize_id(custom_last_move)
    elif pokemon.last_move:
        last_move_id = pokemon.last_move.id

    if cond == 1 and last_move_id:
        for move_idx, move in enumerate(move_slots):
            if move is not None and move.id == last_move_id:
                row[32 + move_idx] = 1.0
                break

    row[36] = min(pokemon.status_counter, 5) / 5.0

    row[42] = pokemon.preparing

    level_stats, stat_provenance = _get_pokemon_level_stats(pokemon, is_opponent, precomputed_stats)
    row[43] = level_stats[0] / 300.0
    row[44] = level_stats[1] / 300.0
    row[45] = level_stats[2] / 300.0
    row[46] = level_stats[3] / 300.0
    row[47] = level_stats[4] / 300.0
    row[48] = level_stats[5] / 300.0
    row[49] = float(stat_provenance == Provenance.SELF_KNOWN)
    row[NUM_PROVENANCE_START : NUM_PROVENANCE_START + 6] = stat_provenance

    # action legality (allies only, the action mask is otherwise invisible to the
    # network, hiding choice lock / disable / trapping / force switches)
    if active_idx is not None and not is_opponent and not battle.teampreview:
        move_legal, can_switch_out = _ally_legality(battle, active_idx, move_slots)
        row[50] = move_legal[0]
        row[51] = move_legal[1]
        row[52] = move_legal[2]
        row[53] = move_legal[3]
        row[54] = can_switch_out

    row[55] = pokemon.revealed


def _pokemon_effects_into(
    pokemon: Pokemon | None,
    tok: PokemonTokenizer,
    categorical: np.ndarray,
    numerical: np.ndarray,
) -> None:
    if pokemon is None:
        return
    effects = []
    for effect, counter in pokemon.effects.items():
        effect_id = tok.volatiles.get(effect, 0)
        remaining_known = effect.name.startswith(("YAWN", "PERISH"))
        kind = (
            CounterKind.KNOWN_REMAINING
            if remaining_known
            else CounterKind.ACTION_COUNT
            if counter
            else CounterKind.PRESENCE_ONLY
        )
        effects.append(
            (
                EffectNamespace.POKEMON,
                effect_id,
                kind,
                float(counter),
                0.0,
                remaining_known,
                float(counter) if remaining_known else 0.0,
            )
        )
    _write_effects(effects, categorical, numerical)


def _ally_legality(
    battle: DoubleBattle, active_idx: int, move_slots: list[Move | None]
) -> tuple[list[float], float]:
    # mirrors MegaEnv.single_action_mask (env.py) without the action encoding
    move_legal = [0.0] * MOVE_SLOTS
    if battle._wait or (any(battle.force_switch) and not battle.force_switch[active_idx]):
        return move_legal, 0.0

    can_switch_out = float(
        bool(battle.available_switches[active_idx])
        and not (battle.trapped[active_idx] or battle.maybe_trapped[active_idx])
    )

    available_move_ids = {move.id for move in battle.available_moves[active_idx]}
    for i, move in enumerate(move_slots):
        if move is not None and move.id in available_move_ids:
            move_legal[i] = 1.0

    return move_legal, can_switch_out


def _pad_team(
    res: list[tuple[Pokemon | None, int, int | None]],
) -> list[tuple[Pokemon | None, int, int | None]]:
    overflow = len(res) - TEAM_SIZE
    if overflow > 0:
        # only happens when an active slot placeholder pushes a 6-mon team list
        # (opponent with open team sheet) over the row budget.prefer dropping
        # mons that are confirmed to have not been brought
        actives, rest = res[:2], res[2:]
        for i in range(len(rest) - 1, -1, -1):
            if overflow == 0:
                break
            mon = rest[i][0]
            if mon is not None and not mon.revealed and not mon.fainted:
                rest.pop(i)
                overflow -= 1
        del rest[len(rest) - overflow :]
        res = actives + rest

    pad_len = TEAM_SIZE - len(res)
    if pad_len > 0:
        res.extend([(None, -1, None)] * pad_len)
    return res


def _selected_ally_pokemon(battle: DoubleBattle) -> set[Pokemon]:
    """Return the persistent set of allies selected at team preview."""
    if battle.teampreview:
        return set(battle.team.values())

    selected = {mon for mon in battle.team.values() if mon.selected_in_teampreview}

    # battle state authoritative, previous is fallback for trapped situations
    selected.update(mon for mon in battle.active_pokemon if mon is not None)
    selected.update(mon for switches in battle.available_switches for mon in switches)
    selected.update(mon for mon in battle.team.values() if mon.fainted)
    return selected


def _get_ordered_pokemon(
    battle: DoubleBattle,
    is_opponent: bool,
    selected_allies: set[Pokemon] | None = None,
    orig_idx_map: Mapping[Pokemon, int] | None = None,
) -> list[tuple[Pokemon | None, int, int | None]]:
    # returns list of (pokemon, orig_id, active_id)
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    # both sides: dict insertion order is stable across the battle, so this gives
    # every mon a persistent identity even as the active-first ordering reshuffles
    if orig_idx_map is None:
        orig_idx_map = {mon: i for i, mon in enumerate(team.values())}

    if is_opponent:
        if battle.teampreview:
            res = [(mon, orig_idx_map.get(mon, -1), None) for mon in team.values()]
            return _pad_team(res)

        # active slots are positional: left always at index 0, right at index 1
        res: list[tuple[Pokemon | None, int, int | None]] = []
        assigned: set[Pokemon] = set()
        for mon in active:
            if mon is None:
                res.append((None, -1, None))
            else:
                res.append((mon, orig_idx_map.get(mon, -1), None))
                assigned.add(mon)
        res += [
            (mon, orig_idx_map.get(mon, -1), None) for mon in team.values() if mon not in assigned
        ]
        return _pad_team(res)

    if battle.teampreview:
        res = [(mon, orig_idx_map.get(mon, -1), None) for mon in team.values()]
        return _pad_team(res)

    if selected_allies is None:
        selected_allies = _selected_ally_pokemon(battle)

    res = []
    assigned = set()
    for active_idx, mon in enumerate(active):
        if mon is None:
            res.append((None, -1, None))
        else:
            res.append((mon, orig_idx_map.get(mon, -1), active_idx))
            assigned.add(mon)

    bench, dropped = [], []
    for mon in team.values():
        if mon in assigned:
            continue
        idx = orig_idx_map.get(mon, -1)
        if mon in selected_allies:
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
    selected_allies: set[Pokemon] | None = None,
) -> int:
    if mon is None:
        return 0
    if battle.teampreview:
        return 2
    if mon.fainted:
        return 3
    if seq_idx < 2:
        return 1
    if is_opponent:
        return 2
    if selected_allies is None:
        selected_allies = _selected_ally_pokemon(battle)
    return 2 if mon in selected_allies else -1


def _global_field_token_into(
    battle: DoubleBattle,
    tok: PokemonTokenizer,
    categorical: np.ndarray,
    numerical: np.ndarray,
) -> None:
    effects = []
    for weather, start_turn in battle.weather.items():
        effects.append(
            (
                EffectNamespace.WEATHER,
                tok.weathers.get(weather, 0),
                CounterKind.TURN_AGE,
                float(max(0, battle.turn - start_turn)),
                0.0,
                False,
                0.0,
            )
        )
    for field, start_turn in battle.fields.items():
        remaining_known = field in {Field.TRICK_ROOM, Field.MAGIC_ROOM, Field.WONDER_ROOM}
        age = float(max(0, battle.turn - start_turn))
        effects.append(
            (
                EffectNamespace.FIELD,
                tok.fields.get(field, 0),
                CounterKind.TURN_AGE,
                age,
                0.0,
                remaining_known,
                max(0.0, 5.0 - age) if remaining_known else 0.0,
            )
        )
    _write_effects(effects, categorical, numerical)
    numerical[2] = float(battle.teampreview)
    numerical[3] = battle.turn / 24.0


def _side_token_into(
    battle: DoubleBattle,
    conditions: dict[SideCondition, int],
    tok: PokemonTokenizer,
    fainted_count: int,
    mega_available: bool,
    cat: np.ndarray,
    num: np.ndarray,
) -> None:
    effects = []
    for condition, stored_value in conditions.items():
        stackable = condition in _STACKABLE_SIDE_EFFECTS
        effect_value = tok.side_conditions.get(condition, 0)
        effect_id = int(effect_value)
        remaining_known = condition == SideCondition.TAILWIND
        age = 0.0 if stackable else float(max(0, battle.turn - stored_value))
        effects.append(
            (
                EffectNamespace.SIDE,
                effect_id,
                CounterKind.STACK_COUNT if stackable else CounterKind.TURN_AGE,
                age,
                float(stored_value) if stackable else 0.0,
                remaining_known,
                max(0.0, 4.0 - age) if remaining_known else 0.0,
            )
        )
    _write_effects(effects, cat, num)
    num[3] = float(fainted_count) / float(TEAM_SIZE)
    num[4] = float(mega_available)


def _side_mega_available(
    battle: DoubleBattle,
    *,
    is_opponent: bool,
    selected_allies: set[Pokemon] | None = None,
) -> bool:
    if is_opponent:
        if battle.opponent_used_mega_evolve:
            return False
        candidates = battle.opponent_team.values()
    else:
        if battle.used_mega_evolve:
            return False
        candidates = _selected_ally_pokemon(battle) if selected_allies is None else selected_allies

    return any(
        PokemonTokenizer.normalize_id(mon.item) in _MEGA_ITEMS and not _is_mega_form(mon)
        for mon in candidates
    )


def _validate_output(out: StructuredObservation) -> None:
    expected = (
        ("token_type_ids", out.token_type_ids, (SEQUENCE_LENGTH,), torch.long),
        ("side_ids", out.side_ids, (SEQUENCE_LENGTH,), torch.long),
        ("slot_ids", out.slot_ids, (SEQUENCE_LENGTH,), torch.long),
        (
            "categorical",
            out.categorical,
            (SEQUENCE_LENGTH, CATEGORICAL_WIDTH),
            torch.long,
        ),
        (
            "numerical",
            out.numerical,
            (SEQUENCE_LENGTH, NUMERICAL_WIDTH),
            torch.float32,
        ),
        (
            "events_cat",
            out.events_cat,
            (EVENT_COUNT, EVENT_CATEGORICAL_WIDTH),
            torch.long,
        ),
        (
            "events_num",
            out.events_num,
            (EVENT_COUNT, EVENT_NUMERICAL_WIDTH),
            torch.float32,
        ),
        ("events_side_ids", out.events_side_ids, (EVENT_COUNT,), torch.long),
        ("events_slot_ids", out.events_slot_ids, (EVENT_COUNT,), torch.long),
    )
    for name, tensor, shape, dtype in expected:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"from_battle_into requires CPU output tensors; {name} is on {tensor.device}."
            )
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                f"Invalid {name}: expected shape {shape} and dtype {dtype}, "
                f"got shape {tuple(tensor.shape)} and dtype {tensor.dtype}."
            )


def _event_location(
    battle: DoubleBattle,
    event: BattleEvent,
    pokemon_to_slot: dict[Pokemon, tuple[SideId, int]],
) -> tuple[SideId, int]:
    entity_id = event.entity_id
    if entity_id is None:
        return SideId.NONE, 0

    if entity_id in ("p1", "p2"):
        side = SideId.ALLY if entity_id == battle.player_role else SideId.OPPONENT
        return side, 0

    try:
        pokemon = battle.get_pokemon(entity_id)
    except (AssertionError, IndexError, KeyError, ValueError):
        return SideId.NONE, 0
    return pokemon_to_slot.get(pokemon, (SideId.NONE, 0))


def _event_target_location(
    battle: DoubleBattle,
    event: BattleEvent,
    pokemon_to_slot: dict[Pokemon, tuple[SideId, int]],
) -> tuple[SideId, int]:
    if event.target_id is None:
        return SideId.NONE, 0
    try:
        pokemon = battle.get_pokemon(event.target_id)
    except (AssertionError, IndexError, KeyError, ValueError):
        return SideId.NONE, 0
    return pokemon_to_slot.get(pokemon, (SideId.NONE, 0))


def from_battle_into(
    battle: AbstractBattle,
    out: StructuredObservation,
    tok: PokemonTokenizer | None = None,
    stat_overrides: Mapping[Pokemon, PrecomputedStats] | None = None,
) -> None:
    assert isinstance(battle, DoubleBattle)
    _validate_output(out)
    tok = tok or tokenizer

    token_types = out.token_type_ids.numpy()
    sides = out.side_ids.numpy()
    slots = out.slot_ids.numpy()
    categorical = out.categorical.numpy()
    numerical = out.numerical.numpy()

    token_types.fill(0)
    sides.fill(0)
    slots.fill(0)
    categorical.fill(0)
    numerical.fill(0)
    token_types[0] = TokenType.CLS
    sides[0] = SideId.NONE

    selected_allies = _selected_ally_pokemon(battle)
    ally_orig_idx = {mon: i for i, mon in enumerate(battle.team.values())}
    opponent_orig_idx = {mon: i for i, mon in enumerate(battle.opponent_team.values())}
    stat_cache = _BATTLE_STAT_CACHE.setdefault(battle, {})

    pokemon_to_slot = {}

    idx = 1
    for side, is_opponent, orig_idx_map in (
        (SideId.ALLY, False, ally_orig_idx),
        (SideId.OPPONENT, True, opponent_orig_idx),
    ):
        ordered = _get_ordered_pokemon(
            battle,
            is_opponent,
            selected_allies if not is_opponent else None,
            orig_idx_map,
        )
        for slot_idx, (mon, orig_idx, active_idx) in enumerate(ordered):
            cond = _slot_condition(
                battle, mon, slot_idx, is_opponent, selected_allies if not is_opponent else None
            )
            slot_id = slot_idx + 1
            if mon is not None:
                pokemon_to_slot[mon] = (side, slot_id)
            move_slots = _iter_move_slots(mon)

            token_types[idx] = TokenType.POKEMON_SUPER
            sides[idx] = side
            slots[idx] = slot_id
            _pokemon_categorical_into(mon, tok, move_slots, categorical[idx])
            idx += 1

            token_types[idx] = TokenType.POKEMON_NUMERIC
            sides[idx] = side
            slots[idx] = slot_id
            precomputed = stat_overrides.get(mon) if stat_overrides and mon else None
            if (
                precomputed is None
                and mon is not None
                and (is_opponent or not _has_exact_stats(mon))
            ):
                precomputed = _cached_imputed_stats(mon, stat_cache)
            _pokemon_numeric_into(
                mon,
                battle,
                cond,
                orig_idx,
                move_slots,
                numerical[idx],
                active_idx,
                is_opponent=is_opponent,
                precomputed_stats=precomputed,
            )
            _pokemon_effects_into(mon, tok, categorical[idx - 1], numerical[idx])
            idx += 1

    token_types[idx] = TokenType.FIELD_SUPER
    sides[idx] = SideId.NONE
    _global_field_token_into(battle, tok, categorical[idx], numerical[idx + 1])
    idx += 1
    token_types[idx] = TokenType.FIELD_NUMERIC
    sides[idx] = SideId.NONE
    idx += 1

    ally_fainted = sum(mon.fainted for mon in battle.team.values())
    ally_mega_available = _side_mega_available(
        battle,
        is_opponent=False,
        selected_allies=selected_allies,
    )
    token_types[idx] = TokenType.FIELD_SUPER
    sides[idx] = SideId.ALLY
    _side_token_into(
        battle,
        battle.side_conditions,
        tok,
        ally_fainted,
        ally_mega_available,
        categorical[idx],
        numerical[idx + 1],
    )
    idx += 1
    token_types[idx] = TokenType.FIELD_NUMERIC
    sides[idx] = SideId.ALLY
    idx += 1

    opp_fainted = sum(mon.fainted for mon in battle.opponent_team.values())
    opp_mega_available = _side_mega_available(battle, is_opponent=True)
    token_types[idx] = TokenType.FIELD_SUPER
    sides[idx] = SideId.OPPONENT
    _side_token_into(
        battle,
        battle.opponent_side_conditions,
        tok,
        opp_fainted,
        opp_mega_available,
        categorical[idx],
        numerical[idx + 1],
    )
    idx += 1
    token_types[idx] = TokenType.FIELD_NUMERIC
    sides[idx] = SideId.OPPONENT
    idx += 1

    if idx != SEQUENCE_LENGTH:
        raise RuntimeError(f"Structured observation length drifted to {idx}")

    untruncated_events = EventCollector.consume_events(battle)
    event_overflow = max(0, len(untruncated_events) - EVENT_COUNT)
    events = EventCollector.truncate_events(untruncated_events, limit=EVENT_COUNT)

    events_cat = out.events_cat.numpy()
    events_num = out.events_num.numpy()
    events_side_ids = out.events_side_ids.numpy()
    events_slot_ids = out.events_slot_ids.numpy()

    events_cat.fill(0)
    events_num.fill(0)
    events_side_ids.fill(0)
    events_slot_ids.fill(0)

    for event_idx, event in enumerate(events):
        side_id, slot_id = _event_location(battle, event, pokemon_to_slot)
        target_side_id, target_slot_id = _event_target_location(battle, event, pokemon_to_slot)

        events_cat[event_idx, 0] = event.event_type
        events_cat[event_idx, 1] = event.move_id
        events_cat[event_idx, 2] = event.item_id
        events_cat[event_idx, 3] = event.status_id
        events_cat[event_idx, 4] = min(event.order + 1, EVENT_COUNT)
        events_cat[event_idx, 5] = event.effect_id
        events_cat[event_idx, 6] = event.ability_id
        events_cat[event_idx, 7] = event.flags
        events_cat[event_idx, 8] = target_side_id
        events_cat[event_idx, 9] = target_slot_id

        events_num[event_idx, 0] = event.value
        events_num[event_idx, 1] = event.order / float(EVENT_COUNT)
        events_num[event_idx, 2] = float(event_overflow)

        events_side_ids[event_idx] = side_id
        events_slot_ids[event_idx] = slot_id


def from_battle(
    battle: AbstractBattle,
    tok: PokemonTokenizer | None = None,
    stat_overrides: Mapping[Pokemon, PrecomputedStats] | None = None,
) -> StructuredObservation:
    obs = StructuredObservation.empty_batch(1)[0]
    from_battle_into(battle, obs, tok, stat_overrides)
    return obs


def _is_mega_form(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    species = pokemon.species
    if not species:
        return False
    return PokemonTokenizer.normalize_id(species) in _MEGA_FORMS


def _can_mega(pokemon: Pokemon | None, battle: DoubleBattle, active_idx: int | None = None) -> bool:
    if pokemon is None:
        return False
    if active_idx is not None:
        return battle.can_mega_evolve[active_idx]
    # fallback if the above doesnt work
    item = pokemon.item
    if not item:
        return False
    return PokemonTokenizer.normalize_id(item) in _MEGA_ITEMS and not _is_mega_form(pokemon)
