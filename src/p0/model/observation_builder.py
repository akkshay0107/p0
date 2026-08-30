"""Builds structured battle observations and per-slot action masks from poke-env battles.

Implements the observation builder that serializes a DoubleBattle into the
StructuredObservation contract (entities, categoricals, numericals, action mask and
team-preview state) consumed by the policy and the runtime.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from p0.battle.events import (
    SPATIAL_CATEGORICAL_WIDTH,
    SPATIAL_NUMERICAL_WIDTH,
    SPATIAL_SLOT_COUNT,
)
from p0.battle.views import BattleView, MoveView, PokemonView, TransformedPokemonView
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_IDENTITY_KNOWNNESS,
    CAT_IDX_MECHANIC_STATE,
    CAT_IDX_NATURE,
    CAT_IDX_PRESENCE_STATUS,
    CAT_IDX_STAT_PROVENANCE,
    CAT_IDX_STATUS_COUNTER_KIND,
    CATEGORICAL_WIDTH,
    MAX_EFFECTS,
    MOVE_SLOTS,
    NUM_IDX_CAN_MEGA,
    NUM_IDX_CAN_SWITCH_OUT,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUM_IDX_HP_FRACTION,
    NUM_IDX_LEGALITY_UNKNOWN,
    NUM_IDX_LEVEL_STATS,
    NUM_IDX_MOVE_LEGAL,
    NUM_IDX_PREPARING,
    NUM_IDX_REVEALED,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_STAT_PROVENANCE,
    NUM_IDX_STATUS_COUNTER,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TEAM_SIZE,
    CounterKind,
    EffectNamespace,
    IdentityKnownness,
    MechanicState,
    PresenceStatus,
    SideId,
    StatProvenance,
    StructuredObservation,
    TokenType,
    effect_cat_slice,
    effect_num_slice,
)
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.spread_usage import load_spread_table_file
from p0.teams.stat_points import BaseStats, calculate_stats

# The format's level clause pins every Pokemon to level 50.
FORMAT_LEVEL = 50

_DEFAULT_RESOURCES = default_runtime_resources()
_MEGA_ITEMS = _DEFAULT_RESOURCES.mega_items
_MEGA_FORMS = _DEFAULT_RESOURCES.mega_forms

# Named after the poke-env protocol enums the adapters hand over; matching on
# member names keeps this module free of the poke-env dependency itself.
_STACKABLE_SIDE_CONDITION_NAMES = frozenset({"SPIKES", "TOXIC_SPIKES"})
_KNOWN_DURATION_FIELD_NAMES = frozenset({"TRICK_ROOM", "MAGIC_ROOM", "WONDER_ROOM"})
_EMPTY_MOVE_SLOTS: tuple[None, ...] = (None,) * MOVE_SLOTS

_TOKEN_TYPE_LAYOUT = np.asarray(
    (
        *(TokenType.POKEMON for _ in range(TEAM_SIZE * 2)),
        TokenType.FIELD,
        TokenType.FIELD,
        TokenType.FIELD,
    ),
    dtype=np.int64,
)
_SIDE_LAYOUT = np.asarray(
    (
        *(SideId.ALLY for _ in range(TEAM_SIZE)),
        *(SideId.OPPONENT for _ in range(TEAM_SIZE)),
        SideId.NONE,
        SideId.ALLY,
        SideId.OPPONENT,
    ),
    dtype=np.int64,
)
_SLOT_LAYOUT = np.asarray(
    (
        *(slot for _ in range(2) for slot in range(1, TEAM_SIZE + 1)),
        0,
        0,
        0,
    ),
    dtype=np.int64,
)


def _status_counter_kind(status: object | None) -> CounterKind:
    """StatusRecord counter semantics.

    SLP counts public turns already slept (never the hidden RNG total duration);
    TOX is the badly-poisoned stage that scales the damage tick, the same role
    Spikes layers play for hazards. Everything else carries no counter.
    """
    name = getattr(status, "name", None)
    if name == "SLP":
        return CounterKind.TURN_AGE
    if name == "TOX":
        return CounterKind.STACK_COUNT
    return CounterKind.PRESENCE_ONLY


def _safe_fraction(num: float | int | None, den: float | int | None) -> float:
    if not den:
        return 0.0
    return (num or 0) / den


def _iter_move_slots(pokemon: PokemonView | None) -> tuple[MoveView | None, ...]:
    if pokemon is None:
        return _EMPTY_MOVE_SLOTS
    moves = tuple(pokemon.moves.values())[:MOVE_SLOTS]
    return moves + _EMPTY_MOVE_SLOTS[: MOVE_SLOTS - len(moves)]


def _pad_team(
    res: list[tuple[PokemonView | None, int, int | None]],
) -> list[tuple[PokemonView | None, int, int | None]]:
    overflow = len(res) - TEAM_SIZE
    if overflow > 0:
        # only happens when an active slot placeholder pushes a 6-mon team list
        # (opponent with open team sheet) over the row budget. Eviction must stay
        # decidable from public state, so it never consults team selection.
        actives, rest = res[:2], res[2:]
        for predicate in (
            lambda mon: mon.fainted,
            lambda mon: not mon.revealed,
        ):
            for i in range(len(rest) - 1, -1, -1):
                if overflow == 0:
                    break
                mon = rest[i][0]
                if mon is not None and predicate(mon):
                    rest.pop(i)
                    overflow -= 1
        del rest[len(rest) - overflow :]
        res = actives + rest

    pad_len = TEAM_SIZE - len(res)
    if pad_len > 0:
        res.extend([(None, -1, None)] * pad_len)
    return res


def _selected_ally_pokemon(battle: Any) -> set[Any]:
    """Return the persistent set of allies selected at team preview."""
    if battle.teampreview:
        return set(battle.team.values())

    selected = {mon for mon in battle.team.values() if mon.selected_in_teampreview is True}

    # battle state authoritative, previous is fallback for trapped situations
    selected.update(mon for mon in battle.active_pokemon if mon is not None)
    selected.update(mon for switches in battle.available_switches for mon in switches)
    selected.update(mon for mon in battle.team.values() if mon.fainted)
    return selected


def _get_ordered_pokemon(
    battle: Any,
    is_opponent: bool,
    selected_allies: set[Any] | None = None,
    orig_idx_map: Mapping[Any, int] | None = None,
) -> list[tuple[Any | None, int, int | None]]:
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
        res: list[tuple[PokemonView | None, int, int | None]] = []
        assigned: set[Any] = set()
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

    res = []
    assigned = set()
    for active_idx, mon in enumerate(active):
        if mon is None:
            res.append((None, -1, None))
        else:
            res.append((mon, orig_idx_map.get(mon, -1), active_idx))
            assigned.add(mon)

    # Row order must not depend on team selection: a public replay cannot know it
    # until a reserve appears, and an order that drifts would break replay parity.
    res += [(mon, orig_idx_map.get(mon, -1), None) for mon in team.values() if mon not in assigned]

    return _pad_team(res)


def _slot_condition(
    battle: Any,
    mon: Any | None,
    seq_idx: int,
    is_opponent: bool,
    selected_allies: set[Any] | None = None,
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
    if mon.selected_in_teampreview is None:
        return 0
    if selected_allies is None:
        selected_allies = _selected_ally_pokemon(battle)
    return 2 if mon in selected_allies else -1


def _imputed_stats(pokemon: PokemonView) -> tuple[int, int, int, int, int, int] | None:
    """Estimate level-50 stats from the usage priors, or None when unrecoverable.

    Only the species and nature are required: the usage prior is keyed on those, and
    move categories matter solely for the fallback used on uncovered species. A
    partially revealed opponent therefore still gets a usage-backed estimate.
    """
    species = pokemon.species
    nature = pokemon.nature
    if not species or not nature:
        return None

    categories = tuple(move.category.name.lower() for move in pokemon.moves.values())
    estimate = load_spread_table_file().resolve(species, str(nature), categories)
    if estimate is None:
        return None

    return calculate_stats(
        BaseStats.from_mapping(pokemon.base_stats), estimate.points, str(nature), FORMAT_LEVEL
    )


def _cached_imputed_stats(
    pokemon: PokemonView, cache: dict[Any, tuple[int, int, int, int, int, int]]
) -> tuple[int, int, int, int, int, int] | None:
    cache_key = (id(pokemon), getattr(pokemon, "species", None))
    result = cache.get(cache_key)
    if result is not None:
        return result
    result = _imputed_stats(pokemon)
    if result is None:
        return None
    cache[cache_key] = result
    return result


def _has_exact_stats(pokemon: PokemonView) -> bool:
    stats = pokemon.stats
    return stats is not None and all(
        stats.get(key) is not None for key in ("hp", "atk", "def", "spa", "spd", "spe")
    )


def _get_pokemon_level_stats(
    pokemon: PokemonView,
    is_opponent: bool,
    precomputed: tuple[int, int, int, int, int, int] | None,
) -> tuple[tuple[float, ...], StatProvenance]:
    stats = pokemon.stats
    if not is_opponent and stats is not None:
        values = [stats.get(key) for key in ("hp", "atk", "def", "spa", "spd", "spe")]
        if all(value is not None for value in values):
            return tuple(float(value) for value in values), StatProvenance.KNOWN  # type: ignore

    if precomputed is not None:
        return tuple(float(value) for value in precomputed), StatProvenance.IMPUTED
    return (0.0,) * 6, StatProvenance.UNKNOWN


def _resolve_stats(
    pokemon: PokemonView | None,
    is_opponent: bool,
    cache: dict[Any, tuple[int, int, int, int, int, int]],
    overrides: Mapping[Any, tuple[int, int, int, int, int, int] | None] | None,
) -> tuple[int, int, int, int, int, int] | None:
    if pokemon is None:
        return None
    if overrides is not None and pokemon in overrides:
        return overrides[pokemon]
    if not is_opponent and _has_exact_stats(pokemon):
        return None
    return _cached_imputed_stats(pokemon, cache)


def _is_mega_form(pokemon: PokemonView | None) -> bool:
    if pokemon is None:
        return False
    species = pokemon.species
    if not species:
        return False
    return PokemonTokenizer.normalize_id(species) in _MEGA_FORMS


def _can_mega(pokemon: PokemonView | None, battle: Any, active_idx: int | None = None) -> bool:
    if pokemon is None:
        return False
    if active_idx is not None:
        return battle.can_mega_evolve[active_idx]
    # Fallback if the attribute above is unavailable.
    item = pokemon.item
    if not item:
        return False
    return PokemonTokenizer.normalize_id(item) in _MEGA_ITEMS and not _is_mega_form(pokemon)


def _side_mega_available(
    battle: Any,
    *,
    is_opponent: bool,
    selected_allies: set[Any] | None = None,
) -> tuple[bool, bool]:
    """Whether the side still holds a mega stone, and whether that is knowable.

    A replay cannot see an unbrought reserve's item, so a side whose only mega-stone
    holder has not been revealed reports unknown instead of a false negative.
    """
    if is_opponent:
        if battle.opponent_used_mega_evolve:
            return False, True
        candidates = battle.opponent_team.values()
    else:
        if battle.used_mega_evolve:
            return False, True
        candidates = _selected_ally_pokemon(battle) if selected_allies is None else selected_allies

    available = any(
        PokemonTokenizer.normalize_id(mon.item) in _MEGA_ITEMS and not _is_mega_form(mon)
        for mon in candidates
    )
    if available or is_opponent:
        return available, True

    unresolved = any(
        mon.selected_in_teampreview is None
        and PokemonTokenizer.normalize_id(mon.item) in _MEGA_ITEMS
        and not _is_mega_form(mon)
        for mon in battle.team.values()
    )
    return False, not unresolved


def _write_effects(
    entries: list[tuple[EffectNamespace, int, CounterKind, float, float, bool, float]],
    categorical: np.ndarray,
    numerical: np.ndarray,
) -> None:
    if not entries:
        return
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


def _pokemon_effects_into(
    pokemon: PokemonView | None,
    tok: PokemonTokenizer,
    categorical: np.ndarray,
    numerical: np.ndarray,
) -> None:
    if pokemon is None or not pokemon.effects:
        return
    effects = []
    for effect, counter in pokemon.effects.items():
        effect_id = tok.volatiles[effect]
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


def _pokemon_categorical_into(
    pokemon: PokemonView | None,
    tok: PokemonTokenizer,
    move_slots: tuple[MoveView | None, ...],
    row: np.ndarray,
    cond: int = 0,
    stat_provenance: StatProvenance = StatProvenance.PAD,
) -> None:
    if pokemon is None:
        row[CAT_IDX_IDENTITY_KNOWNNESS] = IdentityKnownness.PAD
        row[CAT_IDX_STAT_PROVENANCE] = StatProvenance.PAD
        row[CAT_IDX_PRESENCE_STATUS] = PresenceStatus.PAD
        row[CAT_IDX_MECHANIC_STATE] = MechanicState.NORMAL
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
    row[CAT_IDX_NATURE] = tok.nature_id(pokemon)
    row[CAT_IDX_STATUS_COUNTER_KIND] = _status_counter_kind(pokemon.status)

    # Distinguish missing species from names the current vocabulary cannot encode.
    species_name = pokemon.species
    if not species_name:
        try:
            species_name = pokemon.base_species
        except (KeyError, AttributeError):
            species_name = None

    if not species_name:
        row[CAT_IDX_IDENTITY_KNOWNNESS] = IdentityKnownness.UNKNOWN
    elif row[0] > 0:
        row[CAT_IDX_IDENTITY_KNOWNNESS] = IdentityKnownness.KNOWN
    else:
        row[CAT_IDX_IDENTITY_KNOWNNESS] = IdentityKnownness.OOV

    # Record whether level stats are exact, estimated, or unavailable.
    row[CAT_IDX_STAT_PROVENANCE] = stat_provenance

    # Summarize what the observer knows about this Pokemon's battle presence.
    if cond == 1:
        row[CAT_IDX_PRESENCE_STATUS] = PresenceStatus.ACTIVE
    elif cond == -1:
        row[CAT_IDX_PRESENCE_STATUS] = PresenceStatus.UNBROUGHT_CONFIRMED
    elif pokemon.revealed:
        row[CAT_IDX_PRESENCE_STATUS] = PresenceStatus.BENCH_REVEALED
    else:
        row[CAT_IDX_PRESENCE_STATUS] = PresenceStatus.RESERVE_UNCONFIRMED

    # Transform is explicit in the view. Illusion is present only while an
    # Illusion user is still presenting another Pokemon as its base species.
    try:
        base_species_id = PokemonTokenizer.normalize_id(pokemon.base_species)
    except (KeyError, AttributeError):
        base_species_id = ""

    if isinstance(pokemon, TransformedPokemonView) or getattr(pokemon, "is_transformed", False):
        row[CAT_IDX_MECHANIC_STATE] = MechanicState.TRANSFORMED
    elif (
        PokemonTokenizer.normalize_id(pokemon.ability or "") == "illusion"
        and base_species_id != "zoroark"
    ):
        row[CAT_IDX_MECHANIC_STATE] = MechanicState.ILLUSION_DISGUISED
    else:
        row[CAT_IDX_MECHANIC_STATE] = MechanicState.NORMAL


def _ally_legality(
    battle: BattleView, active_idx: int, move_slots: tuple[MoveView | None, ...]
) -> tuple[list[float], float, bool]:
    """Per-move legality, can-switch-out, and whether the source could prove them."""
    decision = battle.decision
    slot = decision.slots[active_idx]

    # A source without the authoritative request emits zeros and lets the gate say so;
    # a concrete illegal value here would be indistinguishable from a proven restriction.
    if not slot.legality_known:
        return [0.0] * MOVE_SLOTS, 0.0, False

    any_force = decision.slots[0].force_switch or decision.slots[1].force_switch
    if decision.wait or (any_force and not slot.force_switch):
        return [0.0] * MOVE_SLOTS, 0.0, True

    move_legal = [
        float(index < len(slot.move_targets) and bool(slot.move_targets[index]))
        for index in range(MOVE_SLOTS)
    ]
    return move_legal, float(bool(slot.switch_slots) and not slot.trapped), True


def _pokemon_numeric_into(
    pokemon: PokemonView | None,
    battle: BattleView,
    cond: int,
    orig_idx: int,
    move_slots: tuple[MoveView | None, ...],
    row: np.ndarray,
    level_stats: tuple[float, ...],
    stat_provenance: StatProvenance,
    active_idx: int | None = None,
    is_opponent: bool = False,
) -> None:
    row[cond + 1] = 1.0

    if pokemon is None:
        return

    row[NUM_IDX_HP_FRACTION] = float(pokemon.current_hp_fraction)

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
    mega_known = active_idx is None or battle.decision.slots[active_idx].legality_known
    row[NUM_IDX_CAN_MEGA] = _can_mega(pokemon, battle, active_idx) if mega_known else 0.0
    row[31] = _is_mega_form(pokemon)

    last_move_id = None
    custom_last_move = battle.last_move(pokemon)
    if custom_last_move:
        last_move_id = PokemonTokenizer.normalize_id(custom_last_move)
    elif pokemon.last_move:
        last_move_id = pokemon.last_move.id

    if cond == 1 and last_move_id:
        for move_idx, move in enumerate(move_slots):
            if move is not None and move.id == last_move_id:
                row[32 + move_idx] = 1.0
                break

    row[NUM_IDX_STATUS_COUNTER] = min(pokemon.status_counter, 5) / 5.0

    row[NUM_IDX_PREPARING] = pokemon.preparing

    row[NUM_IDX_LEVEL_STATS : NUM_IDX_LEVEL_STATS + 6] = [stat / 300.0 for stat in level_stats]
    row[NUM_IDX_STAT_PROVENANCE] = float(stat_provenance == StatProvenance.KNOWN)

    # action legality (allies only, the action mask is otherwise invisible to the
    # network, hiding choice lock / disable / trapping / force switches)
    if active_idx is not None and not is_opponent and not battle.teampreview:
        move_legal, can_switch_out, legality_known = _ally_legality(battle, active_idx, move_slots)
        row[NUM_IDX_MOVE_LEGAL : NUM_IDX_MOVE_LEGAL + MOVE_SLOTS] = move_legal
        row[NUM_IDX_CAN_SWITCH_OUT] = can_switch_out
        row[NUM_IDX_LEGALITY_UNKNOWN] = float(not legality_known)

    row[NUM_IDX_REVEALED] = pokemon.revealed


def _global_field_token_into(
    battle: Any,
    tok: PokemonTokenizer,
    categorical: np.ndarray,
    numerical: np.ndarray,
) -> None:
    if not battle.weather and not battle.fields:
        numerical[2] = float(battle.teampreview)
        numerical[3] = battle.turn / 24.0
        return
    effects = []
    for weather, start_turn in battle.weather.items():
        effects.append(
            (
                EffectNamespace.WEATHER,
                tok.weathers[weather],
                CounterKind.TURN_AGE,
                float(max(0, battle.turn - start_turn)),
                0.0,
                False,
                0.0,
            )
        )
    for field, start_turn in battle.fields.items():
        remaining_known = field.name in _KNOWN_DURATION_FIELD_NAMES
        age = float(max(0, battle.turn - start_turn))
        effects.append(
            (
                EffectNamespace.FIELD,
                tok.fields[field],
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
    battle: Any,
    conditions: Mapping[Any, int],
    tok: PokemonTokenizer,
    fainted_count: int,
    mega_available: bool,
    cat: np.ndarray,
    num: np.ndarray,
) -> None:
    if not conditions:
        num[3] = float(fainted_count) / float(TEAM_SIZE)
        num[4] = float(mega_available)
        return
    effects = []
    for condition, stored_value in conditions.items():
        stackable = condition.name in _STACKABLE_SIDE_CONDITION_NAMES
        effect_id = int(tok.side_conditions[condition])
        remaining_known = condition.name == "TAILWIND"
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


def _write_spatial_events(
    battle: BattleView,
    out: StructuredObservation,
) -> None:
    spatial_cat = out.spatial_cat.numpy()
    spatial_num = out.spatial_num.numpy()
    spatial_cat.fill(0)
    spatial_num.fill(0)

    turn_records = getattr(battle, "spatial_turn", None)
    if turn_records is None:
        return

    for slot_idx, record in enumerate(turn_records[:SPATIAL_SLOT_COUNT]):
        spatial_cat[slot_idx] = (
            record.action_type,
            record.move_id,
            record.target_slot,
        )
        spatial_num[slot_idx] = (
            record.order_rank,
            record.hp_delta,
            record.damage_dealt,
            record.net_boost_delta,
            record.landed_crit,
            record.took_crit,
            record.move_failed,
            record.item_consumed,
        )


def _write_observation(
    battle: BattleView,
    out: StructuredObservation,
    tok: PokemonTokenizer,
    stat_overrides: Mapping[Any, tuple[int, int, int, int, int, int] | None] | None = None,
) -> None:
    """Serialize a battle view into a pre-allocated StructuredObservation in place.

    Arguments:
      battle: battle view providing the entity, categorical, and numerical features
      out: pre-allocated observation buffer to populate in place
      tok: tokenizer mapping string identifiers to categorical vocabulary indices
      stat_overrides: optional per-species base stat overrides keyed by species identifier

    Returns:
      None; writes directly into the out buffers
    """
    token_types = out.token_type_ids.numpy()
    sides = out.side_ids.numpy()
    slots = out.slot_ids.numpy()
    categorical = out.categorical.numpy()
    numerical = out.numerical.numpy()

    token_types[:] = _TOKEN_TYPE_LAYOUT
    sides[:] = _SIDE_LAYOUT
    slots[:] = _SLOT_LAYOUT
    categorical.fill(0)
    numerical.fill(0)

    selected_allies = set(_selected_ally_pokemon(battle))
    ally_orig_idx = {mon: i for i, mon in enumerate(battle.team.values())}
    opponent_orig_idx = {mon: i for i, mon in enumerate(battle.opponent_team.values())}
    stat_cache = battle.stat_cache

    pokemon_to_slot = {}

    idx = 0
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
                precomputed = _resolve_stats(mon, is_opponent, stat_cache, stat_overrides)
                level_stats, stat_prov = _get_pokemon_level_stats(mon, is_opponent, precomputed)
            else:
                level_stats, stat_prov = (0.0,) * 6, StatProvenance.PAD

            move_slots = _iter_move_slots(mon)
            _pokemon_categorical_into(
                mon, tok, move_slots, categorical[idx], cond=cond, stat_provenance=stat_prov
            )
            _pokemon_numeric_into(
                mon,
                battle,
                cond,
                orig_idx,
                move_slots,
                numerical[idx],
                level_stats=level_stats,
                stat_provenance=stat_prov,
                active_idx=active_idx,
                is_opponent=is_opponent,
            )
            _pokemon_effects_into(mon, tok, categorical[idx], numerical[idx])
            idx += 1

    _global_field_token_into(battle, tok, categorical[idx], numerical[idx])
    idx += 1

    ally_fainted = sum(mon.fainted for mon in battle.team.values())
    ally_mega_available, ally_mega_known = _side_mega_available(
        battle,
        is_opponent=False,
        selected_allies=selected_allies,
    )
    _side_token_into(
        battle,
        battle.side_conditions,
        tok,
        ally_fainted,
        ally_mega_available,
        categorical[idx],
        numerical[idx],
    )
    # Mega availability is legality-shaped: the same gate marks it unproven.
    numerical[idx][NUM_IDX_LEGALITY_UNKNOWN] = float(not ally_mega_known)
    # The joint action-mask token is decision-level, so its provenance lives on the ally
    # side row rather than on an active Pokemon row that may be an empty placeholder.
    if not battle.teampreview:
        for position, slot in enumerate(battle.decision.slots):
            numerical[idx][NUM_IDX_SLOT_LEGALITY_UNKNOWN + position] = float(
                not slot.legality_known
            )
    idx += 1

    opp_fainted = sum(mon.fainted for mon in battle.opponent_team.values())
    opp_mega_available, _ = _side_mega_available(battle, is_opponent=True)
    _side_token_into(
        battle,
        battle.opponent_side_conditions,
        tok,
        opp_fainted,
        opp_mega_available,
        categorical[idx],
        numerical[idx],
    )
    idx += 1

    if idx != SEQUENCE_LENGTH:
        raise RuntimeError(f"Structured observation length drifted to {idx}")

    _write_spatial_events(battle, out)


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
            "spatial_cat",
            out.spatial_cat,
            (SPATIAL_SLOT_COUNT, SPATIAL_CATEGORICAL_WIDTH),
            torch.long,
        ),
        (
            "spatial_num",
            out.spatial_num,
            (SPATIAL_SLOT_COUNT, SPATIAL_NUMERICAL_WIDTH),
            torch.float32,
        ),
    )
    for name, tensor, shape, dtype in expected:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"ObservationBuilder.build_into requires CPU output tensors; "
                f"{name} is on {tensor.device}."
            )
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                f"Invalid {name}: expected shape {shape} and dtype {dtype}, "
                f"got shape {tuple(tensor.shape)} and dtype {tensor.dtype}."
            )


class ObservationBuilder:
    """One resource-bound orchestrator for live and reconstructed battle views."""

    def __init__(
        self,
        resources: RuntimeResources,
    ):
        self.resources = resources
        self.tokenizer = resources.tokenizer

    def build_into(
        self,
        battle: BattleView,
        out: StructuredObservation,
        stat_overrides: Mapping[Any, tuple[int, int, int, int, int, int] | None] | None = None,
    ) -> None:
        self.validate_output(out)
        self.build_into_prevalidated(battle, out, stat_overrides)

    def build_into_prevalidated(
        self,
        battle: BattleView,
        out: StructuredObservation,
        stat_overrides: Mapping[Any, tuple[int, int, int, int, int, int] | None] | None = None,
    ) -> None:
        _write_observation(battle, out, self.tokenizer, stat_overrides)

    @staticmethod
    def validate_output(out: StructuredObservation) -> None:
        _validate_output(out)

    def build(
        self,
        battle: BattleView,
        stat_overrides: Mapping[Any, tuple[int, int, int, int, int, int] | None] | None = None,
    ) -> StructuredObservation:
        obs = StructuredObservation.empty_batch(1)[0]
        self.build_into_prevalidated(battle, obs, stat_overrides)
        return obs
