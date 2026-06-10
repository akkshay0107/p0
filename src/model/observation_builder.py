from __future__ import annotations

import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

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

# mega stones end in -ite(x/y) (z soon)
# replacing the substring check for mega items
_MEGA_ITEMS = frozenset(
    item
    for item in tokenizer.items
    if item.endswith("ite") or item.endswith("itex") or item.endswith("itey")
)

# Nature stat impacts: (boosted_stat, hindered_stat)
# 1: Atk, 2: Def, 3: SpA, 4: SpD, 5: Spe
nature_impacts = {
    "adamant": (1, 3),
    "brave": (1, 5),
    "lonely": (1, 2),
    "naughty": (1, 4),
    "bold": (2, 1),
    "relaxed": (2, 5),
    "impish": (2, 3),
    "lax": (2, 4),
    "modest": (3, 1),
    "quiet": (3, 5),
    "mild": (3, 2),
    "rash": (3, 4),
    "calm": (4, 1),
    "gentle": (4, 2),
    "sassy": (4, 5),
    "careful": (4, 3),
    "timid": (5, 1),
    "hasty": (5, 2),
    "jolly": (5, 3),
    "naive": (5, 4),
}


def _get_turns_left(battle: DoubleBattle, start_turn: int, duration: int = 5) -> float:
    if start_turn < 0:
        return 0.0
    return max(0.0, duration - (battle.turn - start_turn)) / float(duration)


def _safe_fraction(num: float | int | None, den: float | int | None) -> float:
    if not den:
        return 0.0
    return (num or 0) / den


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
        tok.nature_id(pokemon),
    ]


def _estimate_stat_by_nature(pokemon: Pokemon, battle: DoubleBattle):
    evs = [0] * 6
    ivs = [31] * 6
    nature = pokemon.nature or "serious"

    if nature in nature_impacts:
        boosted, _ = nature_impacts[nature]
        if boosted in (1, 3, 5):  # Attacking or Speed
            evs[boosted] = 252
            if boosted == 5:  # Speed boost; pick best attack stat
                if pokemon.base_stats["atk"] >= pokemon.base_stats["spa"]:
                    evs[1] = 252
                else:
                    evs[3] = 252
            else:
                evs[5] = 252
            evs[0] = 4
        elif boosted in (2, 4):  # Defensive
            evs[boosted] = 252
            evs[0] = 252
            evs[5] = 4
    else:
        # assume 252 hp and nothing else
        evs[0] = 252

    gen_data = GenData.from_gen(battle.gen)
    return compute_raw_stats(
        pokemon.species,
        evs,
        ivs,
        pokemon.level or 50,
        nature,
        gen_data,
    )


def _get_pokemon_level_stats(
    pokemon: Pokemon, battle: DoubleBattle, is_opponent: bool
) -> tuple[list[float], float]:
    stats = pokemon.stats
    if not is_opponent and stats is not None:
        values = [stats.get(key) for key in ("hp", "atk", "def", "spa", "spd", "spe")]
        if all(value is not None for value in values):
            return [float(value) for value in values], 1.0  # type: ignore

    raw_stats = _estimate_stat_by_nature(pokemon, battle)
    return [float(x) for x in raw_stats], 0.0


def _pokemon_numeric(
    pokemon: Pokemon | None,
    battle: DoubleBattle,
    cond: int,
    orig_idx: int,
    move_slots: list[Move | None],
    active_idx: int | None = None,
    is_opponent: bool = False,
) -> list[float]:
    row = [0.0] * NUMERICAL_WIDTH
    row[cond + 1] = 1.0

    if pokemon is None:
        return row

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

    if cond == 1 and pokemon.last_move:
        last_move_id = pokemon.last_move.id
        for move_idx, move in enumerate(move_slots):
            if move is not None and move.id == last_move_id:
                row[32 + move_idx] = 1.0
                break

    row[36] = min(pokemon.status_counter, 5) / 5.0

    for i, effect in enumerate(_VOLATILE_ORDER):
        val = pokemon.effects.get(effect, 0)
        max_dur = _VOLATILE_MAX_DURATIONS[effect]
        row[37 + i] = min(val, max_dur) / max_dur

    row[42] = pokemon.preparing

    level_stats, stats_exact = _get_pokemon_level_stats(pokemon, battle, is_opponent)
    row[43] = level_stats[0] / 300.0
    row[44] = level_stats[1] / 300.0
    row[45] = level_stats[2] / 300.0
    row[46] = level_stats[3] / 300.0
    row[47] = level_stats[4] / 300.0
    row[48] = level_stats[5] / 300.0
    row[49] = stats_exact

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

    return row


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


def _get_ordered_pokemon(
    battle: DoubleBattle, is_opponent: bool, possible_switches: set[Pokemon] | None = None
) -> list[tuple[Pokemon | None, int, int | None]]:
    # returns list of (pokemon, orig_id, active_id)
    active = battle.opponent_active_pokemon if is_opponent else battle.active_pokemon
    team = battle.opponent_team if is_opponent else battle.team

    # both sides: dict insertion order is stable across the battle, so this gives
    # every mon a persistent identity even as the active-first ordering reshuffles
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

    if possible_switches is None:
        possible_switches = {mon for switches in battle.available_switches for mon in switches}

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
    if mon.fainted:
        return 3
    if seq_idx < 2:
        return 1
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
        battle.turn / 24.0,
    ]
    return categorical, numerical


def _side_token(
    battle: DoubleBattle,
    conditions: dict[SideCondition, int],
    tok: PokemonTokenizer,
    fainted_count: int,
    mega_available: bool,
) -> tuple[list[int], list[float]]:
    cat = [0] * CATEGORICAL_WIDTH
    num = [0.0] * 5

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
    num[4] = float(mega_available)

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
            numerical[idx] = _pokemon_numeric(
                mon, battle, cond, orig_idx, move_slots, active_idx, is_opponent=is_opponent
            )
            idx += 1

    global_cat, global_num = _global_field_token(battle, tok)
    token_types[idx] = TokenType.FIELD_SUPER
    sides[idx] = SideId.NONE
    categorical[idx] = global_cat
    idx += 1
    token_types[idx] = TokenType.FIELD_NUMERIC
    sides[idx] = SideId.NONE
    numerical[idx] = global_num + [0.0] * (NUMERICAL_WIDTH - len(global_num))
    idx += 1

    ally_fainted = sum([1 for mon in battle.team.values() if mon.fainted])
    # mega forms persist after fainting, so "any mega form on the team" == "mega used"
    ally_mega_available = not any(_is_mega_form(mon) for mon in battle.team.values())
    ally_cat, ally_num = _side_token(
        battle, battle.side_conditions, tok, ally_fainted, ally_mega_available
    )
    token_types[idx] = TokenType.FIELD_SUPER
    sides[idx] = SideId.ALLY
    categorical[idx] = ally_cat
    idx += 1
    token_types[idx] = TokenType.FIELD_NUMERIC
    sides[idx] = SideId.ALLY
    numerical[idx] = ally_num + [0.0] * (NUMERICAL_WIDTH - len(ally_num))
    idx += 1

    opp_fainted = sum([1 for mon in battle.opponent_team.values() if mon.fainted])
    opp_mega_available = not any(_is_mega_form(mon) for mon in battle.opponent_team.values())
    opp_cat, opp_num = _side_token(
        battle, battle.opponent_side_conditions, tok, opp_fainted, opp_mega_available
    )
    token_types[idx] = TokenType.FIELD_SUPER
    sides[idx] = SideId.OPPONENT
    categorical[idx] = opp_cat
    idx += 1
    token_types[idx] = TokenType.FIELD_NUMERIC
    sides[idx] = SideId.OPPONENT
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
    return PokemonTokenizer.normalize_id(item) in _MEGA_ITEMS and not _is_mega_form(pokemon)
