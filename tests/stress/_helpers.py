"""Shared scale controls for opt-in stress tests."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from p0.battle.events import RawBattleEvent
from p0.format_config import FORMAT
from p0.model.resources import default_runtime_resources
from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamRecord


@dataclass(frozen=True, slots=True)
class StressSpeciesPool:
    """Validated choices attached to one legal, sendable species form."""

    species: str
    base_species: str
    items: tuple[str, ...]
    abilities: tuple[str, ...]
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StressDexCatalog:
    """Format-legal values and species-aware pools used by stress inputs."""

    species: tuple[str, ...]
    items: tuple[str, ...]
    abilities: tuple[str, ...]
    moves: tuple[str, ...]
    natures: tuple[str, ...]
    species_pools: tuple[StressSpeciesPool, ...]


def _legal_display_names(dex: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    entries = dex[kind]
    legal_ids = frozenset(str(value) for value in dex["legality"][kind])
    return tuple(
        str(entry["name"])
        for entry in entries
        if isinstance(entry, Mapping) and str(entry.get("id", "")) in legal_ids
    )


def _load_species_pools() -> tuple[StressSpeciesPool, ...]:
    """Load species-specific ability and learnset pools from the stress catalog."""
    catalog_path = Path(__file__).parent / "stress_team_catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Stress team catalog JSON not found: {catalog_path}")

    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Stress team catalog contains malformed JSON") from exc

    if not isinstance(payload, Mapping) or payload.get("schemaVersion") != 1:
        raise RuntimeError("Stress team catalog has an unsupported schema")
    if payload.get("format") != FORMAT.bo3_format:
        raise RuntimeError("Stress team catalog was built for the wrong format")

    raw_pools = payload.get("species")
    if not isinstance(raw_pools, list):
        raise RuntimeError("Stress team catalog has no species pools")

    pools: list[StressSpeciesPool] = []
    for raw_pool in raw_pools:
        if not isinstance(raw_pool, Mapping):
            raise RuntimeError("Stress team catalog contains a malformed species pool")

        values: dict[str, tuple[str, ...]] = {}
        for field in ("items", "abilities", "moves"):
            raw_values = raw_pool.get(field)
            if not isinstance(raw_values, list) or not all(
                isinstance(value, str) and value for value in raw_values
            ):
                raise RuntimeError(f"Stress team catalog has an invalid {field} pool")
            values[field] = tuple(raw_values)

        species = raw_pool.get("name")
        base_species = raw_pool.get("baseSpecies")
        if not isinstance(species, str) or not species:
            raise RuntimeError("Stress team catalog has a species without a name")
        if not isinstance(base_species, str) or not base_species:
            raise RuntimeError(f"Stress team catalog has no base species for {species}")
        if len(set(values["items"])) != len(values["items"]):
            raise RuntimeError(f"Stress team catalog repeats items for {species}")
        if len(set(values["moves"])) != len(values["moves"]):
            raise RuntimeError(f"Stress team catalog repeats moves for {species}")

        pools.append(
            StressSpeciesPool(
                species=species,
                base_species=base_species,
                items=values["items"],
                abilities=values["abilities"],
                moves=values["moves"],
            )
        )

    if len({pool.species for pool in pools}) != len(pools):
        raise RuntimeError("Stress team catalog repeats species forms")
    if len({pool.base_species for pool in pools}) < 6:
        raise RuntimeError("Stress team catalog has fewer than six base species")
    return tuple(pools)


@lru_cache(maxsize=1)
def stress_dex_catalog() -> StressDexCatalog:
    """Load the active format's legal dex values once for all stress generators."""
    dex = default_runtime_resources().dex
    species_pools = _load_species_pools()

    catalog = StressDexCatalog(
        species=tuple(pool.species for pool in species_pools),
        items=_legal_display_names(dex, "items"),
        abilities=_legal_display_names(dex, "abilities"),
        moves=_legal_display_names(dex, "moves"),
        natures=_legal_display_names(dex, "natures"),
        species_pools=species_pools,
    )
    if not all((catalog.species, catalog.items, catalog.abilities, catalog.moves, catalog.natures)):
        raise AssertionError("The active format dex has an empty legal category")
    return catalog


def _random_stat_points(rng: random.Random) -> StatPoints:
    """Sample random EV point allocations respecting the maximum total EV budget constraint.

    Generates 6 stat points in [0, 32] such that their sum does not exceed 66 total points
    (representing the 508 EV cap in Showdown VGC / Singles).
    """
    while True:
        values = [rng.randrange(33) for _ in range(6)]
        if sum(values) <= 66 and any(values):
            return StatPoints(
                hp=values[0],
                atk=values[1],
                defense=values[2],
                spa=values[3],
                spd=values[4],
                spe=values[5],
            )


def _assign_unique_items(
    rng: random.Random,
    pools: tuple[StressSpeciesPool, ...],
) -> tuple[str, ...]:
    """Find one distinct admitted item for every selected species pool."""
    if len(pools) != 6:
        raise ValueError(f"Expected six species pools, got {len(pools)}")

    # Process the most constrained pools first. The augmenting-path assignment
    # makes item uniqueness a solved constraint, not a reason to discard a team.
    order = list(range(len(pools)))
    rng.shuffle(order)
    order.sort(key=lambda index: len(pools[index].items))
    item_owner: dict[str, int] = {}
    assigned: list[str | None] = [None] * len(pools)

    def assign(pool_index: int, seen: set[str]) -> bool:
        candidates = list(pools[pool_index].items)
        rng.shuffle(candidates)
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            previous_owner = item_owner.get(item)
            if previous_owner is None or assign(previous_owner, seen):
                item_owner[item] = pool_index
                assigned[pool_index] = item
                return True
        return False

    for pool_index in order:
        if not assign(pool_index, set()):
            names = ", ".join(pool.species for pool in pools)
            raise RuntimeError(f"No six-item matching exists for validated pools: {names}")

    if any(item is None for item in assigned):
        raise AssertionError("Item matching completed without assigning every team member")
    return tuple(item for item in assigned if item is not None)


def _random_team_members(rng: random.Random) -> tuple[TeamMember, ...]:
    """Sample six Showdown-valid members from distinct Species Clause groups."""
    catalog = stress_dex_catalog()
    by_base_species: dict[str, list[StressSpeciesPool]] = {}
    for pool in catalog.species_pools:
        by_base_species.setdefault(pool.base_species, []).append(pool)

    selected_bases = rng.sample(tuple(by_base_species), 6)
    selected_pools = tuple(
        rng.choice(by_base_species[base_species]) for base_species in selected_bases
    )
    items = _assign_unique_items(rng, selected_pools)

    return tuple(
        TeamMember(
            species=pool.species,
            item=items[index],
            ability=rng.choice(pool.abilities),
            moves=tuple(
                rng.sample(
                    pool.moves,
                    1 if len(pool.moves) == 1 else rng.randint(2, min(4, len(pool.moves))),
                )
            ),
            nature=rng.choice(catalog.natures),
            level=50,
        )
        for index, pool in enumerate(selected_pools)
    )


def stress_random_team_record(rng: random.Random, *, label: str = "stress") -> TeamRecord:
    """Build a six-member team admitted by the active format's species-aware dex pools."""
    members = _random_team_members(rng)
    return TeamRecord(
        team=CanonicalTeam(members),
        spreads=tuple(_random_stat_points(rng) for _ in members),
        metadata=TeamMetadata(
            source_series=(label,),
            source_replays=(f"{label}-game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        ),
    )


def stress_random_replay_teams(
    rng: random.Random,
) -> tuple[tuple[TeamMember, ...], tuple[TeamMember, ...]]:
    """Return two independent six-member teams for a synthetic replay payload."""
    return _random_team_members(rng), _random_team_members(rng)


def stress_series_id(parent: str, players: tuple[str, str] = ("Alice", "Bob")) -> str:
    """Calculate the deterministic grouping identity used by the replay compiler.

    Hashes the format ID, parent series ID, and casefolded player names to produce
    a 24-character hexadecimal series identifier.
    """
    value = "\n".join((FORMAT.bo3_format, parent, *(player.casefold() for player in players)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _showteam_json(team: tuple[TeamMember, ...]) -> str:
    """Serialize team members to Showdown '|showteam|' JSON format."""
    return json.dumps(
        [
            {
                "species": member.species,
                "item": member.item,
                "ability": member.ability,
                "moves": list(member.moves),
                "nature": member.nature,
            }
            for member in team
        ],
        separators=(",", ":"),
    )


def stress_random_replay_payload(
    rng: random.Random,
    replay_id: str,
    *,
    series_id: str,
    game_number: int = 1,
    winner: str | None = None,
    teams: tuple[tuple[TeamMember, ...], tuple[TeamMember, ...]] | None = None,
) -> dict[str, Any]:
    """Build a replay-shaped payload with random values from the active legal dex.

    Emits simulated Showdown protocol lines including teampreview, showteam headers,
    lead switches, double battle moves, and match conclusion.
    """
    if teams is None:
        teams = stress_random_replay_teams(rng)
    first_team, second_team = teams
    players = ("Alice", "Bob")
    winner = winner or rng.choice(players)

    p1a, p1b = first_team[:2]
    p2a, p2b = second_team[:2]
    lines = [
        "|start",
        "|teampreview",
        f"|showteam|p1|{_showteam_json(first_team)}",
        f"|showteam|p2|{_showteam_json(second_team)}",
        "|",
        f"|switch|p1a: {p1a.species}|{p1a.species}, L50|100/100",
        f"|switch|p1b: {p1b.species}|{p1b.species}, L50|100/100",
        f"|switch|p2a: {p2a.species}|{p2a.species}, L50|100/100",
        f"|switch|p2b: {p2b.species}|{p2b.species}, L50|100/100",
        "|turn|1",
        "|",
        f"|move|p1a: {p1a.species}|{p1a.moves[0]}|p2a: {p2a.species}",
        f"|move|p1b: {p1b.species}|{p1b.moves[0]}|p2b: {p2b.species}",
        f"|move|p2a: {p2a.species}|{p2a.moves[0]}|p1a: {p1a.species}",
        f"|move|p2b: {p2b.species}|{p2b.moves[0]}|p1b: {p1b.species}",
        "|",
        f"|win|{winner}",
    ]
    return {
        "id": replay_id,
        "formatid": FORMAT.bo3_format,
        "p1": players[0],
        "p2": players[1],
        "uploadtime": 1_750_000_000 + rng.randrange(1_000_000),
        "roomid": replay_id,
        "parent": series_id,
        "game_number": game_number,
        "log": "\n".join(lines),
    }


def stress_random_replay_payloads(
    rng: random.Random,
    count: int,
    *,
    replay_prefix: str,
    series_prefix: str,
) -> tuple[dict[str, Any], ...]:
    """Build independent one-game payloads for scale tests."""
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    return tuple(
        stress_random_replay_payload(
            rng,
            f"{replay_prefix}-{index}",
            series_id=f"{series_prefix}-{index}",
        )
        for index in range(count)
    )


def stress_random_bo3_payloads(
    rng: random.Random,
    series_count: int,
    *,
    replay_prefix: str,
    series_prefix: str,
) -> tuple[dict[str, Any], ...]:
    """Build paired games (game 1 & game 2) per series while preserving each series' team roster.

    Ensures that both games in a Best-of-3 series share the same team compositions across
    players, mimicking actual tournament series conditions for replay grouping tests.
    """
    if series_count < 1:
        raise ValueError(f"series_count must be positive, got {series_count}")

    payloads: list[dict[str, Any]] = []
    for index in range(series_count):
        # Sample shared teams for this BO3 series
        teams = stress_random_replay_teams(rng)
        series_id = f"{series_prefix}-{index}"
        payloads.extend(
            (
                stress_random_replay_payload(
                    rng,
                    f"{replay_prefix}-{index}-1",
                    series_id=series_id,
                    game_number=1,
                    teams=teams,
                ),
                stress_random_replay_payload(
                    rng,
                    f"{replay_prefix}-{index}-2",
                    series_id=series_id,
                    game_number=2,
                    winner="Bob",
                    teams=teams,
                ),
            )
        )
    return tuple(payloads)


def _random_entity(rng: random.Random) -> str:
    """Generate a random battle slot entity identifier (e.g. 'p1a: Flutter Mane')."""
    return f"{rng.choice(('p1a', 'p1b', 'p2a', 'p2b'))}: {rng.choice(stress_dex_catalog().species)}"


def _random_target(rng: random.Random) -> str:
    """Generate a random move target entity identifier."""
    return _random_entity(rng)


def _random_hp_status(rng: random.Random) -> str:
    """Generate a random Showdown HP status string with optional status condition (e.g. '75/100 par')."""
    value = rng.randrange(1, 101)
    suffix = rng.choice(("", "g", "y"))
    return f"{value}/100{suffix}"


def _random_raw_event(case: str, rng: random.Random) -> RawBattleEvent:
    """Build one valid parser case with fresh identifiers from the active dex."""
    catalog = stress_dex_catalog()
    entity = _random_entity(rng)
    target = _random_target(rng)
    move = rng.choice(catalog.moves)
    item = rng.choice(catalog.items)
    ability = rng.choice(catalog.abilities)
    species = rng.choice(catalog.species)

    common_boost = (
        entity,
        rng.choice(("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")),
        str(rng.randrange(1, 4)),
    )
    status = rng.choice(("par", "slp", "frz", "brn", "psn", "tox"))
    field = rng.choice(("Trick Room", "Electric Terrain", "Grassy Terrain"))
    volatile = rng.choice(("Protect", "Substitute", "Destiny Bond"))

    # Map each parser branch to its corresponding raw protocol event tuple
    cases: dict[str, RawBattleEvent] = {
        "move": RawBattleEvent(("", "move", entity, move, target)),
        "unknown_move": RawBattleEvent(("", "move", entity, "not-a-real-move", target)),
        "switch": RawBattleEvent(("", "switch", entity, f"{species}, L50", _random_hp_status(rng))),
        "drag": RawBattleEvent(("", "drag", entity, f"{species}, L50", _random_hp_status(rng))),
        "swap": RawBattleEvent(("", "swap", entity, target)),
        "faint": RawBattleEvent(("", "faint", entity)),
        "damage": RawBattleEvent(("", "-damage", entity, _random_hp_status(rng)), pre_hp=0.75),
        "damage_missing_pre_hp": RawBattleEvent(("", "-damage", entity, "50/100")),
        "heal": RawBattleEvent(("", "-heal", entity, _random_hp_status(rng)), pre_hp=0.25),
        "boost": RawBattleEvent(("", "-boost", *common_boost)),
        "unboost": RawBattleEvent(("", "-unboost", *common_boost)),
        "status": RawBattleEvent(("", "-status", entity, status)),
        "curestatus": RawBattleEvent(("", "-curestatus", entity, status)),
        "enditem": RawBattleEvent(("", "-enditem", entity, item)),
        "item": RawBattleEvent(("", "-item", entity, item)),
        "item_transfer": RawBattleEvent(("", "-item", entity, item, "[from] move: Trick")),
        "ability": RawBattleEvent(("", "-ability", entity, ability)),
        "weather_start": RawBattleEvent(("", "-weather", "RainDance")),
        "weather_end": RawBattleEvent(("", "-weather", "none")),
        "weather_upkeep": RawBattleEvent(("", "-weather", "RainDance", "[upkeep]")),
        "fieldstart": RawBattleEvent(("", "-fieldstart", f"move: {field}")),
        "fieldend": RawBattleEvent(("", "-fieldend", f"move: {field}")),
        "sidestart": RawBattleEvent(("", "-sidestart", rng.choice(("p1", "p2")), f"move: {move}")),
        "sideend": RawBattleEvent(("", "-sideend", rng.choice(("p1", "p2")), f"move: {move}")),
        "start": RawBattleEvent(("", "-start", entity, f"move: {volatile}")),
        "end": RawBattleEvent(("", "-end", entity, f"move: {volatile}")),
        "formechange": RawBattleEvent(("", "-formechange", entity, species)),
        "detailschange": RawBattleEvent(("", "detailschange", entity, species)),
        "fail": RawBattleEvent(("", "-fail", entity)),
        "immune": RawBattleEvent(("", "-immune", entity)),
        "miss": RawBattleEvent(("", "-miss", entity, target)),
        "activate_protect": RawBattleEvent(("", "-activate", entity, "move: Protect")),
        "activate_ability": RawBattleEvent(("", "-activate", entity, f"ability: {ability}")),
        "activate_item": RawBattleEvent(("", "-activate", entity, f"item: {item}")),
        "activate_effect": RawBattleEvent(("", "-activate", entity, f"move: {volatile}")),
        "crit": RawBattleEvent(("", "-crit", entity)),
        "mega": RawBattleEvent(("", "-mega", entity)),
        "cant_status": RawBattleEvent(("", "cant", entity, status, move)),
        "cant_effect": RawBattleEvent(("", "cant", entity, "flinch", move)),
        "prepare": RawBattleEvent(("", "-prepare", entity, move, target)),
        "singlemove": RawBattleEvent(("", "-singlemove", entity, f"move: {volatile}")),
        "setboost": RawBattleEvent(("", "-setboost", *common_boost)),
        "clearboost": RawBattleEvent(("", "-clearboost", entity)),
        "clearnegativeboost": RawBattleEvent(("", "-clearnegativeboost", entity)),
        "clearpositiveboost": RawBattleEvent(("", "-clearpositiveboost", entity)),
        "clearallboost": RawBattleEvent(("", "-clearallboost")),
        "swapboost": RawBattleEvent(("", "-swapboost", entity, target, "atk, def")),
        "invertboost": RawBattleEvent(("", "-invertboost", entity)),
        "copyboost": RawBattleEvent(("", "-copyboost", entity, target)),
        "transform": RawBattleEvent(("", "-transform", entity, target)),
        "endability": RawBattleEvent(("", "-endability", entity, ability)),
        "fieldactivate": RawBattleEvent(("", "-fieldactivate", f"move: {field}")),
        "notarget": RawBattleEvent(("", "-notarget", entity)),
        "ignored": RawBattleEvent(("", "chat", "ignored")),
    }
    return cases[case]


def stress_random_raw_events(rng: random.Random) -> tuple[RawBattleEvent, ...]:
    """Create a shuffled event corpus covering every parser branch with dex-wide values."""
    cases = (
        "move",
        "unknown_move",
        "switch",
        "drag",
        "swap",
        "faint",
        "damage",
        "damage_missing_pre_hp",
        "heal",
        "boost",
        "unboost",
        "status",
        "curestatus",
        "enditem",
        "item",
        "item_transfer",
        "ability",
        "weather_start",
        "weather_end",
        "weather_upkeep",
        "fieldstart",
        "fieldend",
        "sidestart",
        "sideend",
        "start",
        "end",
        "formechange",
        "detailschange",
        "fail",
        "immune",
        "miss",
        "activate_protect",
        "activate_ability",
        "activate_item",
        "activate_effect",
        "crit",
        "mega",
        "cant_status",
        "cant_effect",
        "prepare",
        "singlemove",
        "setboost",
        "clearboost",
        "clearnegativeboost",
        "clearpositiveboost",
        "clearallboost",
        "swapboost",
        "invertboost",
        "copyboost",
        "transform",
        "endability",
        "fieldactivate",
        "notarget",
        "ignored",
    )
    events = [_random_raw_event(case, rng) for case in cases]
    events.extend(_random_raw_event(rng.choice(cases), rng) for _ in range(rng.randrange(1, 8)))
    rng.shuffle(events)
    return tuple(events)


def stress_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer stress-test control from the environment."""
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def stress_batch_sizes() -> tuple[int, ...]:
    """Return configured model batch sizes, preserving declaration order."""
    values = tuple(
        int(value.strip())
        for value in os.getenv("P0_STRESS_BATCHES", "1,8,32").split(",")
        if value.strip()
    )
    if not values or any(value < 1 for value in values):
        raise ValueError("P0_STRESS_BATCHES must contain positive integers")
    return tuple(dict.fromkeys(values))


def stress_repetitions(default: int = 32) -> int:
    """Return the number of repeated operations for lifecycle tests."""
    return stress_int("P0_STRESS_REPETITIONS", default)


def stress_count(name: str, default: int) -> int:
    """Return a configurable workload count."""
    return stress_int(name, default)


def stress_rng(name: str = "P0_STRESS_SEED", default: int = 20260805) -> random.Random:
    """Return a deterministic RNG for randomized stress inputs."""
    return random.Random(stress_int(name, default, minimum=0))
