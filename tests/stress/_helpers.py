"""Shared scale controls for opt-in stress tests."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from p0.format_config import FORMAT
from p0.model.resources import default_runtime_resources
from p0.paths import DEFAULT_PATHS
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
    catalog_path = DEFAULT_PATHS.data_root / "stress_team_catalog.json"
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
    """
    Sample random EV point allocations respecting the maximum total EV budget constraint.

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
    """
    Calculate the deterministic grouping identity used by the replay compiler.

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
    """
    Build a replay-shaped payload with random values from the active legal dex.

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
    """
    Build paired games (game 1 & game 2) per series while preserving each series' team roster.

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


def stress_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    raw_value: str | None = None,
) -> int:
    """Read a positive integer stress-test control from the environment."""
    configured_value = os.getenv(name, str(default)) if raw_value is None else raw_value
    value = int(configured_value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def stress_batch_sizes(raw_value: str | None = None) -> tuple[int, ...]:
    """Return configured model batch sizes, preserving declaration order."""
    configured_value = os.getenv("P0_STRESS_BATCHES", "1,8,32") if raw_value is None else raw_value
    values = tuple(int(value.strip()) for value in configured_value.split(",") if value.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError("P0_STRESS_BATCHES must contain positive integers")
    return tuple(dict.fromkeys(values))


def stress_devices(raw_value: str | None = None) -> tuple[torch.device, ...]:
    """Return requested stress-test devices, filtering unavailable CUDA and duplicates."""
    configured_value = (
        os.getenv("P0_STRESS_DEVICES", "cpu,cuda") if raw_value is None else raw_value
    )
    requested = tuple(
        value.strip().lower() for value in configured_value.split(",") if value.strip()
    )
    devices: list[torch.device] = []
    for name in requested:
        if name == "cpu":
            devices.append(torch.device("cpu"))
        elif name == "cuda" and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        elif name != "cuda":
            raise ValueError(f"Unsupported stress-test device {name!r}")
    if not devices:
        return (torch.device("cpu"),)
    return tuple(dict.fromkeys(devices))


def stress_repetitions(default: int = 32) -> int:
    """Return the number of repeated operations for lifecycle tests."""
    return stress_int("P0_STRESS_REPETITIONS", default)


def stress_count(name: str, default: int) -> int:
    """Return a configurable workload count."""
    return stress_int(name, default)


def stress_rng(name: str = "P0_STRESS_SEED", default: int = 20260805) -> random.Random:
    """Return a deterministic RNG for randomized stress inputs."""
    return random.Random(stress_int(name, default, minimum=0))
