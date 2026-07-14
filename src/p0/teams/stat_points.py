"""Champions Stat Point calculation and deterministic spread imputation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Mapping

STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
STAT_POINT_LIMIT = 32
STAT_POINT_TOTAL_LIMIT = 66
STAT_POINT_IMPUTER_VERSION = 1

NATURE_IMPACTS: dict[str, tuple[str, str]] = {
    "adamant": ("atk", "spa"),
    "brave": ("atk", "spe"),
    "lonely": ("atk", "def"),
    "naughty": ("atk", "spd"),
    "bold": ("def", "atk"),
    "relaxed": ("def", "spe"),
    "impish": ("def", "spa"),
    "lax": ("def", "spd"),
    "modest": ("spa", "atk"),
    "quiet": ("spa", "spe"),
    "mild": ("spa", "def"),
    "rash": ("spa", "spd"),
    "calm": ("spd", "atk"),
    "gentle": ("spd", "def"),
    "sassy": ("spd", "spe"),
    "careful": ("spd", "spa"),
    "timid": ("spe", "atk"),
    "hasty": ("spe", "def"),
    "jolly": ("spe", "spa"),
    "naive": ("spe", "spd"),
}


@dataclass(frozen=True, slots=True)
class StatPoints:
    hp: int = 0
    atk: int = 0
    defense: int = 0
    spa: int = 0
    spd: int = 0
    spe: int = 0

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(type(value) is not int for value in values):
            raise TypeError("Stat Points must be integers")
        if any(not 0 <= value <= STAT_POINT_LIMIT for value in values):
            raise ValueError("Each Stat Point value must be in [0, 32]")
        if sum(values) > STAT_POINT_TOTAL_LIMIT:
            raise ValueError("A Stat Point spread may use at most 66 points")

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.hp, self.atk, self.defense, self.spa, self.spd, self.spe)

    def as_dict(self) -> dict[str, int]:
        return dict(zip(STAT_NAMES, self.as_tuple(), strict=True))


@dataclass(frozen=True, slots=True)
class BaseStats:
    hp: int
    atk: int
    defense: int
    spa: int
    spd: int
    spe: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in self.as_tuple()):
            raise ValueError("Base stats must be positive integers")

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.hp, self.atk, self.defense, self.spa, self.spd, self.spe)

    @classmethod
    def from_mapping(cls, stats: Mapping[str, int]) -> BaseStats:
        try:
            return cls(*(int(stats[name]) for name in STAT_NAMES))
        except KeyError as exc:
            raise ValueError(f"Missing base stat: {exc.args[0]}") from exc


def _modify_nature(stat: int, stat_name: str, nature: str) -> int:
    impact = NATURE_IMPACTS.get(nature.lower())
    if impact is None:
        return stat
    if impact[0] == stat_name:
        return stat * 110 // 100
    if impact[1] == stat_name:
        return stat * 90 // 100
    return stat


@lru_cache(maxsize=8192)
def calculate_stats(
    base_stats: BaseStats,
    points: StatPoints,
    nature: str,
    level: int = 50,
) -> tuple[int, int, int, int, int, int]:
    """Match the pinned Champions level-clause statModify implementation."""
    if not 1 <= level <= 100:
        raise ValueError("Level must be in [1, 100]")
    result: list[int] = []
    for stat_name, base, stat_points in zip(
        STAT_NAMES, base_stats.as_tuple(), points.as_tuple(), strict=True
    ):
        iv_contribution = max(2 * stat_points - 1, 0)
        stat = (2 * base + 31 + iv_contribution) * level // 100
        if stat_name == "hp":
            result.append(stat + level + 10)
        else:
            result.append(_modify_nature(stat + 5, stat_name, nature))
    return tuple(result)  # type: ignore[return-value]


class Role(StrEnum):
    PHYSICAL = "physical-attacker"
    SPECIAL = "special-attacker"
    MIXED = "mixed-attacker"
    SPEED_CONTROL = "speed-control"
    TRICK_ROOM = "trick-room"
    SUPPORT = "support"
    BULKY_SETUP = "bulky-setup"


@dataclass(frozen=True, slots=True)
class ImputationInput:
    species: str
    nature: str
    item: str
    ability: str
    moves: tuple[str, ...]
    move_categories: tuple[str, ...]
    base_stats: BaseStats
    level: int = 50

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError("Species is required for Stat Point imputation")
        if len(self.moves) != len(self.move_categories):
            raise ValueError("Moves and move categories must be aligned")


@dataclass(frozen=True, slots=True)
class SpreadCandidate:
    points: StatPoints
    weight: int
    role: Role


@dataclass(frozen=True, slots=True)
class PrecomputedStats:
    """Six exact level stats prepared outside observation construction."""

    values: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if len(self.values) != len(STAT_NAMES) or any(value <= 0 for value in self.values):
            raise ValueError("Precomputed stats must contain six positive values")


_SPEED_CONTROL = frozenset({"tailwind", "icywind", "electroweb", "trickroom"})
_SUPPORT = frozenset(
    {
        "protect",
        "detect",
        "wideguard",
        "fakeout",
        "followme",
        "ragepowder",
        "helpinghand",
        "spore",
        "willowisp",
        "recover",
        "roost",
        "slackoff",
    }
)
_SETUP = frozenset({"calmmind", "bulkup", "coil", "curse", "irondefense", "nastyplot"})
_RECOVERY = frozenset({"recover", "roost", "slackoff", "synthesis", "moonlight", "softboiled"})
_PRIORITY = frozenset(
    {"aquajet", "bulletpunch", "extremespeed", "fakeout", "iceshard", "suckerpunch"}
)
_SPEED_ABILITIES = frozenset({"chlorophyll", "sandrush", "swiftswim", "surgesurfer"})
_OFFENSE_ITEMS = frozenset({"choiceband", "choicespecs", "lifeorb"})


def _normalized_id(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def classify_role(value: ImputationInput) -> Role:
    moves = {_normalized_id(move) for move in value.moves}
    physical = sum(category.lower() == "physical" for category in value.move_categories)
    special = sum(category.lower() == "special" for category in value.move_categories)
    if "trickroom" in moves or value.nature.lower() in {"brave", "quiet", "relaxed", "sassy"}:
        return Role.TRICK_ROOM
    if moves & _SPEED_CONTROL:
        return Role.SPEED_CONTROL
    if moves & _SETUP:
        return Role.BULKY_SETUP
    if len(moves & _SUPPORT) >= 2 and physical + special <= 1:
        return Role.SUPPORT
    if physical and special:
        return Role.MIXED
    if physical:
        return Role.PHYSICAL
    if special:
        return Role.SPECIAL
    return Role.SUPPORT


def _candidate_weight(
    value: ImputationInput, points: StatPoints, role: Role, base_weight: int
) -> int:
    allocation = points.as_dict()
    moves = {_normalized_id(move) for move in value.moves}
    item = _normalized_id(value.item)
    ability = _normalized_id(value.ability)
    boosted = NATURE_IMPACTS.get(value.nature.lower(), ("", ""))[0]
    score = base_weight + allocation.get(boosted, 0)
    if item in _OFFENSE_ITEMS:
        score += max(allocation["atk"], allocation["spa"])
    if ability in _SPEED_ABILITIES:
        score += allocation["spe"]
    if moves & _PRIORITY:
        score += allocation["hp"] // 2
    if moves & _RECOVERY or role in {Role.SUPPORT, Role.BULKY_SETUP}:
        score += (allocation["hp"] + allocation["def"] + allocation["spd"]) // 3
    if role == Role.TRICK_ROOM:
        score += STAT_POINT_LIMIT - allocation["spe"]
    defense_total = value.base_stats.defense + value.base_stats.spd
    offense_total = value.base_stats.atk + value.base_stats.spa
    if defense_total > offense_total:
        score += allocation["hp"] // 2
    return max(1, score)


def impute_candidates(value: ImputationInput) -> tuple[SpreadCandidate, ...]:
    """Return a small deterministic set of legal, weighted candidate spreads."""
    role = classify_role(value)
    attack = "atk" if value.base_stats.atk >= value.base_stats.spa else "spa"
    if role == Role.PHYSICAL:
        attack = "atk"
    elif role == Role.SPECIAL:
        attack = "spa"

    def spread(**points: int) -> StatPoints:
        fields = {"def": "defense"}
        return StatPoints(**{fields.get(key, key): amount for key, amount in points.items()})

    if role == Role.TRICK_ROOM:
        shapes = (
            (spread(hp=32, **{attack: 32}, defense=2), 100),
            (spread(hp=32, defense=17, spd=17), 55),
        )
    elif role in {Role.SUPPORT, Role.SPEED_CONTROL, Role.BULKY_SETUP}:
        shapes = (
            (spread(hp=32, defense=17, spd=17), 100),
            (spread(hp=32, spe=32, defense=2), 65),
        )
    elif role == Role.MIXED:
        shapes = (
            (spread(atk=32, spa=32, hp=2), 100),
            (spread(hp=32, atk=17, spa=17), 60),
        )
    else:
        shapes = (
            (spread(**{attack: 32}, spe=32, hp=2), 100),
            (spread(hp=32, **{attack: 32}, defense=2), 55),
        )
    return tuple(
        SpreadCandidate(points, _candidate_weight(value, points, role, weight), role)
        for points, weight in shapes
    )


def select_candidate(value: ImputationInput, seed: int | None = None) -> SpreadCandidate:
    candidates = impute_candidates(value)
    if seed is None:
        return candidates[0]
    weights = [item.weight for item in candidates]
    return random.Random(seed).choices(candidates, weights=weights, k=1)[0]


@lru_cache(maxsize=8192)
def imputed_stats(value: ImputationInput) -> PrecomputedStats:
    candidate = select_candidate(value)
    stats = calculate_stats(value.base_stats, candidate.points, value.nature, value.level)
    return PrecomputedStats(stats)
