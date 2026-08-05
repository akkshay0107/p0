"""Champions Stat Point calculation and deterministic spread imputation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Mapping, NamedTuple

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


class BaseStats(NamedTuple):
    hp: int
    atk: int
    defense: int
    spa: int
    spd: int
    spe: int

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


class SpreadCandidate(NamedTuple):
    points: StatPoints
    weight: int
    role: Role


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


def classify_role(nature: str, moves: tuple[str, ...], move_categories: tuple[str, ...]) -> Role:
    nature_lower = nature.lower()
    moves_set = {_normalized_id(move) for move in moves}
    physical = sum(category.lower() == "physical" for category in move_categories)
    special = sum(category.lower() == "special" for category in move_categories)

    if "trickroom" in moves_set or nature_lower in {"brave", "quiet", "relaxed", "sassy"}:
        return Role.TRICK_ROOM

    if moves_set & _SPEED_CONTROL:
        return Role.SPEED_CONTROL

    if moves_set & _SETUP:
        return Role.BULKY_SETUP

    if len(moves_set & _SUPPORT) >= 2 and physical + special <= 1:
        return Role.SUPPORT

    if physical and special:
        return Role.MIXED

    if physical:
        return Role.PHYSICAL

    if special:
        return Role.SPECIAL

    return Role.SUPPORT


def _candidate_weight(
    nature: str,
    moves: tuple[str, ...],
    item: str,
    ability: str,
    base_stats: BaseStats,
    points: StatPoints,
    role: Role,
    base_weight: int,
) -> int:
    allocation = points.as_dict()
    moves_set = {_normalized_id(move) for move in moves}
    item_norm = _normalized_id(item)
    ability_norm = _normalized_id(ability)
    boosted = NATURE_IMPACTS.get(nature.lower(), ("", ""))[0]
    score = base_weight + allocation.get(boosted, 0)

    if item_norm in _OFFENSE_ITEMS:
        score += max(allocation["atk"], allocation["spa"])

    if ability_norm in _SPEED_ABILITIES:
        score += allocation["spe"]

    if moves_set & _PRIORITY:
        score += allocation["hp"] // 2

    if moves_set & _RECOVERY or role in {Role.SUPPORT, Role.BULKY_SETUP}:
        score += (allocation["hp"] + allocation["def"] + allocation["spd"]) // 3

    if role == Role.TRICK_ROOM:
        score += STAT_POINT_LIMIT - allocation["spe"]

    defense_total = base_stats.defense + base_stats.spd
    offense_total = base_stats.atk + base_stats.spa

    if defense_total > offense_total:
        score += allocation["hp"] // 2

    return max(1, score)


def _spread(**points: int) -> StatPoints:
    """Construct a StatPoints value using Showdown's ``def`` spelling."""
    fields = {"def": "defense"}
    return StatPoints(**{fields.get(key, key): amount for key, amount in points.items()})


def impute_candidates(
    nature: str,
    moves: tuple[str, ...],
    move_categories: tuple[str, ...],
    item: str,
    ability: str,
    base_stats: BaseStats,
) -> tuple[SpreadCandidate, ...]:
    """Return a small deterministic set of legal, weighted candidate spreads."""
    role = classify_role(nature, moves, move_categories)
    attack = "atk" if base_stats.atk >= base_stats.spa else "spa"
    if role == Role.PHYSICAL:
        attack = "atk"
    elif role == Role.SPECIAL:
        attack = "spa"

    if role == Role.TRICK_ROOM:
        shapes = (
            (_spread(hp=32, **{attack: 32}, defense=2), 100),
            (_spread(hp=32, defense=17, spd=17), 55),
        )
    elif role in {Role.SUPPORT, Role.SPEED_CONTROL, Role.BULKY_SETUP}:
        shapes = (
            (_spread(hp=32, defense=17, spd=17), 100),
            (_spread(hp=32, spe=32, defense=2), 65),
        )
    elif role == Role.MIXED:
        shapes = (
            (_spread(atk=32, spa=32, hp=2), 100),
            (_spread(hp=32, atk=17, spa=17), 60),
        )
    else:
        shapes = (
            (_spread(**{attack: 32}, spe=32, hp=2), 100),
            (_spread(hp=32, **{attack: 32}, defense=2), 55),
        )
    return tuple(
        SpreadCandidate(
            points,
            _candidate_weight(nature, moves, item, ability, base_stats, points, role, weight),
            role,
        )
        for points, weight in shapes
    )


def select_candidate(
    nature: str,
    moves: tuple[str, ...],
    move_categories: tuple[str, ...],
    item: str,
    ability: str,
    base_stats: BaseStats,
    seed: int | None = None,
) -> SpreadCandidate:
    candidates = impute_candidates(nature, moves, move_categories, item, ability, base_stats)
    if seed is None:
        return candidates[0]
    weights = [item.weight for item in candidates]
    return random.Random(seed).choices(candidates, weights=weights, k=1)[0]


@lru_cache(maxsize=8192)
def imputed_stats(
    nature: str,
    moves: tuple[str, ...],
    move_categories: tuple[str, ...],
    item: str,
    ability: str,
    base_stats: BaseStats,
    level: int = 50,
) -> tuple[int, int, int, int, int, int]:
    candidate = select_candidate(nature, moves, move_categories, item, ability, base_stats)
    return calculate_stats(base_stats, candidate.points, nature, level)
