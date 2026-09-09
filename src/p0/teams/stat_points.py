"""
Champions Stat Point arithmetic and the spread fallback used without usage data.

Stat Point spreads are hidden in Champions, so opponent stats are always estimated.
Empirical priors live in p0.teams.spread_usage; this module owns the level-clause
stat arithmetic those priors feed, plus the move-category fallback used for species
the usage exports do not cover.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, NamedTuple

from p0.format_config import active_global_contract

STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
STAT_POINT_LIMIT = 32
STAT_POINT_TOTAL_LIMIT = 66
STAT_POINT_IMPUTER_VERSION = active_global_contract().payload("teams", "major")[
    "stat_point_imputer_version"
]

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

    @classmethod
    def from_dict(cls, data: Mapping[str, int]) -> StatPoints:
        try:
            return cls(*(int(data[name]) for name in STAT_NAMES))
        except KeyError as exc:
            raise ValueError(f"Missing stat point field: {exc.args[0]}") from exc


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

    bases = base_stats.as_tuple()
    sps = points.as_tuple()
    hp = (2 * bases[0] + 31 + max(2 * sps[0] - 1, 0)) * level // 100 + level + 10

    def _stat(i: int) -> int:
        return _modify_nature(
            (2 * bases[i] + 31 + max(2 * sps[i] - 1, 0)) * level // 100 + 5,
            STAT_NAMES[i],
            nature,
        )

    return (hp, _stat(1), _stat(2), _stat(3), _stat(4), _stat(5))


MOVE_CATEGORY_PHYSICAL = "physical"
MOVE_CATEGORY_SPECIAL = "special"
MOVE_CATEGORY_STATUS = "status"

# Both fallback shapes spend the full 66-point budget.
_FALLBACK_PHYSICAL = StatPoints(hp=32, atk=32, spe=2)
_FALLBACK_SPECIAL = StatPoints(hp=32, spa=32, spe=2)
_FALLBACK_STATUS = StatPoints(hp=32, defense=17, spd=17)


def fallback_points(move_categories: tuple[str, ...]) -> StatPoints | None:
    """
    Guess a spread from move categories alone, for species with no usage data.

    Categories are tested in a fixed physical, special, status order rather than by
    base stat, so the result depends only on the moves and never on the species.

    Returns None when no category reaches two moves, which happens only when fewer
    than four moves are known: four moves across three categories always leave one
    category with at least two. Callers treat None as an explicit UNKNOWN rather
    than substituting a blind guess.
    """
    counts = Counter(category.lower() for category in move_categories)

    if counts[MOVE_CATEGORY_PHYSICAL] >= 2:
        return _FALLBACK_PHYSICAL
    if counts[MOVE_CATEGORY_SPECIAL] >= 2:
        return _FALLBACK_SPECIAL
    if counts[MOVE_CATEGORY_STATUS] >= 2:
        return _FALLBACK_STATUS
    return None
