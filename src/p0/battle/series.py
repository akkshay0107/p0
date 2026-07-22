"""Causal per-game series summaries for Bo3 context.

Both offline replay reconstruction and the live runtime produce these
records, and the series-context encoder in p0.model consumes them, so the
schema lives in the pure battle layer. A game's summary never contains
later-game information; series context for Game N is re-encoded from the
summaries of games before N, encoded afresh as explicit series context.

All identifiers are normalized ids (the output form of
p0.teams.team.normalize_id: lowercase alphanumeric), never vocabulary ids,
so summaries survive vocabulary and observation-schema changes. This module
must not import p0.teams, so producers normalize and this schema validates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SERIES_SUMMARY_SCHEMA_VERSION = 1

# A Bo3 game can be conditioned on at most the two games before it.
MAX_PRIOR_GAMES = 2


def _require_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


def _require_normalized(value: str, owner: str, *, allow_empty: bool = False) -> None:
    if value == "" and allow_empty:
        return
    if not value or not value.isalnum() or value != value.lower():
        raise ValueError(f"{owner} must be a normalized lowercase alphanumeric id, got {value!r}")


@dataclass(frozen=True, slots=True)
class SideGameSummary:
    """What one side revealed and did during a single completed game."""

    leads: tuple[str, str]
    brought: tuple[str, ...]
    mega_species: str
    moves_used: Mapping[str, tuple[str, ...]]
    revealed_items: Mapping[str, str]
    revealed_abilities: Mapping[str, str]
    revealed_formes: tuple[str, ...]
    switch_count: int
    pivot_count: int
    plan_tags: tuple[str, ...] = ()

    _FIELDS = frozenset(
        {
            "leads",
            "brought",
            "mega_species",
            "moves_used",
            "revealed_items",
            "revealed_abilities",
            "revealed_formes",
            "switch_count",
            "pivot_count",
            "plan_tags",
        }
    )

    def __post_init__(self) -> None:
        if len(self.leads) != 2:
            raise ValueError("SideGameSummary.leads must name both opening actives")
        for species in self.leads:
            _require_normalized(species, "SideGameSummary lead species")
        if len(self.brought) > 4 or len(set(self.brought)) != len(self.brought):
            raise ValueError("SideGameSummary.brought must be at most four distinct species")
        for species in self.brought:
            _require_normalized(species, "SideGameSummary brought species")
        # brought lists observed members only, but the leads are always observed.
        if self.brought and not set(self.leads) <= set(self.brought):
            raise ValueError("SideGameSummary.brought must include the leads")
        _require_normalized(self.mega_species, "SideGameSummary.mega_species", allow_empty=True)
        for species, moves in self.moves_used.items():
            _require_normalized(species, "SideGameSummary moves_used species")
            for move in moves:
                _require_normalized(move, "SideGameSummary move id")
        for mapping, owner in (
            (self.revealed_items, "revealed_items"),
            (self.revealed_abilities, "revealed_abilities"),
        ):
            for species, revealed in mapping.items():
                _require_normalized(species, f"SideGameSummary {owner} species")
                _require_normalized(revealed, f"SideGameSummary {owner} value")
        for forme in self.revealed_formes:
            _require_normalized(forme, "SideGameSummary revealed forme")
        for name, count in (("switch_count", self.switch_count), ("pivot_count", self.pivot_count)):
            if type(count) is not int or count < 0:
                raise ValueError(f"SideGameSummary.{name} must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "leads": list(self.leads),
            "brought": list(self.brought),
            "mega_species": self.mega_species,
            "moves_used": {
                species: list(self.moves_used[species]) for species in sorted(self.moves_used)
            },
            "revealed_items": {
                species: self.revealed_items[species] for species in sorted(self.revealed_items)
            },
            "revealed_abilities": {
                species: self.revealed_abilities[species]
                for species in sorted(self.revealed_abilities)
            },
            "revealed_formes": list(self.revealed_formes),
            "switch_count": self.switch_count,
            "pivot_count": self.pivot_count,
            "plan_tags": list(self.plan_tags),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SideGameSummary:
        _require_fields(value, cls._FIELDS, "SideGameSummary")
        try:
            leads = tuple(str(species) for species in value["leads"])
            return cls(
                leads=(leads[0], leads[1]),
                brought=tuple(str(species) for species in value["brought"]),
                mega_species=str(value["mega_species"]),
                moves_used={
                    str(species): tuple(str(move) for move in moves)
                    for species, moves in value["moves_used"].items()
                },
                revealed_items={
                    str(species): str(item) for species, item in value["revealed_items"].items()
                },
                revealed_abilities={
                    str(species): str(ability)
                    for species, ability in value["revealed_abilities"].items()
                },
                revealed_formes=tuple(str(forme) for forme in value["revealed_formes"]),
                switch_count=int(value["switch_count"]),
                pivot_count=int(value["pivot_count"]),
                plan_tags=tuple(str(tag) for tag in value["plan_tags"]),
            )
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(f"Invalid serialized SideGameSummary: {exc}") from exc


@dataclass(frozen=True, slots=True)
class GameSummary:
    """Causal structured summary of one completed game in a series.

    series_score is the score after this game, ordered by canonical player
    index; sides is likewise (player 0, player 1).
    """

    game_number: int
    winner: int
    series_score: tuple[int, int]
    turns: int
    sides: tuple[SideGameSummary, SideGameSummary]
    speed_observations: tuple[str, ...] = ()
    summary_schema: int = SERIES_SUMMARY_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "game_number",
            "winner",
            "series_score",
            "turns",
            "sides",
            "speed_observations",
            "summary_schema",
        }
    )

    def __post_init__(self) -> None:
        if self.summary_schema != SERIES_SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GameSummary schema {self.summary_schema!r}; "
                f"expected {SERIES_SUMMARY_SCHEMA_VERSION}"
            )
        if type(self.game_number) is not int or not 1 <= self.game_number <= MAX_PRIOR_GAMES + 1:
            raise ValueError("GameSummary.game_number must be 1, 2, or 3")
        if self.winner not in (-1, 0, 1):
            raise ValueError("GameSummary.winner must be 0, 1, or -1 for no result")
        wins = self.series_score
        if len(wins) != 2 or any(type(count) is not int or count < 0 for count in wins):
            raise ValueError("GameSummary.series_score must be two nonnegative win counts")
        if sum(wins) > self.game_number or max(wins) > 2:
            raise ValueError("GameSummary.series_score is inconsistent with the game number")
        if self.winner in (0, 1) and wins[self.winner] < 1:
            raise ValueError("GameSummary.series_score must credit the winner")
        if type(self.turns) is not int or self.turns < 0:
            raise ValueError("GameSummary.turns must be a nonnegative integer")
        if len(self.sides) != 2:
            raise ValueError("GameSummary.sides must cover both players")

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_number": self.game_number,
            "winner": self.winner,
            "series_score": list(self.series_score),
            "turns": self.turns,
            "sides": [side.to_dict() for side in self.sides],
            "speed_observations": list(self.speed_observations),
            "summary_schema": self.summary_schema,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GameSummary:
        _require_fields(value, cls._FIELDS, "GameSummary")
        try:
            score = tuple(int(count) for count in value["series_score"])
            sides = tuple(SideGameSummary.from_dict(side) for side in value["sides"])
            if len(sides) != 2:
                raise ValueError("GameSummary.sides must cover both players")
            return cls(
                game_number=int(value["game_number"]),
                winner=int(value["winner"]),
                series_score=(score[0], score[1]),
                turns=int(value["turns"]),
                sides=(sides[0], sides[1]),
                speed_observations=tuple(str(item) for item in value["speed_observations"]),
                summary_schema=int(value["summary_schema"]),
            )
        except (IndexError, TypeError) as exc:
            raise ValueError(f"Invalid serialized GameSummary: {exc}") from exc
