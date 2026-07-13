"""Independently versioned durable league metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, cast

LEAGUE_STATE_SCHEMA = 1


@dataclass(slots=True)
class LeagueState:
    shadow_id: str | None = None
    anchor_ids: list[str] = field(default_factory=list)
    regular_ids: list[str] = field(default_factory=list)
    win_rates: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)
    snapshots_since_anchor: int = 0

    def active_ids(self) -> list[str]:
        result = [] if self.shadow_id is None else [self.shadow_id]
        return result + self.anchor_ids + self.regular_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "league_schema": LEAGUE_STATE_SCHEMA,
            "shadow_id": self.shadow_id,
            "anchor_ids": self.anchor_ids,
            "regular_ids": self.regular_ids,
            "win_rates": self.win_rates,
            "games": self.games,
            "snapshots_since_anchor": self.snapshots_since_anchor,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> LeagueState:
        if value.get("league_schema") != LEAGUE_STATE_SCHEMA:
            raise ValueError("Unsupported or missing league state schema")
        try:
            state = cls(
                shadow_id=cast(str | None, value["shadow_id"]),
                anchor_ids=[str(item) for item in cast(list[object], value["anchor_ids"])],
                regular_ids=[str(item) for item in cast(list[object], value["regular_ids"])],
                win_rates={
                    str(key): float(cast(float, item))
                    for key, item in cast(dict[object, object], value["win_rates"]).items()
                },
                games={
                    str(key): int(cast(int, item))
                    for key, item in cast(dict[object, object], value["games"]).items()
                },
                snapshots_since_anchor=int(cast(int, value["snapshots_since_anchor"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed league state") from exc
        ids = state.active_ids()
        if len(ids) != len(set(ids)) or state.snapshots_since_anchor < 0:
            raise ValueError("Malformed league state identifiers or counters")
        if set(state.win_rates) != set(ids) or set(state.games) != set(ids):
            raise ValueError("League statistics do not align with active opponents")
        return state
