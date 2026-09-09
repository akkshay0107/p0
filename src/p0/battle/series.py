"""Series perspective identity.

Identifies a player across games in a best-of-three series. Used by
training and live play to track prior-game history separately for each player.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeriesPerspectiveKey:
    """Stable identity for one player's state within a series."""

    series_id: str
    canonical_player: int

    def __post_init__(self) -> None:
        if not self.series_id:
            raise ValueError("SeriesPerspectiveKey.series_id must be non-empty")
        if type(self.canonical_player) is not int or self.canonical_player not in (0, 1):
            raise ValueError("SeriesPerspectiveKey.canonical_player must be 0 or 1")
