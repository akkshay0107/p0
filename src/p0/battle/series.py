"""Identity for one player's state across the games of a Bo3 series.

Series context itself is continuous, not symbolic: a completed game is
summarized by compressing its per-decision local history tokens through
DynamicSeriesResampler. Those tokens are a function of the current
weights, so they are always produced in process and never persisted. This
key is what behaviour cloning, self-play, and live play use to keep each
canonical player's prior-game state apart while they do it.
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
