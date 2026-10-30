"""Tests for battle-series perspective identity."""

from __future__ import annotations

import pytest

from p0.battle.series import SeriesPerspectiveKey


class TestSeries:
    def test_series_perspective_key_validation(self) -> None:
        """Verify series IDs and player slots are validated."""
        key = SeriesPerspectiveKey(
            series_id="test-series-123",
            canonical_player=0,
        )
        assert key.series_id == "test-series-123"
        assert key.canonical_player == 0

        with pytest.raises(ValueError, match="must be non-empty"):
            SeriesPerspectiveKey(series_id="", canonical_player=0)

        with pytest.raises(ValueError, match="must be 0 or 1"):
            SeriesPerspectiveKey(series_id="test", canonical_player=2)
