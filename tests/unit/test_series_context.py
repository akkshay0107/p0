"""Tests for dynamic prior-game series resampling."""

from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.series_context import DynamicSeriesResampler

D_MODEL = 32


def _resampler() -> DynamicSeriesResampler:
    torch.manual_seed(0)
    return DynamicSeriesResampler(
        d_model=D_MODEL,
        nhead=4,
        dim_feedforward=64,
        num_summary_tokens=SERIES_TOKENS_PER_GAME,
        num_layers=2,
    )


class TestDynamicSeriesResampler:
    def test_padded_resampling_matches_independent_game_histories(self) -> None:
        """Verify vectorized masked resampling matches independent game resampling."""
        resampler = _resampler()
        short = torch.randn(5, D_MODEL)
        long = torch.randn(9, D_MODEL)
        padded = torch.zeros((2, 9, D_MODEL))
        padded[0, :5] = short
        padded[1] = long
        mask = torch.arange(9).unsqueeze(0) < torch.tensor((5, 9)).unsqueeze(1)

        actual = resampler(padded, mask)
        expected = torch.cat(
            (
                resampler(short.unsqueeze(0), torch.ones((1, 5), dtype=torch.bool)),
                resampler(long.unsqueeze(0), torch.ones((1, 9), dtype=torch.bool)),
            )
        )

        torch.testing.assert_close(actual, expected)

    def test_series_resampler_rejects_empty_game(self) -> None:
        """Verify empty games are handled outside the tensor-only neural kernel."""
        resampler = _resampler()
        empty_history = torch.zeros(2, 1, D_MODEL)
        empty_mask = torch.zeros(2, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="at least one"):
            resampler(empty_history, empty_mask)
