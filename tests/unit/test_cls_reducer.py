"""Tests for reducer history packing and memory attention."""

from __future__ import annotations

import pytest
import torch

from p0.model.architecture_contract import (
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    SERIES_SLOTS,
)
from p0.model.cls_reducer import MemoryReducer, pack_history_tokens


class TestPackHistoryTokens:
    def test_history_packing_right_aligns_short_history_and_rejects_oversized_input(
        self,
    ) -> None:
        history = torch.tensor([[[3.0, 5.0], [7.0, 11.0]]])

        packed, mask = pack_history_tokens(history)

        assert packed.shape == (1, 48, 2)
        torch.testing.assert_close(
            packed,
            torch.tensor([[[0.0, 0.0]] * 46 + [[3.0, 5.0], [7.0, 11.0]]]),
        )
        assert mask.tolist() == [[False] * 46 + [True, True]]

        with pytest.raises(ValueError, match="N <= 48"):
            pack_history_tokens(torch.zeros((1, 49, 2)))


class TestMemoryReducer:
    def test_reducer_keeps_current_and_history_summaries_trainable(self) -> None:
        """Verify differentiable training reconstruction reaches current and history tokens."""
        reducer = MemoryReducer(32, 4, 1, 64)
        current = torch.randn(2, CURRENT_TOKEN_COUNT, 32, requires_grad=True)
        local_summary = reducer.local_summary(current)
        history = torch.randn(2, HISTORY_WINDOW, 32, requires_grad=True)
        series = torch.zeros(2, SERIES_SLOTS, 32)
        series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
        history_mask = torch.ones(2, HISTORY_WINDOW, dtype=torch.bool)

        output = reducer.reduce(
            local_summary,
            current,
            series,
            series_mask,
            history,
            history_mask,
        )
        output.cls.square().mean().backward()

        assert current.grad is not None and torch.isfinite(current.grad).all()
        assert torch.count_nonzero(current.grad) > 0
        assert history.grad is not None and torch.isfinite(history.grad).all()
        assert torch.count_nonzero(history.grad) > 0

    def test_reducer_rejects_a_local_summary_that_does_not_match_the_batch(self) -> None:
        """Verify MemoryReducer validates batch size and dtype alignment on local_summary arguments."""
        reducer = MemoryReducer(32, 4, 1, 64)
        current = torch.randn(2, CURRENT_TOKEN_COUNT, 32)
        summary = reducer.local_summary(current)
        series = torch.zeros(2, SERIES_SLOTS, 32)
        series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
        history = torch.zeros(2, HISTORY_WINDOW, 32)
        history_mask = torch.zeros(2, HISTORY_WINDOW, dtype=torch.bool)

        with pytest.raises(ValueError, match="local summary"):
            reducer.reduce(
                summary[:1],
                current,
                series,
                series_mask,
                history,
                history_mask,
            )
        with pytest.raises(ValueError, match="local summary"):
            reducer.reduce(
                summary.to(torch.float64),
                current,
                series,
                series_mask,
                history,
                history_mask,
            )

    def test_reducer_uses_fixed_padding_only_memory_attention(self) -> None:
        """Verify MemoryReducer ignores masked padding tokens in series and history memory banks."""
        torch.manual_seed(0)
        reducer = MemoryReducer(32, 4, 1, 64)
        current = torch.randn(2, CURRENT_TOKEN_COUNT, 32)
        series = torch.randn(2, SERIES_SLOTS, 32)
        series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
        series_mask[1, :4] = True
        history = torch.randn(2, HISTORY_WINDOW, 32)
        history_mask = torch.zeros(2, HISTORY_WINDOW, dtype=torch.bool)
        first = reducer.reduce(
            reducer.local_summary(current),
            current,
            series,
            series_mask,
            history,
            history_mask,
        )

        # Mutating unmasked series padding tokens must not change reduced CLS token
        changed_series_padding = series.clone()
        changed_series_padding[0] = 1000.0
        second = reducer.reduce(
            reducer.local_summary(current),
            current,
            changed_series_padding,
            series_mask,
            history,
            history_mask,
        )
        torch.testing.assert_close(first.cls, second.cls)

        # Mutating unmasked history padding tokens must not change reduced CLS token
        changed_history_padding = history.clone()
        changed_history_padding[0] = 1000.0
        third = reducer.reduce(
            reducer.local_summary(current),
            current,
            series,
            series_mask,
            changed_history_padding,
            history_mask,
        )
        torch.testing.assert_close(first.cls, third.cls)

        # Mutating valid unmasked history tokens alters reduced representation
        changed_valid_history = history.clone()
        changed_valid_history[1, -1] = 1000.0
        changed_mask = history_mask.clone()
        changed_mask[1, -1] = True
        fourth = reducer.reduce(
            reducer.local_summary(current),
            current,
            series,
            series_mask,
            changed_valid_history,
            changed_mask,
        )
        assert not torch.allclose(first.cls[1], fourth.cls[1])
        assert first.pokemon.shape == (2, 12, 32)
        assert first.local_history_token.shape == (2, 32)
