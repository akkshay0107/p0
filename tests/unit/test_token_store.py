"""Tests for completed-game token storage."""

import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.token_store import SeriesTokenStore


class TestSeriesTokenStore:
    def test_reused_source_storage_preserves_each_perspective_snapshot(self) -> None:
        store = SeriesTokenStore(d_model=2)
        first_player = SeriesPerspectiveKey("match", 0)
        second_player = SeriesPerspectiveKey("match", 1)
        source = torch.ones(4, 2)

        store.append(first_player, source)
        source.fill_(2.0)
        store.append(second_player, source)
        source.fill_(3.0)
        tokens, mask = store.get_tokens([first_player, second_player], torch.device("cpu"))

        torch.testing.assert_close(tokens[0, :4], torch.ones(4, 2))
        torch.testing.assert_close(tokens[1, :4], torch.full((4, 2), 2.0))
        assert mask[:, :4].all()
        assert not mask[:, 4:].any()
