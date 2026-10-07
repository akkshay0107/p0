"""Generated isolation checks for large batches of series keys."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.token_store import SeriesTokenStore
from tests.stress._helpers import stress_count


class TestSeriesTokenStore:
    @pytest.mark.stress
    @settings(max_examples=16, deadline=None)
    @given(
        key_count=st.integers(min_value=2, max_value=stress_count("P0_STRESS_SERIES_KEYS", 1024)),
        drop_index=st.integers(min_value=0, max_value=2**16),
        reverse=st.booleans(),
    )
    def test_generated_key_batches_preserve_identity_after_drop(
        self, key_count: int, drop_index: int, reverse: bool
    ) -> None:
        """Distinct per-key values survive reordered retrieval and deletion of another key."""
        store = SeriesTokenStore(d_model=1, max_games=2)
        keys = tuple(
            SeriesPerspectiveKey(f"series-{index}", index % 2) for index in range(key_count)
        )
        for index, key in enumerate(keys):
            store.append(key, torch.full((SERIES_TOKENS_PER_GAME, 1), float(index + 1)))
        selected = tuple(reversed(keys)) if reverse else keys
        expected_values = torch.arange(1, key_count + 1, dtype=torch.float32)
        if reverse:
            expected_values = expected_values.flip(0)
        tokens, mask = store.get_tokens(selected, torch.device("cpu"))
        torch.testing.assert_close(
            tokens[:, :SERIES_TOKENS_PER_GAME, 0],
            expected_values[:, None].expand(-1, SERIES_TOKENS_PER_GAME),
            rtol=0,
            atol=0,
        )
        assert mask[:, :SERIES_TOKENS_PER_GAME].all()
        assert not mask[:, SERIES_TOKENS_PER_GAME:].any()
        assert not tokens[:, SERIES_TOKENS_PER_GAME:].any()

        removed = drop_index % key_count
        store.drop(keys[removed])
        after, after_mask = store.get_tokens(selected, torch.device("cpu"))
        removed_row = selected.index(keys[removed])
        assert not after[removed_row].any()
        assert not after_mask[removed_row].any()
        retained_rows = torch.arange(key_count) != removed_row
        torch.testing.assert_close(after[retained_rows], tokens[retained_rows], rtol=0, atol=0)
        assert torch.equal(after_mask[retained_rows], mask[retained_rows])
