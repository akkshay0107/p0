from __future__ import annotations

import pytest
import torch

from p0.model.swiglu_encoder import AttentionPool


class TestAttentionPool:
    def test_pool_is_permutation_invariant_and_trainable(self) -> None:
        torch.manual_seed(17)
        pool = AttentionPool(d_model=32, nhead=4)
        source = torch.randn(3, 7, 32, requires_grad=True)

        pooled = pool(source)
        permuted = pool(source[:, torch.tensor([4, 1, 6, 0, 3, 5, 2])])

        assert pooled.shape == (3, 32)
        torch.testing.assert_close(pooled, permuted)
        pooled.square().mean().backward()
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert pool.query.grad is not None
        assert torch.isfinite(pool.query.grad).all()

    @pytest.mark.parametrize(
        "source",
        (
            torch.empty(2, 0, 32),
            torch.empty(2, 4, 16),
            torch.empty(2, 32),
        ),
    )
    def test_pool_rejects_invalid_source_shapes(self, source: torch.Tensor) -> None:
        pool = AttentionPool(d_model=32, nhead=4)

        with pytest.raises(ValueError, match="source must have shape"):
            pool(source)
