"""Tests for model architecture configuration."""

from __future__ import annotations

import pytest

from p0.model.config import ModelConfig


class TestModelConfig:
    def test_model_config_is_checkpoint_local_and_validated(self) -> None:
        """Verify ModelConfig enforces divisibility requirements (d_model % nhead == 0)."""
        config = ModelConfig.baseline()
        assert config.d_model == 384
        assert config.reducer_layers == 8
        assert config.dim_feedforward == 1536
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(63, 8, 1, 256)

    def test_model_config_has_only_scaling_fields(self) -> None:
        """Verify ModelConfig accepts valid scaling architectures and rejects deprecated parameters."""
        config = ModelConfig.baseline()
        assert config.dim_feedforward == 1536
        assert ModelConfig.from_dict(config.to_dict()) == config
        enabled = ModelConfig(
            d_model=64,
            nhead=4,
            reducer_layers=1,
            dim_feedforward=128,
        )
        assert ModelConfig.from_dict(enabled.to_dict()) == enabled
        stale = config.to_dict()
        stale["history_tokens"] = 8
        with pytest.raises(ValueError, match=r"unknown=.*history_tokens"):
            ModelConfig.from_dict(stale)
        with pytest.raises(ValueError, match="low-width event channel"):
            ModelConfig(d_model=96, nhead=3, reducer_layers=1, dim_feedforward=128)
