"""Tests for model architecture configuration."""

from __future__ import annotations

import pytest

from p0.model.config import ModelConfig


class TestModelConfig:
    def test_serialization_pins_architecture_and_rejects_invalid_configs(self) -> None:
        """Pin serialized fields, defaults, and invalid architecture inputs."""
        config = ModelConfig.baseline()
        assert config.d_model == 384
        assert config.nhead == 8
        assert config.reducer_layers == 8
        assert config.dim_feedforward == 1536
        baseline_fields = {
            "d_model": 384,
            "nhead": 8,
            "reducer_layers": 8,
            "dim_feedforward": 1536,
        }
        assert config.to_dict() == baseline_fields
        assert ModelConfig.from_dict(baseline_fields) == config

        stale = dict(baseline_fields)
        stale["history_tokens"] = 8
        with pytest.raises(ValueError, match=r"unknown=.*history_tokens"):
            ModelConfig.from_dict(stale)

        invalid_configs = (
            (63, 8, 1, 256, "divisible"),
            (96, 3, 1, 128, "low-width event channel"),
        )
        for d_model, nhead, reducer_layers, dim_feedforward, error in invalid_configs:
            with pytest.raises(ValueError, match=error):
                ModelConfig(d_model, nhead, reducer_layers, dim_feedforward)
