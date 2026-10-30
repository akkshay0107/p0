"""Tests for policy factory and series-context construction."""

from __future__ import annotations

from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources


class TestFactoryAndSeries:
    def test_compile_policy_device_guard(self) -> None:
        """Verify compile_policy leaves CPU policy nets unmodified to prevent torch.compile overhead during CPU tests."""
        config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
        resources = default_runtime_resources()
        policy = build_policy(config, resources).to("cpu")

        compiled_cpu = compile_policy(policy, enable=True)
        assert compiled_cpu is policy

    def test_factory_preserves_resources_and_model_config(self) -> None:
        """Verify the policy and observation builder keep the supplied resources."""
        resources = default_runtime_resources()
        config = ModelConfig(32, 2, 1, 128)
        policy = build_policy(config, resources)
        builder = ObservationBuilder(resources=resources)
        assert policy.resources is builder.resources is resources
        assert policy.config == config == ModelConfig.from_dict(config.to_dict())
