"""Tests for training utilities: precision, seeding, scheduler, and optimizer grouping."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from p0.training.config import TrainingConfig
from p0.training.utils import (
    PPOScheduler,
    adamw_param_groups,
    default_device,
    seed_everything,
    select_optimization_precision,
)


class TestTrainingPrecision:
    def test_optimization_precision_waterfall(self) -> None:
        """Verify BF16, FP16, and FP32 are selected in preference order."""
        assert select_optimization_precision(True, torch.device("cpu")) == (
            torch.float32,
            False,
            False,
        )
        assert select_optimization_precision(True, torch.device("cuda"), bf16_supported=True) == (
            torch.bfloat16,
            True,
            False,
        )
        assert select_optimization_precision(True, torch.device("cuda"), bf16_supported=False) == (
            torch.float16,
            True,
            True,
        )
        assert select_optimization_precision(False, torch.device("cuda")) == (
            torch.float32,
            False,
            False,
        )


class TestPPOScheduler:
    def test_warmup_and_cosine_decay(self) -> None:
        """Verify linear warmup ramp followed by cosine decay to minimum LR."""
        config = TrainingConfig(
            num_episodes=100,
            lr=1e-3,
            ramp_up_phase=0.2,
            magnet_alpha=0.05,
        )
        scheduler = PPOScheduler(config)

        assert scheduler.alpha(0) == pytest.approx(0.05)
        assert scheduler.alpha(50) == pytest.approx(0.05)

        # At t=0: minimum LR (0.1 * lr_max)
        assert scheduler.lr(0) == pytest.approx(1e-4)

        # At t=ramp_up_end (20): peak LR
        assert scheduler.lr(20) == pytest.approx(1e-3)

        # At midpoint of warmup (10): linear interpolation
        assert scheduler.lr(10) == pytest.approx(0.55 * 1e-3)

        # At final episode (99): decays back to minimum LR
        assert scheduler.lr(99) == pytest.approx(1e-4)

        # Clamping beyond bounds
        assert scheduler.lr(100) == pytest.approx(1e-4)

    def test_invalid_episode_rejected(self) -> None:
        """Verify invalid or negative episode values are rejected."""
        scheduler = PPOScheduler(TrainingConfig())
        with pytest.raises(ValueError, match="non-negative integer"):
            scheduler.lr(-1)
        with pytest.raises(ValueError, match="non-negative integer"):
            scheduler.lr(True)  # type: ignore[arg-type]

    def test_state_dict_round_trip(self) -> None:
        """Verify scheduler configuration exports and validates matching restore."""
        config = TrainingConfig(num_episodes=50, lr=2e-4, ramp_up_phase=0.1)
        scheduler = PPOScheduler(config)

        state = scheduler.state_dict()
        assert state["lr_max"] == 2e-4
        assert state["ramp_up_end"] == 5

        # Matching configuration restores cleanly
        scheduler.load_state_dict(state)

        # Mismatched configuration is rejected
        tampered = dict(state)
        tampered["lr_max"] = 5e-4
        with pytest.raises(ValueError, match="does not match"):
            scheduler.load_state_dict(tampered)


class TestAdamWParamGroups:
    def test_param_groups_weight_decay_filtering(self) -> None:
        """Verify weight decay applies only to Linear weights and skips frozen parameters."""

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(8, 8, bias=True)
                self.frozen_linear = nn.Linear(8, 8, bias=False)
                self.frozen_linear.weight.requires_grad = False
                self.norm = nn.LayerNorm(8)
                self.embed = nn.Embedding(10, 8)

        model = DummyModel()
        groups = adamw_param_groups(model, weight_decay=1e-2)

        assert len(groups) == 2
        decay_group, no_decay_group = groups

        assert decay_group["weight_decay"] == 1e-2
        assert no_decay_group["weight_decay"] == 0.0

        # Only active Linear weight is in decay group
        assert len(decay_group["params"]) == 1
        assert decay_group["params"][0] is model.linear.weight

        # Linear bias, norm weight/bias, embed weight in no-decay group
        no_decay_ids = {id(p) for p in no_decay_group["params"]}
        assert id(model.linear.bias) in no_decay_ids
        assert id(model.norm.weight) in no_decay_ids
        assert id(model.norm.bias) in no_decay_ids
        assert id(model.embed.weight) in no_decay_ids

        # Frozen weight is not in either group
        assert id(model.frozen_linear.weight) not in no_decay_ids
        assert id(model.frozen_linear.weight) not in {id(p) for p in decay_group["params"]}

    def test_negative_weight_decay_rejected(self) -> None:
        """Verify negative weight decay is rejected."""
        model = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="non-negative"):
            adamw_param_groups(model, weight_decay=-0.01)


class TestSeedEverything:
    def test_seed_everything_deterministic(self) -> None:
        """Verify seeding produces identical sequences across random, numpy, and torch."""
        seed_everything(12345)
        py_val1 = random.random()
        np_val1 = float(np.random.rand())
        torch_val1 = float(torch.rand(1).item())

        seed_everything(12345)
        py_val2 = random.random()
        np_val2 = float(np.random.rand())
        torch_val2 = float(torch.rand(1).item())

        assert py_val1 == py_val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2

    def test_negative_seed_rejected(self) -> None:
        """Verify negative seed values are rejected."""
        with pytest.raises(ValueError, match="nonnegative integer"):
            seed_everything(-1)


class TestDefaultDevice:
    def test_default_device_resolution(self) -> None:
        """Verify default_device returns cuda when available, otherwise cpu."""
        device = default_device()
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        assert device.type == expected_type
