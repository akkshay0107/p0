"""Training utilities: precision selection, random seeding, schedules, and optimizers."""

from __future__ import annotations

import math
import random
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from p0.training.config import TrainingConfig


class OptimizationPrecision(NamedTuple):
    """Resolved compute precision for one device."""

    dtype: torch.dtype
    autocast: bool
    grad_scaler: bool


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed random number generators for the training process."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_optimization_precision(
    enable_optim: bool,
    device: torch.device,
    *,
    bf16_supported: bool | None = None,
) -> OptimizationPrecision:
    """Prefer BF16, then FP16, and fall back to FP32 on unsupported devices."""
    if not enable_optim or device.type != "cuda":
        return OptimizationPrecision(torch.float32, False, False)
    if bf16_supported is None:
        bf16_supported = torch.cuda.is_bf16_supported()
    if bf16_supported:
        return OptimizationPrecision(torch.bfloat16, True, False)
    return OptimizationPrecision(torch.float16, True, True)


class PPOScheduler:
    """Linear warmup followed by cosine learning rate decay."""

    def __init__(self, config: TrainingConfig) -> None:
        self.alpha_value = config.magnet_alpha
        self.lr_max = config.lr
        self.lr_min = 0.1 * config.lr
        self.ramp_up_end = int(config.ramp_up_phase * config.num_episodes)
        self.decay_len = config.num_episodes - 1 - self.ramp_up_end

    def alpha(self, t: int) -> float:
        """Return the magnet loss coefficient at episode t."""
        del t
        return self.alpha_value

    def state_dict(self) -> dict[str, float | int]:
        """Export scheduler parameters for checkpointing."""
        return {
            "alpha_value": self.alpha_value,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "ramp_up_end": self.ramp_up_end,
            "decay_len": self.decay_len,
        }

    def load_state_dict(self, value: dict[str, float | int]) -> None:
        """Restore scheduler parameters and verify they match the active run."""
        if value != self.state_dict():
            raise ValueError("PPO scheduler configuration does not match the checkpoint")

    def lr(self, t: int) -> float:
        """Compute the learning rate at episode t."""
        if not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise ValueError("episode t must be a non-negative integer")

        if t <= self.ramp_up_end:
            prog = min(max(t / self.ramp_up_end, 0.0), 1.0)
            return (1 - prog) * self.lr_min + prog * self.lr_max

        prog = min(max((t - self.ramp_up_end) / self.decay_len, 0.0), 1.0)
        delta = self.lr_max - self.lr_min
        return self.lr_min + 0.5 * delta * (1 + math.cos(math.pi * prog))


def adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Apply weight decay only to Linear weights and skip frozen parameters."""
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    linear_weights = {
        id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)
    }
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in linear_weights:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
