import math
import random

import numpy as np
import torch
import torch.nn as nn

from p0.training.config import TrainingConfig


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed process-level generators at a training composition root."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_enabled(config: TrainingConfig, device: torch.device) -> bool:
    """Enable FP16 autocast only on CUDA devices."""
    return config.enable_optim and device.type == "cuda"


class PPOScheduler:
    def __init__(self, config: TrainingConfig):
        self.alpha_value = config.magnet_alpha

        self.lr_max = config.lr
        self.lr_min = 0.1 * config.lr
        self.ramp_up_end = int(config.ramp_up_phase * config.num_episodes)
        self.decay_len = config.num_episodes - 1 - self.ramp_up_end

    def alpha(self, t: int) -> float:
        """MMD magnet coefficient.

        Annealing alpha downward late in training pushes the fixed point
        toward Nash; Constant for a QRE fixed point.
        """
        del t
        return self.alpha_value

    def state_dict(self) -> dict[str, float | int]:
        """Expose the PyTorch-standard state_dict interface for checkpointing.

        Since this scheduler computes the learning rate dynamically based on the
        episode 't' and has no mutable internal state, this just saves the static
        configuration parameters into the checkpoint.
        """
        return {
            "alpha_value": self.alpha_value,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "ramp_up_end": self.ramp_up_end,
            "decay_len": self.decay_len,
        }

    def load_state_dict(self, value: dict[str, float | int]) -> None:
        """Restore the scheduler state from a checkpoint.

        Because there is no mutable state to overwrite, this method acts purely as a
        safeguard to ensure we don't accidentally resume a training run using a
        completely different learning rate schedule than what it started with.
        """
        if value != self.state_dict():
            raise ValueError("PPO scheduler configuration does not match the checkpoint")

    def lr(self, t: int):
        """
        Start at the minimum LR, ramp linearly, then decay to the minimum.
        """
        if t <= self.ramp_up_end:
            prog = t / self.ramp_up_end
            prog = min(max(prog, 0.0), 1.0)  # clamp to [0, 1]
            return (1 - prog) * self.lr_min + prog * self.lr_max

        prog = (t - self.ramp_up_end) / self.decay_len
        prog = min(max(prog, 0.0), 1.0)  # clamp to [0, 1]
        delta = self.lr_max - self.lr_min
        return self.lr_min + 0.5 * delta * (1 + math.cos(math.pi * prog))


def adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Apply weight decay only to Linear weights."""
    linear_weights = {
        id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)
    }
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if id(param) in linear_weights:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
