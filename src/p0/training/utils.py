import math

import torch
import torch.nn as nn

from p0.training.config import TrainingConfig


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(config: TrainingConfig, device: torch.device) -> bool:
    """Enable FP16 autocast only on CUDA devices."""
    return config.enable_optim and device.type == "cuda"


class PPOScheduler:
    def __init__(self, config: TrainingConfig):
        self.alpha_value = config.magnet_alpha

        self.lr_max = config.lr
        self.lr_min = 0.1 * config.lr
        self.warmup_episodes = config.warmup_episodes
        self.ramp_up_end = int(config.ramp_up_phase * config.num_episodes)
        self.decay_len = config.num_episodes - self.ramp_up_end

    def alpha(self, t: int) -> float:
        """MMD magnet coefficient. Constant for a QRE fixed point.

        Annealing ``alpha`` downward late in training pushes the fixed point
        toward Nash; kept constant here as the first-cut default.
        """
        del t
        return self.alpha_value

    def lr(self, t: int):
        """
        Learning rate scheduling. Constant high LR for value warmup,
        then linear increase for policy, into cosine decay.
        """
        if t < self.warmup_episodes:
            return self.lr_max

        if self.ramp_up_end > 0 and t <= self.ramp_up_end:
            prog = t / self.ramp_up_end
            prog = min(max(prog, 0.0), 1.0)  # clamp to [0, 1]
            return (1 - prog) * self.lr_min + prog * self.lr_max

        if self.decay_len <= 0:
            return self.lr_max

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
