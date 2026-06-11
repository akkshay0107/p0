import math
from pathlib import Path

import torch
import torch.nn as nn

from src.model.policy import PolicyNet
from src.train.config import PPOConfig


class PPOScheduler:
    def __init__(self, config: PPOConfig):
        self.ent_max = config.entropy_coef
        self.ent_min = 0.1 * config.entropy_coef
        self.ramp_down_start = int((1 - config.ramp_down_phase) * config.num_episodes)
        self.ramp_down_len = config.num_episodes - self.ramp_down_start

        self.lr_max = config.lr
        self.lr_min = 0.1 * config.lr
        self.ramp_up_end = int(config.ramp_up_phase * config.num_episodes)  # also the length
        self.decay_len = config.num_episodes - self.ramp_up_end

    def entropy_coef(self, t: int):
        """
        Entropy coefficient scheduling. Flat into linear decay
        """
        if t < self.ramp_down_start:
            return self.ent_max

        prog = (t - self.ramp_down_start) / self.ramp_down_len
        prog = min(max(prog, 0.0), 1.0)  # clamp to [0, 1]
        return prog * self.ent_min + (1 - prog) * self.ent_max

    def lr(self, t: int):
        """
        Learning rate scheduling. Linear increase into cosine decay.
        """
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


def save_checkpoint(path: Path, episode: int, policy: PolicyNet, optimizer=None, scheduler=None, scaler=None):
    state = {
        "episode": episode,
        "model_state_dict": policy.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: Path, policy: PolicyNet, optimizer=None, scheduler=None, scaler=None):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("episode", None)
