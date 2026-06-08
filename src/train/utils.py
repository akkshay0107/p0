import math
from pathlib import Path

import torch

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
        if t <= self.ramp_up_end:
            prog = t / self.ramp_up_end
            prog = min(max(prog, 0.0), 1.0)  # clamp to [0, 1]
            return (1 - prog) * self.lr_min + prog * self.lr_max
        else:
            prog = (t - self.ramp_up_end) / self.decay_len
            prog = min(max(prog, 0.0), 1.0)  # clamp to [0, 1]
            delta = self.lr_max - self.lr_min
            return self.lr_min + 0.5 * delta * (1 + math.cos(math.pi * prog))


def initial_state(model: PolicyNet, batch_size: int, device: torch.device):
    reducer = model.actor.reducer
    hg = reducer.hg_init.detach().expand(batch_size, -1, -1).to(device)
    return hg


def save_checkpoint(path: Path, episode: int, policy: PolicyNet, optimizer=None, scheduler=None):
    state = {
        "episode": episode,
        "model_state_dict": policy.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: Path, policy: PolicyNet, optimizer=None, scheduler=None):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("episode", None)
