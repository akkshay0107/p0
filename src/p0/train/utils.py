import math
from pathlib import Path

import torch
import torch.nn as nn

from p0.format_config import (
    policy_model_config,
    runtime_manifest_sha256,
    validate_artifact_manifest_reference,
)
from p0.model.policy import PolicyNet
from p0.train.config import TrainingConfig


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOScheduler:
    def __init__(self, config: TrainingConfig):
        self.ent_max = config.entropy_coef
        self.ent_min = 0.1 * config.entropy_coef
        self.ramp_down_start = int((1 - config.ramp_down_phase) * config.num_episodes)
        self.ramp_down_len = config.num_episodes - self.ramp_down_start

        self.lr_max = config.lr
        self.lr_min = 0.1 * config.lr
        self.warmup_episodes = config.warmup_episodes
        self.ramp_up_end = int(config.ramp_up_phase * config.num_episodes)
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


def save_checkpoint(
    path: Path, episode: int, policy: PolicyNet, optimizer=None, scheduler=None, scaler=None
):
    state = {
        "episode": episode,
        "model_state_dict": policy.state_dict(),
        "runtime_manifest_sha256": runtime_manifest_sha256(),
        "model_config": policy_model_config(policy),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def policy_from_checkpoint(path: Path, device: torch.device | str) -> PolicyNet:
    """Construct a policy from its serialized architecture before loading weights."""
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    validate_artifact_manifest_reference(checkpoint)
    config_value = checkpoint.get("model_config")
    if not isinstance(config_value, dict):
        raise ValueError(f"Checkpoint {path} has no valid serialized model configuration")
    config = dict(config_value)
    if not config:
        raise ValueError(f"Checkpoint {path} has no serialized model configuration")
    try:
        config["obs_dim"] = tuple(config["obs_dim"])
        policy = PolicyNet(**config).to(device)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model configuration in checkpoint {path}") from exc
    return policy


def load_checkpoint(path: Path, policy: PolicyNet, optimizer=None, scheduler=None, scaler=None):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location=policy.device)
    validate_artifact_manifest_reference(checkpoint)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config != policy_model_config(policy):
        raise ValueError(f"Checkpoint {path} model configuration does not match the policy")
    policy.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("episode", None)
