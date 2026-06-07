from pathlib import Path

import torch

from src.model.policy import PolicyNet

# utils now only contains the general utility functions
# rest of the stuff moved to its own file


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
