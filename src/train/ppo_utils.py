import functools
import json
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Self

import torch

from lookups import ACT_SIZE, OBS_DIM
from policy import PolicyNet


def unwrap_policy(policy: PolicyNet) -> PolicyNet:
    return getattr(policy, "_orig_mod", policy)


def reducer_of(model: PolicyNet):
    return unwrap_policy(model).reducer


def initial_state(model: PolicyNet, batch_size: int, device: torch.device):
    reducer = reducer_of(model)
    cls = reducer.cls_base.detach().expand(batch_size, -1, -1).squeeze(1).to(device)
    hg = reducer.hg_init.detach().expand(batch_size, -1, -1).to(device)
    return cls, hg


@dataclass
class PPOConfig:
    num_episodes: int = 12500
    num_envs: int = 4
    n_jobs: int = 8
    rollouts_per_episode: int = 128

    gamma: float = 0.97
    gae_lambda: float = 0.95
    lr: float = 3e-5
    batch_size: int = 32
    clip_range: float = 0.25
    entropy_coef: float = 0.06
    entropy_coef_floor: float = 0.02
    entropy_anneal_start: int = 3000
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.05
    ppo_epochs: int = 4
    warmup_episodes: int = 100
    min_lr: float = 1e-6

    checkpoint_path: Path = Path(__file__).parent.parent / "checkpoints" / "ppo_checkpoint.pt"
    pool_dir: Path = Path(__file__).parent.parent / "checkpoints" / "pool"
    pool_size: int = 40
    snapshot_interval: int = 50
    # EMA smoothing factor for per-opponent win-rate tracking (lower = smoother).
    pool_win_rate_smoothing: float = 0.1
    # Minimum sampling weight for any opponent (prevents starvation).
    min_pool_win_rate_weight: float = 0.1
    pool_cache_size: int = 20
    self_play_prob: float = 0.5
    compile_policy: bool = False
    # Skew importance of team preview step
    teampreview_loss_mult: float = 1.5
    teampreview_entropy_mult: float = 2.0


def load_config(config_path: str = ".ppoconfig") -> PPOConfig:
    """
    Loads a PPOConfig from a flat key=value file.
    Merges specified values for hyperparameters with the default values
    for unspecified hyperparameters.

    Defaults to returning default PPOConfig on any error.
    """
    path = Path(config_path)
    if not path.exists():
        return PPOConfig()

    config_dict = {}
    valid_fields = {f.name: f.type for f in fields(PPOConfig)}

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key in valid_fields:
                    target_type = valid_fields[key]
                    try:
                        if target_type is int:
                            config_dict[key] = int(float(value))
                        elif target_type is float:
                            config_dict[key] = float(value)
                        elif target_type is bool:
                            config_dict[key] = value.lower() in ("true", "1", "yes")
                        elif target_type is Path:
                            config_dict[key] = Path(value)
                        else:
                            config_dict[key] = value
                    except (ValueError, TypeError):
                        continue

        return PPOConfig(**config_dict)
    except Exception:
        return PPOConfig()


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectories: list[dict[str, torch.Tensor]] = []

    _FIELDS = (
        "obs",
        "actions",
        "log_probs",
        "values",
        "rewards",
        "dones",
        "action_masks",
        "is_team_preview",
    )

    def add_episode(self, trajectory: list[dict]):
        if not trajectory:
            return
        episode = {f: torch.cat([step[f] for step in trajectory], dim=0) for f in self._FIELDS}
        episode["length"] = len(trajectory)
        self.trajectories.append(episode)

    def get_batches(self, device: torch.device, config: PPOConfig):
        all_episodes = []
        all_advantages = []

        for ep in self.trajectories:
            rewards = ep["rewards"].float()
            values = ep["values"].float()
            dones = ep["dones"].float()
            T = ep["length"]

            adv = torch.zeros_like(rewards)
            gae = torch.zeros(1, dtype=torch.float32)

            for t in reversed(range(T)):
                next_value = values[t + 1] if t + 1 < T else torch.zeros_like(values[t])
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + config.gamma * next_value * nonterminal - values[t]
                gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
                adv[t] = gae

            ret = adv + values

            episode_data = {
                "obs": ep["obs"].to(device),
                "actions": ep["actions"].to(device),
                "log_probs": ep["log_probs"].to(device),
                "action_masks": ep["action_masks"].to(device),
                "values": values.to(device),
                "advantages": adv.to(device),
                "returns": ret.to(device),
                "is_team_preview": ep["is_team_preview"].to(device),
                "length": T,
            }
            all_episodes.append(episode_data)
            all_advantages.append(episode_data["advantages"])

        if all_advantages:
            flat_adv = torch.cat(all_advantages, dim=0)
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std().clamp_min(1e-8)
            for ep in all_episodes:
                ep["advantages"] = (ep["advantages"] - adv_mean) / adv_std

        return all_episodes


def save_checkpoint(path: Path, episode: int, policy: PolicyNet, optimizer=None, scheduler=None):
    model = unwrap_policy(policy)
    state = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def save_best_checkpoint(path: Path, episode: int, win_rate: float, policy: PolicyNet):
    model = unwrap_policy(policy)
    state = {
        "episode": episode,
        "win_rate": win_rate,
        "model_state_dict": model.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path: Path, policy: PolicyNet, optimizer=None, scheduler=None):
    if not path.exists():
        return None
    model = unwrap_policy(policy)
    checkpoint = torch.load(path, map_location=model.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("episode", None)


# AlphaStar style pool of checkpoints for diverse training
class OpponentPool:
    def __init__(self, pool_dir: Path, config: PPOConfig):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        # Ordered list of opponent ids (strings); first = oldest.
        self.opponent_ids: list[str] = []
        # EMA win-rate of the training policy against each opponent.
        # Starts at 0.5 (neutral) for newly added opponents.
        self.win_rates: dict[str, float] = {}
        # LRU cache wrapper for loading policies
        self._load_policy_cached = functools.lru_cache(maxsize=config.pool_cache_size)(
            self._load_policy_impl
        )

    def save_state(self) -> None:
        state = {
            "opponent_ids": self.opponent_ids,
            "win_rates": self.win_rates,
        }
        with open(self.pool_dir / "pool_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        path = self.pool_dir / "pool_state.json"
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.opponent_ids = state.get("opponent_ids", [])
            self.win_rates = state.get("win_rates", {})

        # Sync with filesystem for any missing .pt files (e.g. if json was lost)
        for pt_file in sorted(self.pool_dir.glob("*.pt")):
            opponent_id = pt_file.stem
            if opponent_id not in self.opponent_ids:
                self.opponent_ids.append(opponent_id)
                if opponent_id not in self.win_rates:
                    self.win_rates[opponent_id] = 0.5

        # Cleanup: remove any IDs that no longer have a corresponding .pt file
        existing_ids = {pt.stem for pt in self.pool_dir.glob("*.pt")}
        self.opponent_ids = [oid for oid in self.opponent_ids if oid in existing_ids]
        self.win_rates = {oid: wr for oid, wr in self.win_rates.items() if oid in existing_ids}

    @classmethod
    def load_or_create(cls, pool_dir: Path, config: PPOConfig) -> Self:
        """Load an existing pool from disk, or create an empty one."""
        pool = cls(pool_dir, config)
        pool._load_state()
        return pool

    def add(self, policy: PolicyNet, opponent_id: str) -> None:
        model = unwrap_policy(policy)

        # evict highest winrate (easiest opponents) to keep the pool challenging
        while len(self.opponent_ids) >= self.config.pool_size:
            max_i = 0
            max_wr = self.win_rates.get(self.opponent_ids[max_i], 0.5)
            for i, opp_id in enumerate(self.opponent_ids):
                wr = self.win_rates.get(opp_id, 0.5)
                if wr > max_wr:
                    max_i = i
                    max_wr = wr

            evicted_id = self.opponent_ids.pop(max_i)
            self.win_rates.pop(evicted_id, None)

            evicted_path = self.pool_dir / f"{evicted_id}.pt"
            if evicted_path.exists():
                evicted_path.unlink()

        torch.save({"model_state_dict": model.state_dict()}, self.pool_dir / f"{opponent_id}.pt")
        self.opponent_ids.append(opponent_id)
        # Neutral starting win-rate.
        self.win_rates[opponent_id] = 0.5
        self._load_policy_cached.cache_clear()

    def update_win_rate(self, opponent_id: str, won: bool) -> None:
        if opponent_id not in self.win_rates:
            return
        alpha = self.config.pool_win_rate_smoothing
        self.win_rates[opponent_id] = (1 - alpha) * self.win_rates[opponent_id] + alpha * float(won)

    def sample(self) -> tuple[PolicyNet, str]:
        # Returns frozen snapshot of a policy from the pool (can't train it)
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample().")

        floor = self.config.min_pool_win_rate_weight
        # Weight = 1 - ema_win_rate, clamped to [floor, 1.0].
        # Opponents we beat easily get low weight; hard ones get high weight.
        weights = [max(floor, 1.0 - self.win_rates.get(oid, 0.5)) for oid in self.opponent_ids]
        (opponent_id,) = random.choices(self.opponent_ids, weights=weights, k=1)
        return self._load_policy_cached(opponent_id), opponent_id

    def _load_policy_impl(self, opponent_id: str) -> PolicyNet:
        checkpoint = torch.load(
            self.pool_dir / f"{opponent_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        net = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        return net

    def __len__(self) -> int:
        return len(self.opponent_ids)

    def __repr__(self) -> str:
        return f"OpponentPool(size={len(self)}/{self.config.pool_size}, ids={self.opponent_ids})"
