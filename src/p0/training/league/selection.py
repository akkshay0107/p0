"""Pure PFSP weighting and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from p0.training.config import PoolConfig, TrainingConfig


class OpponentRoster(Protocol):
    def __len__(self) -> int: ...

    def sample_many(self, count: int) -> list[str]: ...


def assign_pool_opponents(n_envs: int, roster: list[str]) -> list[str]:
    if not roster:
        raise ValueError("Pool opponent roster must not be empty.")
    return [roster[env_id % len(roster)] for env_id in range(n_envs)]


@dataclass(slots=True)
class EnvPartition:
    self_idx: torch.Tensor
    pool_idx: torch.Tensor
    opponent_ids: list[str]
    self_mask_cpu: torch.Tensor

    def pool_groups(self) -> tuple[tuple[str, torch.Tensor], ...]:
        grouped: dict[str, list[int]] = {}
        for env_id in self.pool_idx.tolist():
            grouped.setdefault(self.opponent_ids[env_id], []).append(env_id)
        return tuple(
            (opponent_id, torch.tensor(indices, device=self.pool_idx.device))
            for opponent_id, indices in grouped.items()
        )


def build_partition(
    config: TrainingConfig,
    pool: OpponentRoster,
    device: torch.device,
) -> EnvPartition:
    n_self = config.n_envs if len(pool) == 0 else config.n_self_envs
    self_idx = torch.arange(n_self, device=device)
    pool_idx = torch.arange(n_self, config.n_envs, device=device)
    self_mask_cpu = torch.arange(config.n_envs) < n_self
    opponent_ids = ["self"] * config.n_envs
    if pool_idx.numel():
        roster = pool.sample_many(config.n_pool_opponents)
        assignments = assign_pool_opponents(pool_idx.numel(), roster)
        for env_id, opponent_id in zip(pool_idx.tolist(), assignments, strict=True):
            opponent_ids[env_id] = opponent_id
    return EnvPartition(self_idx, pool_idx, opponent_ids, self_mask_cpu)


def pfsp_weight(win_rate: float, games: int, config: PoolConfig) -> float:
    win_rate = max(config.pool_wr_floor, win_rate)
    competitive = win_rate * (1 - win_rate) + 0.3 * win_rate
    exploration = config.pool_explore_coef / (1 + 0.2 * games)
    return competitive + exploration


def sample_opponents(
    roster: list[str],
    win_rates: dict[str, float],
    games: dict[str, int],
    count: int,
    config: PoolConfig,
) -> list[str]:
    if count <= 0:
        raise ValueError("Opponent count must be greater than zero.")
    if not roster:
        raise RuntimeError("Opponent pool is empty. Add an opponent before sampling.")
    count = min(count, len(roster))
    weights = torch.tensor(
        [pfsp_weight(win_rates[item], games.get(item, 0), config) for item in roster],
        dtype=torch.float32,
    )
    indices = torch.multinomial(weights, count, replacement=False)
    return [roster[index] for index in indices.tolist()]
