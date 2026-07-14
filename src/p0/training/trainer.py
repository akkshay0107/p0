"""Episode-level PPO training lifecycle."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import torch

from p0.model.policy import PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.training.checkpoint import PolicyStore
from p0.training.config import PoolConfig, TrainingConfig
from p0.training.league.league import OpponentPool
from p0.training.ppo import PPOUpdater
from p0.training.rollout import RolloutCollector
from p0.training.utils import PPOScheduler

MetricSink = Callable[[Mapping[str, float], int, str], None]


def build_reference_batch(collector: RolloutCollector, size: int = 64) -> dict[str, torch.Tensor]:
    trajectories = collector.buffer.trajectories
    flat_obs = StructuredObservation.cat(
        [trajectory.observations for trajectory in trajectories], dim=0
    )
    flat_masks = torch.cat([trajectory.action_masks for trajectory in trajectories], dim=0)
    indices = torch.randperm(flat_masks.size(0))[: min(size, flat_masks.size(0))]
    result = {
        name: tensor[indices].clone()
        for name, tensor in zip(StructuredObservation._FIELD_NAMES, flat_obs.tensors(), strict=True)
    }
    result["action_masks"] = flat_masks[indices].clone()
    return result


class PPOTrainer:
    def __init__(
        self,
        *,
        policy: PolicyNet,
        policy_store: PolicyStore,
        checkpoint_path: Path,
        collector: RolloutCollector,
        updater: PPOUpdater,
        league: OpponentPool,
        scheduler: PPOScheduler,
        training_config: TrainingConfig,
        pool_config: PoolConfig,
        metric_sink: MetricSink = lambda metrics, step, phase: None,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.policy = policy
        self.policy_store = policy_store
        self.checkpoint_path = checkpoint_path
        self.collector = collector
        self.updater = updater
        self.league = league
        self.scheduler = scheduler
        self.training_config = training_config
        self.pool_config = pool_config
        self.metric_sink = metric_sink
        self.cancel_requested = cancel_requested

    def run(self, start_episode: int = 0) -> None:
        self.collector.vector_env.reset()
        for episode in range(start_episode, self.training_config.num_episodes):
            if self.cancel_requested():
                self._save(episode)
                return
            for group in self.updater.optimizer.param_groups:
                group["lr"] = self.scheduler.lr(episode)
            entropy = self.scheduler.entropy_coef(episode)
            self.collector.reset_completed()
            self.policy.eval()
            started = time.monotonic()
            pool_wins, pool_games = self.collector.collect()
            rollout_seconds = time.monotonic() - started
            trajectories = self.collector.buffer.get_batches(
                self.policy.device, self.training_config
            )
            if not trajectories:
                logging.warning("No trajectories collected, skipping update")
                continue
            stats = self.updater.update(trajectories, episode, entropy)
            self.league.update_shadow(self.policy)
            if (episode + 1) % self.pool_config.snapshot_interval == 0:
                self.league.add(self.policy, f"ep{episode + 1}")
                self.league.maybe_promote(build_reference_batch(self.collector))
                self.league.save_state()
            metrics = {
                key: float(value) for key, value in stats.items() if isinstance(value, (int, float))
            }
            metrics.update(
                {
                    "pool_win_rate": pool_wins / pool_games if pool_games else 0.0,
                    "rollout_seconds": rollout_seconds,
                    "learning_rate": float(self.updater.optimizer.param_groups[0]["lr"]),
                    "trajectory_count": float(len(trajectories)),
                }
            )
            phase = "warmup" if episode < self.training_config.warmup_episodes else "train"
            self.metric_sink(metrics, episode + 1, phase)
            if (episode + 1) % 10 == 0:
                self._save(episode + 1)

    def _save(self, episode: int) -> None:
        self.policy_store.save_training_state(
            self.checkpoint_path,
            episode,
            self.policy,
            optimizer=self.updater.optimizer,
            scaler=self.updater.scaler,
        )
