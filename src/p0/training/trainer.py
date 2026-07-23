"""Episode-level PPO training lifecycle."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from p0.model.policy import PolicyNet
from p0.training.checkpoint import PolicyStore
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import PPOUpdater
from p0.training.rollout import RolloutCollector
from p0.training.utils import PPOScheduler

MetricSink = Callable[[Mapping[str, float], int, str], None]


class PPOTrainer:
    def __init__(
        self,
        *,
        policy: PolicyNet,
        policy_store: PolicyStore,
        checkpoint_path: Path,
        collector: RolloutCollector,
        updater: PPOUpdater,
        magnet: Magnet,
        scheduler: PPOScheduler,
        training_config: TrainingConfig,
        metric_sink: MetricSink = lambda metrics, step, phase: None,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.policy = policy
        self.policy_store = policy_store
        self.checkpoint_path = checkpoint_path
        self.collector = collector
        self.updater = updater
        self.magnet = magnet
        self.scheduler = scheduler
        self.training_config = training_config
        self.metric_sink = metric_sink
        self.cancel_requested = cancel_requested

    def run(self, start_episode: int = 0) -> None:
        self.collector.vector_env.reset()
        refresh_interval = self.training_config.magnet_refresh_interval
        completed_episode = start_episode
        for episode in range(start_episode, self.training_config.num_episodes):
            if self.cancel_requested():
                self._save(episode)
                return
            for group in self.updater.optimizer.param_groups:
                group["lr"] = self.scheduler.lr(episode)
            alpha = self.scheduler.alpha(episode)
            self.collector.reset_completed()
            self.policy.eval()
            started = time.monotonic()
            self.collector.collect()
            rollout_seconds = time.monotonic() - started
            trajectories = self.collector.buffer.get_batches(
                self.policy.device, self.training_config
            )
            if not trajectories:
                logging.warning("No trajectories collected, skipping update")
                completed_episode = episode + 1
                continue
            stats = self.updater.update(trajectories, episode, alpha)
            if (episode + 1) % refresh_interval == 0:
                self.magnet.refresh(self.policy)
                logging.info(f"Refreshed magnet at episode {episode + 1}")
            metrics = {
                key: float(value) for key, value in stats.items() if isinstance(value, (int, float))
            }
            metrics.update(
                {
                    "rollout_seconds": rollout_seconds,
                    "learning_rate": float(self.updater.optimizer.param_groups[0]["lr"]),
                    "trajectory_count": float(len(trajectories)),
                }
            )
            phase = "warmup" if episode < self.training_config.warmup_episodes else "train"
            self.metric_sink(metrics, episode + 1, phase)
            completed_episode = episode + 1
            if (episode + 1) % 10 == 0:
                self._save(episode + 1)
        if completed_episode % 10 != 0:
            self._save(completed_episode)

    def _save(self, episode: int) -> None:
        self.policy_store.save_training_state(
            self.checkpoint_path,
            episode,
            self.policy,
            optimizer=self.updater.optimizer,
            scheduler=self.scheduler,
            scaler=self.updater.scaler,
            magnet=self.magnet,
            trainer_kind="ppo",
        )
