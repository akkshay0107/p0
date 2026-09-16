"""PPO training loop and checkpoint management."""

from __future__ import annotations

import logging
from collections.abc import Callable

from torch.amp import GradScaler
from torch.optim import Optimizer

from p0.model.policy import PolicyNet
from p0.training.config import TrainingConfig
from p0.training.files import TrainingRun
from p0.training.magnet import Magnet
from p0.training.ppo import ppo_update
from p0.training.rollout import RolloutCollector
from p0.training.trajectory import PreparedTrajectory
from p0.training.utils import PPOScheduler

LOGGER = logging.getLogger(__name__)

PPO_BOARD_METRICS = (
    "policy_loss",
    "value_loss",
    "kl_divergence",
    "clip_fraction",
    "normalized_entropy",
    "explained_variance",
    "magnet_kl",
    "grad_norm",
    "mean_game_length",
    "timeout_or_truncation_rate",
)


def _rollout_metrics(trajectories: list[PreparedTrajectory]) -> dict[str, float]:
    """Summarize completed self-play games for the PPO board."""
    if not trajectories:
        raise ValueError("Cannot summarize an empty PPO rollout")

    lengths = [int(trajectory.length) for trajectory in trajectories]
    truncated = sum(trajectory.truncated for trajectory in trajectories)
    return {
        "mean_game_length": sum(lengths) / len(lengths),
        "timeout_or_truncation_rate": truncated / len(lengths),
    }


class PPOTrainer:
    def __init__(
        self,
        *,
        policy: PolicyNet,
        files: TrainingRun,
        collector: RolloutCollector,
        optimizer: Optimizer,
        scaler: GradScaler,
        magnet: Magnet,
        scheduler: PPOScheduler,
        training_config: TrainingConfig,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.policy = policy
        self.files = files
        self.collector = collector
        self.optimizer = optimizer
        self.scaler = scaler
        self.magnet = magnet
        self.scheduler = scheduler
        self.training_config = training_config
        self.cancel_requested = cancel_requested

    def run(self, start_episode: int = 0) -> None:
        self.collector.vector_env.reset()
        refresh_interval = self.training_config.magnet_refresh_interval
        completed_episode = start_episode
        for episode in range(start_episode, self.training_config.num_episodes):
            if self.cancel_requested():
                self._save(episode)
                return
            for group in self.optimizer.param_groups:
                group["lr"] = self.scheduler.lr(episode)
            alpha = self.scheduler.alpha(episode)
            self.collector.reset_completed()
            self.policy.eval()
            self.collector.collect(self.cancel_requested)
            trajectories = self.collector.get_batches(self.policy.device)
            if self.cancel_requested():
                del trajectories
                self._save(episode)
                return
            if not trajectories:
                raise RuntimeError("PPO rollout completed without a finished trajectory")
            trajectory_count = len(trajectories)
            rollout_metrics = _rollout_metrics(trajectories)
            try:
                stats = ppo_update(
                    trajectories,
                    self.policy,
                    self.magnet,
                    self.optimizer,
                    self.scaler,
                    self.training_config,
                    episode,
                    alpha,
                    cancel_requested=self.cancel_requested,
                )
            finally:
                # The next rollout does not need the previous update's GPU copies.
                del trajectories
            if self.cancel_requested() and int(stats["optimizer_updates"]) == 0:
                self._save(episode)
                return
            if int(stats["optimizer_updates"]) == 0:
                raise RuntimeError("PPO completed an episode without an optimizer update")
            if (episode + 1) % refresh_interval == 0:
                self.magnet.refresh(self.policy)
                LOGGER.info("Refreshed magnet at episode %s", episode + 1)
            metrics = {name: float(stats[name]) for name in PPO_BOARD_METRICS if name in stats}
            metrics.update(rollout_metrics)
            self.files.record(
                episode + 1, {"trajectory_count": trajectory_count, **metrics}, {"train": metrics}
            )
            completed_episode = episode + 1
            if self.cancel_requested():
                self._save(completed_episode)
                return
            if (episode + 1) % 10 == 0:
                self._save(episode + 1)
        if completed_episode > start_episode and completed_episode % 10 != 0:
            self._save(completed_episode)

    def _save(self, episode: int) -> None:
        self.collector.prepare_for_checkpoint()
        metadata: dict[str, object] = {
            "gamma": self.training_config.gamma,
            "value_target_semantics": "discounted_terminal_outcome.v1",
            "environment_state": self.collector.vector_env.training_state(),
            "collector_state": self.collector.training_state(),
        }
        self.files.save(
            episode,
            self.policy,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            magnet=self.magnet,
            metadata=metadata,
        )
