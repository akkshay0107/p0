"""Collected and PPO-prepared trajectory lifecycles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW, MAX_PRIOR_GAMES
from p0.model.structured_observation import StructuredObservation
from p0.training.series_history import SeriesHistorySnapshot


def _validate_trajectory(
    observations: StructuredObservation,
    action_masks: torch.Tensor,
    actions: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    length: int,
) -> None:
    if type(length) is not int or length <= 0:
        raise ValueError("Completed trajectories must contain at least one step")
    if any(
        tensor.size(0) != length
        for tensor in (action_masks, actions, log_probs, values, rewards, dones)
    ):
        raise ValueError("Trajectory tensor lengths do not match")
    observations.validate(batch_rank=1)


def _validate_series_snapshot(series_history: SeriesHistorySnapshot) -> None:
    if not isinstance(series_history, tuple) or len(series_history) > MAX_PRIOR_GAMES:
        raise ValueError("series_history must be a tuple containing at most two games")
    for history in series_history:
        if (
            not isinstance(history, torch.Tensor)
            or history.dim() != 2
            or history.size(0) == 0
            or history.size(1) == 0
        ):
            raise ValueError("series_history entries must be non-empty two-dimensional tensors")


@dataclass(frozen=True, slots=True)
class CollectedTrajectory:
    """Rollout-complete trajectory before GAE preparation."""

    observations: StructuredObservation
    action_masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    length: int
    bootstrap_value: float
    series_history: SeriesHistorySnapshot

    def __post_init__(self) -> None:
        _validate_trajectory(
            self.observations,
            self.action_masks,
            self.actions,
            self.log_probs,
            self.values,
            self.rewards,
            self.dones,
            self.length,
        )
        _validate_series_snapshot(self.series_history)


@dataclass(frozen=True, slots=True)
class PreparedTrajectory:
    """Trajectory prepared with returns and advantages for PPO."""

    observations: StructuredObservation
    action_masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    length: int
    series_history: SeriesHistorySnapshot
    returns: torch.Tensor
    advantages: torch.Tensor
    explained_variance: float
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("Prepared trajectories must contain at least one step")
        if any(
            tensor.size(0) != self.length
            for tensor in (
                self.action_masks,
                self.actions,
                self.log_probs,
                self.returns,
                self.advantages,
            )
        ):
            raise ValueError("Prepared trajectory target lengths do not match")
        self.observations.validate(batch_rank=1)
        if not isinstance(self.explained_variance, float):
            raise ValueError("explained_variance must be a float")
        if type(self.truncated) is not bool:
            raise ValueError("truncated must be a bool")
        _validate_series_snapshot(self.series_history)


@dataclass(frozen=True, slots=True)
class TrajectoryStorage:
    step_counts: torch.Tensor
    observations: StructuredObservation
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_masks: torch.Tensor
    history_tokens: torch.Tensor
    history_offsets: torch.Tensor

    @classmethod
    def allocate(
        cls,
        n_envs: int,
        max_steps: int,
        d_model: int,
        history_device: torch.device | str,
    ) -> TrajectoryStorage:
        if n_envs <= 0 or max_steps <= 0 or d_model <= 0:
            raise ValueError("Trajectory dimensions must be positive")

        flat = StructuredObservation.empty_batch(n_envs * max_steps)
        observations = StructuredObservation._from_values(
            [value.reshape(n_envs, max_steps, *value.shape[1:]) for value in flat.tensors()]
        )
        return cls(
            step_counts=torch.zeros(n_envs, dtype=torch.long),
            observations=observations,
            actions=torch.zeros((n_envs, max_steps, 2), dtype=torch.long),
            log_probs=torch.zeros((n_envs, max_steps), dtype=torch.float32),
            values=torch.zeros((n_envs, max_steps), dtype=torch.float32),
            rewards=torch.zeros((n_envs, max_steps), dtype=torch.float32),
            dones=torch.zeros((n_envs, max_steps), dtype=torch.float32),
            action_masks=torch.zeros((n_envs, max_steps, 2, FORMAT.action_size), dtype=torch.bool),
            history_tokens=torch.zeros(
                (n_envs, max_steps, d_model), dtype=torch.float32, device=history_device
            ),
            history_offsets=torch.arange(-HISTORY_WINDOW, 0, device=history_device),
        )

    def ensure_capacity(self, env_ids: torch.Tensor) -> None:
        max_steps = self.actions.size(1)
        overflowing = env_ids[self.step_counts[env_ids] >= max_steps]
        if overflowing.numel():
            raise OverflowError(
                f"Trajectory exceeded {max_steps} steps for environments {overflowing.tolist()}"
            )

    def record(
        self,
        env_ids: torch.Tensor,
        observations: StructuredObservation,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        action_masks: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Store one decision for each selected environment and return its step indices."""
        self.ensure_capacity(env_ids)
        steps = self.step_counts[env_ids]
        if history_tokens.shape != (env_ids.numel(), self.history_tokens.size(2)):
            raise ValueError("History token batch does not match selected environments")
        history_env_ids = env_ids.to(self.history_tokens.device)
        history_steps = steps.to(self.history_tokens.device)
        for destination, source in zip(
            self.observations.tensors(), observations.tensors(), strict=True
        ):
            destination[env_ids, steps] = source
        self.actions[env_ids, steps] = actions
        self.log_probs[env_ids, steps] = log_probs
        self.values[env_ids, steps] = values
        self.action_masks[env_ids, steps] = action_masks
        self.history_tokens[history_env_ids, history_steps] = history_tokens.detach().to(
            self.history_tokens
        )
        self.step_counts[env_ids] += 1
        return steps

    def history_inputs(
        self,
        env_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return recent game history tokens and mask for selected environments."""
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        env_ids_device = env_ids_cpu.to(self.history_tokens.device)
        lengths = self.step_counts[env_ids_cpu].to(self.history_tokens.device)
        indices = lengths.unsqueeze(1) + self.history_offsets.unsqueeze(0)
        mask = indices >= 0
        history = self.history_tokens[
            env_ids_device.unsqueeze(1), indices.clamp_min(0)
        ].masked_fill(~mask.unsqueeze(-1), 0.0)
        return history.to(device=device, dtype=dtype), mask.to(device)

    def full_history(self, env_id: int) -> torch.Tensor | None:
        """Return the full game history for an environment."""
        length = int(self.step_counts[env_id].item())
        if not length:
            return None
        return self.history_tokens[env_id, :length].unsqueeze(0)

    def complete(
        self,
        env_id: int,
        bootstrap_value: float,
        series_history: SeriesHistorySnapshot,
    ) -> CollectedTrajectory:
        """Detach one completed environment trajectory from active storage."""
        length = int(self.step_counts[env_id].item())
        if length == 0:
            raise ValueError(f"Environment {env_id} has no trajectory steps to complete")

        trajectory = CollectedTrajectory(
            observations=self.observations[env_id, :length].clone(),
            actions=self.actions[env_id, :length].clone(),
            log_probs=self.log_probs[env_id, :length].clone(),
            values=self.values[env_id, :length].clone(),
            rewards=self.rewards[env_id, :length].clone(),
            dones=self.dones[env_id, :length].clone(),
            action_masks=self.action_masks[env_id, :length].clone(),
            length=length,
            bootstrap_value=bootstrap_value,
            series_history=series_history,
        )
        self.step_counts[env_id] = 0
        return trajectory


def compute_gae_batch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: torch.Tensor,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation over padded trajectories."""
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching padded shapes.")

    if rewards.dim() != 2 or lengths.shape != (rewards.size(0),):
        raise ValueError("Expected (batch, time) tensors and one length per batch row.")

    batch_size, max_steps = rewards.shape
    lengths = lengths.to(rewards.device)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    bootstrap_values = bootstrap_values.to(rewards.device)

    extended_values = torch.zeros(
        (batch_size, max_steps + 1), dtype=values.dtype, device=values.device
    )
    extended_values[:, :max_steps] = values

    batch_indices = torch.arange(batch_size, device=values.device)
    extended_values[batch_indices, lengths] = bootstrap_values

    for step in reversed(range(max_steps)):
        active = step < lengths
        nonterminal = 1.0 - dones[:, step]
        next_value = extended_values[:, step + 1]

        delta = rewards[:, step] + gamma * next_value * nonterminal - values[:, step]
        gae = torch.where(active, delta + gamma * gae_lambda * nonterminal * gae, 0.0)
        advantages[:, step] = gae

    return advantages


def prepare_trajectory_batches(
    trajectories: list[CollectedTrajectory],
    device: torch.device,
    *,
    gamma: float,
    gae_lambda: float,
) -> list[PreparedTrajectory]:
    """Compute normalized GAE and create the mandatory prepared PPO lifecycle."""
    if not trajectories:
        return []

    rewards = torch.nn.utils.rnn.pad_sequence(
        [trajectory.rewards for trajectory in trajectories], batch_first=True
    )
    values = torch.nn.utils.rnn.pad_sequence(
        [trajectory.values for trajectory in trajectories], batch_first=True
    )
    dones = torch.nn.utils.rnn.pad_sequence(
        [trajectory.dones for trajectory in trajectories], batch_first=True
    )
    lengths = torch.tensor([trajectory.length for trajectory in trajectories])
    bootstrap_values = torch.tensor(
        [trajectory.bootstrap_value for trajectory in trajectories], dtype=torch.float32
    )

    advantages = compute_gae_batch(
        rewards, values, dones, lengths, gamma, gae_lambda, bootstrap_values
    )
    all_advantages = torch.cat(
        [advantages[index, : trajectory.length] for index, trajectory in enumerate(trajectories)]
    )
    mean = all_advantages.mean()
    std = all_advantages.std(unbiased=False).clamp_min(1e-8)

    all_values = torch.cat([trajectory.values for trajectory in trajectories])
    all_returns = all_advantages + all_values
    var_y = torch.var(all_returns, unbiased=False)
    if var_y > 1e-8:
        explained_variance = float(
            (1.0 - torch.var(all_returns - all_values, unbiased=False) / var_y).item()
        )
    else:
        explained_variance = 0.0

    prepared: list[PreparedTrajectory] = []
    for index, trajectory in enumerate(trajectories):
        advantage = advantages[index, : trajectory.length]
        prepared.append(
            PreparedTrajectory(
                observations=trajectory.observations.to(device),
                action_masks=trajectory.action_masks.to(device),
                actions=trajectory.actions.to(device),
                log_probs=trajectory.log_probs.to(device),
                length=trajectory.length,
                series_history=trajectory.series_history,
                returns=(advantage + trajectory.values).to(device),
                advantages=((advantage - mean) / std).to(device),
                explained_variance=explained_variance,
                truncated=not bool(trajectory.dones[-1].item()),
            )
        )
    return prepared
