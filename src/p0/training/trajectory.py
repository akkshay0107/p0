"""Typed active trajectory storage and completed PPO batches."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import SERIES_SLOTS
from p0.model.structured_observation import StructuredObservation


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    observations: StructuredObservation
    action_masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    length: int
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    series_tokens: torch.Tensor | None = None
    series_mask: torch.Tensor | None = None
    bootstrap_value: float = 0.0
    explained_variance: float | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Completed trajectories must contain at least one step")

        if any(
            tensor.size(0) != self.length
            for tensor in (
                self.action_masks,
                self.actions,
                self.log_probs,
                self.values,
                self.rewards,
                self.dones,
            )
        ):
            raise ValueError("Trajectory tensor lengths do not match")

        for name, tensor in (
            ("returns", self.returns),
            ("advantages", self.advantages),
            ("series_tokens", self.series_tokens),
            ("series_mask", self.series_mask),
        ):
            if tensor is not None and tensor.size(0) != self.length:
                raise ValueError(f"Trajectory {name} length does not match")

        self.observations.validate(batch_rank=1)

    def to(self, device: torch.device | str) -> TrajectoryBatch:
        return replace(
            self,
            observations=self.observations.to(device),
            action_masks=self.action_masks.to(device),
            actions=self.actions.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            returns=None if self.returns is None else self.returns.to(device),
            advantages=None if self.advantages is None else self.advantages.to(device),
            series_tokens=None if self.series_tokens is None else self.series_tokens.to(device),
            series_mask=None if self.series_mask is None else self.series_mask.to(device),
            bootstrap_value=self.bootstrap_value,
        )

    def to_ppo_device(self, device: torch.device | str) -> TrajectoryBatch:
        """Move only tensors consumed by the PPO update to device."""
        return replace(
            self,
            observations=self.observations.to(device),
            action_masks=self.action_masks.to(device),
            actions=self.actions.to(device),
            log_probs=self.log_probs.to(device),
            returns=None if self.returns is None else self.returns.to(device),
            advantages=None if self.advantages is None else self.advantages.to(device),
            series_tokens=None if self.series_tokens is None else self.series_tokens.to(device),
            series_mask=None if self.series_mask is None else self.series_mask.to(device),
        )


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
    series_tokens: torch.Tensor
    series_mask: torch.Tensor
    max_steps: int

    @classmethod
    def allocate(
        cls,
        n_envs: int,
        max_steps: int,
        d_model: int,
        device: torch.device | str = "cpu",
    ) -> TrajectoryStorage:
        if n_envs <= 0 or max_steps <= 0:
            raise ValueError("n_envs and max_steps must be positive")

        flat = StructuredObservation.empty_batch(n_envs * max_steps).to(device)
        observations = StructuredObservation._from_values(
            [value.reshape(n_envs, max_steps, *value.shape[1:]) for value in flat.tensors()]
        )
        return cls(
            step_counts=torch.zeros(n_envs, dtype=torch.long, device=device),
            observations=observations,
            actions=torch.zeros((n_envs, max_steps, 2), dtype=torch.long, device=device),
            log_probs=torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
            values=torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
            rewards=torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
            dones=torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
            action_masks=torch.zeros(
                (n_envs, max_steps, 2, FORMAT.action_size), dtype=torch.bool, device=device
            ),
            series_tokens=torch.zeros(
                (n_envs, max_steps, SERIES_SLOTS, d_model), dtype=torch.float32, device=device
            ),
            series_mask=torch.zeros(
                (n_envs, max_steps, SERIES_SLOTS), dtype=torch.bool, device=device
            ),
            max_steps=max_steps,
        )

    def ensure_capacity(self, env_ids: torch.Tensor) -> None:
        overflowing = env_ids[self.step_counts[env_ids] >= self.max_steps]
        if overflowing.numel():
            raise OverflowError(
                f"Trajectory exceeded {self.max_steps} steps for environments "
                f"{overflowing.tolist()}"
            )

    def record(
        self,
        env_ids: torch.Tensor,
        observations: StructuredObservation,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        action_masks: torch.Tensor,
        series_tokens: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Store one decision for each selected environment and return its step indices."""
        self.ensure_capacity(env_ids)
        steps = self.step_counts[env_ids]
        for destination, source in zip(
            self.observations.tensors(), observations.tensors(), strict=True
        ):
            destination[env_ids, steps] = source
        self.actions[env_ids, steps] = actions
        self.log_probs[env_ids, steps] = log_probs
        self.values[env_ids, steps] = values
        self.action_masks[env_ids, steps] = action_masks
        self.series_tokens[env_ids, steps] = series_tokens
        self.series_mask[env_ids, steps] = series_mask
        self.step_counts[env_ids] += 1
        return steps

    def complete(self, env_id: int, bootstrap_value: float = 0.0) -> TrajectoryBatch:
        length = int(self.step_counts[env_id].item())
        if length == 0:
            raise ValueError(f"Environment {env_id} has no trajectory steps to complete")

        self.step_counts[env_id] = 0
        return TrajectoryBatch(
            observations=self.observations[env_id, :length].clone(),
            actions=self.actions[env_id, :length].clone(),
            log_probs=self.log_probs[env_id, :length].clone(),
            values=self.values[env_id, :length].clone(),
            rewards=self.rewards[env_id, :length].clone(),
            dones=self.dones[env_id, :length].clone(),
            action_masks=self.action_masks[env_id, :length].clone(),
            series_tokens=self.series_tokens[env_id, :length].clone(),
            series_mask=self.series_mask[env_id, :length].clone(),
            length=length,
            bootstrap_value=bootstrap_value,
        )


def compute_gae_batch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    bootstrap_values: torch.Tensor,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE) over a batch of trajectories."""
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching padded shapes.")

    if rewards.dim() != 2 or lengths.shape != (rewards.size(0),):
        raise ValueError("Expected (batch, time) tensors and one length per batch row.")

    batch_size, max_steps = rewards.shape
    lengths = lengths.to(rewards.device)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    bootstrap_values = bootstrap_values.to(rewards.device)

    # Pad with an extra slot containing bootstrap values for games that do not
    # end within the allotted turns. Completed games receive a zero bootstrap,
    # so this slot does not affect their advantages.
    extended_values = torch.zeros(
        (batch_size, max_steps + 1), dtype=values.dtype, device=values.device
    )
    extended_values[:, :max_steps] = values

    # step < lengths is our active mask, step + 1 == lengths evaluates to the exact boundary.
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
    trajectories: list[TrajectoryBatch],
    device: torch.device,
    *,
    gamma: float,
    gae_lambda: float,
) -> list[TrajectoryBatch]:
    """Pad and compute normalized GAE advantages for a list of trajectory batches."""
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
    completed = []

    for index, trajectory in enumerate(trajectories):
        advantage = advantages[index, : trajectory.length]
        completed.append(
            replace(
                trajectory,
                returns=advantage + trajectory.values,
                advantages=advantage,
            )
        )

    flat = torch.cat([batch.advantages for batch in completed if batch.advantages is not None])
    mean, std = flat.mean(), flat.std(unbiased=False).clamp_min(1e-8)

    all_returns = torch.cat([batch.returns for batch in completed if batch.returns is not None])
    all_values = torch.cat([batch.values for batch in completed])
    var_y = torch.var(all_returns, unbiased=False)
    if var_y > 1e-8:
        explained_variance = float(
            (1.0 - torch.var(all_returns - all_values, unbiased=False) / var_y).item()
        )
    else:
        explained_variance = 0.0

    return [
        replace(
            batch,
            advantages=(batch.advantages - mean) / std,
            explained_variance=explained_variance,
        ).to_ppo_device(device)
        for batch in completed
        if batch.advantages is not None
    ]
