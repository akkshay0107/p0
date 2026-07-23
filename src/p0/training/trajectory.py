"""Typed active trajectory storage and completed PPO batches."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from p0.format_config import FORMAT
from p0.model.structured_observation import StructuredObservation


@dataclass(slots=True)
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
        for name, tensor in (("returns", self.returns), ("advantages", self.advantages)):
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
        )

    def target_slices(self, target_size: int) -> list[slice]:
        """Return independent target windows for bounded PPO recomputation."""
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        return [
            slice(start, min(start + target_size, self.length))
            for start in range(0, self.length, target_size)
        ]


@dataclass(slots=True)
class TrajectoryStorage:
    step_counts: torch.Tensor
    observations: StructuredObservation
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_masks: torch.Tensor
    max_steps: int

    @classmethod
    def allocate(
        cls,
        n_envs: int,
        max_steps: int,
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
        self.step_counts[env_ids] += 1
        return steps

    def complete(self, env_id: int) -> TrajectoryBatch:
        length = int(self.step_counts[env_id].item())
        if length == 0:
            raise ValueError(f"Environment {env_id} has no trajectory steps to complete")
        batch = TrajectoryBatch(
            observations=self.observations[env_id, :length].clone(),
            actions=self.actions[env_id, :length].clone(),
            log_probs=self.log_probs[env_id, :length].clone(),
            values=self.values[env_id, :length].clone(),
            rewards=self.rewards[env_id, :length].clone(),
            dones=self.dones[env_id, :length].clone(),
            action_masks=self.action_masks[env_id, :length].clone(),
            length=length,
        )
        self.step_counts[env_id] = 0
        return batch


def compute_gae_batch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching padded shapes.")
    if rewards.dim() != 2 or lengths.shape != (rewards.size(0),):
        raise ValueError("Expected (batch, time) tensors and one length per batch row.")
    batch_size, max_steps = rewards.shape
    lengths = lengths.to(rewards.device)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(max_steps)):
        active = step < lengths
        next_value = (
            torch.where(step + 1 < lengths, values[:, step + 1], 0.0)
            if step + 1 < max_steps
            else torch.zeros_like(gae)
        )
        nonterminal = 1.0 - dones[:, step]
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
    advantages = compute_gae_batch(rewards, values, dones, lengths, gamma, gae_lambda)
    completed = []
    for index, trajectory in enumerate(trajectories):
        advantage = advantages[index, : trajectory.length]
        completed.append(
            replace(
                trajectory,
                returns=advantage + trajectory.values,
                advantages=advantage,
            ).to(device)
        )
    flat = torch.cat([batch.advantages for batch in completed if batch.advantages is not None])
    mean, std = flat.mean(), flat.std(unbiased=False).clamp_min(1e-8)
    return [
        replace(batch, advantages=(batch.advantages - mean) / std)
        for batch in completed
        if batch.advantages is not None
    ]
