"""Typed active trajectory storage and completed PPO batches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import SERIES_SLOTS
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
    series_tokens: torch.Tensor | None = None
    series_mask: torch.Tensor | None = None
    series_features: Any | None = None
    series_id: str | None = None
    game_number: int = 1
    player: int | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("Completed trajectories must contain at least one step")
        if self.game_number < 1:
            raise ValueError("Trajectory game_number must be positive")
        if self.player is not None and self.player not in (0, 1):
            raise ValueError("Trajectory player must be 0 or 1 when provided")
        if self.series_id is not None and not self.series_id:
            raise ValueError("Trajectory series_id must be non-empty when provided")
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
        if (self.series_tokens is None) != (self.series_mask is None):
            raise ValueError("series_tokens and series_mask must be provided together")
        if self.series_tokens is not None:
            if self.series_mask is None:
                raise ValueError("series_mask is required with series_tokens")
            if self.series_tokens.dim() != 2 or self.series_mask.shape != (
                self.series_tokens.size(0),
            ):
                raise ValueError("Invalid fixed series context shapes")
        if self.series_features is not None and self.series_tokens is not None:
            raise ValueError("Use raw series_features or encoded series_tokens, not both")
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
            series_features=self.series_features,
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
    series_tokens: torch.Tensor | None = None
    series_masks: torch.Tensor | None = None
    series_features: list[Any | None] | None = None
    series_ids: list[str | None] | None = None
    game_numbers: torch.Tensor | None = None
    player_index: int | None = None

    @classmethod
    def allocate(
        cls,
        n_envs: int,
        max_steps: int,
        device: torch.device | str = "cpu",
        d_model: int | None = None,
        player_index: int | None = None,
    ) -> TrajectoryStorage:
        if n_envs <= 0 or max_steps <= 0:
            raise ValueError("n_envs and max_steps must be positive")
        flat = StructuredObservation.empty_batch(n_envs * max_steps).to(device)
        observations = StructuredObservation._from_values(
            [value.reshape(n_envs, max_steps, *value.shape[1:]) for value in flat.tensors()]
        )
        if d_model is not None and d_model <= 0:
            raise ValueError("d_model must be positive when provided")
        if player_index is not None and player_index not in (0, 1):
            raise ValueError("player_index must be 0 or 1 when provided")
        series_tokens = (
            torch.zeros((n_envs, SERIES_SLOTS, d_model), dtype=torch.float32, device=device)
            if d_model is not None
            else None
        )
        series_masks = (
            torch.zeros((n_envs, SERIES_SLOTS), dtype=torch.bool, device=device)
            if d_model is not None
            else None
        )
        series_features: list[Any | None] | None = None
        series_ids: list[str | None] | None = None
        game_numbers: torch.Tensor | None = None
        if d_model is not None:
            series_features = [None for _ in range(n_envs)]
        series_ids = [None for _ in range(n_envs)]
        game_numbers = torch.ones(n_envs, dtype=torch.long, device=device)
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
            series_tokens=series_tokens,
            series_masks=series_masks,
            series_features=series_features,
            series_ids=series_ids,
            game_numbers=game_numbers,
            player_index=player_index,
        )

    def set_context(
        self,
        env_ids: torch.Tensor,
        series_ids: list[str | None],
        game_numbers: torch.Tensor,
    ) -> None:
        """Set the causal boundary metadata for the next recorded decisions."""
        if self.series_ids is None or self.game_numbers is None:
            raise RuntimeError("TrajectoryStorage was not initialized with context metadata")
        if len(series_ids) != env_ids.numel() or game_numbers.shape != (env_ids.numel(),):
            raise ValueError("Trajectory context metadata must match selected environments")
        if game_numbers.dtype != torch.long:
            raise ValueError("Trajectory game_numbers must use torch.long")
        if torch.any(game_numbers < 1):
            raise ValueError("Trajectory game_numbers must be positive")
        for env_id, series_id, game_number in zip(
            env_ids.tolist(), series_ids, game_numbers.tolist(), strict=True
        ):
            if series_id is not None and not series_id:
                raise ValueError("Trajectory series_id must be non-empty when provided")
            self.series_ids[env_id] = series_id
            self.game_numbers[env_id] = game_number

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
        series_tokens: torch.Tensor | None = None,
        series_mask: torch.Tensor | None = None,
        series_features: list[Any | None] | None = None,
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
        if (
            self.series_tokens is not None
            and self.series_masks is not None
            and series_tokens is not None
            and series_mask is not None
        ):
            self.series_tokens[env_ids] = series_tokens.to(self.series_tokens)
            self.series_masks[env_ids] = series_mask.to(self.series_masks)
        if self.series_features is not None and series_features is not None:
            if len(series_features) != env_ids.numel():
                raise ValueError("series_features must have one entry per selected environment")
            for env_id, features in zip(env_ids.tolist(), series_features, strict=True):
                self.series_features[env_id] = features
        self.step_counts[env_ids] += 1
        return steps

    def complete(self, env_id: int) -> TrajectoryBatch:
        length = int(self.step_counts[env_id].item())
        if length == 0:
            raise ValueError(f"Environment {env_id} has no trajectory steps to complete")
        raw_series_features = None if self.series_features is None else self.series_features[env_id]
        batch = TrajectoryBatch(
            observations=self.observations[env_id, :length].clone(),
            actions=self.actions[env_id, :length].clone(),
            log_probs=self.log_probs[env_id, :length].clone(),
            values=self.values[env_id, :length].clone(),
            rewards=self.rewards[env_id, :length].clone(),
            dones=self.dones[env_id, :length].clone(),
            action_masks=self.action_masks[env_id, :length].clone(),
            length=length,
            series_tokens=(
                None
                if raw_series_features is not None or self.series_tokens is None
                else self.series_tokens[env_id].clone()
            ),
            series_mask=(
                None
                if raw_series_features is not None or self.series_masks is None
                else self.series_masks[env_id].clone()
            ),
            series_features=raw_series_features,
            series_id=None if self.series_ids is None else self.series_ids[env_id],
            game_number=(1 if self.game_numbers is None else int(self.game_numbers[env_id].item())),
            player=self.player_index,
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
