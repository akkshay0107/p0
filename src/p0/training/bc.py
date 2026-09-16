"""Behavior-cloning trainer."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from p0.battle.actions import TEAM_SIZE
from p0.model.policy import MemoryInputs, PolicyNet, PreparedDecision
from p0.replays.dataset import LazyReplayDataset, ReplayGameChunk
from p0.runtime.process_context import PROCESS_CONTEXT
from p0.training._bc_batch import (
    BCDecisionBatch,
    BCGameWindow,
    _BCUpdateDataset,
    collate_bc_batches,
)
from p0.training._bc_history import (
    commit_history_updates,
    prepare_series_context,
)
from p0.training._bc_metrics import (
    BCEvaluationMetrics,
    BCObjective,
    _BCEvaluationAccumulator,
    _policy_loss_sum,
    _ragged_logsumexp,
    _validate_objective_inputs,
    compute_bc_objective,
)
from p0.training.config import BCConfig
from p0.training.series_history import SeriesHistoryStore
from p0.training.utils import select_optimization_precision

LOGGER = logging.getLogger(__name__)


def _expand_team_preview_orbits(
    candidate_values: Tensor,
    candidate_offsets: Tensor,
    team_preview: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return each row's distinct candidates, adding preview pair orientations."""
    if candidate_values.numel() == 0:
        return candidate_values, candidate_offsets

    if candidate_values.device.type != "cpu" or candidate_offsets.device.type != "cpu":
        raise ValueError("BC candidate expansion expects CPU tensors")

    values = candidate_values.tolist()
    offsets = candidate_offsets.tolist()
    preview_rows = team_preview.to(device="cpu", dtype=torch.bool).tolist()
    expanded: list[tuple[int, int]] = []
    expanded_offsets = [0]

    for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        unique: dict[tuple[int, int], None] = {}
        for first, second in values[start:stop]:
            unique[(first, second)] = None
            if preview_rows[row]:
                first_swap = (first % TEAM_SIZE) * TEAM_SIZE + first // TEAM_SIZE
                second_swap = (second % TEAM_SIZE) * TEAM_SIZE + second // TEAM_SIZE
                unique[(first_swap, second)] = None
                unique[(first, second_swap)] = None
                unique[(first_swap, second_swap)] = None
        expanded.extend(unique)
        expanded_offsets.append(len(expanded))

    return (
        candidate_values.new_tensor(expanded).reshape(-1, 2),
        candidate_offsets.new_tensor(expanded_offsets),
    )


class BCCancelled(RuntimeError):
    """Raised between batches so callers keep the last completed epoch checkpoint."""


def _empty_training_totals() -> dict[str, Any]:
    return {
        "loss": 0.0,
        "loss_weight": 0.0,
        "decisions": 0,
        "updates": 0,
        "games": 0,
        "grad_norm_sum": 0.0,
    }


def _seed_bc_worker(worker_id: int) -> None:
    """Derive independent Python and NumPy worker streams from Torch's seed."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _discounted_outcome_targets(
    outcome: Tensor,
    decision_index: Tensor,
    game_length: Tensor,
    gamma: Tensor,
) -> Tensor:
    """Discount terminal outcomes by the number of following recorded decisions."""
    exponent = (game_length - 1 - decision_index).to(outcome.dtype)
    return torch.pow(gamma, exponent) * outcome


class BCTrainer:
    """Train a policy on complete games with gathered immutable history."""

    def __init__(
        self,
        policy: PolicyNet,
        dataset: Iterable[ReplayGameChunk],
        config: BCConfig,
        *,
        device: torch.device | str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.policy = policy.to(device)
        self.dataset = dataset
        self.config = config
        self.device = torch.device(device)
        self._gamma = torch.tensor(config.gamma, device=self.device, dtype=torch.float32)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.precision = select_optimization_precision(config.enable_optim, self.device)
        self.scaler = GradScaler(device=self.device.type, enabled=self.precision.grad_scaler)
        self.batch_decisions = config.batch_decisions
        self._series_history = SeriesHistoryStore(policy.d_model)
        self.cancel_requested = cancel_requested

    def train(self) -> dict[str, float | int]:
        """Run configured epochs over the streaming dataset."""
        if self.config.epochs > 1 and iter(self.dataset) is self.dataset:
            raise ValueError("BC datasets must be re-iterable when epochs is greater than one")

        totals = _empty_training_totals()
        for _ in range(self.config.epochs):
            epoch_totals = self._train_epoch_totals()
            for key in totals:
                totals[key] += epoch_totals[key]
        return self._metrics(totals)

    def train_epoch(self) -> dict[str, float | int]:
        """Train exactly one epoch and return its decision-weighted metrics."""
        return self._metrics(self._train_epoch_totals())

    def _train_epoch_totals(self) -> dict[str, Any]:
        self.policy.train()
        self._series_history.clear()

        totals = _empty_training_totals()
        chunk_size = min(self.batch_decisions, self.config.max_chunk_size)
        self.optimizer.zero_grad(set_to_none=True)
        num_workers = self.config.num_workers
        if num_workers > 0 and not isinstance(self.dataset, LazyReplayDataset):
            raise ValueError(
                "BC num_workers greater than zero requires a worker-sharded LazyReplayDataset"
            )
        update_groups = _BCUpdateDataset(self.dataset, self.batch_decisions, chunk_size)
        source: Iterable[tuple[BCDecisionBatch, ...]] = update_groups
        if num_workers > 0:
            source = DataLoader(
                update_groups,
                num_workers=num_workers,
                batch_size=None,
                prefetch_factor=self.config.prefetch_factor,
                multiprocessing_context=PROCESS_CONTEXT,
                worker_init_fn=_seed_bc_worker,
                generator=torch.Generator().manual_seed(self.config.seed),
            )

        try:
            for update_batches in source:
                if self.cancel_requested():
                    raise BCCancelled("Behavior-cloning training was cancelled")
                self._train_update(update_batches, totals)

            if self._series_history.has_partial_games:
                raise ValueError("BC dataset ended with an incomplete perspective-game")
        finally:
            self._series_history.clear()
        return totals

    def _train_update(
        self,
        batches: tuple[BCDecisionBatch, ...],
        totals: dict[str, Any],
    ) -> None:
        """Backpropagate and step once for one complete-game update group."""
        history_state = self._series_history.snapshot_keys(
            window.series_key for batch in batches for window in batch.windows
        )
        policy_weight = sum(float(batch.loss_mask.sum()) for batch in batches)
        value_count = sum(int(batch.outcome_valid.sum()) for batch in batches)
        try:
            update_loss = self._backward_chunk(batches[0], policy_weight, value_count)
            for batch in batches[1:]:
                update_loss += self._backward_chunk(batch, policy_weight, value_count)

            grad_norm = 0.0
            if policy_weight or value_count:
                grad_norm = self._step_optimizer()
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self._series_history.restore_keys(history_state)
            raise

        totals["loss"] += update_loss
        totals["loss_weight"] += policy_weight
        totals["decisions"] += sum(batch.decisions for batch in batches)
        totals["games"] += sum(batch.completed_game_count for batch in batches)
        if policy_weight or value_count:
            totals["updates"] += 1
            totals["grad_norm_sum"] += grad_norm

    def _metrics(self, totals: dict[str, Any]) -> dict[str, float | int]:
        updates = int(totals["updates"])
        return {
            "overall_nll": (
                float(totals["loss"]) / float(totals["loss_weight"])
                if totals["loss_weight"]
                else 0.0
            ),
            "grad_norm": (float(totals["grad_norm_sum"]) / updates if updates else 0.0),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "updates": updates,
            "games": int(totals["games"]),
            "decisions": int(totals["decisions"]),
        }

    def _prepare_model_inputs(
        self, batch: BCDecisionBatch
    ) -> tuple[PreparedDecision, Tensor, Tensor, Tensor, Tensor]:
        observations = batch.observations.to(self.device)
        context_action_mask = batch.context_action_mask.to(self.device)
        encoded = self.policy.encode(observations, context_action_mask)
        # Build a local summary for every decision from its current tokens only.
        # The target row is later reduced with its history; the local summary remains
        # the snapshot used to populate history windows for other rows.
        local_tokens = encoded.local_history_token
        target_indices = batch.target_indices.to(self.device)
        target_local_tokens = local_tokens[target_indices]
        target_encoded = encoded[target_indices]
        history_indices = batch.history_indices.to(self.device)
        history_mask = batch.history_mask.to(self.device)
        history_tokens = local_tokens[history_indices] * history_mask.unsqueeze(-1)
        candidate_values = batch.candidate_values
        candidate_offsets = batch.candidate_offsets
        candidate_values, candidate_offsets = _expand_team_preview_orbits(
            candidate_values,
            candidate_offsets,
            batch.observations.is_teampreview()[batch.target_indices],
        )
        candidate_values = candidate_values.to(self.device)
        candidate_offsets = candidate_offsets.to(self.device)
        series_tokens, series_mask = prepare_series_context(
            self._series_history,
            batch.windows,
            target_local_tokens,
            self.policy.series,
        )
        memory = MemoryInputs(
            series_tokens=series_tokens,
            series_mask=series_mask,
            history_tokens=history_tokens,
            history_mask=history_mask,
        )
        return (
            self.policy.prepare(target_encoded, memory),
            batch.action_mask.to(self.device),
            candidate_values,
            candidate_offsets,
            target_local_tokens,
        )

    def _step_optimizer(self) -> float:
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        previous_scale = self.scaler.get_scale()
        if not bool(torch.isfinite(grad_norm).item()):
            LOGGER.warning(
                "Non-finite BC gradient norm detected; discarding the accumulated update "
                f"(loss scale={previous_scale:.0f})"
            )
            self.scaler.update()
            raise FloatingPointError("BC update has non-finite gradients and was discarded")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() < previous_scale:
            raise FloatingPointError("BC update has non-finite gradients and was discarded")
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.item())

    def _backward_chunk(
        self,
        batch: BCDecisionBatch,
        policy_weight: float,
        effective_value_count: int,
    ) -> Tensor:
        """
        Backpropagate one validated decision chunk and update running totals.

        Arguments:
          batch: Collated CPU batch containing one contiguous set of decisions.
          policy_weight: Total policy label weight for the update.
          effective_value_count: Total valid outcome count for the update.

        Returns:
          Detached policy-loss numerator. Gradients are accumulated for the update.
        """
        _validate_objective_inputs(
            batch.candidate_values.size(0),
            batch.candidate_offsets,
            batch.label_kind,
            batch.loss_mask,
        )
        candidate_values = batch.candidate_values
        if candidate_values.dim() != 2 or candidate_values.shape[1] != 2:
            raise ValueError("candidate_values must have shape (candidates, 2)")
        if candidate_values.dtype != torch.long:
            raise ValueError("candidate_values must use torch.long action ids")
        if candidate_values.numel() and torch.any(
            (candidate_values < 0) | (candidate_values >= self.policy.actor.act_size)
        ):
            raise ValueError("candidate action ids are outside the action contract")
        loss_mask = batch.loss_mask.to(self.device)
        with autocast(
            device_type=self.device.type,
            enabled=self.precision.autocast,
            dtype=self.precision.dtype,
        ):
            prepared, action_mask, candidate_values, candidate_offsets, history_tokens = (
                self._prepare_model_inputs(batch)
            )
            log_probs = self.policy.score_candidates(
                prepared,
                action_mask,
                candidate_values,
                candidate_offsets,
                validated=True,
            )
            value_predictions = self.policy.critic(prepared.reduced.cls)
        log_probs = log_probs.float()
        value_predictions = value_predictions.float()
        marginal_log_probs = _ragged_logsumexp(log_probs, candidate_offsets)
        policy_sum, batch_policy_weight = _policy_loss_sum(marginal_log_probs, loss_mask)
        outcome = batch.outcome.to(device=self.device, dtype=torch.float32)
        decision_index = batch.decision_index.to(self.device)
        game_length = batch.game_length.to(self.device)
        value_mask = batch.outcome_valid.to(self.device)
        value_targets = _discounted_outcome_targets(
            outcome,
            decision_index,
            game_length,
            self._gamma,
        )
        value_error = value_predictions - value_targets
        batch_value_count = int(batch.outcome_valid.sum())
        value_sum = (
            value_error.square()[value_mask].sum()
            if batch_value_count
            else value_predictions.sum() * 0.0
        )
        policy_term = policy_sum / policy_weight if policy_weight else policy_sum * 0.0
        value_term = value_sum / effective_value_count if effective_value_count else value_sum * 0.0
        total_loss = policy_term + self.config.value_coef * value_term
        if batch_policy_weight or batch_value_count:
            if not bool(torch.isfinite(total_loss).item()):
                raise FloatingPointError("BC update has a non-finite loss and was discarded")
            self.scaler.scale(total_loss).backward()

        commit_history_updates(self._series_history, batch.windows, history_tokens)
        return policy_sum.detach()

    @torch.inference_mode()
    def evaluate(
        self,
        dataset: Iterable[ReplayGameChunk] | None = None,
    ) -> BCEvaluationMetrics:
        """Evaluate exact and partial replay labels without changing parameters."""
        self.policy.eval()
        self._series_history.clear()
        source = self.dataset if dataset is None else dataset
        accumulator = _BCEvaluationAccumulator.create(self.device)
        chunk_size = min(self.batch_decisions, self.config.max_chunk_size)

        try:
            for batch in collate_bc_batches(source, chunk_size):
                validated_masks = _validate_objective_inputs(
                    batch.candidate_values.size(0),
                    batch.candidate_offsets,
                    batch.label_kind,
                    batch.loss_mask,
                )
                (
                    prepared,
                    action_mask,
                    candidate_values,
                    candidate_offsets,
                    history_tokens,
                ) = self._prepare_model_inputs(batch)
                encoded = prepared.encoded
                reduced = prepared.reduced
                candidate_log_probs = self.policy.score_candidates(
                    prepared,
                    action_mask,
                    candidate_values,
                    candidate_offsets,
                    validated=True,
                ).float()
                value_predictions = self.policy.critic(reduced.cls).float()
                accumulator.add_legality(
                    self.policy.unmasked_first_slot_logits(prepared),
                    action_mask,
                    encoded.numerical,
                )
                outcome = batch.outcome.to(device=self.device, dtype=torch.float32)
                decision_index = batch.decision_index.to(self.device)
                game_length = batch.game_length.to(self.device)
                value_mask = batch.outcome_valid.to(self.device)
                value_targets = _discounted_outcome_targets(
                    outcome,
                    decision_index,
                    game_length,
                    self._gamma,
                )
                masks = tuple(mask.to(self.device) for mask in validated_masks)
                marginal_nll = -_ragged_logsumexp(candidate_log_probs, candidate_offsets)
                greedy = self.policy.act(prepared, action_mask, deterministic=True)
                predicted, best_scores = greedy.actions, greedy.log_probs
                accumulator.add(
                    exact_actions=batch.exact_action.to(self.device),
                    masks=masks,  # pyright: ignore[reportArgumentType]
                    marginal_nll=marginal_nll,
                    predicted=predicted,
                    best_scores=best_scores,
                    team_preview=encoded.phase,
                )
                accumulator.add_value(value_predictions, value_targets, value_mask)
                commit_history_updates(self._series_history, batch.windows, history_tokens)

            metrics = accumulator.finalize()
        finally:
            self._series_history.clear()

        if metrics.non_finite_values:
            raise FloatingPointError("BC evaluation contains non-finite predictions or values")
        return metrics


__all__ = [
    "BCCancelled",
    "BCDecisionBatch",
    "BCEvaluationMetrics",
    "BCGameWindow",
    "BCObjective",
    "BCTrainer",
    "collate_bc_batches",
    "compute_bc_objective",
]
