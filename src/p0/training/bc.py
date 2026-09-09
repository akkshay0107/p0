"""Behavior-cloning trainer."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from p0.battle.actions import TEAM_SIZE
from p0.model.policy import MemoryInputs, PolicyNet, PreparedDecision
from p0.replays.dataset import ReplayGameChunk
from p0.runtime.process_context import PROCESS_CONTEXT
from p0.training._bc_batch import (
    BCBatchDataset,
    BCDecisionBatch,
    BCGameWindow,
    collate_bc_batches,
)
from p0.training._bc_history import (
    _BCHistoryUpdate,
    commit_history_updates,
    prepare_series_context,
)
from p0.training._bc_metrics import (
    BCEvaluationMetrics,
    BCObjective,
    _BCEvaluationAccumulator,
    _compute_bc_objective_unchecked,
    _ragged_logsumexp,
    _validate_objective_inputs,
    compute_bc_objective,
)
from p0.training.checkpoint import DEFAULT_POLICY_STORE, CheckpointStore
from p0.training.config import BCConfig
from p0.training.series_history import SeriesHistoryStore
from p0.training.utils import select_optimization_precision

LOGGER = logging.getLogger(__name__)


def _expand_team_preview_orbits(
    candidate_values: Tensor,
    candidate_offsets: Tensor,
    team_preview: Tensor,
) -> tuple[Tensor, Tensor]:
    """Add all four within-pair orientations for team-preview candidates."""
    if candidate_values.numel() == 0:
        return candidate_values, candidate_offsets

    counts = candidate_offsets[1:] - candidate_offsets[:-1]
    team_preview = team_preview.to(device=candidate_values.device, dtype=torch.bool)
    first, second = candidate_values.unbind(dim=-1)
    first_swap = (first % TEAM_SIZE) * TEAM_SIZE + first // TEAM_SIZE
    second_swap = (second % TEAM_SIZE) * TEAM_SIZE + second // TEAM_SIZE
    orbit_values = torch.stack(
        (
            candidate_values,
            torch.stack((first_swap, second), dim=-1),
            torch.stack((first, second_swap), dim=-1),
            torch.stack((first_swap, second_swap), dim=-1),
        ),
        dim=1,
    )
    row_preview = torch.repeat_interleave(team_preview, counts)
    keep = torch.arange(4, device=candidate_values.device).unsqueeze(0) < torch.where(
        row_preview, 4, 1
    ).unsqueeze(-1)
    expanded_values = orbit_values[keep]
    expanded_counts = counts * torch.where(team_preview, 4, 1)
    expanded_offsets = torch.cat(
        (
            candidate_offsets.new_zeros(1),
            expanded_counts.cumsum(0).to(dtype=candidate_offsets.dtype),
        )
    )
    return expanded_values, expanded_offsets


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
        "grad_norm_count": 0,
    }


def _seed_bc_worker(worker_id: int) -> None:
    """Derive independent Python and NumPy worker streams from Torch's seed."""
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


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
        checkpoint_store: CheckpointStore = DEFAULT_POLICY_STORE,
        provenance: Mapping[str, object] | None = None,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> None:
        self.policy = policy.to(device)
        self.dataset = dataset
        self.config = config
        self.device = torch.device(device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.precision = select_optimization_precision(config.enable_optim, self.device)
        self.scaler = GradScaler(device=self.device.type, enabled=self.precision.grad_scaler)
        self.checkpoint_store = checkpoint_store
        self.provenance = dict(provenance or {})
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
        accumulated_decisions = 0
        update_batches: list[BCDecisionBatch] = []
        pending_game_keys: set[tuple[object, int]] = set()
        chunk_size = min(self.batch_decisions, self.config.max_chunk_size)
        self.optimizer.zero_grad(set_to_none=True)
        num_workers = self.config.num_workers
        prefetch_factor = self.config.prefetch_factor if num_workers > 0 else None
        dataloader = DataLoader(
            BCBatchDataset(self.dataset, chunk_size),
            num_workers=num_workers,
            batch_size=None,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=PROCESS_CONTEXT if num_workers > 0 else None,
            worker_init_fn=_seed_bc_worker if num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        for batch in dataloader:
            if self.cancel_requested():
                raise BCCancelled("Behaviour-cloning training was cancelled")
            update_batches.append(batch)
            accumulated_decisions += batch.decisions
            for window in batch.windows:
                game_key = (window.series_key, window.game_number)
                if window.is_game_end:
                    pending_game_keys.discard(game_key)
                else:
                    pending_game_keys.add(game_key)
            if (
                batch.windows
                and accumulated_decisions >= self.batch_decisions
                and not pending_game_keys
            ):
                policy_weight = sum(float(item.loss_mask.sum()) for item in update_batches)
                value_count = sum(int(item.outcome_valid.sum()) for item in update_batches)
                for item in update_batches:
                    self._backward_chunk(item, totals, policy_weight, value_count)
                if policy_weight or value_count:
                    updated, grad_norm = self._step_optimizer(1.0)
                    if updated:
                        totals["updates"] += 1
                        totals["grad_norm_sum"] += grad_norm
                        totals["grad_norm_count"] += 1
                update_batches.clear()
                accumulated_decisions = 0

        if update_batches:
            policy_weight = sum(float(item.loss_mask.sum()) for item in update_batches)
            value_count = sum(int(item.outcome_valid.sum()) for item in update_batches)
            for item in update_batches:
                self._backward_chunk(item, totals, policy_weight, value_count)
            if policy_weight or value_count:
                updated, grad_norm = self._step_optimizer(1.0)
                if updated:
                    totals["updates"] += 1
                    totals["grad_norm_sum"] += grad_norm
                    totals["grad_norm_count"] += 1
        if pending_game_keys or self._series_history.has_partial_games:
            raise ValueError("BC dataset ended with an incomplete perspective-game")
        self._series_history.clear()
        return totals

    def _metrics(self, totals: dict[str, Any]) -> dict[str, float | int]:
        updates = int(totals["updates"])
        grad_norm_count = int(totals["grad_norm_count"])
        return {
            "overall_nll": float(totals["loss"]) / max(float(totals["loss_weight"]), 1.0),
            "grad_norm": (
                float(totals["grad_norm_sum"]) / grad_norm_count if grad_norm_count else 0.0
            ),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "updates": updates,
            "games": int(totals["games"]),
            "decisions": int(totals["decisions"]),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        selection_state: Mapping[str, object] | None = None,
    ) -> None:
        """Persist the policy and optimizer state through the checkpoint seam."""
        metadata = dict(self.provenance)
        if selection_state is not None:
            metadata["selection_state"] = dict(selection_state)
        self.checkpoint_store.save_training_state(
            Path(path),
            epoch,
            self.policy,
            optimizer=self.optimizer,
            scaler=self.scaler,
            metadata=metadata,
            trainer_kind="bc",
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore a BC training state and return its completed epoch."""
        return self.checkpoint_store.load_training_state(
            Path(path),
            self.policy,
            optimizer=self.optimizer,
            scaler=self.scaler,
            expected_trainer_kind="bc",
            expected_metadata=self.provenance,
            require_training_state=True,
        )

    def load_selection_state(self, path: str | Path) -> Mapping[str, object]:
        """Load persisted best-policy selection state for resume-safe validation."""
        metadata = self.checkpoint_store.load_metadata(Path(path))
        state = metadata.get("selection_state", {})
        if not isinstance(state, Mapping):
            raise ValueError("BC checkpoint selection_state must be a mapping")
        return state

    def _prepare_model_inputs(
        self, batch: BCDecisionBatch
    ) -> tuple[PreparedDecision, Tensor, Tensor, Tensor, tuple[_BCHistoryUpdate, ...]]:
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
        candidate_values = batch.candidate_values.to(self.device)
        candidate_offsets = batch.candidate_offsets.to(self.device)
        candidate_values, candidate_offsets = _expand_team_preview_orbits(
            candidate_values,
            candidate_offsets,
            target_encoded.phase,
        )
        series_tokens, series_mask, history_updates = prepare_series_context(
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
            history_updates,
        )

    def _forward_batch(
        self,
        batch: BCDecisionBatch,
    ) -> tuple[Tensor, Tensor, tuple[_BCHistoryUpdate, ...], Tensor]:
        prepared, action_mask, candidate_values, candidate_offsets, history_updates = (
            self._prepare_model_inputs(batch)
        )
        return (
            self.policy.score_candidates(
                prepared,
                action_mask,
                candidate_values,
                candidate_offsets,
                validated=True,
            ),
            self.policy.critic(prepared.reduced.cls),
            history_updates,
            candidate_offsets,
        )

    def _step_optimizer(self, loss_weight: float) -> tuple[bool, float]:
        if loss_weight <= 0:
            raise ValueError("loss_weight must be positive before an optimizer step")
        self.scaler.unscale_(self.optimizer)
        inverse_weight = 1.0 / loss_weight
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(inverse_weight)
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
            self.optimizer.zero_grad(set_to_none=True)
            return False, 0.0
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        updated = self.scaler.get_scale() >= previous_scale
        return updated, float(grad_norm.item())

    def _backward_chunk(
        self,
        batch: BCDecisionBatch,
        totals: dict[str, Any],
        policy_weight: float,
        effective_value_count: int,
    ) -> None:
        """
        Backpropagate one validated decision chunk and update running totals.

        Arguments:
          batch: Collated CPU batch containing one contiguous set of decisions.
          totals: Mutable epoch totals receiving detached reporting tensors.
          policy_weight: Effective update's total fractional policy-label weight.
          effective_value_count: Effective update's total valid outcome count.

        Returns:
          None. Gradients are accumulated for the effective update.
        """
        validated_masks = _validate_objective_inputs(
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
        exact_cpu, partial_cpu, _, labeled_cpu = validated_masks
        exact_count = exact_cpu.sum()
        partial_count = partial_cpu.sum()
        labeled_count = labeled_cpu.sum()
        loss_weight = batch.loss_mask.sum()
        loss_mask = batch.loss_mask.to(self.device)
        with autocast(
            device_type=self.device.type,
            enabled=self.precision.autocast,
            dtype=self.precision.dtype,
        ):
            (
                log_probs,
                value_predictions,
                history_updates,
                candidate_offsets,
            ) = self._forward_batch(batch)
        exact = exact_cpu.to(self.device)
        partial = partial_cpu.to(self.device)
        labeled = labeled_cpu.to(self.device)
        objective = _compute_bc_objective_unchecked(
            log_probs,
            candidate_offsets,
            loss_mask,
            exact,
            partial,
            labeled,
            exact_count=exact_count,
            partial_count=partial_count,
            labeled_count=labeled_count,
            loss_weight=loss_weight,
        )
        outcome = batch.outcome.to(self.device)
        decision_index = batch.decision_index.to(self.device)
        game_length = batch.game_length.to(self.device)
        value_mask = batch.outcome_valid.to(self.device)
        gamma = torch.as_tensor(self.config.gamma, device=self.device, dtype=outcome.dtype)
        value_targets = (
            torch.pow(gamma, (game_length - 1 - decision_index).to(outcome.dtype)) * outcome
        )
        value_error = value_predictions - value_targets
        batch_value_count = int(batch.outcome_valid.sum())
        policy_sum = objective.loss * objective.loss_weight
        value_sum = (
            value_error.square()[value_mask].sum()
            if batch_value_count
            else value_predictions.sum() * 0.0
        )
        policy_term = policy_sum / policy_weight if policy_weight else policy_sum * 0.0
        value_term = value_sum / effective_value_count if effective_value_count else value_sum * 0.0
        total_loss = policy_term + self.config.value_coef * value_term
        if objective.labeled_count or batch_value_count:
            self.scaler.scale(total_loss).backward()

        commit_history_updates(self._series_history, history_updates)
        # Keep the reported policy NLL independent from the auxiliary value loss;
        # the optimizer still receives their weighted sum above.
        totals["loss"] += policy_sum.detach()
        totals["loss_weight"] += objective.loss_weight
        totals["decisions"] += batch.decisions
        totals["games"] += batch.completed_game_count

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
                history_updates,
            ) = self._prepare_model_inputs(batch)
            encoded = prepared.encoded
            reduced = prepared.reduced
            candidate_log_probs = self.policy.score_candidates(
                prepared,
                action_mask,
                candidate_values,
                candidate_offsets,
                validated=True,
            )
            value_predictions = self.policy.critic(reduced.cls)
            accumulator.add_legality(
                self.policy.unmasked_first_slot_logits(prepared),
                action_mask,
                encoded.numerical,
            )
            outcome = batch.outcome.to(self.device)
            decision_index = batch.decision_index.to(self.device)
            game_length = batch.game_length.to(self.device)
            value_mask = batch.outcome_valid.to(self.device)
            gamma = torch.as_tensor(self.config.gamma, device=self.device, dtype=outcome.dtype)
            value_targets = (
                torch.pow(gamma, (game_length - 1 - decision_index).to(outcome.dtype)) * outcome
            )
            masks = tuple(mask.to(self.device) for mask in validated_masks)
            marginal_nll = -_ragged_logsumexp(candidate_log_probs, candidate_offsets)
            safe_nll = torch.where(
                torch.isfinite(marginal_nll),
                marginal_nll,
                torch.zeros_like(marginal_nll),
            )
            greedy = self.policy.act(prepared, action_mask, deterministic=True)
            predicted, best_scores = greedy.actions, greedy.log_probs
            accumulator.add(
                exact_actions=batch.exact_action.to(self.device),
                masks=masks,  # pyright: ignore[reportArgumentType]
                marginal_nll=marginal_nll,
                safe_nll=safe_nll,
                predicted=predicted,
                best_scores=best_scores,
            )
            accumulator.add_value(value_predictions, value_targets, value_mask)
            commit_history_updates(self._series_history, history_updates)

        self._series_history.clear()
        return accumulator.finalize()


__all__ = [
    "BCCancelled",
    "BCBatchDataset",
    "BCDecisionBatch",
    "BCEvaluationMetrics",
    "BCGameWindow",
    "BCObjective",
    "BCTrainer",
    "collate_bc_batches",
    "compute_bc_objective",
]
