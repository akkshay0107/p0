"""Behavior-cloning trainer and compatibility façade."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from p0.model.policy import EncodedObs, PolicyNet
from p0.replays.dataset import ReplayGameChunk
from p0.training._bc_batch import (
    BCBatchDataset,
    BCDecisionBatch,
    BCGameWindow,
    collate_bc_batches,
)
from p0.training._bc_history import _BCHistoryUpdate, _BCSeriesHistory
from p0.training._bc_metrics import (
    BCEvaluationMetrics,
    BCObjective,
    _BCEvaluationAccumulator,
    _ragged_logsumexp,
    _validate_objective_inputs,
    compute_bc_objective,
)
from p0.training.checkpoint import DEFAULT_POLICY_STORE, CheckpointStore
from p0.training.config import BCConfig


class BCCancelled(RuntimeError):
    """Raised between batches so callers keep the last completed epoch checkpoint."""


BCModelInputs = tuple[EncodedObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class _PreparedBCBatch:
    model_inputs: BCModelInputs
    history_updates: tuple[_BCHistoryUpdate, ...]


def _empty_training_totals() -> dict[str, float | int]:
    return {
        "loss": 0.0,
        "loss_weight": 0.0,
        "exact_nll": 0.0,
        "partial_nll": 0.0,
        "decisions": 0,
        "labeled_decisions": 0,
        "exact_decisions": 0,
        "partial_decisions": 0,
        "updates": 0,
        "games": 0,
    }


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
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.amp_enabled)
        self.checkpoint_store = checkpoint_store
        self.provenance = dict(provenance or {})
        self.batch_decisions = config.batch_decisions
        self._series_history = _BCSeriesHistory(policy.d_model)
        self.cancel_requested = cancel_requested
        torch.manual_seed(config.seed)

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

    def _train_epoch_totals(self) -> dict[str, float | int]:
        self.policy.train()
        self._series_history.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        totals = _empty_training_totals()
        accumulated_decisions = 0
        accumulated_loss_weight = 0.0
        chunk_size = min(self.batch_decisions, self.config.max_chunk_size)
        self.optimizer.zero_grad(set_to_none=True)
        num_workers = self.config.num_workers
        prefetch_factor = self.config.prefetch_factor if num_workers > 0 else None
        dataloader = DataLoader(
            BCBatchDataset(self.dataset, chunk_size),
            num_workers=num_workers,
            batch_size=None,
            prefetch_factor=prefetch_factor,
        )

        for batch in dataloader:
            if self.cancel_requested():
                raise BCCancelled("Behaviour-cloning training was cancelled")
            accumulated_loss_weight += self._backward_chunk(batch, totals)
            accumulated_decisions += batch.decisions
            if (
                accumulated_decisions >= self.batch_decisions
                and not self._series_history.has_partial_games
            ):
                if accumulated_loss_weight > 0:
                    totals["updates"] += int(self._step_optimizer(accumulated_loss_weight))
                accumulated_decisions = 0
                accumulated_loss_weight = 0.0

        if self._series_history.has_partial_games:
            raise ValueError("BC dataset ended with an incomplete perspective-game")
        if accumulated_loss_weight > 0:
            totals["updates"] += int(self._step_optimizer(accumulated_loss_weight))
        self._series_history.clear()
        return totals

    def _metrics(self, totals: dict[str, float | int]) -> dict[str, float | int]:
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        updates = int(totals["updates"])
        games = int(totals["games"])
        decisions = int(totals["decisions"])
        return {
            "loss": float(totals["loss"]) / max(float(totals["loss_weight"]), 1.0),
            "exact_nll": float(totals["exact_nll"]) / max(int(totals["exact_decisions"]), 1),
            "partial_nll": float(totals["partial_nll"]) / max(int(totals["partial_decisions"]), 1),
            "decisions": decisions,
            "labeled_decisions": int(totals["labeled_decisions"]),
            "exact_decisions": int(totals["exact_decisions"]),
            "partial_decisions": int(totals["partial_decisions"]),
            "updates": updates,
            "games": games,
            "decisions_per_update": decisions / updates if updates else 0.0,
            "games_per_update": games / updates if updates else 0.0,
            "peak_memory_bytes": peak_memory,
        }

    def save_checkpoint(self, path: str | Path, *, epoch: int) -> None:
        """Persist the policy and optimizer state through the checkpoint seam."""
        self.checkpoint_store.save_training_state(
            Path(path),
            epoch,
            self.policy,
            optimizer=self.optimizer,
            scaler=self.scaler,
            metadata=self.provenance,
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

    def _prepare_model_inputs(self, batch: BCDecisionBatch) -> _PreparedBCBatch:
        observations = batch.observations.to(self.device)
        context_action_mask = batch.context_action_mask.to(self.device)
        encoded = self.policy.encode(observations, context_action_mask)
        local_tokens = self.policy.local_history_tokens(encoded)
        target_indices = batch.target_indices.to(self.device)
        target_local_tokens = local_tokens[target_indices]
        target_encoded = EncodedObs(
            encoded.tokens[target_indices],
            encoded.aux[target_indices],
            encoded.numerical[target_indices],
        )
        history_indices = batch.history_indices.to(self.device)
        history_mask = batch.history_mask.to(self.device)
        history_tokens = local_tokens[history_indices] * history_mask.unsqueeze(-1)
        history_age_ids = batch.history_age_ids.to(self.device)
        series_context = self._series_history.prepare(
            batch.windows,
            target_local_tokens,
            self.policy.series.resample_single_game,
        )
        return _PreparedBCBatch(
            model_inputs=(
                target_encoded,
                batch.action_mask.to(self.device),
                series_context.tokens,
                series_context.mask,
                history_tokens,
                history_mask,
                history_age_ids,
            ),
            history_updates=series_context.updates,
        )

    def _forward_batch(
        self,
        batch: BCDecisionBatch,
    ) -> tuple[Tensor, tuple[_BCHistoryUpdate, ...]]:
        prepared = self._prepare_model_inputs(batch)
        log_probs = self.policy.actor.score_joint_candidates(
            *prepared.model_inputs,
            batch.candidate_values.to(self.device),
            batch.candidate_offsets.to(self.device),
        )
        return log_probs, prepared.history_updates

    def _greedy_actions(self, model_inputs: BCModelInputs) -> tuple[Tensor, Tensor]:
        actions, log_probs, _, _ = self.policy.actor.greedy(*model_inputs)
        return actions, log_probs

    def _step_optimizer(self, loss_weight: float) -> bool:
        if loss_weight <= 0:
            raise ValueError("loss_weight must be positive before an optimizer step")
        self.scaler.unscale_(self.optimizer)
        inverse_weight = 1.0 / loss_weight
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(inverse_weight)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return self.scaler.get_scale() >= previous_scale

    def _backward_chunk(
        self,
        batch: BCDecisionBatch,
        totals: dict[str, float | int],
    ) -> float:
        labels = batch.label_kind.to(self.device)
        loss_mask = batch.loss_mask.to(self.device)
        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            log_probs, history_updates = self._forward_batch(batch)
        objective = compute_bc_objective(
            log_probs,
            batch.candidate_offsets.to(self.device),
            labels,
            loss_mask,
        )
        loss_sum = 0.0
        if objective.labeled_count:
            loss = objective.loss * objective.loss_weight
            if not torch.isfinite(loss):
                raise ValueError("Non-finite BC loss in a collated decision batch")
            self.scaler.scale(loss).backward()
            loss_sum = objective.loss.detach().item() * objective.loss_weight

        self._series_history.apply(history_updates)
        totals["loss"] += loss_sum
        totals["loss_weight"] += objective.loss_weight
        totals["exact_nll"] += objective.exact_nll.detach().item() * objective.exact_count
        totals["partial_nll"] += objective.partial_nll.detach().item() * objective.partial_count
        totals["decisions"] += batch.decisions
        totals["labeled_decisions"] += objective.labeled_count
        totals["exact_decisions"] += objective.exact_count
        totals["partial_decisions"] += objective.partial_count
        totals["games"] += batch.completed_game_count
        return objective.loss_weight

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
            prepared = self._prepare_model_inputs(batch)
            model_inputs = prepared.model_inputs
            candidate_offsets = batch.candidate_offsets.to(self.device)
            candidate_log_probs = self.policy.actor.score_joint_candidates(
                *model_inputs,
                batch.candidate_values.to(self.device),
                candidate_offsets,
            )
            validated_masks = _validate_objective_inputs(
                candidate_log_probs.numel(),
                batch.candidate_offsets,
                batch.label_kind,
                batch.loss_mask,
            )
            masks = tuple(mask.to(self.device) for mask in validated_masks)
            marginal_nll = -_ragged_logsumexp(candidate_log_probs, candidate_offsets)
            safe_nll = torch.where(
                torch.isfinite(marginal_nll),
                marginal_nll,
                torch.zeros_like(marginal_nll),
            )
            predicted, best_scores = self._greedy_actions(model_inputs)
            accumulator.add(
                batch,
                candidate_offsets=candidate_offsets,
                masks=masks,  # pyright: ignore[reportArgumentType]
                marginal_nll=marginal_nll,
                safe_nll=safe_nll,
                predicted=predicted,
                best_scores=best_scores,
            )
            self._series_history.apply(prepared.history_updates)

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
