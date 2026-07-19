"""Behaviour-cloning objectives over exact and ragged replay labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast

from p0.model.policy import EncodedObs, PolicyNet
from p0.model.series_context import SeriesFeatures, tensorize_series
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.training.checkpoint import DEFAULT_POLICY_STORE, CheckpointStore
from p0.training.config import BCConfig


@dataclass(frozen=True, slots=True)
class BCObjective:
    """Loss and detached reporting values for one candidate-scored batch."""

    loss: Tensor
    exact_nll: Tensor
    partial_nll: Tensor
    marginal_log_probs: Tensor
    exact_count: int
    partial_count: int
    labeled_count: int


@dataclass(frozen=True, slots=True)
class BCTrainMetrics:
    """Aggregated metrics from a bounded recurrent BC run."""

    loss: float
    exact_nll: float
    partial_nll: float
    decisions: int
    labeled_decisions: int
    exact_decisions: int
    partial_decisions: int
    updates: int
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class _ChunkMetrics:
    loss: float
    exact_nll: float
    partial_nll: float
    decisions: int
    labeled_decisions: int
    exact_decisions: int
    partial_decisions: int


class BCTrainer:
    """Train a policy on complete game chunks with detached truncated BPTT."""

    def __init__(
        self,
        policy: PolicyNet,
        dataset: Iterable[ReplayGameChunk],
        config: BCConfig,
        *,
        device: torch.device | str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        checkpoint_store: CheckpointStore = DEFAULT_POLICY_STORE,
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
        torch.manual_seed(config.seed)

    def train(self) -> BCTrainMetrics:
        """Run configured epochs over the streaming dataset."""
        self.policy.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        totals = {
            "loss": 0.0,
            "exact_nll": 0.0,
            "partial_nll": 0.0,
            "decisions": 0,
            "labeled_decisions": 0,
            "exact_decisions": 0,
            "partial_decisions": 0,
            "updates": 0,
        }
        for _ in range(self.config.epochs):
            for game in self.dataset:
                self._train_game(game, totals)
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        labeled = max(totals["labeled_decisions"], 1)
        exact = max(totals["exact_decisions"], 1)
        partial = max(totals["partial_decisions"], 1)
        return BCTrainMetrics(
            loss=totals["loss"] / labeled,
            exact_nll=totals["exact_nll"] / exact,
            partial_nll=totals["partial_nll"] / partial,
            decisions=totals["decisions"],
            labeled_decisions=totals["labeled_decisions"],
            exact_decisions=totals["exact_decisions"],
            partial_decisions=totals["partial_decisions"],
            updates=totals["updates"],
            peak_memory_bytes=peak_memory,
        )

    def save_checkpoint(self, path: str | Path, *, epoch: int) -> None:
        """Persist the policy and optimizer state through the checkpoint seam."""
        self.checkpoint_store.save_training_state(
            Path(path),
            epoch,
            self.policy,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore a BC training state and return its completed epoch."""
        return self.checkpoint_store.load_training_state(
            Path(path),
            self.policy,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )

    def _initial_state(self, game: ReplayGameChunk) -> Tensor:
        if not self.policy.config.series_context_enabled:
            return self.policy.initial_state(1)
        features = tensorize_series(
            game.summary_inputs,
            player_index=game.player,
            tokenizer=self.policy.resources.tokenizer,
        )
        return self.policy.initial_series_state(SeriesFeatures.stack([features]))

    def _train_game(self, game: ReplayGameChunk, totals: dict[str, Any]) -> None:
        state = self._initial_state(game)
        chunk_length = min(self.config.chunk_length, self.config.batch_decisions)
        for start in range(0, game.length, chunk_length):
            end = min(start + chunk_length, game.length)
            state, metrics = self._train_chunk(game, start, end, state)
            for name in (
                "loss",
                "exact_nll",
                "partial_nll",
                "decisions",
                "labeled_decisions",
                "exact_decisions",
                "partial_decisions",
            ):
                value = getattr(metrics, name)
                if name == "loss":
                    value *= metrics.labeled_decisions
                totals[name] += value
            totals["updates"] += int(metrics.labeled_decisions > 0)

    def _train_chunk(
        self,
        game: ReplayGameChunk,
        start: int,
        end: int,
        state: Tensor,
    ) -> tuple[Tensor, _ChunkMetrics]:
        observations = game.observations[start:end].to(self.device)
        action_mask = game.action_mask[start:end].to(self.device)
        candidate_start = int(game.candidate_offsets[start].item())
        candidate_end = int(game.candidate_offsets[end].item())
        candidates = game.candidate_values[candidate_start:candidate_end].to(self.device)
        offsets = game.candidate_offsets[start : end + 1] - candidate_start
        offsets = offsets.to(self.device)
        labels = game.label_kind[start:end].to(self.device)
        loss_mask = game.loss_mask[start:end].to(self.device)
        encoded = self.policy.encode(observations, action_mask)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss: Tensor | None = None
        exact_nll = 0.0
        partial_nll = 0.0
        labeled_decisions = 0
        exact_decisions = 0
        partial_decisions = 0
        for index in range(end - start):
            candidate_left = int(offsets[index].item())
            candidate_right = int(offsets[index + 1].item())
            one_encoded = EncodedObs(
                encoded.tokens[index : index + 1],
                encoded.aux[index : index + 1],
                encoded.numerical[index : index + 1],
            )
            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                log_probs, state = self.policy.actor.score_joint_candidates_with_state(
                    one_encoded,
                    action_mask[index : index + 1],
                    state,
                    candidates[candidate_left:candidate_right],
                    torch.tensor(
                        [0, candidate_right - candidate_left],
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
                objective = compute_bc_objective(
                    log_probs,
                    torch.tensor(
                        [0, candidate_right - candidate_left],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    labels[index : index + 1],
                    loss_mask[index : index + 1],
                )
            total_loss = (
                objective.loss * objective.labeled_count
                if total_loss is None
                else total_loss + objective.loss * objective.labeled_count
            )
            exact_nll += objective.exact_nll.detach().item() * objective.exact_count
            partial_nll += objective.partial_nll.detach().item() * objective.partial_count
            labeled_decisions += objective.labeled_count
            exact_decisions += objective.exact_count
            partial_decisions += objective.partial_count
        if total_loss is not None and labeled_decisions:
            loss = total_loss / labeled_decisions
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite BC loss in series {game.series_id}, game {game.game_number}"
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_value = loss.detach().item()
        else:
            loss_value = 0.0
        return state.detach(), _ChunkMetrics(
            loss=loss_value,
            exact_nll=exact_nll,
            partial_nll=partial_nll,
            decisions=end - start,
            labeled_decisions=labeled_decisions,
            exact_decisions=exact_decisions,
            partial_decisions=partial_decisions,
        )


def _validate_objective_inputs(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> Tensor:
    if candidate_log_probs.dim() != 1:
        raise ValueError("candidate_log_probs must be one-dimensional")
    if candidate_offsets.dim() != 1 or candidate_offsets.dtype != torch.long:
        raise ValueError("candidate_offsets must be a one-dimensional torch.long tensor")
    if label_kind.dim() != 1 or loss_mask.dim() != 1:
        raise ValueError("label_kind and loss_mask must be one-dimensional")
    if label_kind.numel() + 1 != candidate_offsets.numel():
        raise ValueError("candidate_offsets must have one boundary per decision")
    if label_kind.numel() != loss_mask.numel():
        raise ValueError("label_kind and loss_mask must have matching lengths")
    if candidate_offsets.device != candidate_log_probs.device:
        candidate_offsets = candidate_offsets.to(candidate_log_probs.device)
    if (
        label_kind.device != candidate_log_probs.device
        or loss_mask.device != candidate_log_probs.device
    ):
        raise ValueError("objective tensors must share a device")
    if (
        candidate_offsets[0].item() != 0
        or candidate_offsets[-1].item() != candidate_log_probs.numel()
    ):
        raise ValueError("candidate_offsets must start at zero and end at candidate count")
    if torch.any(candidate_offsets[1:] < candidate_offsets[:-1]):
        raise ValueError("candidate_offsets must be nondecreasing")
    if torch.any((loss_mask < 0) | (loss_mask > 1)):
        raise ValueError("loss_mask values must be in [0, 1]")
    return candidate_offsets


def _ragged_logsumexp(candidate_log_probs: Tensor, offsets: Tensor) -> Tensor:
    decision_count = offsets.numel() - 1
    counts = offsets[1:] - offsets[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(decision_count, device=candidate_log_probs.device), counts
    )
    finite_scores = torch.where(
        torch.isfinite(candidate_log_probs),
        candidate_log_probs,
        torch.full_like(candidate_log_probs, float("-inf")),
    )
    row_max = torch.full(
        (decision_count,),
        float("-inf"),
        dtype=candidate_log_probs.dtype,
        device=candidate_log_probs.device,
    )
    if finite_scores.numel():
        row_max.scatter_reduce_(0, row_ids, finite_scores, reduce="amax", include_self=True)
    safe_max = row_max[row_ids] if row_ids.numel() else row_max.new_empty((0,))
    safe_delta = torch.where(
        torch.isfinite(safe_max),
        finite_scores - safe_max,
        torch.zeros_like(finite_scores),
    )
    row_sum = torch.zeros(
        (decision_count,), dtype=candidate_log_probs.dtype, device=candidate_log_probs.device
    )
    if safe_delta.numel():
        row_sum.scatter_add_(0, row_ids, torch.exp(safe_delta))
    result = row_max + torch.log(row_sum)
    return torch.where(
        (counts > 0) & torch.isfinite(row_max),
        result,
        torch.full_like(result, float("-inf")),
    )


def compute_bc_objective(
    candidate_log_probs: Tensor,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> BCObjective:
    """Compute exact and candidate-marginalized NLL without dropping unknown steps."""
    offsets = _validate_objective_inputs(
        candidate_log_probs, candidate_offsets, label_kind, loss_mask
    )
    exact = label_kind == int(LabelKind.EXACT)
    partial = label_kind == int(LabelKind.PARTIAL)
    unknown = label_kind == int(LabelKind.UNKNOWN)
    if torch.any(~(exact | partial | unknown)):
        raise ValueError("label_kind contains an unsupported label")
    counts = offsets[1:] - offsets[:-1]
    if torch.any(exact & (counts != 1)):
        raise ValueError("EXACT labels must have exactly one candidate")
    if torch.any(partial & (counts < 2)):
        raise ValueError("PARTIAL labels must have at least two candidates")
    if torch.any(unknown & (counts != 0)):
        raise ValueError("UNKNOWN labels must not have candidates")
    if torch.any(unknown & (loss_mask != 0)):
        raise ValueError("UNKNOWN labels must have a zero loss mask")
    if torch.any((exact | partial) & (loss_mask == 0)):
        raise ValueError("Labeled decisions must have a nonzero loss mask")

    marginal_log_probs = _ragged_logsumexp(candidate_log_probs, offsets)
    exact_scores = candidate_log_probs[offsets[:-1][exact]]
    exact_score_rows = torch.zeros_like(marginal_log_probs)
    exact_score_rows[exact] = exact_scores
    selected_log_probs = torch.where(exact, exact_score_rows, marginal_log_probs)
    per_decision_loss = torch.where(
        loss_mask > 0,
        -selected_log_probs * loss_mask,
        torch.zeros_like(selected_log_probs),
    )
    labeled_count = int(loss_mask.sum().item())
    denominator = loss_mask.sum().clamp_min(1.0)
    exact_count = int(exact.sum().item())
    partial_count = int(partial.sum().item())
    exact_nll = (-exact_scores).sum() / max(exact_count, 1)
    partial_nll = (-marginal_log_probs[partial]).sum() / max(partial_count, 1)
    return BCObjective(
        loss=per_decision_loss.sum() / denominator,
        exact_nll=exact_nll,
        partial_nll=partial_nll,
        marginal_log_probs=marginal_log_probs,
        exact_count=exact_count,
        partial_count=partial_count,
        labeled_count=labeled_count,
    )


__all__ = ["BCObjective", "compute_bc_objective"]
