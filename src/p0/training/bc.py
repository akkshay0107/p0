"""Behaviour-cloning objectives over exact and ragged replay labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast

from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.cls_reducer import pack_history_tokens
from p0.model.policy import PolicyNet
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
    """Aggregated metrics from a stateless complete-game BC run."""

    loss: float
    exact_nll: float
    partial_nll: float
    decisions: int
    labeled_decisions: int
    exact_decisions: int
    partial_decisions: int
    updates: int
    peak_memory_bytes: int


@dataclass(slots=True)
class _RunTotals:
    """Running sums over chunks. loss, exact_nll and partial_nll are weighted sums."""

    loss: float = 0.0
    exact_nll: float = 0.0
    partial_nll: float = 0.0
    decisions: int = 0
    labeled_decisions: int = 0
    exact_decisions: int = 0
    partial_decisions: int = 0
    updates: int = 0

    def add(self, chunk: _RunTotals) -> None:
        self.loss += chunk.loss
        self.exact_nll += chunk.exact_nll
        self.partial_nll += chunk.partial_nll
        self.decisions += chunk.decisions
        self.labeled_decisions += chunk.labeled_decisions
        self.exact_decisions += chunk.exact_decisions
        self.partial_decisions += chunk.partial_decisions
        self.updates += int(chunk.labeled_decisions > 0)


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
        if self.config.epochs > 1 and iter(self.dataset) is self.dataset:
            raise ValueError("BC datasets must be re-iterable when epochs is greater than one")
        self.policy.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        totals = _RunTotals()
        for _ in range(self.config.epochs):
            for game in self.dataset:
                self._train_game(game, totals)
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        return BCTrainMetrics(
            loss=totals.loss / max(totals.labeled_decisions, 1),
            exact_nll=totals.exact_nll / max(totals.exact_decisions, 1),
            partial_nll=totals.partial_nll / max(totals.partial_decisions, 1),
            decisions=totals.decisions,
            labeled_decisions=totals.labeled_decisions,
            exact_decisions=totals.exact_decisions,
            partial_decisions=totals.partial_decisions,
            updates=totals.updates,
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

    def _series_inputs(self, game: ReplayGameChunk) -> tuple[Tensor, Tensor]:
        histories = getattr(game, "prior_game_histories", None)
        series, mask = self.policy.encode_series(histories)
        return series.to(self.device), mask.to(self.device)

    def _history_inputs(
        self,
        local_tokens: Tensor,
        target_slice: slice | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        start = 0 if target_slice is None or target_slice.start is None else target_slice.start
        stop = (
            local_tokens.size(0)
            if target_slice is None or target_slice.stop is None
            else target_slice.stop
        )
        windows: list[tuple[Tensor, Tensor, Tensor]] = []
        for target in range(start, stop):
            left = max(0, target - HISTORY_WINDOW)
            windows.append(pack_history_tokens(local_tokens[left:target].unsqueeze(0)))
        if not windows:
            raise ValueError("A replay game must contain at least one decision")
        packed = torch.cat([window[0] for window in windows], dim=0)
        masks = torch.cat([window[1] for window in windows], dim=0)
        ages = torch.cat([window[2] for window in windows], dim=0)
        return packed, masks, ages

    def _train_game(self, game: ReplayGameChunk, totals: _RunTotals) -> None:
        for start in range(0, game.length, self.config.batch_decisions):
            target_slice = slice(start, min(start + self.config.batch_decisions, game.length))
            self._train_window(game, target_slice, totals)

    def _train_window(
        self,
        game: ReplayGameChunk,
        target_slice: slice,
        totals: _RunTotals,
    ) -> None:
        start = target_slice.start
        stop = target_slice.stop
        if start is None or stop is None or not 0 <= start < stop <= game.length:
            raise ValueError("target_slice must select a non-empty in-game decision window")

        context_start = max(0, start - HISTORY_WINDOW)
        observations = game.observations[context_start:stop].to(self.device)
        action_mask = game.action_mask[context_start:stop].to(self.device)
        candidate_start = int(game.candidate_offsets[start])
        candidate_end = int(game.candidate_offsets[stop])
        candidate_values = game.candidate_values[candidate_start:candidate_end].to(self.device)
        candidate_offsets = (game.candidate_offsets[start : stop + 1] - candidate_start).to(
            self.device
        )
        labels = game.label_kind[start:stop].to(self.device)
        loss_mask = game.loss_mask[start:stop].to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            encoded = self.policy.encode(observations, action_mask)
            local_tokens = self.policy.local_history_tokens(encoded)
            relative_start = start - context_start
            relative_stop = stop - context_start
            target_encoded = encoded._replace(
                tokens=encoded.tokens[relative_start:relative_stop],
                aux=encoded.aux[relative_start:relative_stop],
                numerical=encoded.numerical[relative_start:relative_stop],
            )
            target_mask = action_mask[relative_start:relative_stop]
            history_tokens, history_mask, history_age_ids = self._history_inputs(
                local_tokens, slice(relative_start, relative_stop)
            )
            series_tokens, series_mask = self._series_inputs(game)
            target_count = stop - start
            log_probs = self.policy.actor.score_joint_candidates(
                target_encoded,
                target_mask,
                series_tokens.expand(target_count, -1, -1),
                series_mask.expand(target_count, -1),
                history_tokens,
                history_mask,
                history_age_ids,
                candidate_values,
                candidate_offsets,
            )
        objective = compute_bc_objective(
            log_probs,
            candidate_offsets,
            labels,
            loss_mask,
        )
        labeled_decisions = objective.labeled_count
        exact_decisions = objective.exact_count
        partial_decisions = objective.partial_count
        loss_sum = 0.0
        if labeled_decisions:
            loss = objective.loss
            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite BC loss in series {game.series_id}, game {game.game_number}"
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum = loss.detach().item() * labeled_decisions
        totals.add(
            _RunTotals(
                loss=loss_sum,
                exact_nll=objective.exact_nll.detach().item() * exact_decisions,
                partial_nll=objective.partial_nll.detach().item() * partial_decisions,
                decisions=target_count,
                labeled_decisions=labeled_decisions,
                exact_decisions=exact_decisions,
                partial_decisions=partial_decisions,
            )
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
    row_max = torch.full(
        (decision_count,),
        float("-inf"),
        dtype=candidate_log_probs.dtype,
        device=candidate_log_probs.device,
    )
    row_max.scatter_reduce_(0, row_ids, candidate_log_probs, reduce="amax", include_self=True)
    # A decision with no candidates keeps row_max at -inf; shifting by it would give nan.
    gathered_max = row_max[row_ids]
    shifted = torch.where(
        torch.isfinite(gathered_max),
        candidate_log_probs - gathered_max,
        torch.zeros_like(candidate_log_probs),
    )
    row_sum = torch.zeros_like(row_max)
    row_sum.scatter_add_(0, row_ids, torch.exp(shifted))
    return torch.where(
        torch.isfinite(row_max),
        row_max + torch.log(row_sum),
        torch.full_like(row_max, float("-inf")),
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

    # EXACT decisions carry exactly one candidate, so their marginal is that candidate's
    # log-probability and the exact and marginal objectives coincide.
    marginal_log_probs = _ragged_logsumexp(candidate_log_probs, offsets)
    # UNKNOWN decisions have no candidates, so their marginal is -inf; the mask must zero
    # them by selection rather than by multiplication, which would give nan.
    per_decision_loss = torch.where(
        loss_mask > 0,
        -marginal_log_probs * loss_mask,
        torch.zeros_like(marginal_log_probs),
    )
    mask_total = loss_mask.sum()
    labeled_count = int(mask_total.item())
    exact_count = int(exact.sum().item())
    partial_count = int(partial.sum().item())
    exact_nll = (-marginal_log_probs[exact]).sum() / max(exact_count, 1)
    partial_nll = (-marginal_log_probs[partial]).sum() / max(partial_count, 1)
    return BCObjective(
        loss=per_decision_loss.sum() / mask_total.clamp_min(1.0),
        exact_nll=exact_nll,
        partial_nll=partial_nll,
        marginal_log_probs=marginal_log_probs,
        exact_count=exact_count,
        partial_count=partial_count,
        labeled_count=labeled_count,
    )


__all__ = ["BCObjective", "compute_bc_objective"]
