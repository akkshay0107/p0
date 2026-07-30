"""Behaviour-cloning objectives over exact and ragged replay labels.

This module provides the core components for behavior cloning (BC) training,
including data collation from game chunks, the BCTrainer for running epochs
over the dataset, and custom ragged-tensor Negative Log Likelihood (NLL)
objectives to support both exact and partial candidate scoring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import (
    HISTORY_WINDOW,
    MAX_PRIOR_GAMES,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import DecisionType, LabelKind
from p0.training.checkpoint import DEFAULT_POLICY_STORE, CheckpointStore
from p0.training.config import BCConfig


class BCCancelled(RuntimeError):
    """Raised between batches so callers keep the last completed epoch checkpoint."""


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
    loss_weight: float


@dataclass(frozen=True, slots=True)
class BCGameWindow:
    """Compact identity and target span for one perspective-game window."""

    series_key: SeriesPerspectiveKey
    game_number: int
    batch_start: int
    batch_stop: int
    is_game_end: bool
    is_series_end: bool


@dataclass(frozen=True, slots=True)
class BCDecisionBatch:
    """Target decisions plus game-local context descriptions for one update."""

    observations: StructuredObservation
    context_action_mask: Tensor
    action_mask: Tensor
    label_kind: Tensor
    label_confidence: Tensor
    loss_mask: Tensor
    decision_type: Tensor
    exact_action: Tensor
    candidate_values: Tensor
    candidate_offsets: Tensor
    target_indices: Tensor
    history_indices: Tensor
    history_mask: Tensor
    history_age_ids: Tensor
    windows: tuple[BCGameWindow, ...]

    @property
    def decisions(self) -> int:
        return int(self.label_kind.numel())

    @property
    def games(self) -> int:
        return len({(window.series_key, window.game_number) for window in self.windows})

    @property
    def completed_game_count(self) -> int:
        return sum(window.is_game_end for window in self.windows)


@dataclass(frozen=True, slots=True)
class _BCGameKey:
    series_key: SeriesPerspectiveKey
    game_number: int


@dataclass(frozen=True, slots=True)
class _BCGameHistoryUpdate:
    game_key: _BCGameKey
    tokens: Tensor
    is_game_end: bool
    is_series_end: bool


BCModelInputs = tuple[EncodedObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class _PreparedBCBatch:
    model_inputs: BCModelInputs
    history_updates: tuple[_BCGameHistoryUpdate, ...]


class _BCSeriesHistoryStore:
    """Retain detached raw game tokens between truncated BC backward passes."""

    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self._partial: dict[_BCGameKey, list[Tensor]] = {}
        self._completed: dict[SeriesPerspectiveKey, dict[int, Tensor]] = {}
        self._ended: set[SeriesPerspectiveKey] = set()

    def is_ended(self, series_key: SeriesPerspectiveKey) -> bool:
        return series_key in self._ended

    @property
    def has_partial_games(self) -> bool:
        return bool(self._partial)

    def partial_fragments(self, game_key: _BCGameKey) -> tuple[Tensor, ...]:
        return tuple(self._partial.get(game_key, ()))

    def completed_before(
        self,
        series_key: SeriesPerspectiveKey,
        game_number: int,
    ) -> dict[_BCGameKey, Tensor]:
        games = self._completed.get(series_key, {})
        return {
            _BCGameKey(series_key, prior_number): tokens
            for prior_number, tokens in games.items()
            if prior_number < game_number
        }

    def apply(self, updates: tuple[_BCGameHistoryUpdate, ...]) -> None:
        for update in updates:
            tokens = update.tokens.detach().to(device="cpu", dtype=torch.float32)
            if tokens.dim() != 2 or tokens.shape[1] != self.d_model:
                raise ValueError("BC game history tokens do not match the policy width")

            fragments = self._partial.setdefault(update.game_key, [])
            fragments.append(tokens)
            if update.is_game_end:
                game_tokens = torch.cat(self._partial.pop(update.game_key), dim=0)
                completed = self._completed.setdefault(update.game_key.series_key, {})
                if update.game_key.game_number in completed:
                    raise ValueError("A BC perspective-game completed more than once")
                completed[update.game_key.game_number] = game_tokens

            if update.is_series_end:
                if not update.is_game_end:
                    raise ValueError("A BC series can end only at a game boundary")
                self.drop(update.game_key.series_key)
                self._ended.add(update.game_key.series_key)

    def drop(self, series_key: SeriesPerspectiveKey) -> None:
        self._completed.pop(series_key, None)
        stale = [key for key in self._partial if key.series_key == series_key]
        for game_key in stale:
            self._partial.pop(game_key)

    def clear(self) -> None:
        self._partial.clear()
        self._completed.clear()
        self._ended.clear()


@dataclass(frozen=True, slots=True)
class BCEvaluationMetrics:
    overall_nll: float
    exact_nll: float
    partial_nll: float
    exact_joint_accuracy: float
    decisions: int
    labeled_decisions: int
    unknown_decisions: int
    exact_decisions: int
    partial_decisions: int
    illegal_predictions: int
    non_finite_values: int
    by_decision_type: Mapping[str, Mapping[str, float | int]]
    confidence_buckets: Mapping[str, Mapping[str, float | int]]
    candidate_set_sizes: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_nll": self.overall_nll,
            "exact_nll": self.exact_nll,
            "partial_nll": self.partial_nll,
            "exact_joint_accuracy": self.exact_joint_accuracy,
            "decisions": self.decisions,
            "labeled_decisions": self.labeled_decisions,
            "unknown_decisions": self.unknown_decisions,
            "exact_decisions": self.exact_decisions,
            "partial_decisions": self.partial_decisions,
            "illegal_predictions": self.illegal_predictions,
            "non_finite_values": self.non_finite_values,
            "by_decision_type": dict(self.by_decision_type),
            "confidence_buckets": dict(self.confidence_buckets),
            "candidate_set_sizes": dict(self.candidate_set_sizes),
        }


_CONFIDENCE_BUCKET_NAMES = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]")
_DECISION_TYPE_BIN_COUNT = max(int(decision_type) for decision_type in DecisionType) + 1


def _binned_evaluation_totals(
    bin_ids: Tensor,
    labeled_mask: Tensor,
    safe_nll: Tensor,
    *,
    minlength: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    if bin_ids.dim() != 1 or labeled_mask.shape != bin_ids.shape or safe_nll.shape != bin_ids.shape:
        raise ValueError("Evaluation bin inputs must be aligned one-dimensional tensors")
    if bin_ids.dtype != torch.long or labeled_mask.dtype != torch.bool:
        raise ValueError("Evaluation bin ids and labeled mask have invalid dtypes")
    if bin_ids.device != labeled_mask.device or bin_ids.device != safe_nll.device:
        raise ValueError("Evaluation bin inputs must share a device")

    decisions = torch.bincount(bin_ids, minlength=minlength)
    labeled = torch.bincount(bin_ids[labeled_mask], minlength=decisions.numel())
    labeled_nll = torch.where(
        labeled_mask,
        safe_nll.to(dtype=torch.float64),
        torch.zeros((), dtype=torch.float64, device=safe_nll.device),
    )
    nll_sums = torch.bincount(
        bin_ids,
        weights=labeled_nll,
        minlength=decisions.numel(),
    )
    return decisions, labeled, nll_sums


def _merge_bin_totals(
    current: tuple[Tensor, Tensor, Tensor],
    update: tuple[Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if current[0].numel() == update[0].numel():
        for current_values, update_values in zip(current, update, strict=True):
            current_values.add_(update_values)
        return current

    width = max(current[0].numel(), update[0].numel())
    merged: list[Tensor] = []
    for current_values, update_values in zip(current, update, strict=True):
        if current_values.numel() < width:
            current_values = torch.cat(
                (
                    current_values,
                    current_values.new_zeros(width - current_values.numel()),
                )
            )
        if update_values.numel() < width:
            update_values = torch.cat(
                (
                    update_values,
                    update_values.new_zeros(width - update_values.numel()),
                )
            )
        merged.append(current_values + update_values)
    return merged[0], merged[1], merged[2]


def _finalize_evaluation_bins(
    totals: tuple[Tensor, Tensor, Tensor],
    *,
    names: tuple[str, ...] | None = None,
) -> dict[str, Mapping[str, float | int]]:
    decision_values = totals[0].cpu().tolist()
    labeled_values = totals[1].cpu().tolist()
    nll_values = totals[2].cpu().tolist()
    result: dict[str, Mapping[str, float | int]] = {}
    for index, decision_count in enumerate(decision_values):
        if not decision_count:
            continue
        key = names[index] if names is not None else str(index)
        labeled_count = int(labeled_values[index])
        result[key] = {
            "decisions": int(decision_count),
            "labeled": labeled_count,
            "nll": float(nll_values[index]) / max(labeled_count, 1),
        }
    return result


@dataclass(slots=True)
class _BCEvaluationAccumulator:
    counts: Tensor
    nll_sums: Tensor
    decision_type_totals: tuple[Tensor, Tensor, Tensor]
    confidence_totals: tuple[Tensor, Tensor, Tensor]
    confidence_boundaries: Tensor
    candidate_counts: Tensor

    @classmethod
    def create(cls, device: torch.device) -> _BCEvaluationAccumulator:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        return cls(
            counts=torch.zeros(8, dtype=torch.long, device=device),
            nll_sums=torch.zeros(3, dtype=torch.float64, device=device),
            decision_type_totals=(
                torch.zeros(_DECISION_TYPE_BIN_COUNT, dtype=torch.long, device=device),
                torch.zeros(_DECISION_TYPE_BIN_COUNT, dtype=torch.long, device=device),
                torch.zeros(_DECISION_TYPE_BIN_COUNT, dtype=torch.float64, device=device),
            ),
            confidence_totals=(
                torch.zeros(4, dtype=torch.long, device=device),
                torch.zeros(4, dtype=torch.long, device=device),
                torch.zeros(4, dtype=torch.float64, device=device),
            ),
            confidence_boundaries=torch.tensor((0.25, 0.5, 0.75), device=device),
            candidate_counts=empty_long.clone(),
        )

    def add(
        self,
        batch: BCDecisionBatch,
        *,
        candidate_offsets: Tensor,
        exact: Tensor,
        partial: Tensor,
        unknown: Tensor,
        labeled_mask: Tensor,
        marginal_nll: Tensor,
        safe_nll: Tensor,
        predicted: Tensor,
        best_scores: Tensor,
    ) -> None:
        exact_correct = torch.all(
            predicted[exact] == batch.exact_action.to(predicted.device)[exact],
            dim=1,
        ).sum()
        self.counts += torch.stack(
            (
                exact.sum() + partial.sum() + unknown.sum(),
                labeled_mask.sum(),
                unknown.sum(),
                exact.sum(),
                partial.sum(),
                exact_correct,
                (~torch.isfinite(best_scores)).sum(),
                ((~torch.isfinite(marginal_nll)) & labeled_mask).sum(),
            )
        )
        safe_nll_64 = safe_nll.to(dtype=torch.float64)
        self.nll_sums += torch.stack(
            (
                safe_nll_64[labeled_mask].sum(),
                safe_nll_64[exact].sum(),
                safe_nll_64[partial].sum(),
            )
        )

        decision_types = batch.decision_type.to(exact.device)
        self.decision_type_totals = _merge_bin_totals(
            self.decision_type_totals,
            _binned_evaluation_totals(
                decision_types,
                labeled_mask,
                safe_nll,
                minlength=_DECISION_TYPE_BIN_COUNT,
            ),
        )

        confidence = batch.label_confidence.to(exact.device)
        bucket_ids = torch.bucketize(confidence, self.confidence_boundaries)
        self.confidence_totals = _merge_bin_totals(
            self.confidence_totals,
            _binned_evaluation_totals(
                bucket_ids,
                labeled_mask,
                safe_nll,
                minlength=len(_CONFIDENCE_BUCKET_NAMES),
            ),
        )

        candidate_sizes = candidate_offsets[1:] - candidate_offsets[:-1]
        batch_candidate_counts = torch.bincount(candidate_sizes)
        width = max(self.candidate_counts.numel(), batch_candidate_counts.numel())
        if self.candidate_counts.numel() < width:
            self.candidate_counts = torch.cat(
                (
                    self.candidate_counts,
                    self.candidate_counts.new_zeros(width - self.candidate_counts.numel()),
                )
            )
        if batch_candidate_counts.numel() < width:
            batch_candidate_counts = torch.cat(
                (
                    batch_candidate_counts,
                    batch_candidate_counts.new_zeros(width - batch_candidate_counts.numel()),
                )
            )
        self.candidate_counts.add_(batch_candidate_counts)

    def finalize(self) -> BCEvaluationMetrics:
        counts = self.counts.cpu().tolist()
        nll_sums = self.nll_sums.cpu().tolist()
        candidate_counts = self.candidate_counts.cpu().tolist()
        decisions, labeled, unknown, exact, partial = (int(value) for value in counts[:5])
        return BCEvaluationMetrics(
            overall_nll=float(nll_sums[0]) / max(labeled, 1),
            exact_nll=float(nll_sums[1]) / max(exact, 1),
            partial_nll=float(nll_sums[2]) / max(partial, 1),
            exact_joint_accuracy=int(counts[5]) / max(exact, 1),
            decisions=decisions,
            labeled_decisions=labeled,
            unknown_decisions=unknown,
            exact_decisions=exact,
            partial_decisions=partial,
            illegal_predictions=int(counts[6]),
            non_finite_values=int(counts[7]),
            by_decision_type=_finalize_evaluation_bins(self.decision_type_totals),
            confidence_buckets=_finalize_evaluation_bins(
                self.confidence_totals,
                names=_CONFIDENCE_BUCKET_NAMES,
            ),
            candidate_set_sizes={
                str(size): int(count) for size, count in enumerate(candidate_counts) if count
            },
        )


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


def _compact_observations(
    observations: list[StructuredObservation],
) -> StructuredObservation:
    if len(observations) == 1:
        return observations[0].clone()
    return StructuredObservation.cat(observations)


def _compact_tensors(tensors: list[Tensor], *, dim: int = 0) -> Tensor:
    if len(tensors) == 1:
        return tensors[0].clone()
    return torch.cat(tensors, dim=dim)


def _collate_bc_window(
    source_windows: list[tuple[ReplayGameChunk, int, int]],
) -> BCDecisionBatch:
    windows: list[BCGameWindow] = []
    observations: list[StructuredObservation] = []
    context_action_masks: list[Tensor] = []
    action_masks: list[Tensor] = []
    label_kinds: list[Tensor] = []
    label_confidences: list[Tensor] = []
    loss_masks: list[Tensor] = []
    decision_types: list[Tensor] = []
    exact_actions: list[Tensor] = []

    candidate_values: list[Tensor] = []
    candidate_offsets = [0]
    target_indices: list[Tensor] = []
    history_indices: list[Tensor] = []
    history_masks: list[Tensor] = []
    history_age_ids: list[Tensor] = []
    candidate_base = 0
    context_base = 0
    batch_start = 0

    for game, start, stop in source_windows:
        batch_stop = batch_start + stop - start
        context_start = max(0, start - HISTORY_WINDOW)
        context_length = stop - context_start
        relative_start = start - context_start
        relative_stop = stop - context_start
        observations.append(game.observations[context_start:stop])
        context_action_masks.append(game.action_mask[context_start:stop])
        windows.append(
            BCGameWindow(
                series_key=game.series_key,
                game_number=game.game_number,
                batch_start=batch_start,
                batch_stop=batch_stop,
                is_game_end=stop == game.length,
                is_series_end=stop == game.length and game.is_series_end,
            )
        )

        action_masks.append(game.action_mask[start:stop])
        label_kinds.append(game.label_kind[start:stop])
        label_confidences.append(game.label_confidence[start:stop])
        loss_masks.append(game.loss_mask[start:stop])
        decision_types.append(game.decision_type[start:stop])
        exact_actions.append(game.exact_action[start:stop])

        first_candidate = int(game.candidate_offsets[start])
        last_candidate = int(game.candidate_offsets[stop])
        candidate_values.append(game.candidate_values[first_candidate:last_candidate])
        local_offsets = game.candidate_offsets[start + 1 : stop + 1] - first_candidate
        candidate_offsets.extend(candidate_base + int(offset) for offset in local_offsets)
        candidate_base += last_candidate - first_candidate

        local_targets = torch.arange(relative_start, relative_stop, dtype=torch.long)
        target_indices.append(context_base + local_targets)
        local_history = local_targets.unsqueeze(1) + torch.arange(
            -HISTORY_WINDOW,
            0,
            dtype=torch.long,
        ).unsqueeze(0)
        local_mask = local_history >= 0
        history_indices.append(
            torch.where(
                local_mask,
                context_base + local_history,
                0,
            )
        )
        history_masks.append(local_mask)
        ages = torch.arange(HISTORY_WINDOW - 1, -1, -1, dtype=torch.long).unsqueeze(0)
        history_age_ids.append(torch.where(local_mask, ages, 0))

        context_base += context_length
        batch_start = batch_stop

    return BCDecisionBatch(
        observations=_compact_observations(observations),
        context_action_mask=_compact_tensors(context_action_masks),
        action_mask=_compact_tensors(action_masks),
        label_kind=_compact_tensors(label_kinds),
        label_confidence=_compact_tensors(label_confidences),
        loss_mask=_compact_tensors(loss_masks),
        decision_type=_compact_tensors(decision_types),
        exact_action=_compact_tensors(exact_actions),
        candidate_values=_compact_tensors(candidate_values),
        candidate_offsets=torch.tensor(candidate_offsets, dtype=torch.long),
        target_indices=_compact_tensors(target_indices),
        history_indices=_compact_tensors(history_indices),
        history_mask=_compact_tensors(history_masks),
        history_age_ids=_compact_tensors(history_age_ids),
        windows=tuple(windows),
    )


def collate_bc_batches(
    games: Iterable[ReplayGameChunk], batch_decisions: int
) -> Iterator[BCDecisionBatch]:
    """Fill a decision budget across game perspectives without crossing histories."""
    if type(batch_decisions) is not int or batch_decisions <= 0:
        raise ValueError("batch_decisions must be a positive integer")
    windows: list[tuple[ReplayGameChunk, int, int]] = []
    decisions = 0
    for game in games:
        start = 0
        while start < game.length:
            take = min(batch_decisions - decisions, game.length - start)
            windows.append((game, start, start + take))
            decisions += take
            start += take
            if decisions == batch_decisions:
                yield _collate_bc_window(windows)
                windows = []
                decisions = 0

    if windows:
        yield _collate_bc_window(windows)


class BCBatchDataset(IterableDataset):
    """Wrap dataset collation so DataLoader workers can yield pre-assembled batches."""

    def __init__(self, dataset: Iterable[ReplayGameChunk], chunk_size: int):
        self.dataset = dataset
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[BCDecisionBatch]:
        yield from collate_bc_batches(self.dataset, self.chunk_size)


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
        self._series_history = _BCSeriesHistoryStore(policy.d_model)
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
        return {
            "loss": float(totals["loss"]) / max(float(totals["loss_weight"]), 1.0),
            "exact_nll": float(totals["exact_nll"]) / max(int(totals["exact_decisions"]), 1),
            "partial_nll": float(totals["partial_nll"]) / max(int(totals["partial_decisions"]), 1),
            "decisions": int(totals["decisions"]),
            "labeled_decisions": int(totals["labeled_decisions"]),
            "exact_decisions": int(totals["exact_decisions"]),
            "partial_decisions": int(totals["partial_decisions"]),
            "updates": int(totals["updates"]),
            "games": int(totals["games"]),
            "decisions_per_update": (
                float(totals["decisions"]) / int(totals["updates"])
                if int(totals["updates"])
                else 0.0
            ),
            "games_per_update": (
                float(totals["games"]) / int(totals["updates"]) if int(totals["updates"]) else 0.0
            ),
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
        if start >= stop:
            raise ValueError("A replay game must contain at least one decision")

        device = local_tokens.device
        n = stop - start

        targets = torch.arange(start, stop, device=device).unsqueeze(1)
        offsets = torch.arange(-HISTORY_WINDOW, 0, device=device).unsqueeze(0)
        idx = targets + offsets

        mask = idx >= 0
        valid_idx = torch.where(mask, idx, 0)

        packed = local_tokens[valid_idx]
        packed = packed * mask.unsqueeze(-1)

        ages = torch.arange(HISTORY_WINDOW - 1, -1, -1, device=device).unsqueeze(0).expand(n, -1)
        ages = torch.where(mask, ages, 0)

        return packed, mask, ages

    def _model_inputs(
        self,
        batch: BCDecisionBatch,
    ) -> BCModelInputs:
        """Prepare and immediately commit inputs for isolated diagnostics."""
        prepared = self._prepare_model_inputs(batch)
        self._series_history.apply(prepared.history_updates)
        return prepared.model_inputs

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

        prior_keys, history_by_key, history_updates = self._plan_series_history(
            batch,
            target_local_tokens,
        )
        summaries = self._resample_game_histories(history_by_key)
        series_tokens, series_mask = self._pack_series_context(
            batch.windows,
            prior_keys,
            summaries,
            target_local_tokens,
        )

        return _PreparedBCBatch(
            model_inputs=(
                target_encoded,
                batch.action_mask.to(self.device),
                series_tokens,
                series_mask,
                history_tokens,
                history_mask,
                history_age_ids,
            ),
            history_updates=history_updates,
        )

    def _plan_series_history(
        self,
        batch: BCDecisionBatch,
        target_local_tokens: Tensor,
    ) -> tuple[
        tuple[tuple[_BCGameKey, ...], ...],
        dict[_BCGameKey, Tensor],
        tuple[_BCGameHistoryUpdate, ...],
    ]:
        completed_in_batch: dict[_BCGameKey, Tensor] = {}
        fragments_in_batch: dict[_BCGameKey, list[Tensor]] = {}
        history_by_key: dict[_BCGameKey, Tensor] = {}
        prior_keys: list[tuple[_BCGameKey, ...]] = []
        updates: list[_BCGameHistoryUpdate] = []
        ended_in_batch: set[SeriesPerspectiveKey] = set()

        for window in batch.windows:
            if (
                self._series_history.is_ended(window.series_key)
                or window.series_key in ended_in_batch
            ):
                raise ValueError("A BC batch contains data after a perspective-series ended")
            game_key = _BCGameKey(window.series_key, window.game_number)
            persistent_prior = self._series_history.completed_before(
                window.series_key,
                window.game_number,
            )
            available_prior = {**persistent_prior, **completed_in_batch}
            window_prior = tuple(
                sorted(
                    (
                        key
                        for key in available_prior
                        if key.series_key == window.series_key
                        and key.game_number < window.game_number
                    ),
                    key=lambda key: key.game_number,
                )[-MAX_PRIOR_GAMES:]
            )
            prior_keys.append(window_prior)
            for prior_key in window_prior:
                if prior_key not in history_by_key:
                    history_by_key[prior_key] = available_prior[prior_key].to(
                        device=self.device,
                        dtype=target_local_tokens.dtype,
                    )

            token_chunk = target_local_tokens[window.batch_start : window.batch_stop]
            current_parts = [
                fragment.to(device=self.device, dtype=target_local_tokens.dtype)
                for fragment in self._series_history.partial_fragments(game_key)
            ]
            current_parts.extend(fragments_in_batch.get(game_key, ()))
            current_parts.append(token_chunk)

            if window.is_game_end:
                completed_in_batch[game_key] = torch.cat(current_parts, dim=0)
                fragments_in_batch.pop(game_key, None)
            else:
                fragments_in_batch.setdefault(game_key, []).append(token_chunk)

            updates.append(
                _BCGameHistoryUpdate(
                    game_key=game_key,
                    tokens=token_chunk,
                    is_game_end=window.is_game_end,
                    is_series_end=window.is_series_end,
                )
            )
            if window.is_series_end:
                ended_in_batch.add(window.series_key)

        return tuple(prior_keys), history_by_key, tuple(updates)

    def _resample_game_histories(
        self,
        history_by_key: Mapping[_BCGameKey, Tensor],
    ) -> dict[_BCGameKey, Tensor]:
        if not history_by_key:
            return {}

        game_keys = tuple(history_by_key)
        histories = list(history_by_key.values())
        lengths = torch.tensor(
            [history.size(0) for history in histories],
            device=self.device,
        )
        padded = pad_sequence(histories, batch_first=True)
        positions = torch.arange(padded.size(1), device=self.device)
        history_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        resampled = self.policy.series.resample_single_game(padded, history_mask)
        return dict(zip(game_keys, resampled.unbind(0), strict=True))

    @staticmethod
    def _pack_series_context(
        windows: tuple[BCGameWindow, ...],
        prior_keys: tuple[tuple[_BCGameKey, ...], ...],
        summaries: Mapping[_BCGameKey, Tensor],
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        window_tokens: list[Tensor] = []
        window_masks: list[Tensor] = []
        window_lengths: list[int] = []
        for window, keys in zip(windows, prior_keys, strict=True):
            game_summaries = [summaries[key] for key in keys]
            used_slots = len(game_summaries) * SERIES_TOKENS_PER_GAME
            padding = reference.new_zeros((SERIES_SLOTS - used_slots, reference.size(-1)))
            row = torch.cat((*game_summaries, padding), dim=0)
            mask = torch.arange(SERIES_SLOTS, device=reference.device) < used_slots
            window_tokens.append(row)
            window_masks.append(mask)
            window_lengths.append(window.batch_stop - window.batch_start)

        repeats = torch.tensor(window_lengths, device=reference.device)
        return (
            torch.repeat_interleave(torch.stack(window_tokens), repeats, dim=0),
            torch.repeat_interleave(torch.stack(window_masks), repeats, dim=0),
        )

    def _forward_batch(
        self,
        batch: BCDecisionBatch,
    ) -> tuple[Tensor, tuple[_BCGameHistoryUpdate, ...]]:
        prepared = self._prepare_model_inputs(batch)
        log_probs = self.policy.actor.score_joint_candidates(
            *prepared.model_inputs,
            batch.candidate_values.to(self.device),
            batch.candidate_offsets.to(self.device),
        )
        return log_probs, prepared.history_updates

    def _greedy_actions(
        self,
        model_inputs: BCModelInputs,
    ) -> tuple[Tensor, Tensor]:
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
        labeled_decisions = objective.labeled_count
        exact_decisions = objective.exact_count
        partial_decisions = objective.partial_count
        loss_sum = 0.0

        if labeled_decisions:
            loss = objective.loss * objective.loss_weight
            if not torch.isfinite(loss):
                raise ValueError("Non-finite BC loss in a collated decision batch")
            self.scaler.scale(loss).backward()
            loss_sum = objective.loss.detach().item() * objective.loss_weight

        self._series_history.apply(history_updates)
        totals["loss"] += loss_sum
        totals["loss_weight"] += objective.loss_weight
        totals["exact_nll"] += objective.exact_nll.detach().item() * exact_decisions
        totals["partial_nll"] += objective.partial_nll.detach().item() * partial_decisions
        totals["decisions"] += batch.decisions
        totals["labeled_decisions"] += labeled_decisions
        totals["exact_decisions"] += exact_decisions
        totals["partial_decisions"] += partial_decisions
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

            _validate_objective_inputs(
                candidate_log_probs.numel(),
                batch.candidate_offsets,
                batch.label_kind,
                batch.loss_mask,
            )
            labels = batch.label_kind.to(self.device)
            exact = labels == int(LabelKind.EXACT)
            partial = labels == int(LabelKind.PARTIAL)
            unknown = labels == int(LabelKind.UNKNOWN)
            labeled_mask = exact | partial

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
                exact=exact,
                partial=partial,
                unknown=unknown,
                labeled_mask=labeled_mask,
                marginal_nll=marginal_nll,
                safe_nll=safe_nll,
                predicted=predicted,
                best_scores=best_scores,
            )
            self._series_history.apply(prepared.history_updates)

        self._series_history.clear()
        return accumulator.finalize()


def _validate_objective_inputs(
    candidate_count: int,
    candidate_offsets: Tensor,
    label_kind: Tensor,
    loss_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if candidate_offsets.dim() != 1 or candidate_offsets.dtype != torch.long:
        raise ValueError("candidate_offsets must be a one-dimensional torch.long tensor")

    if label_kind.dim() != 1 or loss_mask.dim() != 1:
        raise ValueError("label_kind and loss_mask must be one-dimensional")

    if label_kind.numel() + 1 != candidate_offsets.numel():
        raise ValueError("candidate_offsets must have one boundary per decision")

    if label_kind.numel() != loss_mask.numel():
        raise ValueError("label_kind and loss_mask must have matching lengths")

    if (
        candidate_offsets.device != label_kind.device
        or candidate_offsets.device != loss_mask.device
    ):
        raise ValueError("candidate label-contract tensors must share a device")

    if candidate_offsets[0].item() != 0 or candidate_offsets[-1].item() != candidate_count:
        raise ValueError("candidate_offsets must start at zero and end at candidate count")

    if torch.any(candidate_offsets[1:] < candidate_offsets[:-1]):
        raise ValueError("candidate_offsets must be nondecreasing")

    if torch.any((loss_mask < 0) | (loss_mask > 1)):
        raise ValueError("loss_mask values must be in [0, 1]")

    exact = label_kind == int(LabelKind.EXACT)
    partial = label_kind == int(LabelKind.PARTIAL)
    unknown = label_kind == int(LabelKind.UNKNOWN)

    if torch.any(~(exact | partial | unknown)):
        raise ValueError("label_kind contains an unsupported label")

    counts = candidate_offsets[1:] - candidate_offsets[:-1]
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

    return candidate_offsets, exact, partial, unknown


def _ragged_logsumexp(candidate_log_probs: Tensor, offsets: Tensor) -> Tensor:
    counts = offsets[1:] - offsets[:-1]

    row_max = torch.segment_reduce(candidate_log_probs, reduce="max", lengths=counts, unsafe=True)
    row_max = torch.where(counts > 0, row_max, float("-inf"))

    row_ids = torch.repeat_interleave(
        torch.arange(counts.numel(), device=candidate_log_probs.device), counts
    )
    gathered_max = row_max[row_ids]

    shifted = torch.where(
        torch.isfinite(gathered_max),
        candidate_log_probs - gathered_max,
        torch.zeros_like(candidate_log_probs),
    )

    row_sum = torch.segment_reduce(torch.exp(shifted), reduce="sum", lengths=counts, unsafe=True)

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
    """Compute exact and candidate-marginalized NLL without dropping unknown steps.

    Arguments:
      candidate_log_probs: A one-dimensional tensor of log probabilities for all candidates.
      candidate_offsets: A one-dimensional tensor marking the start index of candidates for each decision.
      label_kind: A one-dimensional tensor indicating if a decision is exact, partial, or unknown.
      loss_mask: A one-dimensional tensor weighting the loss for each decision.

    Returns:
      A BCObjective dataclass containing the computed loss, metrics, and marginal log probabilities.
    """
    if candidate_log_probs.dim() != 1:
        raise ValueError("candidate_log_probs must be one-dimensional")
    if (
        candidate_offsets.device != candidate_log_probs.device
        or label_kind.device != candidate_log_probs.device
        or loss_mask.device != candidate_log_probs.device
    ):
        raise ValueError("objective tensors must share a device")

    offsets, exact, partial, _ = _validate_objective_inputs(
        candidate_log_probs.numel(),
        candidate_offsets,
        label_kind,
        loss_mask,
    )

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
    labeled_count = int((exact | partial).sum().item())
    loss_weight = float(mask_total.item())
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
        loss_weight=loss_weight,
    )


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
