"""Behaviour-cloning objectives over exact and ragged replay labels.

This module provides the core components for behavior cloning (BC) training,
including data collation from game chunks, the BCTrainer for running epochs
over the dataset, and custom ragged-tensor Negative Log Likelihood (NLL)
objectives to support both exact and partial candidate scoring.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import LabelKind
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


@dataclass(frozen=True, slots=True)
class BCGameWindow:
    """Compact identity and target span for one perspective-game window."""

    series_key: SeriesPerspectiveKey
    game_number: int
    batch_start: int
    batch_stop: int
    is_game_end: bool


@dataclass(frozen=True, slots=True)
class BCCompletedGame:
    """Full observation payload sent once when a perspective-game completes."""

    series_key: SeriesPerspectiveKey
    game_number: int
    observations: StructuredObservation
    action_mask: Tensor


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
    series_keys: tuple[SeriesPerspectiveKey, ...]
    windows: tuple[BCGameWindow, ...]
    completed_games: tuple[BCCompletedGame, ...]

    @property
    def decisions(self) -> int:
        return int(self.label_kind.numel())

    @property
    def games(self) -> int:
        return len({(window.series_key, window.game_number) for window in self.windows})


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
    completed_games: list[BCCompletedGame] = []
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
    series_keys: list[SeriesPerspectiveKey] = []
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
            )
        )
        if stop == game.length:
            completed_games.append(
                BCCompletedGame(
                    series_key=game.series_key,
                    game_number=game.game_number,
                    observations=game.observations,
                    action_mask=game.action_mask,
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
        series_keys.extend([game.series_key] * (stop - start))

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
        series_keys=tuple(series_keys),
        windows=tuple(windows),
        completed_games=tuple(completed_games),
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
        self.series_store = SeriesTokenStore(policy.d_model)
        self.cancel_requested = cancel_requested
        torch.manual_seed(config.seed)

    def train(self) -> dict[str, float | int]:
        """Run configured epochs over the streaming dataset."""
        if self.config.epochs > 1 and iter(self.dataset) is self.dataset:
            raise ValueError("BC datasets must be re-iterable when epochs is greater than one")

        totals: dict[str, float | int] = {
            "loss": 0.0,
            "exact_nll": 0.0,
            "partial_nll": 0.0,
            "decisions": 0,
            "labeled_decisions": 0,
            "exact_decisions": 0,
            "partial_decisions": 0,
            "updates": 0,
            "games": 0,
        }

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
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        totals: dict[str, float | int] = {
            "loss": 0.0,
            "exact_nll": 0.0,
            "partial_nll": 0.0,
            "decisions": 0,
            "labeled_decisions": 0,
            "exact_decisions": 0,
            "partial_decisions": 0,
            "updates": 0,
            "games": 0,
        }
        accumulated_decisions = 0
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
            self._backward_chunk(batch, totals)
            accumulated_decisions += batch.decisions
            self._record_completed_games(batch)

            if accumulated_decisions >= self.batch_decisions:
                self._step_optimizer()
                accumulated_decisions = 0

        if accumulated_decisions > 0:
            self._step_optimizer()

        self.series_store.clear()
        return totals

    def _metrics(self, totals: dict[str, float | int]) -> dict[str, float | int]:
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        )
        return {
            "loss": float(totals["loss"]) / max(int(totals["labeled_decisions"]), 1),
            "exact_nll": float(totals["exact_nll"]) / max(int(totals["exact_decisions"]), 1),
            "partial_nll": float(totals["partial_nll"]) / max(int(totals["partial_decisions"]), 1),
            "decisions": int(totals["decisions"]),
            "labeled_decisions": int(totals["labeled_decisions"]),
            "exact_decisions": int(totals["exact_decisions"]),
            "partial_decisions": int(totals["partial_decisions"]),
            "updates": int(totals["updates"]),
            "games": int(totals["games"]),
            "decisions_per_update": float(totals["decisions"]) / max(int(totals["updates"]), 1),
            "games_per_update": float(totals["games"]) / max(int(totals["updates"]), 1),
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
    ) -> tuple[EncodedObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        observations = batch.observations.to(self.device)
        context_action_mask = batch.context_action_mask.to(self.device)
        encoded = self.policy.encode(observations, context_action_mask)
        local_tokens = self.policy.local_history_tokens(encoded)
        target_indices = batch.target_indices.to(self.device)
        target_encoded = EncodedObs(
            encoded.tokens[target_indices],
            encoded.aux[target_indices],
            encoded.numerical[target_indices],
        )
        history_indices = batch.history_indices.to(self.device)
        history_mask = batch.history_mask.to(self.device)
        history_tokens = local_tokens[history_indices] * history_mask.unsqueeze(-1)
        history_age_ids = batch.history_age_ids.to(self.device)

        series_tokens, series_mask = self.series_store.get_tokens(
            batch.series_keys,
            device=self.device,
        )

        return (
            target_encoded,
            batch.action_mask.to(self.device),
            series_tokens,
            series_mask,
            history_tokens,
            history_mask,
            history_age_ids,
        )

    def _record_completed_games(self, batch: BCDecisionBatch) -> None:
        if not batch.completed_games:
            return

        with (
            torch.no_grad(),
            autocast(device_type=self.device.type, enabled=self.amp_enabled),
        ):
            observations = StructuredObservation.cat(
                [game.observations for game in batch.completed_games]
            ).to(self.device)
            action_mask = torch.cat([game.action_mask for game in batch.completed_games]).to(
                self.device
            )
            encoded = self.policy.encode(observations, action_mask)
            local_tokens = self.policy.local_history_tokens(encoded)

            offset = 0
            for game in batch.completed_games:
                game_stop = offset + game.action_mask.size(0)
                new_tokens = self.policy.series.resample_single_game(
                    local_tokens[offset:game_stop].unsqueeze(0)
                )[0]
                self.series_store.append(game.series_key, new_tokens)
                offset = game_stop

    def _forward_batch(self, batch: BCDecisionBatch) -> Tensor:
        model_inputs = self._model_inputs(batch)
        return self.policy.actor.score_joint_candidates(
            *model_inputs,
            batch.candidate_values.to(self.device),
            batch.candidate_offsets.to(self.device),
        )

    def _greedy_actions(
        self,
        model_inputs: tuple[EncodedObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        actions, log_probs, _, _ = self.policy.actor.greedy(*model_inputs)
        return actions, log_probs

    def _step_optimizer(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _backward_chunk(self, batch: BCDecisionBatch, totals: dict[str, float | int]) -> None:
        labels = batch.label_kind.to(self.device)
        loss_mask = batch.loss_mask.to(self.device)
        with autocast(device_type=self.device.type, enabled=self.amp_enabled):
            log_probs = self._forward_batch(batch)

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
            loss = objective.loss * (batch.decisions / self.batch_decisions)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite BC loss in a collated decision batch")
            self.scaler.scale(loss).backward()
            loss_sum = objective.loss.detach().item() * labeled_decisions

        totals["loss"] += loss_sum
        totals["exact_nll"] += objective.exact_nll.detach().item() * exact_decisions
        totals["partial_nll"] += objective.partial_nll.detach().item() * partial_decisions
        totals["decisions"] += batch.decisions
        totals["labeled_decisions"] += labeled_decisions
        totals["exact_decisions"] += exact_decisions
        totals["partial_decisions"] += partial_decisions
        totals["updates"] += int(labeled_decisions > 0)
        totals["games"] += batch.games

    @torch.inference_mode()
    def evaluate(
        self,
        dataset: Iterable[ReplayGameChunk] | None = None,
    ) -> BCEvaluationMetrics:
        """Evaluate exact and partial replay labels without changing parameters."""
        self.policy.eval()
        source = self.dataset if dataset is None else dataset

        nll_sum = 0.0
        exact_nll_sum = 0.0
        partial_nll_sum = 0.0
        decisions = 0
        labeled = 0
        unknown_count = 0
        exact_count = 0
        partial_count = 0
        exact_correct = 0
        illegal_predictions = 0
        non_finite = 0

        candidate_sizes: Counter[str] = Counter()
        type_totals: dict[str, dict[str, float | int]] = {}
        confidence_totals: dict[str, dict[str, float | int]] = {}
        bucket_names = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]")
        boundaries = torch.tensor((0.25, 0.5, 0.75))

        chunk_size = min(self.batch_decisions, self.config.max_chunk_size)

        for batch in collate_bc_batches(source, chunk_size):
            model_inputs = self._model_inputs(batch)
            candidate_log_probs = self.policy.actor.score_joint_candidates(
                *model_inputs,
                batch.candidate_values.to(self.device),
                batch.candidate_offsets.to(self.device),
            )
            self._record_completed_games(batch)

            objective = compute_bc_objective(
                candidate_log_probs,
                batch.candidate_offsets.to(self.device),
                batch.label_kind.to(self.device),
                batch.loss_mask.to(self.device),
            )

            labels = batch.label_kind.to(self.device)
            exact = labels == int(LabelKind.EXACT)
            partial = labels == int(LabelKind.PARTIAL)
            unknown = labels == int(LabelKind.UNKNOWN)
            labeled_mask = exact | partial

            marginal_nll = -objective.marginal_log_probs
            finite_labeled = torch.isfinite(marginal_nll[labeled_mask])
            non_finite += int((~finite_labeled).sum())

            safe_nll = torch.where(
                torch.isfinite(marginal_nll),
                marginal_nll,
                torch.zeros_like(marginal_nll),
            )

            nll_sum += float(safe_nll[labeled_mask].sum())
            exact_nll_sum += float(safe_nll[exact].sum())
            partial_nll_sum += float(safe_nll[partial].sum())

            predicted, best_scores = self._greedy_actions(model_inputs)
            illegal_predictions += int((~torch.isfinite(best_scores)).sum())

            exact_correct += int(
                torch.all(
                    predicted[exact] == batch.exact_action.to(self.device)[exact],
                    dim=1,
                ).sum()
            )

            batch_decisions = batch.decisions
            decisions += batch_decisions

            exact_batch = int(exact.sum())
            partial_batch = int(partial.sum())
            unknown_batch = int(unknown.sum())

            exact_count += exact_batch
            partial_count += partial_batch
            unknown_count += unknown_batch
            labeled += exact_batch + partial_batch

            counts = (batch.candidate_offsets[1:] - batch.candidate_offsets[:-1]).tolist()
            candidate_sizes.update(str(count) for count in counts)

            bucket_ids = torch.bucketize(batch.label_confidence, boundaries).tolist()
            decision_types = batch.decision_type.tolist()
            labeled_rows = labeled_mask.cpu().tolist()
            nll_rows = safe_nll.cpu().tolist()

            for decision_type, bucket_id, is_labeled, decision_nll in zip(
                decision_types,
                bucket_ids,
                labeled_rows,
                nll_rows,
                strict=True,
            ):
                type_key = str(decision_type)
                type_item = type_totals.setdefault(
                    type_key,
                    {"decisions": 0, "labeled": 0, "nll_sum": 0.0},
                )
                type_item["decisions"] = int(type_item["decisions"]) + 1
                if is_labeled:
                    type_item["labeled"] = int(type_item["labeled"]) + 1
                    type_item["nll_sum"] = float(type_item["nll_sum"]) + decision_nll

                bucket_key = bucket_names[bucket_id]
                bucket_item = confidence_totals.setdefault(
                    bucket_key,
                    {"decisions": 0, "labeled": 0, "nll_sum": 0.0},
                )
                bucket_item["decisions"] = int(bucket_item["decisions"]) + 1
                if is_labeled:
                    bucket_item["labeled"] = int(bucket_item["labeled"]) + 1
                    bucket_item["nll_sum"] = float(bucket_item["nll_sum"]) + decision_nll

        self.series_store.clear()

        def finalized(
            values: Mapping[str, Mapping[str, float | int]],
        ) -> dict[str, Mapping[str, float | int]]:
            result: dict[str, Mapping[str, float | int]] = {}
            for key, item in sorted(values.items()):
                labeled_item = int(item["labeled"])
                result[key] = {
                    "decisions": int(item["decisions"]),
                    "labeled": labeled_item,
                    "nll": float(item["nll_sum"]) / max(labeled_item, 1),
                }
            return result

        return BCEvaluationMetrics(
            overall_nll=nll_sum / max(labeled, 1),
            exact_nll=exact_nll_sum / max(exact_count, 1),
            partial_nll=partial_nll_sum / max(partial_count, 1),
            exact_joint_accuracy=exact_correct / max(exact_count, 1),
            decisions=decisions,
            labeled_decisions=labeled,
            unknown_decisions=unknown_count,
            exact_decisions=exact_count,
            partial_decisions=partial_count,
            illegal_predictions=illegal_predictions,
            non_finite_values=non_finite,
            by_decision_type=finalized(type_totals),
            confidence_buckets=finalized(confidence_totals),
            candidate_set_sizes=dict(
                sorted(candidate_sizes.items(), key=lambda item: int(item[0]))
            ),
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


__all__ = [
    "BCCancelled",
    "BCCompletedGame",
    "BCDecisionBatch",
    "BCEvaluationMetrics",
    "BCGameWindow",
    "BCObjective",
    "BCTrainer",
    "collate_bc_batches",
    "compute_bc_objective",
]
