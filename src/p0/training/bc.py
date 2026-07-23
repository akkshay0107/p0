"""Behaviour-cloning objectives over exact and ragged replay labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast

from p0.model.architecture_contract import HISTORY_WINDOW, SERIES_SLOTS
from p0.model.cls_reducer import pack_history_tokens
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.structured_observation import StructuredObservation
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
    games: int
    decisions_per_update: float
    games_per_update: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            field: getattr(self, field)
            for field in (
                "loss",
                "exact_nll",
                "partial_nll",
                "decisions",
                "labeled_decisions",
                "exact_decisions",
                "partial_decisions",
                "updates",
                "games",
                "decisions_per_update",
                "games_per_update",
                "peak_memory_bytes",
            )
        }


@dataclass(frozen=True, slots=True)
class BCGameWindow:
    game: ReplayGameChunk
    start: int
    stop: int
    batch_start: int
    batch_stop: int


@dataclass(frozen=True, slots=True)
class BCDecisionBatch:
    """Target decisions plus game-local context descriptions for one update."""

    observations: StructuredObservation
    action_mask: Tensor
    label_kind: Tensor
    label_confidence: Tensor
    loss_mask: Tensor
    decision_type: Tensor
    exact_action: Tensor
    candidate_values: Tensor
    candidate_offsets: Tensor
    history_local_indices: Tensor
    windows: tuple[BCGameWindow, ...]

    @property
    def decisions(self) -> int:
        return int(self.label_kind.numel())

    @property
    def games(self) -> int:
        return len({(window.game.series_id, window.game.game_number) for window in self.windows})


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


def _concatenate_observations(
    observations: Iterable[StructuredObservation],
) -> StructuredObservation:
    values = tuple(observations)
    if not values:
        raise ValueError("A BC decision batch must contain observations")
    return StructuredObservation._from_values(
        [
            torch.cat([observation.tensors()[index] for observation in values], dim=0)
            for index in range(len(values[0].tensors()))
        ]
    )


class MultiGameBCCollator:
    """Fill a decision budget across game perspectives without crossing histories."""

    def __init__(self, batch_decisions: int):
        if type(batch_decisions) is not int or batch_decisions <= 0:
            raise ValueError("batch_decisions must be a positive integer")
        self.batch_decisions = batch_decisions

    def __call__(self, games: Iterable[ReplayGameChunk]) -> Iterator[BCDecisionBatch]:
        windows: list[tuple[ReplayGameChunk, int, int]] = []
        decisions = 0
        for game in games:
            start = 0
            while start < game.length:
                take = min(self.batch_decisions - decisions, game.length - start)
                windows.append((game, start, start + take))
                decisions += take
                start += take
                if decisions == self.batch_decisions:
                    yield self._collate(windows)
                    windows = []
                    decisions = 0
        if windows:
            yield self._collate(windows)

    @staticmethod
    def _collate(
        source_windows: list[tuple[ReplayGameChunk, int, int]],
    ) -> BCDecisionBatch:
        windows: list[BCGameWindow] = []
        observations: list[StructuredObservation] = []
        tensor_values: dict[str, list[Tensor]] = {
            name: []
            for name in (
                "action_mask",
                "label_kind",
                "label_confidence",
                "loss_mask",
                "decision_type",
                "exact_action",
            )
        }
        candidate_values: list[Tensor] = []
        candidate_offsets = [0]
        history_rows: list[Tensor] = []
        batch_start = 0
        candidate_base = 0
        for game, start, stop in source_windows:
            batch_stop = batch_start + stop - start
            windows.append(BCGameWindow(game, start, stop, batch_start, batch_stop))
            observations.append(game.observations[start:stop])
            for name in tensor_values:
                tensor_values[name].append(getattr(game, name)[start:stop])
            first_candidate = int(game.candidate_offsets[start])
            last_candidate = int(game.candidate_offsets[stop])
            candidate_values.append(game.candidate_values[first_candidate:last_candidate])
            local_offsets = game.candidate_offsets[start + 1 : stop + 1] - first_candidate
            candidate_offsets.extend(candidate_base + int(offset) for offset in local_offsets)
            candidate_base += last_candidate - first_candidate
            for target in range(start, stop):
                row = torch.full((HISTORY_WINDOW,), -1, dtype=torch.long)
                left = max(0, target - HISTORY_WINDOW)
                count = target - left
                if count:
                    row[-count:] = torch.arange(left, target, dtype=torch.long)
                history_rows.append(row)
            batch_start = batch_stop
        return BCDecisionBatch(
            observations=_concatenate_observations(observations),
            action_mask=torch.cat(tensor_values["action_mask"]),
            label_kind=torch.cat(tensor_values["label_kind"]),
            label_confidence=torch.cat(tensor_values["label_confidence"]),
            loss_mask=torch.cat(tensor_values["loss_mask"]),
            decision_type=torch.cat(tensor_values["decision_type"]),
            exact_action=torch.cat(tensor_values["exact_action"]),
            candidate_values=torch.cat(candidate_values, dim=0),
            candidate_offsets=torch.tensor(candidate_offsets, dtype=torch.long),
            history_local_indices=torch.stack(history_rows),
            windows=tuple(windows),
        )


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
    games: int = 0

    def add(self, chunk: _RunTotals) -> None:
        self.loss += chunk.loss
        self.exact_nll += chunk.exact_nll
        self.partial_nll += chunk.partial_nll
        self.decisions += chunk.decisions
        self.labeled_decisions += chunk.labeled_decisions
        self.exact_decisions += chunk.exact_decisions
        self.partial_decisions += chunk.partial_decisions
        self.updates += chunk.updates
        self.games += chunk.games


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
        self.collator = MultiGameBCCollator(config.batch_decisions)
        self.cancel_requested = cancel_requested
        torch.manual_seed(config.seed)

    def train(self) -> BCTrainMetrics:
        """Run configured epochs over the streaming dataset."""
        if self.config.epochs > 1 and iter(self.dataset) is self.dataset:
            raise ValueError("BC datasets must be re-iterable when epochs is greater than one")
        totals = _RunTotals()
        for _ in range(self.config.epochs):
            totals.add(self._train_epoch_totals())
        return self._metrics(totals)

    def train_epoch(self) -> BCTrainMetrics:
        """Train exactly one epoch and return its decision-weighted metrics."""
        return self._metrics(self._train_epoch_totals())

    def _train_epoch_totals(self) -> _RunTotals:
        self.policy.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        totals = _RunTotals()
        for batch in self.collator(self.dataset):
            if self.cancel_requested():
                raise BCCancelled("Behaviour-cloning training was cancelled")
            self._train_batch(batch, totals)
        return totals

    def _metrics(self, totals: _RunTotals) -> BCTrainMetrics:
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
            games=totals.games,
            decisions_per_update=totals.decisions / max(totals.updates, 1),
            games_per_update=totals.games / max(totals.updates, 1),
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

    def _empty_series_inputs(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        dtype = next(self.policy.parameters()).dtype if dtype is None else dtype
        return (
            torch.zeros(
                (batch_size, SERIES_SLOTS, self.policy.d_model),
                device=self.device,
                dtype=dtype,
            ),
            torch.zeros(
                (batch_size, SERIES_SLOTS),
                device=self.device,
                dtype=torch.bool,
            ),
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

    def _model_inputs(
        self,
        batch: BCDecisionBatch,
    ) -> tuple[EncodedObs, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        encoded_values: list[EncodedObs] = []
        history_values: list[tuple[Tensor, Tensor, Tensor]] = []
        for window in batch.windows:
            context_start = max(0, window.start - HISTORY_WINDOW)
            observations = window.game.observations[context_start : window.stop].to(self.device)
            context_mask = window.game.action_mask[context_start : window.stop].to(self.device)
            encoded = self.policy.encode(observations, context_mask)
            local_tokens = self.policy.local_history_tokens(encoded)
            relative_start = window.start - context_start
            relative_stop = window.stop - context_start
            encoded_values.append(
                EncodedObs(
                    encoded.tokens[relative_start:relative_stop],
                    encoded.aux[relative_start:relative_stop],
                    encoded.numerical[relative_start:relative_stop],
                )
            )
            history_values.append(
                self._history_inputs(
                    local_tokens,
                    slice(relative_start, relative_stop),
                )
            )
        target_encoded = EncodedObs(
            torch.cat([encoded.tokens for encoded in encoded_values]),
            torch.cat([encoded.aux for encoded in encoded_values]),
            torch.cat([encoded.numerical for encoded in encoded_values]),
        )
        history_tokens = torch.cat([history[0] for history in history_values])
        history_mask = torch.cat([history[1] for history in history_values])
        history_age_ids = torch.cat([history[2] for history in history_values])
        series_tokens, series_mask = self._empty_series_inputs(
            batch.decisions,
            dtype=target_encoded.tokens.dtype,
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
        batch_size = model_inputs[0].tokens.shape[0]
        action_count = model_inputs[1].shape[-1]
        actions = torch.arange(action_count, device=self.device)
        pairs = torch.cartesian_prod(actions, actions)
        candidates = pairs.repeat(batch_size, 1)
        pair_count = pairs.shape[0]
        offsets = torch.arange(
            0,
            (batch_size + 1) * pair_count,
            pair_count,
            device=self.device,
            dtype=torch.long,
        )
        scores = self.policy.actor.score_joint_candidates(
            *model_inputs,
            candidates,
            offsets,
        ).reshape(batch_size, pair_count)
        best_indices = torch.argmax(scores, dim=1)
        best_scores = scores.gather(1, best_indices.unsqueeze(1)).squeeze(1)
        return pairs[best_indices], best_scores

    def _train_batch(self, batch: BCDecisionBatch, totals: _RunTotals) -> None:
        labels = batch.label_kind.to(self.device)
        loss_mask = batch.loss_mask.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
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
            loss = objective.loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite BC loss in a collated decision batch")
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
                decisions=batch.decisions,
                labeled_decisions=labeled_decisions,
                exact_decisions=exact_decisions,
                partial_decisions=partial_decisions,
                updates=int(labeled_decisions > 0),
                games=batch.games,
            )
        )

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
        for batch in self.collator(source):
            model_inputs = self._model_inputs(batch)
            candidate_log_probs = self.policy.actor.score_joint_candidates(
                *model_inputs,
                batch.candidate_values.to(self.device),
                batch.candidate_offsets.to(self.device),
            )
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


__all__ = [
    "BCDecisionBatch",
    "BCCancelled",
    "BCEvaluationMetrics",
    "BCObjective",
    "BCTrainMetrics",
    "BCTrainer",
    "MultiGameBCCollator",
    "compute_bc_objective",
]
