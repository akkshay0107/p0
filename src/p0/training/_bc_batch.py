"""Compact behavior-cloning batches and worker-side collation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from p0.battle.series import SeriesPerspectiveKey
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.structured_observation import StructuredObservation
from p0.replays.dataset import ReplayGameChunk


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


def _compact_observations(
    observations: list[StructuredObservation],
) -> StructuredObservation:
    if len(observations) == 1:
        return observations[0].clone()
    return StructuredObservation.cat(observations)


def _compact_tensors(tensors: list[Tensor]) -> Tensor:
    if len(tensors) == 1:
        return tensors[0].clone()
    return torch.cat(tensors)


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
        history_indices.append(torch.where(local_mask, context_base + local_history, 0))
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
    games: Iterable[ReplayGameChunk],
    batch_decisions: int,
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
    """Wrap collation so DataLoader workers yield pre-assembled batches."""

    def __init__(self, dataset: Iterable[ReplayGameChunk], chunk_size: int) -> None:
        self.dataset = dataset
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[BCDecisionBatch]:
        yield from collate_bc_batches(self.dataset, self.chunk_size)
