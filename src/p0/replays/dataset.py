"""Lazy readers and deterministic split manifests for compiled replay shards."""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterable, Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import orjson
import torch
from torch.utils.data import IterableDataset, get_worker_info

from p0.battle.series import SeriesPerspectiveKey
from p0.contracts import require_dataclass_fields
from p0.model.structured_observation import StructuredObservation
from p0.persistence import atomic_json_save
from p0.replays.shards import (
    SHARD_SUMMARY_KEY,
    ShardIndexEntry,
    final_observation_field_specs,
    load_shard_manifest,
    observation_field_specs,
    validate_shard_layout,
    validate_shard_summaries,
)

SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class SeriesSplitManifest:
    """Persisted assignments tied to one dataset and split UUID."""

    seed: int
    assignments: Mapping[str, str]
    dataset_id: str
    split_id: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("SeriesSplitManifest.seed must be an integer")

        for series_id, split in self.assignments.items():
            if not isinstance(series_id, str) or not series_id:
                raise ValueError("Series split ids must be non-empty strings")

            if split not in SPLITS:
                raise ValueError(f"Unsupported series split {split!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "split_id": self.split_id,
            "seed": self.seed,
            "assignments": {
                series_id: self.assignments[series_id] for series_id in sorted(self.assignments)
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SeriesSplitManifest:
        require_dataclass_fields(value, cls)
        assignments = value["assignments"]
        if not isinstance(assignments, Mapping):
            raise ValueError("SeriesSplitManifest.assignments must be an object")

        return cls(
            seed=int(value["seed"]),
            assignments={str(series_id): str(split) for series_id, split in assignments.items()},
            dataset_id=str(value["dataset_id"]),
            split_id=str(value["split_id"]),
        )


def assign_series_splits(
    series_ids: Iterable[str],
    *,
    seed: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    dataset_id: str,
) -> SeriesSplitManifest:
    """Assign complete series deterministically while keeping requested splits populated."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")

    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction plus test_fraction must be less than one")

    unique_ids = sorted(set(series_ids))
    if any(not series_id for series_id in unique_ids):
        raise ValueError("series_ids must contain only non-empty strings")

    ranked = unique_ids
    random.Random(seed).shuffle(ranked)

    requested = int(validation_fraction > 0) + int(test_fraction > 0)

    enough_for_all = len(ranked) >= requested + 1

    test_count = round(len(ranked) * test_fraction)
    validation_count = round(len(ranked) * validation_fraction)

    # Preserve every requested split when the dataset can also retain training data.
    if enough_for_all and test_fraction > 0:
        test_count = max(1, test_count)

    if enough_for_all and validation_fraction > 0:
        validation_count = max(1, validation_count)

    # Fraction rounding must not consume the training split.
    while test_count + validation_count >= len(ranked) and test_count + validation_count:
        if validation_count > int(enough_for_all and validation_fraction > 0):
            validation_count -= 1
        elif test_count > int(enough_for_all and test_fraction > 0):
            test_count -= 1
        else:
            break

    assignments = {
        series_id: (
            "test"
            if index < test_count
            else "validation"
            if index < test_count + validation_count
            else "train"
        )
        for index, series_id in enumerate(ranked)
    }

    return SeriesSplitManifest(
        seed,
        assignments,
        dataset_id=dataset_id,
        split_id=str(uuid.uuid4()),
    )


def write_split_manifest(manifest: SeriesSplitManifest, path: str | Path) -> None:
    """Atomically persist a split manifest."""
    atomic_json_save(Path(path), manifest.to_dict())


def load_split_manifest(
    value: Mapping[str, Any] | str | Path,
) -> SeriesSplitManifest:
    """Load and validate a split manifest before selecting any shard rows."""
    if isinstance(value, (str, Path)):
        try:
            value = orjson.loads(Path(value).read_bytes())
        except (OSError, UnicodeDecodeError, orjson.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read split manifest {value}") from exc

    if not isinstance(value, Mapping):
        raise ValueError("Split manifest must be a JSON object")

    return SeriesSplitManifest.from_dict(value)


@dataclass(frozen=True, slots=True)
class ReplayGameChunk:
    """One complete chronological game perspective from a compiled shard."""

    series_id: str
    game_number: int
    player: int
    canonical_player: int
    observations: StructuredObservation
    action_mask: torch.Tensor
    mask_provenance: torch.Tensor
    label_kind: torch.Tensor
    label_confidence: torch.Tensor
    loss_mask: torch.Tensor
    decision_type: torch.Tensor
    exact_action: torch.Tensor
    candidate_values: torch.Tensor
    candidate_offsets: torch.Tensor
    outcome: torch.Tensor
    # One-row board after the game's last line; it feeds series memory only.
    final_observation: StructuredObservation
    is_series_end: bool = False
    outcome_valid: bool = False

    def __post_init__(self) -> None:
        if (
            self.player not in (0, 1)
            or self.canonical_player not in (0, 1)
            or self.game_number not in (1, 2, 3)
            or type(self.is_series_end) is not bool
        ):
            raise ValueError("ReplayGameChunk has invalid player or game number")

    @property
    def length(self) -> int:
        return self.observations.token_type_ids.shape[0]

    @property
    def series_key(self) -> SeriesPerspectiveKey:
        return SeriesPerspectiveKey(self.series_id, self.canonical_player)


def _series_in_split(manifest: SeriesSplitManifest, split: str) -> frozenset[str]:
    return frozenset(
        series for series, assigned in manifest.assignments.items() if assigned == split
    )


class LazyReplayDataset(IterableDataset):
    """Stream complete game perspectives while keeping one shard resident."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        split_manifest: SeriesSplitManifest | Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        value = orjson.loads(self.manifest_path.read_bytes())
        try:
            self.manifest = load_shard_manifest(value)
        except ValueError as exc:
            raise ValueError(
                f"Cannot use dataset manifest {self.manifest_path}: {exc}. "
                "Run 'p0-replays build-shards' to publish the dataset again"
            ) from exc
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unsupported dataset split {split!r}")

        if split is not None and split_manifest is None:
            raise ValueError("A split_manifest is required when split is selected")

        if split_manifest is None:
            loaded_split = None
        elif isinstance(split_manifest, SeriesSplitManifest):
            loaded_split = split_manifest
        else:
            loaded_split = load_split_manifest(split_manifest)

        if loaded_split is not None and loaded_split.dataset_id != self.manifest.dataset_id:
            raise ValueError("Split and shard manifests reference different datasets")

        self.split = split
        self.split_manifest = loaded_split
        self._root = self.manifest_path.parent.resolve()

        # Resolving the selection here keeps the streaming loop free of split branching.
        self._selected_series: frozenset[str] | None = (
            None if split is None or loaded_split is None else _series_in_split(loaded_split, split)
        )
        self._accepted_series_ids = tuple(entry.series_id for entry in self.manifest.shards)
        if loaded_split is not None and set(loaded_split.assignments) != set(
            self._accepted_series_ids
        ):
            raise ValueError("Split assignments must cover every accepted series")

    def for_split(self, split: str) -> LazyReplayDataset:
        """Reuse a validated dataset without rescanning every shard."""
        if split not in SPLITS or self.split_manifest is None:
            raise ValueError("A valid split and split manifest are required")
        dataset = copy(self)
        dataset.split = split
        dataset._selected_series = _series_in_split(self.split_manifest, split)
        return dataset

    def __iter__(self) -> Iterator[ReplayGameChunk]:
        selected = self._selected_series
        shards = self.manifest.shards

        worker_info = get_worker_info()
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        for entry in shards:
            if selected is not None and entry.series_id not in selected:
                continue
            tensors, summaries = self._load_shard(entry)
            game_offsets = tensors["game_offsets"].tolist()
            final_game_numbers: dict[SeriesPerspectiveKey, int] = {}
            for item in summaries:
                key = SeriesPerspectiveKey(
                    str(item["series_id"]),
                    int(item["canonical_player"]),
                )
                final_game_numbers[key] = max(
                    int(item["game_number"]),
                    final_game_numbers.get(key, 0),
                )
            for game_index, item in enumerate(summaries):
                series_id = str(item["series_id"])
                series_key = SeriesPerspectiveKey(series_id, int(item["canonical_player"]))
                yield self._chunk(
                    tensors,
                    item,
                    game_index,
                    game_offsets[game_index],
                    game_offsets[game_index + 1],
                    is_series_end=int(item["game_number"]) == final_game_numbers[series_key],
                )

    def accepted_series_ids(self) -> tuple[str, ...]:
        """Return series that have games in the shard payloads."""
        return self._accepted_series_ids

    def _load_shard(
        self, entry: ShardIndexEntry
    ) -> tuple[Mapping[str, torch.Tensor], list[Mapping[str, Any]]]:
        path = (self._root / entry.filename).resolve()
        if self._root not in path.parents:
            raise ValueError(f"Shard filename escapes the manifest directory: {entry.filename!r}")

        try:
            payload = torch.load(path, weights_only=True, map_location="cpu")
        except (OSError, RuntimeError, EOFError, UnpicklingError) as exc:
            raise ValueError(f"Unable to load shard {path}; rebuild the dataset") from exc

        if not isinstance(payload, Mapping):
            raise ValueError(f"Malformed shard {path}: expected a mapping")
        if payload.get("runtime_major") != self.manifest.runtime_major:
            raise ValueError(f"Shard vocabulary or encoding is incompatible: {path}")

        tensors = payload.get("tensors")
        summaries = payload.get(SHARD_SUMMARY_KEY)
        if not isinstance(tensors, Mapping) or not isinstance(summaries, list):
            raise ValueError(f"Malformed shard payload {path}")

        try:
            validate_shard_layout(tensors)
        except ValueError as exc:
            raise ValueError(
                f"Shard {path} does not match the current tensor layout; "
                "rebuild it with --force-reconstruct"
            ) from exc

        try:
            validate_shard_summaries(tensors, summaries)
        except ValueError as exc:
            raise ValueError(
                f"Malformed shard summaries in {path}: {exc}; rebuild the dataset"
            ) from exc

        if (
            len(summaries) != entry.games
            or tensors["loss_mask"].shape[0] != entry.decisions
            or any(item["series_id"] != entry.series_id for item in summaries)
        ):
            raise ValueError(f"Shard content does not match its manifest entry: {path}")

        return tensors, summaries

    @staticmethod
    def _chunk(
        tensors: Mapping[str, torch.Tensor],
        item: Mapping[str, Any],
        game_index: int,
        start: int,
        end: int,
        *,
        is_series_end: bool,
    ) -> ReplayGameChunk:
        candidate_bounds = tensors["candidate_offsets"][start : end + 1]
        candidate_start = int(candidate_bounds[0])
        candidate_end = int(candidate_bounds[-1])
        observation = StructuredObservation._from_values(
            [tensors[name][start:end].clone() for name, *_ in observation_field_specs()]
        )
        candidate_offsets = candidate_bounds - candidate_start
        return ReplayGameChunk(
            series_id=str(item["series_id"]),
            game_number=int(item["game_number"]),
            player=int(item["player"]),
            canonical_player=int(item["canonical_player"]),
            observations=observation,
            action_mask=tensors["action_mask"][start:end].clone(),
            mask_provenance=tensors["mask_provenance"][start:end].clone(),
            label_kind=tensors["label_kind"][start:end].clone(),
            label_confidence=tensors["label_confidence"][start:end].clone(),
            loss_mask=tensors["loss_mask"][start:end].clone(),
            decision_type=tensors["decision_type"][start:end].clone(),
            exact_action=tensors["exact_action"][start:end].clone(),
            candidate_values=tensors["candidate_values"][candidate_start:candidate_end].clone(),
            candidate_offsets=candidate_offsets,
            outcome=tensors["outcome"][start:end].clone(),
            final_observation=StructuredObservation._from_values(
                [
                    tensors[name][game_index : game_index + 1].clone()
                    for name, *_ in final_observation_field_specs()
                ]
            ),
            is_series_end=is_series_end,
            outcome_valid=bool(item["outcome_valid"]),
        )


__all__ = [
    "LazyReplayDataset",
    "ReplayGameChunk",
    "SeriesSplitManifest",
    "assign_series_splits",
    "load_split_manifest",
    "write_split_manifest",
]
