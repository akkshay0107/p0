"""Lazy readers and deterministic split manifests for compiled replay shards."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import orjson
import torch
from torch.utils.data import IterableDataset, get_worker_info

from p0.battle.series import SeriesPerspectiveKey
from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    active_global_contract,
    validate_artifact_runtime_contract,
)
from p0.model.structured_observation import StructuredObservation
from p0.persistence import atomic_json_save
from p0.replays.schema import _is_sha256, _require_fields
from p0.replays.shards import (
    SHARD_ARTIFACT_SCHEMA,
    SHARD_SUMMARY_KEY,
    ShardIndexEntry,
    load_shard_manifest,
    observation_field_specs,
    validate_shard_tensors,
)

SPLIT_ARTIFACT_SCHEMA = active_global_contract().payload("replays", "major")[
    "split_artifact_schema"
]
SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class SeriesSplitManifest:
    """Stable series-to-split assignments tied to one runtime contract."""

    global_contract_sha256: str
    seed: int
    assignments: Mapping[str, str]
    dataset_hash: str
    artifact_schema: str = SPLIT_ARTIFACT_SCHEMA

    _FIELDS = frozenset(
        {"artifact_schema", "global_contract_sha256", "dataset_hash", "seed", "assignments"}
    )

    def __post_init__(self) -> None:
        if self.artifact_schema != SPLIT_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unsupported split artifact schema {self.artifact_schema!r}; "
                f"expected {SPLIT_ARTIFACT_SCHEMA}"
            )

        if not _is_sha256(self.global_contract_sha256):
            raise ValueError("SeriesSplitManifest.global_contract_sha256 must be a SHA-256 digest")

        if not _is_sha256(self.dataset_hash):
            raise ValueError("SeriesSplitManifest.dataset_hash must be a SHA-256 digest")

        if type(self.seed) is not int:
            raise ValueError("SeriesSplitManifest.seed must be an integer")

        for series_id, split in self.assignments.items():
            if not isinstance(series_id, str) or not series_id:
                raise ValueError("Series split ids must be non-empty strings")

            if split not in SPLITS:
                raise ValueError(f"Unsupported series split {split!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema": self.artifact_schema,
            "global_contract_sha256": self.global_contract_sha256,
            "dataset_hash": self.dataset_hash,
            "seed": self.seed,
            "assignments": {
                series_id: self.assignments[series_id] for series_id in sorted(self.assignments)
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SeriesSplitManifest:
        _require_fields(value, cls._FIELDS, "SeriesSplitManifest")
        assignments = value["assignments"]
        if not isinstance(assignments, Mapping):
            raise ValueError("SeriesSplitManifest.assignments must be an object")

        return cls(
            artifact_schema=str(value["artifact_schema"]),
            global_contract_sha256=str(value["global_contract_sha256"]),
            seed=int(value["seed"]),
            assignments={str(series_id): str(split) for series_id, split in assignments.items()},
            dataset_hash=str(value["dataset_hash"]),
        )


def assign_series_splits(
    series_ids: Iterable[str],
    *,
    seed: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    global_contract_sha256: str,
    dataset_hash: str,
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

    seed_prefix = f"{seed}:".encode("utf-8")
    hashed_pairs = [
        (hashlib.sha256(seed_prefix + series_id.encode("utf-8")).digest(), series_id)
        for series_id in unique_ids
    ]
    hashed_pairs.sort()

    ranked = [series_id for _, series_id in hashed_pairs]
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
        global_contract_sha256,
        seed,
        assignments,
        dataset_hash=dataset_hash,
    )


def write_split_manifest(manifest: SeriesSplitManifest, path: str | Path) -> None:
    """Atomically persist a split manifest."""
    atomic_json_save(Path(path), manifest.to_dict())


def load_split_manifest(
    value: Mapping[str, Any] | str | Path,
    manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
) -> SeriesSplitManifest:
    """Load and validate a split manifest before selecting any shard rows."""
    if isinstance(value, (str, Path)):
        try:
            value = orjson.loads(Path(value).read_bytes())
        except (OSError, UnicodeDecodeError, orjson.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read split manifest {value}") from exc

    if not isinstance(value, Mapping):
        raise ValueError("Split manifest must be a JSON object")

    validate_artifact_runtime_contract(value, manifest_path)
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

    def to(self, device: torch.device | str) -> ReplayGameChunk:
        return replace(
            self,
            observations=self.observations.to(device),
            action_mask=self.action_mask.to(device),
            mask_provenance=self.mask_provenance.to(device),
            label_kind=self.label_kind.to(device),
            label_confidence=self.label_confidence.to(device),
            loss_mask=self.loss_mask.to(device),
            decision_type=self.decision_type.to(device),
            exact_action=self.exact_action.to(device),
            candidate_values=self.candidate_values.to(device),
            candidate_offsets=self.candidate_offsets.to(device),
            outcome=self.outcome.to(device),
            outcome_valid=self.outcome_valid,
        )


class LazyReplayDataset(IterableDataset):
    """Stream complete game perspectives while keeping one shard resident."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        split_manifest: SeriesSplitManifest | Mapping[str, Any] | str | Path | None = None,
        runtime_manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
        verify_hashes: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        value = orjson.loads(self.manifest_path.read_bytes())
        self.manifest = load_shard_manifest(value, runtime_manifest_path)
        if split is not None and split not in SPLITS:
            raise ValueError(f"Unsupported dataset split {split!r}")

        if split is not None and split_manifest is None:
            raise ValueError("A split_manifest is required when split is selected")

        if split_manifest is None:
            loaded_split = None
        elif isinstance(split_manifest, SeriesSplitManifest):
            loaded_split = split_manifest
        else:
            loaded_split = load_split_manifest(split_manifest, runtime_manifest_path)

        if loaded_split is not None and (
            loaded_split.global_contract_sha256 != self.manifest.global_contract_sha256
        ):
            raise ValueError("Split and shard manifests reference different runtime contracts")

        if loaded_split is not None and loaded_split.dataset_hash != self.manifest.dataset_hash:
            raise ValueError("Split and shard manifests reference different datasets")

        self.split = split
        self.split_manifest = loaded_split
        self.runtime_manifest_path = Path(runtime_manifest_path)
        self.verify_hashes = verify_hashes
        self._root = self.manifest_path.parent

        # Resolving the selection here keeps the streaming loop free of split branching.
        self._selected_series: frozenset[str] | None = (
            None
            if split is None or loaded_split is None
            else frozenset(
                series_id
                for series_id, assigned in loaded_split.assignments.items()
                if assigned == split
            )
        )
        self._accepted_series_ids = self._validate_shard_family()

    def __iter__(self) -> Iterator[ReplayGameChunk]:
        selected = self._selected_series
        shards = self.manifest.shards

        worker_info = get_worker_info()
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        for entry in shards:
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
                if selected is not None and series_id not in selected:
                    continue

                series_key = SeriesPerspectiveKey(series_id, int(item["canonical_player"]))
                yield self._chunk(
                    tensors,
                    item,
                    game_offsets[game_index],
                    game_offsets[game_index + 1],
                    is_series_end=int(item["game_number"]) == final_game_numbers[series_key],
                )

    def accepted_series_ids(self) -> tuple[str, ...]:
        """Return series that have games in the shard payloads."""
        return self._accepted_series_ids

    def _validate_shard_family(self) -> tuple[str, ...]:
        series_shards: dict[str, str] = {}
        replay_counts: dict[str, int] = {}
        for entry in self.manifest.shards:
            _, summaries = self._load_shard(entry)
            for item in summaries:
                series_id = str(item["series_id"])
                previous = series_shards.setdefault(series_id, entry.filename)
                if previous != entry.filename:
                    raise ValueError(f"Series {series_id!r} spans multiple shards")
                replay_id = str(item["source_replay_id"])
                replay_counts[replay_id] = replay_counts.get(replay_id, 0) + 1

        if len(replay_counts) != self.manifest.accepted_games or any(
            count != 2 for count in replay_counts.values()
        ):
            raise ValueError("Shard summaries do not cover every accepted replay exactly twice")
        return tuple(sorted(series_shards))

    def _load_shard(
        self, entry: ShardIndexEntry
    ) -> tuple[Mapping[str, torch.Tensor], list[Mapping[str, Any]]]:
        path = self._root / entry.filename
        root = self._root.resolve()
        resolved_path = path.resolve()
        if root not in resolved_path.parents:
            raise ValueError(f"Shard filename escapes the manifest directory: {entry.filename!r}")

        path = resolved_path
        if not path.is_file():
            raise ValueError(f"Shard file is missing: {path}")

        if self.verify_hashes:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry.sha256:
                raise ValueError(f"Shard hash mismatch for {path}")

        if path.stat().st_size != entry.byte_size:
            raise ValueError(f"Shard byte-size mismatch for {path}")

        try:
            payload = torch.load(path, weights_only=True, map_location="cpu")
        except (OSError, RuntimeError, EOFError, UnpicklingError) as exc:
            raise ValueError(f"Unable to load shard {path}") from exc

        if not isinstance(payload, Mapping):
            raise ValueError(f"Malformed shard {path}: expected a mapping")

        if payload.get("artifact_schema") != SHARD_ARTIFACT_SCHEMA:
            raise ValueError(f"Unsupported shard schema in {path}")

        if payload.get("dataset_hash") != self.manifest.dataset_hash:
            raise ValueError(f"Shard and manifest reference different datasets: {path}")

        if payload.get("global_contract_sha256") != self.manifest.global_contract_sha256:
            raise ValueError(f"Shard and manifest reference different global contracts: {path}")

        tensors = payload.get("tensors")
        summaries = payload.get(SHARD_SUMMARY_KEY)
        if not isinstance(tensors, Mapping) or not isinstance(summaries, list):
            raise ValueError(f"Malformed shard payload {path}")

        validate_shard_tensors(tensors)
        if len(summaries) != tensors["game_offsets"].numel() - 1:
            raise ValueError(f"Shard summary count does not match game offsets in {path}")

        if (
            len(summaries) != entry.games
            or tensors["loss_mask"].shape[0] != entry.decisions
            or tensors["series_offsets"].numel() - 1 != entry.series
        ):
            raise ValueError(f"Shard index metadata does not match payload {path}")

        # outcome_valid gates every value-head decision, so a shard that omitted
        # it would silently train the critic on nothing.
        required_summary_fields = {
            "series_id",
            "game_number",
            "player",
            "canonical_player",
            "source_replay_id",
            "outcome_valid",
        }
        if not all(
            isinstance(item, Mapping) and required_summary_fields <= set(item) for item in summaries
        ):
            raise ValueError(f"Shard summaries must be objects in {path}")

        self._validate_summaries(tensors, summaries, path)

        if (
            tensors["loss_mask"].shape[0] != entry.decisions
            or len(summaries) != entry.games
            or tensors["series_offsets"].numel() - 1 != entry.series
        ):
            raise ValueError(f"Shard payload counts do not match its index entry in {path}")

        return tensors, summaries

    def _validate_summaries(
        self,
        tensors: Mapping[str, torch.Tensor],
        summaries: list[Mapping[str, Any]],
        path: Path,
    ) -> None:
        game_offsets = tensors["game_offsets"].tolist()
        expected_series_offsets: list[int] = []
        seen_series: set[str] = set()
        last_series = ""
        games: dict[tuple[str, int], list[tuple[int, int]]] = {}
        sequences: dict[SeriesPerspectiveKey, list[int]] = {}

        for index, item in enumerate(summaries):
            series_id = item["series_id"]
            replay_id = item["source_replay_id"]
            game_number = item["game_number"]
            player = item["player"]
            canonical_player = item["canonical_player"]
            if (
                not isinstance(series_id, str)
                or not isinstance(replay_id, str)
                or type(game_number) is not int
                or type(player) is not int
                or type(canonical_player) is not int
                or type(item["outcome_valid"]) is not bool
                or player not in (0, 1)
                or canonical_player not in (0, 1)
            ):
                raise ValueError(f"Shard summary has invalid field types in {path}")

            replay_ids = self.manifest.source_series.get(series_id)
            if (
                replay_ids is None
                or not 1 <= game_number <= len(replay_ids)
                or replay_ids[game_number - 1] != replay_id
            ):
                raise ValueError(f"Shard summary does not match source series in {path}")

            if series_id != last_series:
                if series_id in seen_series:
                    raise ValueError(f"Shard series summaries are not contiguous in {path}")
                seen_series.add(series_id)
                expected_series_offsets.append(game_offsets[index])
                last_series = series_id

            games.setdefault((series_id, game_number), []).append((player, canonical_player))
            key = SeriesPerspectiveKey(series_id, canonical_player)
            sequences.setdefault(key, []).append(game_number)

        expected_series_offsets.append(game_offsets[-1])
        if tensors["series_offsets"].tolist() != expected_series_offsets:
            raise ValueError(f"Shard series offsets do not match summaries in {path}")
        if any(sorted(rows) not in ([(0, 0), (1, 1)], [(0, 1), (1, 0)]) for rows in games.values()):
            raise ValueError(f"Shard games must contain both player perspectives in {path}")
        if any(numbers != list(range(1, len(numbers) + 1)) for numbers in sequences.values()):
            raise ValueError(f"Shard game summaries are not chronological in {path}")

    @staticmethod
    def _chunk(
        tensors: Mapping[str, torch.Tensor],
        item: Mapping[str, Any],
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
            is_series_end=is_series_end,
            outcome_valid=bool(item["outcome_valid"]),
        )


__all__ = [
    "LazyReplayDataset",
    "ReplayGameChunk",
    "SPLIT_ARTIFACT_SCHEMA",
    "SeriesSplitManifest",
    "assign_series_splits",
    "load_split_manifest",
    "write_split_manifest",
]
