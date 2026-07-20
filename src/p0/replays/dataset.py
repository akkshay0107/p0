"""Lazy readers and deterministic split manifests for compiled replay shards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import torch

from p0.battle.series import GameSummary
from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
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

SPLIT_ARTIFACT_SCHEMA = "p0.replay_split.v1"
SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class SeriesSplitManifest:
    """Stable series-to-split assignments tied to one runtime contract."""

    runtime_contract_sha256: str
    seed: int
    assignments: Mapping[str, str]
    artifact_schema: str = SPLIT_ARTIFACT_SCHEMA

    _FIELDS = frozenset({"artifact_schema", "runtime_contract_sha256", "seed", "assignments"})

    def __post_init__(self) -> None:
        if self.artifact_schema != SPLIT_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unsupported split artifact schema {self.artifact_schema!r}; "
                f"expected {SPLIT_ARTIFACT_SCHEMA}"
            )
        if not _is_sha256(self.runtime_contract_sha256):
            raise ValueError("SeriesSplitManifest.runtime_contract_sha256 must be a SHA-256 digest")
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
            "runtime_contract_sha256": self.runtime_contract_sha256,
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
            runtime_contract_sha256=str(value["runtime_contract_sha256"]),
            seed=int(value["seed"]),
            assignments={str(series_id): str(split) for series_id, split in assignments.items()},
        )


def assign_series_splits(
    series_ids: Iterable[str],
    *,
    seed: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    runtime_contract_sha256: str,
) -> SeriesSplitManifest:
    """Assign complete series by stable hashing, independent of input order."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction plus test_fraction must be less than one")
    unique_ids = sorted(set(series_ids))
    if any(not series_id for series_id in unique_ids):
        raise ValueError("series_ids must contain only non-empty strings")
    assignments = {}
    for series_id in unique_ids:
        digest = hashlib.sha256(f"{seed}:{series_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / float(2**64)
        if bucket < test_fraction:
            split = "test"
        elif bucket < test_fraction + validation_fraction:
            split = "validation"
        else:
            split = "train"
        assignments[series_id] = split
    return SeriesSplitManifest(runtime_contract_sha256, seed, assignments)


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
            value = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
    summary_inputs: tuple[GameSummary, ...]

    def __post_init__(self) -> None:
        if self.player not in (0, 1) or self.game_number < 1:
            raise ValueError("ReplayGameChunk has invalid player or game number")
        self.observations.validate(batch_rank=1)
        length = self.observations.token_type_ids.shape[0]
        for name, tensor in (
            ("action_mask", self.action_mask),
            ("mask_provenance", self.mask_provenance),
            ("label_kind", self.label_kind),
            ("label_confidence", self.label_confidence),
            ("loss_mask", self.loss_mask),
            ("decision_type", self.decision_type),
            ("exact_action", self.exact_action),
            ("outcome", self.outcome),
        ):
            if tensor.shape[0] != length:
                raise ValueError(f"ReplayGameChunk.{name} length does not match observations")
        if self.candidate_offsets.shape != (length + 1,):
            raise ValueError("ReplayGameChunk.candidate_offsets must have one row per decision")
        if self.candidate_offsets[0].item() != 0 or self.candidate_offsets[-1].item() != len(
            self.candidate_values
        ):
            raise ValueError("ReplayGameChunk.candidate_offsets must bound candidate_values")

    @property
    def length(self) -> int:
        return self.observations.token_type_ids.shape[0]

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
        )


class LazyReplayDataset:
    """Stream complete game perspectives while keeping one shard resident."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str | None = None,
        split_manifest: SeriesSplitManifest | Mapping[str, Any] | str | Path | None = None,
        runtime_manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST,
        verify_hashes: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
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
            loaded_split.runtime_contract_sha256 != self.manifest.runtime_contract_sha256
        ):
            raise ValueError("Split and shard manifests reference different runtime contracts")
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

    def __iter__(self) -> Iterator[ReplayGameChunk]:
        selected = self._selected_series
        # Keyed by game number because both perspectives of a game repeat its summary.
        summaries_by_game: dict[str, dict[int, GameSummary]] = {}
        for entry in self.manifest.shards:
            tensors, summaries = self._load_shard(entry)
            for item in summaries:
                summary_value = item["summary"]
                if summary_value is not None:
                    series = summaries_by_game.setdefault(str(item["series_id"]), {})
                    series[int(item["game_number"])] = GameSummary.from_dict(summary_value)
            game_offsets = tensors["game_offsets"].tolist()
            for game_index, item in enumerate(summaries):
                series_id = str(item["series_id"])
                if selected is not None and series_id not in selected:
                    continue
                game_number = int(item["game_number"])
                history = tuple(
                    summary
                    for prior_number, summary in sorted(
                        summaries_by_game.get(series_id, {}).items()
                    )
                    if prior_number < game_number
                )
                yield self._chunk(
                    tensors,
                    item,
                    game_offsets[game_index],
                    game_offsets[game_index + 1],
                    history,
                )

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
        validate_artifact_runtime_contract(payload, self.runtime_manifest_path)
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
        required_summary_fields = {"series_id", "game_number", "player", "summary"}
        if not all(
            isinstance(item, Mapping) and required_summary_fields <= set(item) for item in summaries
        ):
            raise ValueError(f"Shard summaries must be objects in {path}")
        return tensors, summaries

    @staticmethod
    def _chunk(
        tensors: Mapping[str, torch.Tensor],
        item: Mapping[str, Any],
        start: int,
        end: int,
        history: tuple[GameSummary, ...],
    ) -> ReplayGameChunk:
        candidate_start = int(tensors["candidate_offsets"][start].item())
        candidate_end = int(tensors["candidate_offsets"][end].item())
        observation = StructuredObservation._from_values(
            [tensors[name][start:end].clone() for name, *_ in observation_field_specs()]
        )
        candidate_offsets = (
            tensors["candidate_offsets"][start : end + 1] - candidate_start
        ).clone()
        return ReplayGameChunk(
            series_id=str(item["series_id"]),
            game_number=int(item["game_number"]),
            player=int(item["player"]),
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
            summary_inputs=history,
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
