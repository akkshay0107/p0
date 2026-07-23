"""Compiled tensor-shard artifact contract for streaming behaviour cloning.

This module owns the derived-tensor layer: bounded shard files holding
stacked schema-v4 observations and label tensors for whole chronological
games, plus the manifest and index that tie a compiled corpus to one runtime
contract. It may import torch and the observation schema; p0.replays.schema
must stay torch-free, and nothing here may import p0.runtime.

Compilation and dataset behavior live elsewhere; this module only defines
the layout and validates manifests before any tensor payload is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from p0.battle.actions import ACT_SIZE
from p0.battle.series import SERIES_SUMMARY_SCHEMA_VERSION
from p0.format_config import DEFAULT_RUNTIME_MANIFEST, validate_artifact_runtime_contract
from p0.model.structured_observation import OBSERVATION_SCHEMA_VERSION, StructuredObservation
from p0.replays.schema import (
    REPLAY_IR_SCHEMA_VERSION,
    _is_sha256,
    _require_fields,
    _require_iso_timestamp,
)

SHARD_ARTIFACT_SCHEMA = "p0.replay_shard.v3"
BO1_COMPILATION_SEMANTICS = "independent_bo1_empty_history.v1"

# Non-observation tensors stored per shard. -1 marks a variable dimension:
# T is the shard's decision count and C its total candidate count. Candidates
# use a ragged values-plus-offsets encoding: candidate_offsets has length
# T + 1 and decision t owns candidate_values[offsets[t]:offsets[t + 1]] rows
# of joint action pairs. exact_action rows are meaningful only where
# label_kind is EXACT; loss_mask is zero on UNKNOWN decisions so they keep
# chronological context without contributing policy loss.
# game_offsets and series_offsets delimit whole chronological games and
# series within the shard. outcome is the optional value target.
SHARD_TENSOR_SPECS: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = (
    ("action_mask", (-1, 2, ACT_SIZE), torch.bool),
    ("mask_provenance", (-1,), torch.long),
    ("label_kind", (-1,), torch.long),
    ("label_confidence", (-1,), torch.float32),
    ("loss_mask", (-1,), torch.float32),
    ("decision_type", (-1,), torch.long),
    ("exact_action", (-1, 2), torch.long),
    ("candidate_values", (-1, 2), torch.long),
    ("candidate_offsets", (-1,), torch.long),
    ("game_offsets", (-1,), torch.long),
    ("series_offsets", (-1,), torch.long),
    ("outcome", (-1,), torch.float32),
)

# Per-game GameSummary payloads ride along as JSON strings, not tensors, so
# the series-context encoder owns their tensorization inside each game graph.
SHARD_SUMMARY_KEY = "series_summaries"


def observation_field_specs() -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    """Observation tensors stacked along a leading decision axis.

    Derived from StructuredObservation._FIELD_SPECS so an observation-schema
    change cannot silently diverge from the shard layout.
    """
    return tuple(
        (name, (-1, *shape), dtype) for name, shape, dtype in StructuredObservation._FIELD_SPECS
    )


@dataclass(frozen=True, slots=True)
class ShardIndexEntry:
    filename: str
    sha256: str
    decisions: int
    games: int
    series: int
    byte_size: int

    _FIELDS = frozenset({"filename", "sha256", "decisions", "games", "series", "byte_size"})

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError("ShardIndexEntry.filename must be non-empty")
        if not _is_sha256(self.sha256):
            raise ValueError("ShardIndexEntry.sha256 must be a lowercase SHA-256 digest")
        for name, count in (
            ("decisions", self.decisions),
            ("games", self.games),
            ("series", self.series),
            ("byte_size", self.byte_size),
        ):
            if type(count) is not int or count < 0:
                raise ValueError(f"ShardIndexEntry.{name} must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "decisions": self.decisions,
            "games": self.games,
            "series": self.series,
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShardIndexEntry:
        _require_fields(value, cls._FIELDS, "ShardIndexEntry")
        return cls(
            filename=str(value["filename"]),
            sha256=str(value["sha256"]),
            decisions=int(value["decisions"]),
            games=int(value["games"]),
            series=int(value["series"]),
            byte_size=int(value["byte_size"]),
        )


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """Index and immutable identity for one compiled shard family."""

    runtime_contract_sha256: str
    shards: tuple[ShardIndexEntry, ...]
    diagnostics: Mapping[str, int]
    created_at: str
    dataset_hash: str
    source_format_id: str
    build_config: Mapping[str, Any]
    raw_replays: Mapping[str, str]
    source_series: Mapping[str, tuple[str, ...]]
    source_games: int
    accepted_games: int
    rejected_games: int
    quality_manifest: str
    quality_manifest_sha256: str
    artifact_hashes: Mapping[str, str]
    compilation_semantics: str = BO1_COMPILATION_SEMANTICS
    artifact_schema: str = SHARD_ARTIFACT_SCHEMA
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    replay_ir_schema_version: int = REPLAY_IR_SCHEMA_VERSION
    series_summary_schema_version: int = SERIES_SUMMARY_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "runtime_contract_sha256",
            "dataset_hash",
            "source_format_id",
            "compilation_semantics",
            "build_config",
            "raw_replays",
            "source_series",
            "source_games",
            "accepted_games",
            "rejected_games",
            "quality_manifest",
            "quality_manifest_sha256",
            "artifact_hashes",
            "shards",
            "diagnostics",
            "created_at",
            "artifact_schema",
            "observation_schema_version",
            "replay_ir_schema_version",
            "series_summary_schema_version",
        }
    )

    def __post_init__(self) -> None:
        if self.artifact_schema != SHARD_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unsupported shard artifact schema {self.artifact_schema!r}; "
                f"expected {SHARD_ARTIFACT_SCHEMA}"
            )
        for name, declared, expected in (
            (
                "observation_schema_version",
                self.observation_schema_version,
                OBSERVATION_SCHEMA_VERSION,
            ),
            ("replay_ir_schema_version", self.replay_ir_schema_version, REPLAY_IR_SCHEMA_VERSION),
            (
                "series_summary_schema_version",
                self.series_summary_schema_version,
                SERIES_SUMMARY_SCHEMA_VERSION,
            ),
        ):
            if declared != expected:
                raise ValueError(
                    f"ShardManifest.{name} {declared!r} does not match the active {expected}"
                )
        if not _is_sha256(self.runtime_contract_sha256):
            raise ValueError(
                "ShardManifest.runtime_contract_sha256 must be a lowercase SHA-256 digest"
            )
        if not _is_sha256(self.dataset_hash):
            raise ValueError("ShardManifest.dataset_hash must be a lowercase SHA-256 digest")
        if not self.source_format_id:
            raise ValueError("ShardManifest.source_format_id must be non-empty")
        if self.compilation_semantics != BO1_COMPILATION_SEMANTICS:
            raise ValueError(f"Unsupported compilation semantics {self.compilation_semantics!r}")
        if not isinstance(self.build_config, Mapping):
            raise ValueError("ShardManifest.build_config must be a mapping")
        for replay_id, digest in self.raw_replays.items():
            if not replay_id or not _is_sha256(digest):
                raise ValueError("ShardManifest.raw_replays contains an invalid identity")
        for series_id, replay_ids in self.source_series.items():
            if not series_id or not replay_ids or len(set(replay_ids)) != len(replay_ids):
                raise ValueError("ShardManifest.source_series contains an invalid membership")
        membership_ids = [
            replay_id for replay_ids in self.source_series.values() for replay_id in replay_ids
        ]
        if len(set(membership_ids)) != len(membership_ids) or set(membership_ids) != set(
            self.raw_replays
        ):
            raise ValueError("ShardManifest.source_series must partition all raw replays")
        for name, count in (
            ("source_games", self.source_games),
            ("accepted_games", self.accepted_games),
            ("rejected_games", self.rejected_games),
        ):
            if type(count) is not int or count < 0:
                raise ValueError(f"ShardManifest.{name} must be a nonnegative integer")
        if self.accepted_games + self.rejected_games != self.source_games:
            raise ValueError("Accepted and rejected games must account for every source game")
        if len(self.raw_replays) != self.source_games:
            raise ValueError("ShardManifest.raw_replays must account for every source game")
        if not self.quality_manifest or not _is_sha256(self.quality_manifest_sha256):
            raise ValueError("ShardManifest quality-manifest identity is invalid")
        for filename, digest in self.artifact_hashes.items():
            if not filename or not _is_sha256(digest):
                raise ValueError("ShardManifest.artifact_hashes contains an invalid entry")
        seen: set[str] = set()
        for entry in self.shards:
            if entry.filename in seen:
                raise ValueError(f"Duplicate shard filename {entry.filename!r}")
            seen.add(entry.filename)
        for key, count in self.diagnostics.items():
            if not isinstance(key, str) or type(count) is not int or count < 0:
                raise ValueError("Shard diagnostics must map strings to nonnegative integers")
        _require_iso_timestamp(self.created_at, "ShardManifest.created_at")

    @property
    def decisions(self) -> int:
        return sum(entry.decisions for entry in self.shards)

    @property
    def games(self) -> int:
        return sum(entry.games for entry in self.shards)

    @property
    def series(self) -> int:
        return sum(entry.series for entry in self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema": self.artifact_schema,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "dataset_hash": self.dataset_hash,
            "source_format_id": self.source_format_id,
            "compilation_semantics": self.compilation_semantics,
            "build_config": dict(self.build_config),
            "raw_replays": {
                replay_id: self.raw_replays[replay_id] for replay_id in sorted(self.raw_replays)
            },
            "source_series": {
                series_id: list(self.source_series[series_id])
                for series_id in sorted(self.source_series)
            },
            "source_games": self.source_games,
            "accepted_games": self.accepted_games,
            "rejected_games": self.rejected_games,
            "quality_manifest": self.quality_manifest,
            "quality_manifest_sha256": self.quality_manifest_sha256,
            "artifact_hashes": {
                filename: self.artifact_hashes[filename]
                for filename in sorted(self.artifact_hashes)
            },
            "observation_schema_version": self.observation_schema_version,
            "replay_ir_schema_version": self.replay_ir_schema_version,
            "series_summary_schema_version": self.series_summary_schema_version,
            "shards": [entry.to_dict() for entry in self.shards],
            "diagnostics": {key: self.diagnostics[key] for key in sorted(self.diagnostics)},
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShardManifest:
        _require_fields(value, cls._FIELDS, "ShardManifest")
        diagnostics = value["diagnostics"]
        if not isinstance(diagnostics, Mapping):
            raise ValueError("ShardManifest.diagnostics must be a JSON object")
        source_series = value["source_series"]
        build_config = value["build_config"]
        raw_replays = value["raw_replays"]
        artifact_hashes = value["artifact_hashes"]
        if (
            not isinstance(source_series, Mapping)
            or not isinstance(build_config, Mapping)
            or not isinstance(raw_replays, Mapping)
            or not isinstance(artifact_hashes, Mapping)
        ):
            raise ValueError("ShardManifest identity fields must be JSON objects")
        return cls(
            artifact_schema=str(value["artifact_schema"]),
            runtime_contract_sha256=str(value["runtime_contract_sha256"]),
            dataset_hash=str(value["dataset_hash"]),
            source_format_id=str(value["source_format_id"]),
            compilation_semantics=str(value["compilation_semantics"]),
            build_config=dict(build_config),
            raw_replays={str(replay_id): str(digest) for replay_id, digest in raw_replays.items()},
            source_series={
                str(series_id): tuple(str(replay_id) for replay_id in replay_ids)
                for series_id, replay_ids in source_series.items()
            },
            source_games=int(value["source_games"]),
            accepted_games=int(value["accepted_games"]),
            rejected_games=int(value["rejected_games"]),
            quality_manifest=str(value["quality_manifest"]),
            quality_manifest_sha256=str(value["quality_manifest_sha256"]),
            artifact_hashes={
                str(filename): str(digest) for filename, digest in artifact_hashes.items()
            },
            observation_schema_version=int(value["observation_schema_version"]),
            replay_ir_schema_version=int(value["replay_ir_schema_version"]),
            series_summary_schema_version=int(value["series_summary_schema_version"]),
            shards=tuple(ShardIndexEntry.from_dict(entry) for entry in value["shards"]),
            diagnostics={str(key): int(count) for key, count in diagnostics.items()},
            created_at=str(value["created_at"]),
        )


def load_shard_manifest(
    value: Mapping[str, Any], manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST
) -> ShardManifest:
    """Validate a shard manifest against the active runtime before any tensor load."""
    validate_artifact_runtime_contract(value, manifest_path)
    return ShardManifest.from_dict(value)


def validate_shard_tensors(tensors: Mapping[str, Any]) -> None:
    """Check a shard tensor payload against the frozen layout above.

    Shared by compilation and loading so a writer cannot emit a payload the
    reader would reject.
    """
    expected = {
        name: (shape, dtype)
        for name, shape, dtype in (*observation_field_specs(), *SHARD_TENSOR_SPECS)
    }
    if set(tensors) != set(expected):
        raise ValueError(
            f"Shard tensor fields mismatch; missing={sorted(set(expected) - set(tensors))}, "
            f"unknown={sorted(set(tensors) - set(expected))}"
        )
    for name, (shape, dtype) in expected.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise ValueError(f"Shard tensor {name} has an invalid type or dtype")
        if len(tensor.shape) != len(shape) or any(
            declared != -1 and actual != declared
            for actual, declared in zip(tensor.shape, shape, strict=True)
        ):
            raise ValueError(
                f"Shard tensor {name} has shape {tuple(tensor.shape)}, expected {shape}"
            )
    decisions = tensors["loss_mask"].shape[0]
    candidate_offsets = tensors["candidate_offsets"]
    if (
        candidate_offsets.shape != (decisions + 1,)
        or candidate_offsets[0].item() != 0
        or candidate_offsets[-1].item() != tensors["candidate_values"].shape[0]
        or torch.any(candidate_offsets[1:] < candidate_offsets[:-1])
    ):
        raise ValueError("Shard candidate_offsets must bound every candidate row")
    for name in ("game_offsets", "series_offsets"):
        offsets = tensors[name]
        if (
            offsets[0].item() != 0
            or offsets[-1].item() != decisions
            or torch.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError(f"Shard {name} must be nondecreasing and end at decisions")
    if decisions == 0:
        raise ValueError("Published replay shards must contain at least one decision")
    if not torch.isfinite(tensors["label_confidence"]).all():
        raise ValueError("Shard label_confidence contains non-finite values")
    if (
        not torch.isfinite(tensors["loss_mask"]).all()
        or not torch.isfinite(tensors["outcome"]).all()
    ):
        raise ValueError("Shard scalar targets contain non-finite values")
    for name, _, _ in observation_field_specs():
        tensor = tensors[name]
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Shard observation field {name} contains non-finite values")
    action_mask = tensors["action_mask"]
    if torch.any(~action_mask.any(dim=-1)):
        raise ValueError("Every action slot must contain at least one legal action")
    label_kind = tensors["label_kind"]
    counts = candidate_offsets[1:] - candidate_offsets[:-1]
    exact = label_kind == 1
    partial = label_kind == 2
    unknown = label_kind == 3
    if torch.any(~(exact | partial | unknown)):
        raise ValueError("Shard label_kind contains an unsupported value")
    if torch.any(exact & (counts != 1)):
        raise ValueError("EXACT labels must have exactly one candidate")
    if torch.any(partial & (counts < 2)):
        raise ValueError("PARTIAL labels must have at least two candidates")
    if torch.any(unknown & (counts != 0)):
        raise ValueError("UNKNOWN labels must not have candidates")
    if torch.any(unknown & (tensors["loss_mask"] != 0)):
        raise ValueError("UNKNOWN labels must have zero loss")
    if torch.any((exact | partial) & (tensors["loss_mask"] <= 0)):
        raise ValueError("Labeled decisions must have positive loss")
    candidates = tensors["candidate_values"]
    if torch.any((candidates < 0) | (candidates >= ACT_SIZE)):
        raise ValueError("Shard candidate action ids are outside the action contract")
    if candidates.numel():
        owners = torch.repeat_interleave(torch.arange(decisions), counts)
        legal = action_mask[owners, 0, candidates[:, 0]] & action_mask[owners, 1, candidates[:, 1]]
        same_switch = (
            (candidates[:, 0] >= 1)
            & (candidates[:, 0] <= 6)
            & (candidates[:, 0] == candidates[:, 1])
        )
        mega_first = (candidates[:, 0] >= 27) & (candidates[:, 0] <= 47)
        mega_second = (candidates[:, 1] >= 27) & (candidates[:, 1] <= 47)
        if torch.any(~legal | same_switch | (mega_first & mega_second)):
            raise ValueError("Shard contains an illegal labeled candidate")
