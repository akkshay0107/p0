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

SHARD_ARTIFACT_SCHEMA = "p0.replay_shard.v2"

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
    """Index of one compiled shard family tied to a single runtime contract."""

    runtime_contract_sha256: str
    shards: tuple[ShardIndexEntry, ...]
    diagnostics: Mapping[str, int]
    created_at: str
    artifact_schema: str = SHARD_ARTIFACT_SCHEMA
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    replay_ir_schema_version: int = REPLAY_IR_SCHEMA_VERSION
    series_summary_schema_version: int = SERIES_SUMMARY_SCHEMA_VERSION

    _FIELDS = frozenset(
        {
            "runtime_contract_sha256",
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
        return cls(
            artifact_schema=str(value["artifact_schema"]),
            runtime_contract_sha256=str(value["runtime_contract_sha256"]),
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
