"""Validated team-corpus manifest contract.

Team identity: corpus identity is CanonicalTeam.team_hash, the order- and
spelling-independent SHA-256 of the canonical team JSON. ValidatedTeam's
team_hash (SHA-256 of the packed Showdown string) is a derived runtime
instance identity produced at pack time; each corpus entry stores both so
corpus records map onto runtime sampling without re-parsing.

This module defines only the manifest schema and its validation. Corpus
construction, split assignment, and the corpus-backed TeamSource
implementation live in the corpus workstream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

from p0.format_config import (
    DEFAULT_RUNTIME_MANIFEST,
    canonical_json_sha256,
    validate_artifact_runtime_contract,
)

CORPUS_MANIFEST_SCHEMA = "p0.team_corpus.v1"


class CorpusSplit(IntEnum):
    UNSPECIFIED = 0
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    HELD_OUT_ARCHETYPE = 4


class SamplingPolicy(IntEnum):
    UNSPECIFIED = 0
    USAGE_WEIGHTED = 1
    UNIFORM_CANONICAL = 2
    UNIFORM_ARCHETYPE = 3
    RARE_COVERAGE = 4
    MATCHUP_BALANCED = 5


def _require_fields(value: Mapping[str, Any], expected: frozenset[str], owner: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(f"Invalid {owner} fields; missing={missing}, unknown={unknown}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One admitted team: canonical identity plus its packed runtime form."""

    canonical_hash: str
    packed: str
    packed_sha256: str
    split: CorpusSplit
    usage_count: int
    archetype_tags: tuple[str, ...] = ()
    spread_provenance: str = "imputed"

    _FIELDS = frozenset(
        {
            "canonical_hash",
            "packed",
            "packed_sha256",
            "split",
            "usage_count",
            "archetype_tags",
            "spread_provenance",
        }
    )

    def __post_init__(self) -> None:
        if not _is_sha256(self.canonical_hash):
            raise ValueError("CorpusEntry.canonical_hash must be a lowercase SHA-256 digest")
        if not self.packed:
            raise ValueError("CorpusEntry.packed must be a non-empty packed team")
        actual = hashlib.sha256(self.packed.encode("utf-8")).hexdigest()
        if self.packed_sha256 != actual:
            raise ValueError(
                "CorpusEntry.packed_sha256 does not match the packed team: "
                f"declared={self.packed_sha256}, actual={actual}"
            )
        if self.split is CorpusSplit.UNSPECIFIED:
            raise ValueError("CorpusEntry.split must be assigned before admission")
        if type(self.usage_count) is not int or self.usage_count < 1:
            raise ValueError("CorpusEntry.usage_count must be a positive integer")
        if self.spread_provenance not in ("imputed", "exact"):
            raise ValueError("CorpusEntry.spread_provenance must be 'imputed' or 'exact'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_hash": self.canonical_hash,
            "packed": self.packed,
            "packed_sha256": self.packed_sha256,
            "split": int(self.split),
            "usage_count": self.usage_count,
            "archetype_tags": list(self.archetype_tags),
            "spread_provenance": self.spread_provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorpusEntry:
        _require_fields(value, cls._FIELDS, "CorpusEntry")
        return cls(
            canonical_hash=str(value["canonical_hash"]),
            packed=str(value["packed"]),
            packed_sha256=str(value["packed_sha256"]),
            split=CorpusSplit(value["split"]),
            usage_count=int(value["usage_count"]),
            archetype_tags=tuple(str(tag) for tag in value["archetype_tags"]),
            spread_provenance=str(value["spread_provenance"]),
        )


def corpus_content_hash(entries: tuple[CorpusEntry, ...]) -> str:
    """Identity of the corpus content, deterministic under entry reordering."""
    ordered = sorted(entries, key=lambda entry: (entry.canonical_hash, entry.packed_sha256))
    return canonical_json_sha256([entry.to_dict() for entry in ordered])


@dataclass(frozen=True, slots=True)
class TeamCorpusManifest:
    """The single loadable description of a validated team corpus."""

    runtime_contract_sha256: str
    format_id: str
    corpus_hash: str
    entries: tuple[CorpusEntry, ...]
    created_at: str
    sampling_metadata: Mapping[str, Any]
    artifact_schema: str = CORPUS_MANIFEST_SCHEMA

    _FIELDS = frozenset(
        {
            "runtime_contract_sha256",
            "format_id",
            "corpus_hash",
            "entries",
            "created_at",
            "sampling_metadata",
            "artifact_schema",
        }
    )

    def __post_init__(self) -> None:
        if self.artifact_schema != CORPUS_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported corpus manifest schema {self.artifact_schema!r}; "
                f"expected {CORPUS_MANIFEST_SCHEMA}"
            )
        if not _is_sha256(self.runtime_contract_sha256):
            raise ValueError(
                "TeamCorpusManifest.runtime_contract_sha256 must be a lowercase SHA-256 digest"
            )
        if not self.format_id:
            raise ValueError("TeamCorpusManifest.format_id must be non-empty")
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            key = (entry.canonical_hash, entry.packed_sha256)
            if key in seen:
                raise ValueError(f"Duplicate corpus entry {entry.canonical_hash}")
            seen.add(key)
        actual = corpus_content_hash(self.entries)
        if self.corpus_hash != actual:
            raise ValueError(
                "TeamCorpusManifest.corpus_hash does not match the entries: "
                f"declared={self.corpus_hash}, actual={actual}"
            )
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("TeamCorpusManifest.created_at must be ISO-8601") from exc
        for key in self.sampling_metadata:
            if not isinstance(key, str):
                raise ValueError("Sampling metadata keys must be strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_schema": self.artifact_schema,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "format_id": self.format_id,
            "corpus_hash": self.corpus_hash,
            "entries": [entry.to_dict() for entry in self.entries],
            "created_at": self.created_at,
            "sampling_metadata": dict(self.sampling_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TeamCorpusManifest:
        _require_fields(value, cls._FIELDS, "TeamCorpusManifest")
        metadata = value["sampling_metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("TeamCorpusManifest.sampling_metadata must be a JSON object")
        return cls(
            artifact_schema=str(value["artifact_schema"]),
            runtime_contract_sha256=str(value["runtime_contract_sha256"]),
            format_id=str(value["format_id"]),
            corpus_hash=str(value["corpus_hash"]),
            entries=tuple(CorpusEntry.from_dict(entry) for entry in value["entries"]),
            created_at=str(value["created_at"]),
            sampling_metadata=dict(metadata),
        )


def load_corpus_manifest(
    value: Mapping[str, Any], manifest_path: str | Path = DEFAULT_RUNTIME_MANIFEST
) -> TeamCorpusManifest:
    """Validate a corpus manifest against the active runtime before use."""
    validate_artifact_runtime_contract(value, manifest_path)
    return TeamCorpusManifest.from_dict(value)


@dataclass(frozen=True, slots=True)
class CorpusSourceSpec:
    """Pinned constructor input for the corpus-backed TeamSource.

    The corpus workstream implements CorpusTeamSource(spec) satisfying the
    existing TeamSource protocol in teams/corpus_source.py; this spec is the
    agreed seam so environment configuration and the implementation can land
    independently.
    """

    corpus_path: str
    corpus_hash: str
    format_id: str
    split: CorpusSplit
    seed: int
    sampling_policy: SamplingPolicy
    allow_mirror: bool = True
    curriculum_stage: str = ""

    def __post_init__(self) -> None:
        if not self.corpus_path or not self.format_id:
            raise ValueError("CorpusSourceSpec requires corpus_path and format_id")
        if not _is_sha256(self.corpus_hash):
            raise ValueError("CorpusSourceSpec.corpus_hash must be a lowercase SHA-256 digest")
        if self.split is CorpusSplit.UNSPECIFIED:
            raise ValueError("CorpusSourceSpec.split must be specified")
        if self.sampling_policy is SamplingPolicy.UNSPECIFIED:
            raise ValueError("CorpusSourceSpec.sampling_policy must be specified")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("CorpusSourceSpec.seed must be a nonnegative integer")
