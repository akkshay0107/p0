"""Validated team variants grouped by canonical team UUID."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

from p0.contracts import require_dataclass_fields

CORPUS_MANIFEST_SCHEMA = "p0.team_corpus.v2"


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One admitted team: canonical identity plus its packed runtime form."""

    canonical_id: str
    packed: str
    usage_count: int
    spread_provenance: str = "imputed"

    def __post_init__(self) -> None:
        if not self.canonical_id or not self.packed:
            raise ValueError("Corpus entries require a canonical team ID and packed team")

        if type(self.usage_count) is not int or self.usage_count < 1:
            raise ValueError("CorpusEntry.usage_count must be a positive integer")

        if self.spread_provenance not in ("imputed", "exact"):
            raise ValueError("CorpusEntry.spread_provenance must be 'imputed' or 'exact'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorpusEntry:
        require_dataclass_fields(value, cls)
        return cls(
            canonical_id=str(value["canonical_id"]),
            packed=str(value["packed"]),
            usage_count=int(value["usage_count"]),
            spread_provenance=str(value["spread_provenance"]),
        )


@dataclass(frozen=True, slots=True)
class TeamCorpusManifest:
    """The single loadable description of a validated team corpus."""

    format_id: str
    corpus_id: str
    entries: tuple[CorpusEntry, ...]
    created_at: str
    sampling_metadata: Mapping[str, Any]
    artifact_schema: str = CORPUS_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.artifact_schema != CORPUS_MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported corpus manifest schema {self.artifact_schema!r}; "
                f"expected {CORPUS_MANIFEST_SCHEMA}"
            )

        if not self.format_id:
            raise ValueError("TeamCorpusManifest.format_id must be non-empty")

        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            key = (entry.canonical_id, entry.packed)
            if key in seen:
                raise ValueError(f"Duplicate corpus entry {entry.canonical_id}")
            seen.add(key)

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
            "format_id": self.format_id,
            "corpus_id": self.corpus_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "created_at": self.created_at,
            "sampling_metadata": dict(self.sampling_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TeamCorpusManifest:
        require_dataclass_fields(value, cls)
        metadata = value["sampling_metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("TeamCorpusManifest.sampling_metadata must be a JSON object")
        return cls(
            artifact_schema=str(value["artifact_schema"]),
            format_id=str(value["format_id"]),
            corpus_id=str(value["corpus_id"]),
            entries=tuple(CorpusEntry.from_dict(entry) for entry in value["entries"]),
            created_at=str(value["created_at"]),
            sampling_metadata=dict(metadata),
        )
