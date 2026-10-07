"""Unit tests for team corpus entries, content hashing, and manifest contracts."""

from __future__ import annotations

import hashlib

import pytest

from p0.format_config import active_global_contract
from p0.teams.corpus import (
    CorpusEntry,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)


def _sample_entry(
    packed: str = "packed-pikachu",
    usage: int = 5,
    canonical_hash: str | None = None,
) -> CorpusEntry:
    digest = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    return CorpusEntry(
        canonical_hash=canonical_hash or digest,
        packed=packed,
        packed_sha256=digest,
        usage_count=usage,
    )


class TestCorpusEntry:
    def test_entry_serialization_preserves_exact_and_default_provenance(self) -> None:
        packed_hash = "dbd1bf2dccae769e98729649a1a1c4c9f1a29cc5135e12ea364b98e9295ff5ec"
        entry = CorpusEntry(
            canonical_hash="a" * 64,
            packed="packed-pikachu",
            packed_sha256=packed_hash,
            usage_count=17,
            spread_provenance="exact",
        )
        expected = {
            "canonical_hash": "a" * 64,
            "packed": "packed-pikachu",
            "packed_sha256": packed_hash,
            "usage_count": 17,
            "spread_provenance": "exact",
        }

        assert entry.to_dict() == expected
        assert CorpusEntry.from_dict(expected) == entry

        default_entry = _sample_entry()
        assert default_entry.spread_provenance == "imputed"
        assert default_entry.to_dict()["spread_provenance"] == "imputed"

    def test_packed_hash_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match the packed team"):
            CorpusEntry(
                canonical_hash="a" * 64,
                packed="my-team",
                packed_sha256="b" * 64,
                usage_count=1,
            )

    def test_empty_packed_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty packed team"):
            CorpusEntry(
                canonical_hash="a" * 64,
                packed="",
                packed_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                usage_count=1,
            )

    def test_invalid_usage_count_raises(self) -> None:
        digest = hashlib.sha256(b"team").hexdigest()
        with pytest.raises(ValueError, match="positive integer"):
            CorpusEntry(
                canonical_hash=digest,
                packed="team",
                packed_sha256=digest,
                usage_count=0,
            )

    def test_invalid_provenance_raises(self) -> None:
        digest = hashlib.sha256(b"team").hexdigest()
        with pytest.raises(ValueError, match="spread_provenance"):
            CorpusEntry(
                canonical_hash=digest,
                packed="team",
                packed_sha256=digest,
                usage_count=1,
                spread_provenance="unknown",
            )

    def test_from_dict_rejects_unknown_fields(self) -> None:
        entry = _sample_entry()
        d = entry.to_dict()
        d["unexpected"] = 123
        with pytest.raises(ValueError, match="unknown"):
            CorpusEntry.from_dict(d)


class TestCorpusManifest:
    def test_content_hash_invariance_to_order(self) -> None:
        e1 = _sample_entry("team-alpha")
        e2 = _sample_entry("team-beta")
        changed_entry = _sample_entry("team-gamma")
        assert corpus_content_hash((e1, e2)) == corpus_content_hash((e2, e1))
        assert corpus_content_hash((e1, e2)) != corpus_content_hash((e1, changed_entry))

    def test_manifest_roundtrip_and_validation(self) -> None:
        entry = CorpusEntry(
            canonical_hash="a" * 64,
            packed="packed-pikachu",
            packed_sha256="dbd1bf2dccae769e98729649a1a1c4c9f1a29cc5135e12ea364b98e9295ff5ec",
            usage_count=17,
            spread_provenance="exact",
        )
        entries = (entry,)
        active_sha = active_global_contract().global_sha256
        serialized = {
            "artifact_schema": "p0.team_corpus.v1",
            "global_contract_sha256": active_sha,
            "format_id": "gen9championsvgc2026regmc",
            "corpus_hash": "fcab590ee1e21a55318c3b07fccf5722186e293bb6f261db1941d1919b8c6f05",
            "entries": [
                {
                    "canonical_hash": "a" * 64,
                    "packed": "packed-pikachu",
                    "packed_sha256": "dbd1bf2dccae769e98729649a1a1c4c9f1a29cc5135e12ea364b98e9295ff5ec",
                    "usage_count": 17,
                    "spread_provenance": "exact",
                }
            ],
            "created_at": "2026-08-01T00:00:00Z",
            "sampling_metadata": {"test": True},
        }
        manifest = TeamCorpusManifest(
            global_contract_sha256=active_sha,
            format_id="gen9championsvgc2026regmc",
            corpus_hash="fcab590ee1e21a55318c3b07fccf5722186e293bb6f261db1941d1919b8c6f05",
            entries=entries,
            created_at="2026-08-01T00:00:00Z",
            sampling_metadata={"test": True},
        )
        assert manifest.to_dict() == serialized
        assert TeamCorpusManifest.from_dict(serialized) == manifest
        assert load_corpus_manifest(serialized) == manifest

        incompatible = {**serialized, "global_contract_sha256": "b" * 64}
        with pytest.raises(ValueError, match="incompatible with the active runtime"):
            load_corpus_manifest(incompatible)

    def test_duplicate_entry_raises(self) -> None:
        entry = _sample_entry("team-1")
        active_sha = active_global_contract().global_sha256
        with pytest.raises(ValueError, match="Duplicate corpus entry"):
            TeamCorpusManifest(
                global_contract_sha256=active_sha,
                format_id="gen9championsvgc2026regmc",
                corpus_hash=corpus_content_hash((entry, entry)),
                entries=(entry, entry),
                created_at="2026-08-01T00:00:00Z",
                sampling_metadata={},
            )

    def test_corpus_hash_mismatch_raises(self) -> None:
        entries = (_sample_entry("team-1"),)
        active_sha = active_global_contract().global_sha256
        with pytest.raises(ValueError, match="does not match the entries"):
            TeamCorpusManifest(
                global_contract_sha256=active_sha,
                format_id="gen9championsvgc2026regmc",
                corpus_hash="0" * 64,
                entries=entries,
                created_at="2026-08-01T00:00:00Z",
                sampling_metadata={},
            )

    def test_unsupported_schema_raises(self) -> None:
        entries = (_sample_entry("team-1"),)
        active_sha = active_global_contract().global_sha256
        with pytest.raises(ValueError, match="Unsupported corpus manifest schema"):
            TeamCorpusManifest(
                global_contract_sha256=active_sha,
                format_id="gen9championsvgc2026regmc",
                corpus_hash=corpus_content_hash(entries),
                entries=entries,
                created_at="2026-08-01T00:00:00Z",
                sampling_metadata={},
                artifact_schema="invalid_schema.v99",
            )
