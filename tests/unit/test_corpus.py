"""Unit tests for team corpus entries, content hashing, and manifest contracts."""

from __future__ import annotations

import hashlib

import pytest

from p0.format_config import current_manifest
from p0.teams.corpus import (
    CorpusEntry,
    CorpusSourceSpec,
    CorpusSplit,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)


def _sample_entry(
    packed: str = "packed-pikachu",
    split: CorpusSplit = CorpusSplit.TRAIN,
    usage: int = 5,
    canonical_hash: str | None = None,
) -> CorpusEntry:
    digest = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    return CorpusEntry(
        canonical_hash=canonical_hash or digest,
        packed=packed,
        packed_sha256=digest,
        split=split,
        usage_count=usage,
    )


class TestCorpusEntry:
    def test_valid_entry(self) -> None:
        entry = _sample_entry()
        assert entry.spread_provenance == "imputed"
        assert entry.split is CorpusSplit.TRAIN
        assert entry.usage_count == 5

    def test_packed_hash_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match the packed team"):
            CorpusEntry(
                canonical_hash="a" * 64,
                packed="my-team",
                packed_sha256="b" * 64,
                split=CorpusSplit.TRAIN,
                usage_count=1,
            )

    def test_empty_packed_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty packed team"):
            CorpusEntry(
                canonical_hash="a" * 64,
                packed="",
                packed_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                split=CorpusSplit.TRAIN,
                usage_count=1,
            )

    def test_unspecified_split_raises(self) -> None:
        digest = hashlib.sha256(b"team").hexdigest()
        with pytest.raises(ValueError, match="split must be assigned"):
            CorpusEntry(
                canonical_hash=digest,
                packed="team",
                packed_sha256=digest,
                split=CorpusSplit.UNSPECIFIED,
                usage_count=1,
            )

    def test_invalid_usage_count_raises(self) -> None:
        digest = hashlib.sha256(b"team").hexdigest()
        with pytest.raises(ValueError, match="positive integer"):
            CorpusEntry(
                canonical_hash=digest,
                packed="team",
                packed_sha256=digest,
                split=CorpusSplit.TRAIN,
                usage_count=0,
            )

    def test_invalid_provenance_raises(self) -> None:
        digest = hashlib.sha256(b"team").hexdigest()
        with pytest.raises(ValueError, match="spread_provenance"):
            CorpusEntry(
                canonical_hash=digest,
                packed="team",
                packed_sha256=digest,
                split=CorpusSplit.TRAIN,
                usage_count=1,
                spread_provenance="unknown",
            )

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        entry = _sample_entry()
        d = entry.to_dict()
        assert CorpusEntry.from_dict(d) == entry

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
        assert corpus_content_hash((e1, e2)) == corpus_content_hash((e2, e1))

    def test_manifest_roundtrip_and_validation(self) -> None:
        entries = (_sample_entry("team-1"), _sample_entry("team-2"))
        active_sha = current_manifest().global_sha256
        content_hash = corpus_content_hash(entries)
        manifest = TeamCorpusManifest(
            global_contract_sha256=active_sha,
            format_id="gen9championsvgc2026regmb",
            corpus_hash=content_hash,
            entries=entries,
            created_at="2026-08-01T00:00:00Z",
            sampling_metadata={"test": True},
        )
        assert TeamCorpusManifest.from_dict(manifest.to_dict()) == manifest
        assert load_corpus_manifest(manifest.to_dict()) == manifest

    def test_duplicate_entry_raises(self) -> None:
        entry = _sample_entry("team-1")
        active_sha = current_manifest().global_sha256
        with pytest.raises(ValueError, match="Duplicate corpus entry"):
            TeamCorpusManifest(
                global_contract_sha256=active_sha,
                format_id="gen9championsvgc2026regmb",
                corpus_hash=corpus_content_hash((entry, entry)),
                entries=(entry, entry),
                created_at="2026-08-01T00:00:00Z",
                sampling_metadata={},
            )

    def test_corpus_hash_mismatch_raises(self) -> None:
        entries = (_sample_entry("team-1"),)
        active_sha = current_manifest().global_sha256
        with pytest.raises(ValueError, match="does not match the entries"):
            TeamCorpusManifest(
                global_contract_sha256=active_sha,
                format_id="gen9championsvgc2026regmb",
                corpus_hash="0" * 64,
                entries=entries,
                created_at="2026-08-01T00:00:00Z",
                sampling_metadata={},
            )

    def test_unsupported_schema_raises(self) -> None:
        entries = (_sample_entry("team-1"),)
        active_sha = current_manifest().global_sha256
        with pytest.raises(ValueError, match="Unsupported corpus manifest schema"):
            TeamCorpusManifest(
                global_contract_sha256=active_sha,
                format_id="gen9championsvgc2026regmb",
                corpus_hash=corpus_content_hash(entries),
                entries=entries,
                created_at="2026-08-01T00:00:00Z",
                sampling_metadata={},
                artifact_schema="invalid_schema.v99",
            )


class TestCorpusSourceSpec:
    def test_valid_spec(self) -> None:
        spec = CorpusSourceSpec(
            corpus_path="/path/to/manifest.json",
            corpus_hash="a" * 64,
            format_id="gen9championsvgc2026regmb",
            split=CorpusSplit.TRAIN,
        )
        assert spec.split is CorpusSplit.TRAIN

    def test_unspecified_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be specified"):
            CorpusSourceSpec(
                corpus_path="/path/to/manifest.json",
                corpus_hash="a" * 64,
                format_id="gen9championsvgc2026regmb",
                split=CorpusSplit.UNSPECIFIED,
            )

    def test_invalid_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            CorpusSourceSpec(
                corpus_path="/path/to/manifest.json",
                corpus_hash="not-a-sha256",
                format_id="gen9championsvgc2026regmb",
                split=CorpusSplit.TRAIN,
            )
