"""Tests for UUID-based team corpus manifests."""

from dataclasses import replace

import pytest

from p0.teams.corpus import CorpusEntry, TeamCorpusManifest


def manifest() -> TeamCorpusManifest:
    return TeamCorpusManifest(
        format_id="gen9championsvgc2026regmc",
        corpus_id="corpus-1",
        entries=(CorpusEntry("team-1", "packed-pikachu", 17, "exact"),),
        created_at="2026-08-01T00:00:00Z",
        sampling_metadata={"test": True},
    )


class TestCorpusManifest:
    def test_manifest_roundtrip(self) -> None:
        value = manifest()
        assert TeamCorpusManifest.from_dict(value.to_dict()) == value
        assert set(value.entries[0].to_dict()) == {
            "canonical_id",
            "packed",
            "usage_count",
            "spread_provenance",
        }

    @pytest.mark.parametrize(
        "changes", [{"packed": ""}, {"usage_count": 0}, {"spread_provenance": "unknown"}]
    )
    def test_invalid_entry(self, changes: dict) -> None:
        with pytest.raises(ValueError):
            replace(manifest().entries[0], **changes)

    def test_duplicate_entry(self) -> None:
        value = manifest()
        with pytest.raises(ValueError, match="Duplicate corpus entry"):
            replace(value, entries=value.entries * 2)

    def test_unsupported_schema(self) -> None:
        with pytest.raises(ValueError, match="Unsupported corpus manifest schema"):
            replace(manifest(), artifact_schema="old.v1")

    def test_unknown_entry_fields(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            CorpusEntry.from_dict({**manifest().entries[0].to_dict(), "obsolete": "value"})
