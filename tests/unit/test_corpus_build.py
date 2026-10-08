"""Unit tests for corpus building and audit reports."""

from __future__ import annotations

from pathlib import Path

import orjson

from p0.format_config import FORMAT
from p0.teams.corpus import (
    CorpusEntry,
    TeamCorpusManifest,
)
from p0.teams.corpus_build import (
    write_corpus_manifest,
)


class TestWriteCorpusManifest:
    def test_writes_to_output_dir(self, tmp_path: Path) -> None:
        packed = "Pikachu|Static|Light Ball|Timid|Thunderbolt,Protect"
        entry = CorpusEntry(
            canonical_id="a" * 64,
            packed=packed,
            usage_count=7,
            spread_provenance="exact",
        )
        manifest = TeamCorpusManifest(
            format_id=FORMAT.battle_format,
            corpus_id="corpus-test",
            entries=(entry,),
            created_at="2026-08-01T00:00:00Z",
            sampling_metadata={"source": "unit"},
        )
        out_dir = tmp_path / "pool"
        path = write_corpus_manifest(manifest, out_dir)

        assert path == out_dir / "corpus_manifest.json"
        assert path.is_file()
        loaded = TeamCorpusManifest.from_dict(orjson.loads(path.read_bytes()))
        assert loaded == manifest
