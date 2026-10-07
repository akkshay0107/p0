"""Unit tests for corpus building and audit reports."""

from __future__ import annotations

import hashlib
from pathlib import Path

import orjson

from p0.format_config import FORMAT, active_global_contract
from p0.teams.corpus import (
    CorpusEntry,
    TeamCorpusManifest,
    corpus_content_hash,
    load_corpus_manifest,
)
from p0.teams.corpus_build import (
    write_corpus_manifest,
)


class TestWriteCorpusManifest:
    def test_writes_to_output_dir(self, tmp_path: Path) -> None:
        packed = "Pikachu|Static|Light Ball|Timid|Thunderbolt,Protect"
        entry = CorpusEntry(
            canonical_hash="a" * 64,
            packed=packed,
            packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),
            usage_count=7,
            spread_provenance="exact",
        )
        manifest = TeamCorpusManifest(
            global_contract_sha256=active_global_contract().global_sha256,
            format_id=FORMAT.battle_format,
            corpus_hash=corpus_content_hash((entry,)),
            entries=(entry,),
            created_at="2026-08-01T00:00:00Z",
            sampling_metadata={"source": "unit"},
        )
        out_dir = tmp_path / "pool"
        path = write_corpus_manifest(manifest, out_dir)

        assert path == out_dir / "corpus_manifest.json"
        assert path.is_file()
        loaded = load_corpus_manifest(orjson.loads(path.read_bytes()))
        assert loaded == manifest
