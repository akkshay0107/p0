"""Unit tests for corpus building and audit reports."""

from __future__ import annotations

from pathlib import Path

from p0.format_config import FORMAT
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus_build import (
    audit_corpus,
    build_corpus,
    write_corpus_manifest,
)
from p0.teams.validation import validate_many
from tests.team_fixtures import sample_vocabulary, team_variant


class TestBuildCorpus:
    def test_build_corpus_admits_legal_variants(self) -> None:
        tokenizer = PokemonTokenizer(sample_vocabulary())
        v1 = team_variant("Pikachu", usage_count=5)
        v2 = team_variant("Raichu", usage_count=3)
        manifest, audit = build_corpus(
            (v1, v2),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="a" * 64,
            format_id=FORMAT.battle_format,
        )
        assert len(manifest.entries) == 2
        assert manifest.format_id == FORMAT.battle_format
        assert audit["admitted_count"] == 2
        assert audit["rejected_count"] == 0
        assert set(audit["species_coverage"]) >= {"pikachu", "raichu"}

    def test_build_corpus_rejects_oov_content(self) -> None:
        tokenizer = PokemonTokenizer(sample_vocabulary())
        v_valid = team_variant("Pikachu")
        v_oov = team_variant("Pikachu", item="Leftovers")
        manifest, audit = build_corpus(
            (v_valid, v_oov),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="a" * 64,
        )
        assert len(manifest.entries) == 1
        assert audit["admitted_count"] == 1
        assert audit["rejected_count"] == 1
        assert "oov_item: Leftovers" in audit["rejections_by_reason"]


class TestAuditCorpus:
    def test_audit_aggregates_coverage(self) -> None:
        tokenizer = PokemonTokenizer(sample_vocabulary())
        v1 = team_variant("Pikachu", usage_count=10)
        v2 = team_variant("Raichu", usage_count=5)
        manifest, _ = build_corpus(
            (v1, v2),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="b" * 64,
        )
        audit = audit_corpus(manifest)
        assert audit["admitted_count"] == 2
        assert "pikachu" in audit["species_coverage"]
        assert "raichu" in audit["species_coverage"]


class TestWriteCorpusManifest:
    def test_writes_to_output_dir(self, tmp_path: Path) -> None:
        tokenizer = PokemonTokenizer(sample_vocabulary())
        v = team_variant("Pikachu")
        manifest, _ = build_corpus(
            (v,),
            tokenizer=tokenizer,
            validator=validate_many,
            global_contract_sha256="c" * 64,
        )
        out_dir = tmp_path / "pool"
        path = write_corpus_manifest(manifest, out_dir)
        assert path == out_dir / "corpus_manifest.json"
        assert path.is_file()
