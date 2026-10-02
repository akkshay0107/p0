"""Integration tests for Showdown-backed corpus building and audit reports."""

from __future__ import annotations

import pytest

from p0.format_config import FORMAT
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus_build import build_corpus
from p0.teams.validation import validate_many
from tests.team_fixtures import sample_vocabulary, team_variant


@pytest.mark.heavy
@pytest.mark.integration
class TestBuildCorpus:
    def test_build_corpus_admits_legal_variants(self, showdown_assets: None) -> None:
        del showdown_assets
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

    def test_build_corpus_rejects_oov_content(self, showdown_assets: None) -> None:
        del showdown_assets
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
