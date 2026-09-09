"""Unit tests for corpus building, split assignment, and audit reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from p0.format_config import FORMAT
from p0.model.tokenizer import PokemonTokenizer
from p0.teams.corpus import CorpusSplit
from p0.teams.corpus_build import (
    _component_splits,
    _validate_ratios,
    audit_corpus,
    build_corpus,
    write_corpus_manifest,
)
from p0.teams.validation import validate_many
from tests.team_fixtures import sample_vocabulary, team_variant


class TestSplitRatios:
    def test_valid_ratios(self) -> None:
        _validate_ratios(0.8, 0.1, 0.1)
        _validate_ratios(1.0, 0.0, 0.0)

    def test_negative_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            _validate_ratios(-0.1, 0.6, 0.5)

    def test_sum_not_one_raises(self) -> None:
        with pytest.raises(ValueError, match="must sum to one"):
            _validate_ratios(0.8, 0.1, 0.2)


class TestComponentSplits:
    def test_connected_series_assigned_same_split(self) -> None:
        # Team 1 shares series-A with Team 2; Team 2 shares series-B with Team 3
        # All three must share the same split
        v1 = team_variant("Pikachu", source_series=("series-A",))
        v2 = team_variant("Raichu", source_series=("series-A", "series-B"))
        v3 = team_variant("Charizard", source_series=("series-B",))
        v_unrelated = team_variant("Garchomp", source_series=("series-Z",))

        splits = _component_splits(
            (v1, v2, v3, v_unrelated),
            ratio_train=0.5,
            ratio_val=0.5,
            ratio_test=0.0,
        )
        assert len(splits) == 4
        assert splits[0] == splits[1] == splits[2]

    def test_empty_series_falls_back_to_team_hash(self) -> None:
        v1 = team_variant("Pikachu", source_series=())
        splits = _component_splits(
            (v1,),
            ratio_train=0.8,
            ratio_val=0.1,
            ratio_test=0.1,
        )
        assert len(splits) == 1
        assert splits[0] in {CorpusSplit.TRAIN, CorpusSplit.VALIDATION, CorpusSplit.TEST}


class TestBuildCorpus:
    def test_build_corpus_admits_legal_variants(self) -> None:
        tokenizer = PokemonTokenizer(sample_vocabulary())
        v1 = team_variant("Pikachu", usage_count=5)
        v2 = team_variant("Raichu", source_series=("series-2",), usage_count=3)
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
        v2 = team_variant("Raichu", source_series=("s2",), usage_count=5)
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
        assert sum(audit["split_counts"].values()) == 2


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
