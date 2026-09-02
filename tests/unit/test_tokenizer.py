"""Tests for tokenizer and manifest contracts."""

from __future__ import annotations

import json
from pathlib import Path

from poke_env.battle.status import Status

from p0.format_config import (
    RuntimeManifest,
    active_global_contract,
    current_manifest,
    load_runtime_manifest,
    sha256_file,
)
from p0.model.tokenizer import PokemonTokenizer, Resolution
from p0.paths import DEFAULT_PATHS


def write_config(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def _resources(
    tmp_path: Path, *, extra_species: bool = False, base_power: int = 90
) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    species = {"pikachu": 1}
    if extra_species:
        species["raichu"] = 2
    vocab.write_text(json.dumps({"species": species}), encoding="utf-8")
    dex = tmp_path / "champions_dex.json"
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": base_power}]}))
    return vocab, dex


ROOT = Path(__file__).resolve().parents[2]


def _runtime_files(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    dex = tmp_path / "champions_dex.json"
    vocab.write_text(
        json.dumps({"species": {"pikachu": 1}, "moves": {"tackle": 1}}), encoding="utf-8"
    )
    dex.write_text('{"pikachu":{"base_stats":{"hp":35}}}', encoding="utf-8")
    return vocab, dex


class TestTokenizerContracts:
    def test_runtime_manifest_digest_is_semantic_and_round_trips(self, tmp_path: Path) -> None:
        """Verify runtime manifest digest computation is invariant to key reordering in on-disk JSON."""
        vocab, dex = _runtime_files(tmp_path)
        manifest = current_manifest(vocab_path=vocab, dex_path=dex)
        reordered = json.loads(json.dumps(manifest.to_dict()))
        reordered["contracts"]["actions"]["major"] = {
            key: reordered["contracts"]["actions"]["major"][key]
            for key in reversed(tuple(reordered["contracts"]["actions"]["major"]))
        }
        assert RuntimeManifest.from_dict(reordered) == manifest
        path = tmp_path / "runtime_manifest.json"
        path.write_text(json.dumps(reordered), encoding="utf-8")
        assert load_runtime_manifest(path) == manifest

    def test_tokenizer_aliases_and_resolution_keep_unknown_zero_distinct_from_known_none(
        self,
    ) -> None:
        """Verify tokenizer distinguishes between KNOWN, KNOWN_NONE (valid empty entity), OOV, and UNKNOWN."""
        tokenizer_instance = PokemonTokenizer(
            {
                "weathers": {"raindance": 4},
                "status": {"brn": 5},
                "moves": {"uturn": 7},
            }
        )
        assert tokenizer_instance.id_for("moves", "U-turn") == 7
        assert tokenizer_instance.effect_id_for("status", "status: brn") == 5
        assert tokenizer_instance.resolve("weathers", "rain") == (4, Resolution.KNOWN)
        assert tokenizer_instance.resolve("status", "burn") == (5, Resolution.KNOWN)
        assert tokenizer_instance.resolve("status", "not-a-status") == (0, Resolution.OOV)
        assert tokenizer_instance.resolve("status", None) == (0, Resolution.KNOWN_NONE)
        assert tokenizer_instance.resolve("missing", "rain") == (0, Resolution.UNKNOWN)

    def test_enum_like_tables_lazy_cache_alias_and_missing_member_results(self) -> None:
        """Verify lazy alias dictionary caching on enum-like tables (weathers, status)."""
        tokenizer_instance = PokemonTokenizer({"weathers": {"raindance": 4}, "status": {"brn": 5}})
        assert tokenizer_instance.weathers["R-a-i-n"] == 4
        assert tokenizer_instance.weathers["rain"] == 4
        assert tokenizer_instance.weathers == {"rain": 4}
        assert tokenizer_instance.weathers["unknown-weather"] == 0
        assert "unknownweather" not in tokenizer_instance.weathers
        assert tokenizer_instance.status["burn"] == 5
        assert tokenizer_instance.status[Status.BRN] == 5
        assert tokenizer_instance.status["unknown-status"] == 0
        assert tokenizer_instance.status == {"burn": 5, "brn": 5}
        assert all(isinstance(key, str) for key in tokenizer_instance.status)

    def test_active_contract_rejects_an_unrecorded_spread_table(self) -> None:
        """Verify active global contract checks spread_usage.json checksum."""
        contract = active_global_contract()
        assert contract.spread_usage_sha256 == sha256_file(
            DEFAULT_PATHS.data_root / "spread_usage.json"
        )
