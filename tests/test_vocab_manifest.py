import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from p0.format_config import ACTION_CONTRACT, FORMAT, RuntimeManifest
from p0.model.build_vocab import build
from p0.model.fused_token_encoder import (
    FusedTokenEncoder,
    _load_mechanic_tag_tables,
    _load_move_statics,
    _load_species_statics,
)
from p0.model.resources import RuntimeResources, default_runtime_resources
from p0.model.tokenizer import PokemonTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_active_contract_is_reg_m_b_and_manifest_matches_sources():
    manifest = RuntimeManifest.from_dict(
        json.loads((ROOT / "data/runtime_manifest.json").read_text())
    )
    assert FORMAT.battle_format == "gen9championsvgc2026regmb"
    assert FORMAT.bo3_format == "gen9championsvgc2026regmbbo3"
    assert manifest.battle_format == FORMAT.battle_format
    assert manifest.bo3_format == FORMAT.bo3_format
    assert manifest.action == ACTION_CONTRACT
    assert len(manifest.runtime_contract_sha256) == 64


def test_runtime_resources_allow_updated_dex_content(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("runtime_manifest.json", "vocab.json", "champions_dex.json"):
        shutil.copy2(ROOT / "data" / name, data / name)
    dex_path = data / "champions_dex.json"
    dex = json.loads(dex_path.read_text())
    dex["moves"][0]["basePower"] = int(dex["moves"][0].get("basePower", 0)) + 1
    dex_path.write_text(json.dumps(dex), encoding="utf-8")

    resources = RuntimeResources.from_manifest(data / "runtime_manifest.json")

    assert resources.dex["moves"][0]["basePower"] == dex["moves"][0]["basePower"]


def test_every_legal_content_key_resolves():
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    tokenizer = PokemonTokenizer(vocab)
    for table in ("species", "items", "abilities", "moves"):
        for key in dex["legality"][table]:
            assert tokenizer.resolve(table, key)[1] == "known", (table, key)


def test_mechanics_tables_cover_the_vocab():
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    resources = default_runtime_resources()
    assert _load_move_statics(resources).shape[0] == len(vocab["moves"]) + 1
    assert _load_species_statics(resources).shape[0] == len(vocab["species"]) + 1
    mechanic_tags = _load_mechanic_tag_tables(resources)
    assert mechanic_tags["items"].shape[0] == len(vocab["items"]) + 1
    assert mechanic_tags["abilities"].shape[0] == len(vocab["abilities"]) + 1


def test_item_and_ability_mechanics_are_wired_into_encoder():
    encoder = FusedTokenEncoder(
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        resources=default_runtime_resources(),
    )
    assert encoder.item_mechanic_proj.in_features == encoder._item_mechanic_tags.shape[1]
    assert encoder.ability_mechanic_proj.in_features == encoder._ability_mechanic_tags.shape[1]
    assert encoder._item_mechanic_tags.count_nonzero() > 0
    assert encoder._ability_mechanic_tags.count_nonzero() > 0


def test_field_namespace_and_coverage_audit_are_present(tmp_path):
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    report = build(
        ROOT / "data/champions_dex.json",
        tmp_path / "vocab.json",
        tmp_path / "manifest.json",
    )
    assert "trickroom" in vocab["fields"]
    assert report["missingLegalContent"] == {}
    assert report["unmappedLegalEffects"] == []


def test_reg_mb_legality_inventory_uses_resolved_showdown_rules():
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    assert "pikachu" in dex["legality"]["species"]
    assert "protect" in dex["legality"]["moves"]
    assert "ababo" not in dex["legality"]["species"]
    assert "berserkgene" not in dex["legality"]["items"]


def test_representative_dump_matches_pinned_showdown_runtime():
    script = r"""
const path = require('node:path');
const {Dex} = require(path.resolve('pokemon-showdown/dist/sim/dex'));
const dex = Dex.mod('champions');
const move = dex.moves.get('protect');
const species = dex.species.get('charizardmegax');
const item = dex.items.get('lifeorb');
const ability = dex.abilities.get('intimidate');
console.log(JSON.stringify({
  move: {basePower: move.basePower, pp: move.pp, target: move.target},
  species: {baseSpecies: species.baseSpecies, requiredItem: species.requiredItem, isMega: species.isMega},
  itemTags: Object.keys(item).filter(key => key.startsWith('on')).sort(),
  abilityTags: Object.keys(ability).filter(key => key.startsWith('on')).sort(),
}));
"""
    oracle = json.loads(subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True))
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    tables = {
        name: {entry["id"]: entry for entry in dex[name]}
        for name in ("moves", "species", "items", "abilities")
    }
    assert {key: tables["moves"]["protect"][key] for key in oracle["move"]} == oracle["move"]
    assert {key: tables["species"]["charizardmegax"][key] for key in oracle["species"]} == oracle[
        "species"
    ]
    assert tables["items"]["lifeorb"]["mechanicTags"] == oracle["itemTags"]
    assert tables["abilities"]["intimidate"]["mechanicTags"] == oracle["abilityTags"]


def test_generation_is_deterministic_and_nonlegal_effects_are_reported(tmp_path):
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    dex["protocolEffects"].append("nonlegaltesteffect")
    dex_path = tmp_path / "dex.json"
    dex_path.write_text(json.dumps(dex), encoding="utf-8")
    outputs = []
    for suffix in ("a", "b"):
        vocab = tmp_path / f"vocab-{suffix}.json"
        manifest = tmp_path / f"manifest-{suffix}.json"
        coverage = tmp_path / f"coverage-{suffix}.json"
        build(dex_path, vocab, manifest, coverage)
        outputs.append((vocab.read_bytes(), manifest.read_bytes(), coverage.read_bytes()))
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0][2])
    assert "condition:nonlegaltesteffect" in report["unsupportedNonlegalEffects"]


def test_unknown_legal_effect_namespace_fails_generation(tmp_path):
    dex = deepcopy(json.loads((ROOT / "data/champions_dex.json").read_text()))
    dex["legalProtocolEffects"]["unmapped_family"] = ["reachableeffect"]
    dex_path = tmp_path / "dex.json"
    dex_path.write_text(json.dumps(dex), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown legal protocol-effect namespace"):
        build(
            dex_path,
            tmp_path / "vocab.json",
            tmp_path / "manifest.json",
            tmp_path / "coverage.json",
        )
