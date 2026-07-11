import json
from pathlib import Path

from src.format_config import FORMAT, RuntimeManifest
from src.model.fused_token_encoder import _load_move_statics, _load_species_statics
from src.model.tokenizer import PokemonTokenizer


ROOT = Path(__file__).resolve().parents[1]


def test_active_contract_is_reg_m_b_and_manifest_matches_sources():
    manifest = RuntimeManifest.from_dict(json.loads((ROOT / "data/runtime_manifest.json").read_text()))
    assert FORMAT.battle_format == "gen9championsvgc2026regmb"
    assert FORMAT.bo3_format == "gen9championsvgc2026regmbbo3"
    assert manifest.format == FORMAT
    assert manifest.vocab_schema_version == 1


def test_every_dumped_content_key_resolves():
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    dex = json.loads((ROOT / "data/champions_dex.json").read_text())
    tokenizer = PokemonTokenizer(vocab)
    for table in ("species", "items", "abilities", "moves"):
        for entry in dex[table]:
            key = entry["id"]
            assert tokenizer.resolve(table, key)[1] == "known", (table, key)


def test_mechanics_tables_cover_the_vocab():
    vocab = json.loads((ROOT / "data/vocab.json").read_text())
    assert _load_move_statics().shape[0] == len(vocab["moves"]) + 1
    assert _load_species_statics().shape[0] == len(vocab["species"]) + 1
