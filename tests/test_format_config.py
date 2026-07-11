import json

import pytest

from src.format_config import RuntimeManifest, current_manifest


def test_runtime_manifest_round_trip_and_compatibility(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"species": {"pikachu": 1}}', encoding="utf-8")
    manifest = current_manifest(vocab_path=vocab, model_config={"d_model": 32})

    restored = RuntimeManifest.from_dict(json.loads(manifest.to_json()))
    restored.validate_compatible(manifest)
    with pytest.raises(ValueError, match="model_config"):
        restored.validate_compatible(
            RuntimeManifest(vocab_sha256=manifest.vocab_sha256, model_config={"d_model": 64})
        )


def test_runtime_manifest_rejects_missing_fields(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text("{}", encoding="utf-8")
    manifest = current_manifest(vocab_path=vocab).to_dict()
    del manifest["event_schema_version"]

    with pytest.raises(ValueError, match="missing"):
        RuntimeManifest.from_dict(manifest)


def test_tokenizer_resolution_distinguishes_none_and_oov():
    from src.model.tokenizer import PokemonTokenizer, Resolution

    tokenizer = PokemonTokenizer({"species": {"pikachu": 1}})
    assert tokenizer.resolve("species", None) == (0, Resolution.KNOWN_NONE)
    assert tokenizer.resolve("species", "missingno") == (0, Resolution.OOV)
    assert tokenizer.resolve("species", "pikachu") == (1, Resolution.KNOWN)
