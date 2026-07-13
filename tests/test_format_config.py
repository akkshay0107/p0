import json

import pytest

from p0.format_config import FORMAT, RuntimeManifest, current_manifest


def test_runtime_manifest_round_trip_and_compatibility(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"species": {"pikachu": 1}}', encoding="utf-8")
    manifest = current_manifest(vocab_path=vocab)

    restored = RuntimeManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    assert restored == manifest
    assert "action_schema" not in restored.to_dict()["format"]
    assert list(key for key in restored.to_dict() if key == "action_schema_version") == [
        "action_schema_version"
    ]


def test_runtime_manifest_rejects_missing_fields(tmp_path):
    vocab = tmp_path / "vocab.json"
    vocab.write_text("{}", encoding="utf-8")
    manifest = current_manifest(vocab_path=vocab).to_dict()
    del manifest["event_schema_version"]

    with pytest.raises(ValueError, match="missing"):
        RuntimeManifest.from_dict(manifest)


@pytest.mark.parametrize("field", ["action_schema", "model_config"])
def test_runtime_manifest_rejects_removed_fields(field):
    manifest = RuntimeManifest().to_dict()
    if field == "action_schema":
        manifest["format"][field] = "old"
    else:
        manifest[field] = {}
    with pytest.raises(ValueError, match="unknown"):
        RuntimeManifest.from_dict(manifest)


def test_format_spec_rejects_missing_and_unknown_fields():
    manifest = RuntimeManifest().to_dict()
    del manifest["format"]["action_size"]
    with pytest.raises(ValueError, match="missing"):
        RuntimeManifest.from_dict(manifest)
    manifest = RuntimeManifest().to_dict()
    manifest["format"]["extra"] = FORMAT.action_size
    with pytest.raises(ValueError, match="unknown"):
        RuntimeManifest.from_dict(manifest)


def test_tokenizer_resolution_distinguishes_none_and_oov():
    from p0.model.tokenizer import PokemonTokenizer, Resolution

    tokenizer = PokemonTokenizer({"species": {"pikachu": 1}})
    assert tokenizer.resolve("species", None) == (0, Resolution.KNOWN_NONE)
    assert tokenizer.resolve("species", "missingno") == (0, Resolution.OOV)
    assert tokenizer.resolve("species", "pikachu") == (1, Resolution.KNOWN)
