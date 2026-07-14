import json
from copy import deepcopy

import pytest

from p0.format_config import (
    ACTION_CONTRACT,
    RESOURCE_FEATURE_ABI,
    TENSOR_ABI,
    RuntimeManifest,
    canonical_json_sha256,
    current_manifest,
    validate_artifact_runtime_contract,
)


def _resources(tmp_path, *, extra_species: bool = False, base_power: int = 90):
    vocab = tmp_path / "vocab.json"
    species = {"pikachu": 1}
    if extra_species:
        species["raichu"] = 2
    vocab.write_text(json.dumps({"species": species}), encoding="utf-8")
    dex = tmp_path / "champions_dex.json"
    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": base_power}]}))
    return vocab, dex


def _write_manifest(path, manifest: RuntimeManifest) -> None:
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")


def test_runtime_manifest_round_trips_one_readable_contract(tmp_path):
    vocab, dex = _resources(tmp_path)
    manifest = current_manifest(vocab_path=vocab, dex_path=dex)
    restored = RuntimeManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

    assert restored == manifest
    assert restored.tensor_abi == TENSOR_ABI
    assert restored.resource_feature_abi == RESOURCE_FEATURE_ABI
    assert restored.action == ACTION_CONTRACT
    assert restored.runtime_contract_sha256 == canonical_json_sha256(
        restored.runtime_contract()
    )


def test_canonical_hash_ignores_object_order_but_not_required_semantics():
    first = {"shape": [31, 10], "dtype": "int64"}
    reordered = {"dtype": "int64", "shape": [31, 10]}
    changed = {"dtype": "int64", "shape": [32, 10]}

    assert canonical_json_sha256(first) == canonical_json_sha256(reordered)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)
    with pytest.raises(ValueError, match="unsupported value"):
        canonical_json_sha256({"scale": 0.5})


def test_vocabulary_expansion_breaks_contract_but_dex_change_does_not(tmp_path):
    vocab, dex = _resources(tmp_path, base_power=90)
    original = current_manifest(vocab_path=vocab, dex_path=dex)

    dex.write_text(json.dumps({"moves": [{"id": "test", "basePower": 80}]}))
    rebalanced = current_manifest(vocab_path=vocab, dex_path=dex)
    assert rebalanced.runtime_contract_sha256 == original.runtime_contract_sha256
    assert rebalanced.champions_dex_sha256 != original.champions_dex_sha256

    vocab, dex = _resources(tmp_path, extra_species=True, base_power=80)
    expanded = current_manifest(vocab_path=vocab, dex_path=dex)
    assert expanded.runtime_contract_sha256 != original.runtime_contract_sha256


def test_manifest_rejects_tampered_contract_hash(tmp_path):
    vocab, dex = _resources(tmp_path)
    value = current_manifest(vocab_path=vocab, dex_path=dex).to_dict()
    value["runtime_contract"]["tensor_abi"] = "changed-without-rehashing"
    with pytest.raises(ValueError, match="does not match"):
        RuntimeManifest.from_dict(value)


def test_manifest_rejects_missing_and_unknown_fields(tmp_path):
    vocab, dex = _resources(tmp_path)
    value = current_manifest(vocab_path=vocab, dex_path=dex).to_dict()
    del value["runtime_contract"]["resource_feature_abi"]
    with pytest.raises(ValueError, match="missing"):
        RuntimeManifest.from_dict(value)

    value = current_manifest(vocab_path=vocab, dex_path=dex).to_dict()
    value["runtime_contract"]["unplanned"] = True
    with pytest.raises(ValueError, match="unknown"):
        RuntimeManifest.from_dict(value)


def test_artifact_validation_compares_only_runtime_contract(tmp_path):
    vocab, dex = _resources(tmp_path)
    manifest = current_manifest(vocab_path=vocab, dex_path=dex)
    manifest_path = tmp_path / "runtime_manifest.json"
    _write_manifest(manifest_path, manifest)
    artifact = {"runtime_contract_sha256": manifest.runtime_contract_sha256}
    assert validate_artifact_runtime_contract(artifact, manifest_path) == manifest

    changed_provenance = deepcopy(manifest.to_dict())
    changed_provenance["mechanics_provenance"]["showdown_commit"] = "new-commit"
    manifest_path.write_text(json.dumps(changed_provenance), encoding="utf-8")
    validate_artifact_runtime_contract(artifact, manifest_path)

    artifact["runtime_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incompatible"):
        validate_artifact_runtime_contract(artifact, manifest_path)


def test_legacy_manifest_reference_fails_cleanly(tmp_path):
    vocab, dex = _resources(tmp_path)
    manifest_path = tmp_path / "runtime_manifest.json"
    _write_manifest(manifest_path, current_manifest(vocab_path=vocab, dex_path=dex))
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_artifact_runtime_contract(
            {"runtime_manifest_sha256": "0" * 64}, manifest_path
        )


def test_tokenizer_resolution_distinguishes_none_and_oov():
    from p0.model.tokenizer import PokemonTokenizer, Resolution

    tokenizer = PokemonTokenizer({"species": {"pikachu": 1}})
    assert tokenizer.resolve("species", None) == (0, Resolution.KNOWN_NONE)
    assert tokenizer.resolve("species", "missingno") == (0, Resolution.OOV)
    assert tokenizer.resolve("species", "pikachu") == (1, Resolution.KNOWN)
