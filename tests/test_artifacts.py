import importlib.util
from pathlib import Path

import pytest
import torch

from p0.model.policy import PolicyNet
from p0.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH
from p0.training.checkpoint import CHECKPOINT_SCHEMA, DEFAULT_POLICY_STORE
from p0.training.utils import (
    load_checkpoint,
    policy_from_checkpoint,
    save_checkpoint,
)

_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_training", Path(__file__).parents[1] / "scripts" / "export_training.py"
)
assert _EXPORT_SPEC is not None and _EXPORT_SPEC.loader is not None
_EXPORT_MODULE = importlib.util.module_from_spec(_EXPORT_SPEC)
_EXPORT_SPEC.loader.exec_module(_EXPORT_MODULE)
collect_export_files = _EXPORT_MODULE.collect_export_files


def _small_policy() -> PolicyNet:
    return PolicyNet(
        obs_dim=(SEQUENCE_LENGTH, NUMERICAL_WIDTH),
        act_size=49,
        d_model=32,
        nhead=4,
        nlayer=1,
    )


def test_checkpoint_reconstructs_serialized_policy(tmp_path):
    path = tmp_path / "policy.pt"
    original = _small_policy()
    save_checkpoint(path, 7, original)

    restored = policy_from_checkpoint(path, "cpu")
    assert restored.d_model == original.d_model
    assert len(restored.actor.reducer.encoder.layers) == 1
    assert load_checkpoint(path, restored) == 7


def test_checkpoint_rejects_incompatible_runtime_contract(tmp_path):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, 1, _small_policy())
    artifact = torch.load(path, weights_only=False)
    artifact["runtime_contract_sha256"] = "0" * 64
    torch.save(artifact, path)

    with pytest.raises(ValueError, match="incompatible"):
        policy_from_checkpoint(path, "cpu")


def test_checkpoint_has_contract_reference_envelope_and_local_model_config(tmp_path):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, 1, _small_policy())
    artifact = torch.load(path, weights_only=True)
    assert artifact["artifact_schema"] == CHECKPOINT_SCHEMA
    assert artifact["artifact_type"] == "training"
    assert "runtime_manifest_sha256" not in artifact
    assert len(artifact["runtime_contract_sha256"]) == 64
    assert artifact["model_config"]["d_model"] == 32
    assert artifact["provenance"] == {}


def test_checkpoint_rejects_legacy_dictionary(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "runtime_manifest_sha256": "0" * 64,
            "model_state_dict": _small_policy().state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="legacy checkpoint"):
        policy_from_checkpoint(path, "cpu")


def test_policy_checkpoint_provenance_does_not_control_loading(tmp_path):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_policy(
        path,
        _small_policy(),
        metadata={"showdown_commit": "older", "stat_imputer": "experimental"},
    )
    assert DEFAULT_POLICY_STORE.load_policy(path, "cpu").d_model == 32


def test_checkpoint_does_not_duplicate_dex_derived_static_tables(tmp_path):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
    state = torch.load(path, weights_only=True)["model_state_dict"]
    assert not any(
        name.endswith(
            (
                "_species_statics",
                "_move_statics",
                "_item_mechanic_tags",
                "_ability_mechanic_tags",
            )
        )
        for name in state
    )


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "policy.pt"
    path.write_bytes(b"previous")

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr("p0.training.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
    assert path.read_bytes() == b"previous"


def test_export_includes_interpretation_contracts(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "checkpoint.pt").write_bytes(b"checkpoint")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/runtime_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data/vocab.json").write_text("{}", encoding="utf-8")

    exported = {arcname for _, arcname, _ in collect_export_files(tmp_path, artifacts)}
    assert "artifacts/checkpoint.pt" in exported
    assert "data/runtime_manifest.json" in exported
    assert "data/vocab.json" in exported
