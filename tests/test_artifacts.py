import importlib.util
from pathlib import Path

import pytest
import torch

from p0.model.policy import PolicyNet
from p0.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH
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


def test_checkpoint_rejects_incompatible_manifest(tmp_path):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, 1, _small_policy())
    artifact = torch.load(path, weights_only=False)
    artifact["runtime_manifest_sha256"] = "0" * 64
    torch.save(artifact, path)

    with pytest.raises(ValueError, match="runtime_manifest_sha256"):
        policy_from_checkpoint(path, "cpu")


def test_checkpoint_has_hash_reference_and_local_model_config(tmp_path):
    path = tmp_path / "policy.pt"
    save_checkpoint(path, 1, _small_policy())
    artifact = torch.load(path, weights_only=True)
    assert "runtime_manifest" not in artifact
    assert len(artifact["runtime_manifest_sha256"]) == 64
    assert artifact["model_config"]["d_model"] == 32


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
