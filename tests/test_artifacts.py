import importlib.util
from pathlib import Path

import pytest
import torch

from p0.battle.series import GameSummary, SideGameSummary
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.series_context import SeriesFeatures, tensorize_series
from p0.training.checkpoint import CHECKPOINT_SCHEMA, DEFAULT_POLICY_STORE

_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_training", Path(__file__).parents[1] / "scripts" / "export_training.py"
)
assert _EXPORT_SPEC is not None and _EXPORT_SPEC.loader is not None
_EXPORT_MODULE = importlib.util.module_from_spec(_EXPORT_SPEC)
_EXPORT_SPEC.loader.exec_module(_EXPORT_MODULE)
collect_export_files = _EXPORT_MODULE.collect_export_files


def _small_policy() -> PolicyNet:
    return build_policy(ModelConfig(32, 4, 1, 128), default_runtime_resources())


def test_checkpoint_round_trip_envelope_provenance_and_state_layout(tmp_path):
    path = tmp_path / "policy.pt"
    original = _small_policy()
    DEFAULT_POLICY_STORE.save_training_state(path, 7, original)

    restored = DEFAULT_POLICY_STORE.load_policy(path, "cpu")
    assert restored.d_model == original.d_model
    assert len(restored.actor.reducer.core_layers) == 1
    assert DEFAULT_POLICY_STORE.load_training_state(path, restored) == 7
    artifact = torch.load(path, weights_only=False)
    assert artifact["artifact_schema"] == CHECKPOINT_SCHEMA
    assert artifact["artifact_type"] == "training"
    assert "runtime_manifest_sha256" not in artifact
    assert len(artifact["runtime_contract_sha256"]) == 64
    assert artifact["model_config"]["d_model"] == 32
    assert artifact["provenance"] == {}

    DEFAULT_POLICY_STORE.save_policy(
        path,
        _small_policy(),
        metadata={"showdown_commit": "older", "stat_imputer": "experimental"},
    )
    assert DEFAULT_POLICY_STORE.load_policy(path, "cpu").d_model == 32
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


def test_series_policy_checkpoint_round_trip(tmp_path):
    path = tmp_path / "policy.pt"
    config = ModelConfig(32, 4, 1, 128)
    original = build_policy(config, default_runtime_resources())
    DEFAULT_POLICY_STORE.save_policy(path, original)

    restored = DEFAULT_POLICY_STORE.load_policy(path, "cpu")
    assert restored.config == config

    side = SideGameSummary(
        leads=("charizard", "garchomp"),
        brought=("charizard", "garchomp", "pikachu"),
        mega_species="",
        moves_used={"charizard": ("flamethrower",)},
        revealed_items={},
        revealed_abilities={},
        revealed_formes=(),
        switch_count=1,
        pivot_count=0,
    )
    game = GameSummary(game_number=1, winner=0, series_score=(1, 0), turns=5, sides=(side, side))
    features = SeriesFeatures.stack(
        [tensorize_series((game,), 0, default_runtime_resources().tokenizer)]
    )
    with torch.no_grad():
        assert torch.equal(restored.encode_series(features), original.encode_series(features))


def test_deep_tied_checkpoint_round_trip_is_deterministic(tmp_path):
    path = tmp_path / "policy.pt"
    config = ModelConfig(
        d_model=32,
        nhead=4,
        prelude_layers=1,
        history_tokens=8,
        dim_feedforward=128,
        core_repeats=3,
        core_weights_tied=True,
        pass_embedding_enabled=False,
    )
    original = build_policy(config, default_runtime_resources())
    DEFAULT_POLICY_STORE.save_policy(path, original)

    restored = DEFAULT_POLICY_STORE.load_policy(path, "cpu")

    assert restored.config == config
    assert restored.actor.reducer.core_layers[0] is restored.actor.reducer.core_layers[1]
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])


def test_checkpoint_rejects_incompatible_and_legacy_contracts(tmp_path):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_training_state(path, 1, _small_policy())
    artifact = torch.load(path, weights_only=False)
    artifact["runtime_contract_sha256"] = "0" * 64
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="incompatible"):
        DEFAULT_POLICY_STORE.load_policy(path, "cpu")

    torch.save(
        {
            "runtime_manifest_sha256": "0" * 64,
            "model_state_dict": _small_policy().state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="legacy checkpoint"):
        DEFAULT_POLICY_STORE.load_policy(path, "cpu")

    DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
    legacy = dict(torch.load(path, weights_only=False))
    legacy["artifact_schema"] = "p0.checkpoint.v1"
    torch.save(legacy, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint schema"):
        DEFAULT_POLICY_STORE.load_policy(path, "cpu")


@pytest.mark.parametrize(
    "mutate, message",
    (
        (lambda config: config.pop("prelude_layers"), "Invalid model configuration"),
        (lambda config: config.update({"unknown": 1}), "Invalid model configuration"),
        (lambda config: config.update({"d_model": True}), "Invalid model configuration"),
    ),
)
def test_checkpoint_rejects_malformed_model_configuration(tmp_path, mutate, message):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
    artifact = torch.load(path, weights_only=False)
    mutate(artifact["model_config"])
    torch.save(artifact, path)

    with pytest.raises(ValueError, match=message):
        DEFAULT_POLICY_STORE.load_policy(path, "cpu")


def test_checkpoint_rejects_pre_workstream_5_model_configuration(tmp_path):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
    artifact = torch.load(path, weights_only=False)
    artifact["model_config"] = {
        "d_model": 32,
        "nhead": 4,
        "reducer_layers": 1,
        "history_tokens": 8,
        "dim_feedforward": 128,
        "series_context_enabled": False,
        "series_tokens": 4,
    }
    torch.save(artifact, path)

    with pytest.raises(ValueError, match="Invalid model configuration"):
        DEFAULT_POLICY_STORE.load_policy(path, "cpu")


def test_training_checkpoint_rejects_state_config_mismatch(tmp_path):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_training_state(path, 1, _small_policy())
    artifact = torch.load(path, weights_only=False)
    artifact["model_config"]["d_model"] = 64
    torch.save(artifact, path)

    policy = _small_policy()
    with pytest.raises(ValueError, match="model configuration does not match"):
        DEFAULT_POLICY_STORE.load_training_state(path, policy)


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "policy.pt"
    path.write_bytes(b"previous")

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr("p0.persistence.os.replace", fail_replace)
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
