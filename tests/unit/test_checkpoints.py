from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from p0.model.architecture_contract import CHECKPOINT_ARTIFACT_SCHEMA
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.training.checkpoint import DEFAULT_CHECKPOINT_STORE, CheckpointStore
from p0.training.magnet import Magnet


def _small_policy() -> PolicyNet:
    return build_policy(ModelConfig(32, 4, 1, 128), default_runtime_resources())


def _optimizer(policy: PolicyNet) -> torch.optim.Optimizer:
    return torch.optim.AdamW(policy.parameters())


def _scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cpu", enabled=False)


class TestCheckpoints:
    def test_checkpoint_round_trip_envelope_provenance_and_state_layout(
        self, tmp_path: Path
    ) -> None:
        """
        Verify that checkpoint saving captures required schema metadata and strips immutable runtime statics.

        Verifies that:
        1. Saved artifact contains runtime major/minor identity and configuration envelopes.
        2. Static lookup tables (species stats, move stats, mechanic tags) are omitted from state_dict
           to avoid checkpoint bloat and decouple model weights from static dex data.
        3. Loading restores matching policy architecture and training step count.
        """
        path = tmp_path / "policy.pt"
        original = _small_policy()
        DEFAULT_CHECKPOINT_STORE.save_training(
            path,
            7,
            original,
            trainer_kind="ppo",
            optimizer=_optimizer(original),
            scaler=_scaler(),
            metadata={},
        )

        restored = DEFAULT_CHECKPOINT_STORE.load_policy(path, "cpu")
        assert restored.config == ModelConfig(32, 4, 1, 128)
        assert (
            DEFAULT_CHECKPOINT_STORE.load_training(
                path,
                restored,
                trainer_kind="ppo",
                optimizer=_optimizer(restored),
                scaler=_scaler(),
            )
            == 7
        )
        artifact = DEFAULT_CHECKPOINT_STORE.read(path).artifact
        assert artifact["artifact_schema"] == CHECKPOINT_ARTIFACT_SCHEMA
        assert artifact["artifact_type"] == "training"
        assert "runtime_manifest_sha256" not in artifact
        assert set(artifact["runtime_contract"]) == {"major_sha256", "minor_sha256"}
        assert artifact["model_config"]["d_model"] == 32
        assert artifact["provenance"] == {"trainer_kind": "ppo"}

        DEFAULT_CHECKPOINT_STORE.save_policy(
            path,
            _small_policy(),
            metadata={"showdown_commit": "older", "stat_imputer": "experimental"},
        )
        assert DEFAULT_CHECKPOINT_STORE.load_policy(path, "cpu").d_model == 32
        state = DEFAULT_CHECKPOINT_STORE.read(path).artifact["model_state_dict"]
        # Verify runtime static buffers are excluded from serialized model state dict
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

    def test_deep_checkpoint_round_trip_is_deterministic(self, tmp_path: Path) -> None:
        """Verify deep multi-layer transformer policies restore all layer weights deterministically."""
        path = tmp_path / "policy.pt"
        config = ModelConfig(
            d_model=32,
            nhead=4,
            reducer_layers=3,
            dim_feedforward=128,
        )
        original = build_policy(config, default_runtime_resources())
        DEFAULT_CHECKPOINT_STORE.save_policy(path, original)

        restored = DEFAULT_CHECKPOINT_STORE.load_policy(path, "cpu")

        assert restored.config == config
        for name, parameter in original.state_dict().items():
            torch.testing.assert_close(parameter, restored.state_dict()[name])

        histories = torch.randn(1, 10, config.d_model)
        history_mask = torch.ones(1, 10, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(
                original.series(histories, history_mask),
                restored.series(histories, history_mask),
                rtol=0,
                atol=0,
            )

    @pytest.mark.parametrize(
        "mutate, message",
        (
            (lambda config: config.pop("reducer_layers"), "Invalid model configuration"),
            (lambda config: config.update({"unknown": 1}), "Invalid model configuration"),
            (lambda config: config.update({"d_model": True}), "Invalid model configuration"),
        ),
    )
    def test_checkpoint_rejects_malformed_model_configuration(
        self, tmp_path: Path, mutate: Any, message: str
    ) -> None:
        """Verify that CheckpointStore strictly validates model configuration schema and rejects malformed configs."""
        path = tmp_path / "policy.pt"
        DEFAULT_CHECKPOINT_STORE.save_policy(path, _small_policy())
        artifact = dict(DEFAULT_CHECKPOINT_STORE.read(path).artifact)
        mutate(artifact["model_config"])
        torch.save(artifact, path)

        with pytest.raises(ValueError, match=message):
            DEFAULT_CHECKPOINT_STORE.load_policy(path, "cpu")

    def test_training_checkpoint_rejects_state_config_mismatch(self, tmp_path: Path) -> None:
        """Verify load_training detects architecture mismatch between checkpoint and target model instance."""
        path = tmp_path / "policy.pt"
        saved = _small_policy()
        DEFAULT_CHECKPOINT_STORE.save_training(
            path,
            1,
            saved,
            trainer_kind="ppo",
            optimizer=_optimizer(saved),
            scaler=_scaler(),
            metadata={},
        )
        artifact = dict(DEFAULT_CHECKPOINT_STORE.read(path).artifact)
        artifact["model_config"]["d_model"] = 64
        torch.save(artifact, path)

        policy = _small_policy()
        before_state = {
            name: tensor.detach().clone() for name, tensor in policy.state_dict().items()
        }
        with pytest.raises(ValueError, match="model configuration does not match"):
            DEFAULT_CHECKPOINT_STORE.load_training(
                path,
                policy,
                trainer_kind="ppo",
                optimizer=_optimizer(policy),
                scaler=_scaler(),
            )
        assert all(
            torch.equal(before_state[name], tensor) for name, tensor in policy.state_dict().items()
        )

    def test_training_checkpoint_round_trip_restores_optimizer_magnet_and_provenance(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify that full training checkpoints restore optimizer states, EMA Magnet weights, and step counts."""
        path = tmp_path / "training.pt"
        store = CheckpointStore()
        policy = build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        magnet = Magnet(policy)
        # Perform a dummy training step to populate Adam momentum and variance state buffers
        loss = torch.stack(
            tuple(parameter.square().mean() for parameter in policy.parameters())
        ).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_optimizer_state = optimizer.state_dict()
        with torch.no_grad():
            for parameter in magnet.policy.parameters():
                parameter.add_(0.125)
        store.save_training(
            path,
            17,
            policy,
            optimizer=optimizer,
            magnet=magnet,
            metadata={"dataset_id": "d" * 64},
            trainer_kind="ppo",
            scaler=_scaler(),
        )

        restored = build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.5)
        restored_magnet = Magnet(restored)
        episode = store.load_training(
            path,
            restored,
            trainer_kind="ppo",
            optimizer=restored_optimizer,
            magnet=restored_magnet,
            scaler=_scaler(),
        )
        assert episode == 17
        # Confirm learning rate and Adam moment tensors were restored from checkpoint
        torch.testing.assert_close(restored_optimizer.state_dict(), saved_optimizer_state)
        assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
        for name, parameter in restored.state_dict().items():
            torch.testing.assert_close(parameter, policy.state_dict()[name])
        for name, parameter in restored_magnet.state_dict().items():
            torch.testing.assert_close(parameter, magnet.state_dict()[name])

    def test_checkpoint_without_a_magnet_restarts_it_from_the_restored_policy(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "training.pt"
        store = CheckpointStore()
        policy = _small_policy()
        store.save_training(
            path,
            3,
            policy,
            trainer_kind="ppo",
            optimizer=_optimizer(policy),
            scaler=_scaler(),
            metadata={},
        )

        restored = _small_policy()
        magnet = Magnet(restored)
        with torch.no_grad():
            for parameter in restored.parameters():
                parameter.add_(1.0)
        store.load_training(
            path,
            restored,
            trainer_kind="ppo",
            optimizer=_optimizer(restored),
            magnet=magnet,
            scaler=_scaler(),
        )

        for name, parameter in magnet.state_dict().items():
            torch.testing.assert_close(parameter, policy.state_dict()[name])

    def test_changed_value_objective_warns_and_loads_weights(self, tmp_path: Path, caplog) -> None:
        path = tmp_path / "policy.pt"
        store = CheckpointStore()
        policy = _small_policy()
        store.save_policy(path, policy, metadata={"gamma": 0.99})

        with caplog.at_level("WARNING", logger="p0.training.checkpoint"):
            restored = store.load_policy(path, "cpu", expected_objective={"gamma": 0.9})

        assert "gamma=0.99" in caplog.text
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)

    def test_checkpoint_rejects_incompatible_vocabulary(self, tmp_path: Path) -> None:
        """Verify checkpoint loading rejects files whose vocabulary/encoding identity has been altered."""
        path = tmp_path / "policy.pt"
        store = CheckpointStore()
        store.save_policy(
            path, build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        )
        artifact = dict(store.read(path).artifact)
        artifact["runtime_contract"]["major_sha256"] = "older-vocabulary"
        torch.save(artifact, path)
        with pytest.raises(ValueError, match="incompatible"):
            store.load_policy(path, "cpu")

    def test_checkpoint_rejects_weights_only_resume(self, tmp_path: Path) -> None:
        """Verify that resuming training rejects weights-only inference checkpoints."""
        path = tmp_path / "policy.pt"
        store = CheckpointStore()
        store.save_policy(
            path, build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        )
        target = build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        with pytest.raises(ValueError, match="weights-only"):
            store.load_training(
                path,
                target,
                trainer_kind="ppo",
                optimizer=_optimizer(target),
                scaler=_scaler(),
            )

    def test_training_checkpoint_rejects_other_trainer_but_policy_weights_transfer(
        self, tmp_path: Path
    ) -> None:
        """Verify a BC training checkpoint cannot resume PPO but still supplies inference weights."""
        path = tmp_path / "bc-training.pt"
        store = CheckpointStore()
        policy = _small_policy()
        store.save_training(
            path,
            1,
            policy,
            optimizer=torch.optim.AdamW(policy.parameters()),
            trainer_kind="bc",
            scaler=_scaler(),
            metadata={},
        )

        target = _small_policy()
        with pytest.raises(ValueError, match="trainer"):
            store.load_training(
                path,
                target,
                trainer_kind="ppo",
                optimizer=_optimizer(target),
                scaler=_scaler(),
            )

        restored = store.load_policy(path, "cpu")
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor)

    def test_training_checkpoint_without_optimizer_state_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "training.pt"
        store = CheckpointStore()
        policy = _small_policy()
        store.save_training(
            path,
            1,
            policy,
            trainer_kind="ppo",
            optimizer=_optimizer(policy),
            scaler=_scaler(),
            metadata={},
        )
        artifact = torch.load(path, weights_only=True)
        del artifact["training_state"]["optimizer_state_dict"]
        torch.save(artifact, path)

        target = _small_policy()
        with pytest.raises(ValueError, match="Invalid training state"):
            store.load_training(
                path,
                target,
                trainer_kind="ppo",
                optimizer=_optimizer(target),
                scaler=_scaler(),
            )

    def test_load_episode_helper(self, tmp_path: Path) -> None:
        """Verify load_episode reads episode without full model restoration."""
        store = CheckpointStore()
        policy = _small_policy()

        policy_path = tmp_path / "policy.pt"
        store.save_policy(policy_path, policy)
        assert store.load_episode(policy_path) == 0

        training_path = tmp_path / "training.pt"
        store.save_training(
            training_path,
            42,
            policy,
            trainer_kind="ppo",
            optimizer=_optimizer(policy),
            scaler=_scaler(),
            metadata={},
        )
        assert store.load_episode(training_path) == 42


class TestLoadedCheckpoint:
    def test_loaded_input_survives_atomic_replacement(self, tmp_path: Path) -> None:
        store = CheckpointStore()
        path = tmp_path / "input.pt"
        policy = _small_policy()
        optimizer = _optimizer(policy)
        store.save_training(
            path,
            1,
            policy,
            trainer_kind="ppo",
            optimizer=optimizer,
            scaler=_scaler(),
            metadata={},
        )
        loaded = store.read(path)
        store.save_training(
            path,
            2,
            policy,
            trainer_kind="ppo",
            optimizer=optimizer,
            scaler=_scaler(),
            metadata={},
        )
        assert store.load_episode(loaded) == 1
        assert store.load_episode(path) == 2


class TestRuntimeCompatibility:
    def test_dex_only_change_warns_and_loads_weights(self, tmp_path: Path, caplog) -> None:
        from copy import deepcopy

        from p0.model.resources import RuntimeResources

        original = default_runtime_resources()
        path = tmp_path / "policy.pt"
        CheckpointStore(resources=original).save_policy(
            path, build_policy(ModelConfig(32, 4, 1, 128), original)
        )
        dex = deepcopy(original.dex)
        dex["moves"][0]["basePower"] = 999
        changed = RuntimeResources.from_data(original.vocab, dex)
        with caplog.at_level("WARNING", logger="p0.training.checkpoint"):
            restored = CheckpointStore(resources=changed).load_policy(path, "cpu")
        assert restored.config == ModelConfig(32, 4, 1, 128)
        assert "different dex data" in caplog.text
