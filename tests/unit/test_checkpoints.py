from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.training.checkpoint import CHECKPOINT_SCHEMA, DEFAULT_POLICY_STORE, CheckpointStore
from p0.training.magnet import Magnet
from p0.training.series_history import SeriesHistoryStore


def _small_policy() -> PolicyNet:
    return build_policy(ModelConfig(32, 4, 1, 128), default_runtime_resources())


class TestCheckpoints:
    def test_checkpoint_round_trip_envelope_provenance_and_state_layout(
        self, tmp_path: Path
    ) -> None:
        """
        Verify that checkpoint saving captures required schema metadata and strips immutable runtime statics.

        Verifies that:
        1. Saved artifact contains global contract SHA-256 and configuration envelopes.
        2. Static lookup tables (species stats, move stats, mechanic tags) are omitted from state_dict
           to avoid checkpoint bloat and decouple model weights from static dex data.
        3. Loading restores matching policy architecture and training step count.
        """
        path = tmp_path / "policy.pt"
        original = _small_policy()
        DEFAULT_POLICY_STORE.save_training_state(path, 7, original)

        restored = DEFAULT_POLICY_STORE.load_policy(path, "cpu")
        assert restored.d_model == original.d_model
        assert len(restored.actor.reducer.encoder.layers) == 1
        assert DEFAULT_POLICY_STORE.load_training_state(path, restored) == 7
        artifact = torch.load(path, weights_only=False)
        assert artifact["artifact_schema"] == CHECKPOINT_SCHEMA
        assert artifact["artifact_type"] == "training"
        assert "runtime_manifest_sha256" not in artifact
        assert len(artifact["global_contract_sha256"]) == 64
        assert artifact["global_contract"]["global_sha256"] == artifact["global_contract_sha256"]
        assert artifact["model_config"]["d_model"] == 32
        assert artifact["provenance"] == {}

        DEFAULT_POLICY_STORE.save_policy(
            path,
            _small_policy(),
            metadata={"showdown_commit": "older", "stat_imputer": "experimental"},
        )
        assert DEFAULT_POLICY_STORE.load_policy(path, "cpu").d_model == 32
        state = torch.load(path, weights_only=True)["model_state_dict"]
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

    def test_series_policy_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """Verify that serialized policy restores series encoder weights producing bitwise identical series tokens."""
        path = tmp_path / "policy.pt"
        config = ModelConfig(32, 4, 1, 128)
        original = build_policy(config, default_runtime_resources())
        DEFAULT_POLICY_STORE.save_policy(path, original)

        restored = DEFAULT_POLICY_STORE.load_policy(path, "cpu")
        assert restored.config == config

        histories = torch.randn(1, 10, config.d_model)
        history_mask = torch.ones(1, 10, dtype=torch.bool)
        with torch.no_grad():
            rest_t = restored.series(histories, history_mask)
            orig_t = original.series(histories, history_mask)
            assert torch.equal(rest_t, orig_t)

    def test_deep_checkpoint_round_trip_is_deterministic(self, tmp_path: Path) -> None:
        """Verify deep multi-layer transformer policies restore all layer weights deterministically without alias sharing."""
        path = tmp_path / "policy.pt"
        config = ModelConfig(
            d_model=32,
            nhead=4,
            reducer_layers=3,
            dim_feedforward=128,
        )
        original = build_policy(config, default_runtime_resources())
        DEFAULT_POLICY_STORE.save_policy(path, original)

        restored = DEFAULT_POLICY_STORE.load_policy(path, "cpu")

        assert restored.config == config
        assert len(restored.actor.reducer.encoder.layers) == 3
        # Ensure layers are distinct module instances rather than shared references
        assert (
            restored.actor.reducer.encoder.layers[0] is not restored.actor.reducer.encoder.layers[1]
        )
        for name, parameter in original.state_dict().items():
            torch.testing.assert_close(parameter, restored.state_dict()[name])

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
        DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
        artifact = torch.load(path, weights_only=False)
        mutate(artifact["model_config"])
        torch.save(artifact, path)

        with pytest.raises(ValueError, match=message):
            DEFAULT_POLICY_STORE.load_policy(path, "cpu")

    def test_training_checkpoint_rejects_state_config_mismatch(self, tmp_path: Path) -> None:
        """Verify load_training_state detects architecture mismatch between checkpoint and target model instance."""
        path = tmp_path / "policy.pt"
        DEFAULT_POLICY_STORE.save_training_state(path, 1, _small_policy())
        artifact = torch.load(path, weights_only=False)
        artifact["model_config"]["d_model"] = 64
        torch.save(artifact, path)

        policy = _small_policy()
        with pytest.raises(ValueError, match="model configuration does not match"):
            DEFAULT_POLICY_STORE.load_training_state(path, policy)

    def test_atomic_checkpoint_failure_preserves_previous_target(self, tmp_path: Path) -> None:
        """Verify a failed filesystem replacement leaves the existing checkpoint target untouched."""
        path = tmp_path / "policy.pt"
        path.mkdir()
        sentinel = path / "previous"
        sentinel.write_bytes(b"previous")

        with pytest.raises(OSError):
            DEFAULT_POLICY_STORE.save_policy(path, _small_policy())

        assert sentinel.read_bytes() == b"previous"

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
        with torch.no_grad():
            for parameter in magnet.policy.parameters():
                parameter.add_(0.125)
        store.save_training_state(
            path,
            17,
            policy,
            optimizer=optimizer,
            magnet=magnet,
            metadata={"dataset_hash": "d" * 64},
            trainer_kind="ppo",
        )

        restored = build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.5)
        restored_magnet = Magnet(restored)
        episode = store.load_training_state(
            path,
            restored,
            optimizer=restored_optimizer,
            magnet=restored_magnet,
            expected_trainer_kind="ppo",
            expected_metadata={"dataset_hash": "d" * 64},
            require_training_state=True,
        )
        assert episode == 17
        # Confirm learning rate and Adam moment tensors were restored from checkpoint
        assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
        assert restored_optimizer.state
        for name, parameter in restored.state_dict().items():
            torch.testing.assert_close(parameter, policy.state_dict()[name])
        for name, parameter in restored_magnet.state_dict().items():
            torch.testing.assert_close(parameter, magnet.state_dict()[name])

    def test_training_checkpoint_round_trip_restores_collector_history_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify detached active history survives the checkpoint metadata envelope."""
        path = tmp_path / "collector-training.pt"
        policy = _small_policy()
        history = SeriesHistoryStore(policy.d_model)
        history.append(
            "series",
            1,
            torch.ones((3, policy.d_model), requires_grad=True),
            is_game_end=False,
            is_series_end=False,
        )
        collector_state = {
            "series_store1": {},
            "series_store2": {},
            "series_history1": history.training_state(),
            "series_history2": {},
        }
        store = CheckpointStore()
        store.save_training_state(
            path,
            4,
            policy,
            metadata={"collector_state": collector_state},
            trainer_kind="ppo",
        )

        metadata = store.load_metadata(path)
        raw_collector_state = metadata["collector_state"]
        assert isinstance(raw_collector_state, dict)
        restored = SeriesHistoryStore(policy.d_model)
        raw_history = raw_collector_state["series_history1"]
        assert isinstance(raw_history, dict)
        restored.restore_training_state(raw_history)
        assert restored.has_partial_games
        restored_state = restored.training_state()["series"]
        active_fragments = restored_state["active_fragments"]
        assert isinstance(active_fragments, tuple)
        assert isinstance(active_fragments[0], torch.Tensor)
        torch.testing.assert_close(
            active_fragments[0],
            torch.ones((3, policy.d_model)),
        )

    def test_checkpoint_rejects_global_contract_tampering(self, tmp_path: Path) -> None:
        """Verify checkpoint loading rejects files whose global contract checksum has been altered."""
        path = tmp_path / "policy.pt"
        store = CheckpointStore()
        store.save_policy(
            path, build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        )
        artifact = torch.load(path, weights_only=False)
        artifact["global_contract_sha256"] = "0" * 64
        torch.save(artifact, path)
        with pytest.raises(ValueError, match="global_contract_sha256"):
            store.load_policy(path, "cpu")

    def test_checkpoint_rejects_weights_only_resume_when_training_is_required(
        self, tmp_path: Path
    ) -> None:
        """Verify that requiring training state rejects weights-only inference checkpoints."""
        path = tmp_path / "policy.pt"
        store = CheckpointStore()
        store.save_policy(
            path, build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
        )
        with pytest.raises(ValueError, match="weights-only"):
            store.load_training_state(
                path,
                build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources()),
                require_training_state=True,
            )
