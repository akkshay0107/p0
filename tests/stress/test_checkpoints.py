from __future__ import annotations

from pathlib import Path

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.training.checkpoint import CheckpointStore
from p0.training.magnet import Magnet


def _policy():
    return build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())


@pytest.mark.stress
def test_training_checkpoint_round_trip_restores_optimizer_magnet_and_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.pt"
    store = CheckpointStore()
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    magnet = Magnet(policy)
    loss = torch.stack(tuple(parameter.square().mean() for parameter in policy.parameters())).sum()
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

    restored = _policy()
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
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert restored_optimizer.state
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter, policy.state_dict()[name])
    for name, parameter in restored_magnet.state_dict().items():
        torch.testing.assert_close(parameter, magnet.state_dict()[name])


@pytest.mark.stress
def test_checkpoint_rejects_runtime_contract_tampering(tmp_path: Path) -> None:
    path = tmp_path / "policy.pt"
    store = CheckpointStore()
    store.save_policy(path, _policy())
    artifact = torch.load(path, weights_only=False)
    artifact["runtime_contract_sha256"] = "0" * 64
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="runtime contract"):
        store.load_policy(path, "cpu")


@pytest.mark.stress
def test_checkpoint_rejects_weights_only_resume_when_training_is_required(tmp_path: Path) -> None:
    path = tmp_path / "policy.pt"
    store = CheckpointStore()
    store.save_policy(path, _policy())
    with pytest.raises(ValueError, match="weights-only"):
        store.load_training_state(path, _policy(), require_training_state=True)
