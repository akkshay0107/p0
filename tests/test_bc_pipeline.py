import torch
from test_replay_dataset import _payload

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.replays.compile import compile_payloads, write_tensor_shards
from p0.replays.dataset import LazyReplayDataset
from p0.training.bc import BCTrainer
from p0.training.config import BCConfig


def test_replay_to_series_bc_checkpoint_smoke(tmp_path) -> None:
    result = compile_payloads((_payload("game-1"), _payload("game-2")))
    built = write_tensor_shards(
        result,
        tmp_path / "shards",
        max_decisions_per_shard=8,
        created_at="2026-01-01T00:00:00Z",
    )
    dataset = LazyReplayDataset(built.manifest_path)
    policy = build_policy(
        ModelConfig(
            d_model=64,
            nhead=4,
            reducer_layers=1,
            dim_feedforward=128,
        ),
        default_runtime_resources(),
    )
    trainer = BCTrainer(
        policy,
        dataset,
        BCConfig(
            batch_decisions=2,
            learning_rate=1e-3,
            epochs=1,
            amp=False,
        ),
        device="cpu",
    )

    metrics = trainer.train()

    assert metrics.decisions == 8
    assert metrics.labeled_decisions > 0
    assert torch.isfinite(torch.tensor(metrics.loss))
    checkpoint = tmp_path / "bc.pt"
    trainer.save_checkpoint(checkpoint, epoch=1)

    restored = build_policy(trainer.policy.config, default_runtime_resources())
    restored_trainer = BCTrainer(
        restored,
        (),
        trainer.config,
        device="cpu",
    )
    assert restored_trainer.load_checkpoint(checkpoint) == 1
    for name, parameter in trainer.policy.state_dict().items():
        torch.testing.assert_close(parameter, restored_trainer.policy.state_dict()[name])
