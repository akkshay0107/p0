from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from p0.runtime import showdown
from p0.training.config import TrainingConfig
from p0.training.ppo import compute_ppo_objective
from p0.training.trainer import PPOTrainer
from p0.training.utils import amp_enabled


def test_pure_ppo_objective_clips_and_weights_team_preview():
    config = TrainingConfig(teampreview_loss_mult=2.0, teampreview_alpha_mult=3.0)
    total, policy, value, ratio, log_ratio = compute_ppo_objective(
        torch.log(torch.tensor([2.0, 0.5])),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 0.5]),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2),
        torch.tensor([True, False]),
        config,
        alpha=0.1,
        critic_only=False,
    )
    assert total.shape == policy.shape == value.shape == ratio.shape == log_ratio.shape == (2,)
    assert ratio.tolist() == pytest.approx([2.0, 0.5])
    assert total[0] != total[1]


def test_ppo_amp_is_cuda_only():
    config = TrainingConfig(enable_optim=True)

    assert not amp_enabled(config, torch.device("cpu"))
    assert amp_enabled(config, torch.device("cuda"))
    assert not amp_enabled(TrainingConfig(enable_optim=False), torch.device("cuda"))


def test_showdown_group_rolls_back_servers_when_later_start_fails(monkeypatch):
    events = []

    class FakeServer:
        def __init__(self, port, **kwargs):
            self.port = port

        def __enter__(self):
            events.append(("start", self.port))
            if self.port == 2:
                raise RuntimeError("failed")
            return self

        def __exit__(self, *args):
            events.append(("stop", self.port))

    monkeypatch.setattr(showdown, "ShowdownServer", FakeServer)
    monkeypatch.setattr(showdown, "build_showdown", lambda root: None)
    with pytest.raises(RuntimeError, match="failed"):
        with showdown.start_showdown_servers(2, ports=(1, 2)):
            pass
    assert events == [("start", 1), ("start", 2), ("stop", 1)]


def test_showdown_build_failure_has_bounded_diagnostics(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise __import__("subprocess").CalledProcessError(1, args[0], stderr="x" * 5000)

    monkeypatch.setattr(showdown.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as error:
        showdown.build_showdown(tmp_path)
    assert len(str(error.value)) < 4200


def test_trainer_cancellation_saves_once_before_collecting(tmp_path):
    saved = []

    class Store:
        def save_training_state(self, path, episode, policy, **kwargs):
            saved.append((path, episode, policy, kwargs))

    collector = SimpleNamespace(vector_env=SimpleNamespace(reset=lambda: None))
    updater = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]),
        scaler=object(),
    )
    trainer = PPOTrainer(
        policy=cast(Any, object()),
        policy_store=cast(Any, Store()),
        checkpoint_path=tmp_path / "checkpoint.pt",
        collector=cast(Any, collector),
        updater=cast(Any, updater),
        magnet=cast(Any, object()),
        scheduler=cast(Any, object()),
        training_config=TrainingConfig(
            num_episodes=1, warmup_episodes=0, magnet_refresh_interval=1
        ),
        cancel_requested=lambda: True,
    )
    trainer.run()
    assert [(path, episode) for path, episode, _, _ in saved] == [(tmp_path / "checkpoint.pt", 0)]


def test_trainer_saves_final_completed_episode(tmp_path):
    saved = []

    class Store:
        def save_training_state(self, path, episode, policy, **kwargs):
            saved.append((path, episode))

    collector = SimpleNamespace(vector_env=SimpleNamespace(reset=lambda: None))
    updater = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.0}]),
        scaler=object(),
    )
    trainer = PPOTrainer(
        policy=cast(Any, object()),
        policy_store=cast(Any, Store()),
        checkpoint_path=tmp_path / "checkpoint.pt",
        collector=cast(Any, collector),
        updater=cast(Any, updater),
        magnet=cast(Any, object()),
        scheduler=cast(Any, object()),
        training_config=TrainingConfig(
            num_episodes=2, warmup_episodes=0, magnet_refresh_interval=1
        ),
    )

    trainer.run(start_episode=2)

    assert saved == [(tmp_path / "checkpoint.pt", 2)]
