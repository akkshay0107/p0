import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.runtime import showdown
from p0.training.checkpoint import CHECKPOINT_SCHEMA, DEFAULT_POLICY_STORE, CheckpointStore
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import compute_ppo_objective
from p0.training.trainer import PPOTrainer
from p0.training.utils import amp_enabled


def _small_policy() -> PolicyNet:
    return build_policy(ModelConfig(32, 4, 1, 128), default_runtime_resources())


def test_checkpoint_round_trip_envelope_provenance_and_state_layout(tmp_path):
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

    histories = [torch.randn(1, 10, config.d_model)]
    with torch.no_grad():
        rest_t, rest_m = restored.encode_series(histories)
        orig_t, orig_m = original.encode_series(histories)
        assert torch.equal(rest_t, orig_t)
        assert torch.equal(rest_m, orig_m)


def test_deep_checkpoint_round_trip_is_deterministic(tmp_path):
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
    assert restored.actor.reducer.encoder.layers[0] is not restored.actor.reducer.encoder.layers[1]
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
def test_checkpoint_rejects_malformed_model_configuration(tmp_path, mutate, message):
    path = tmp_path / "policy.pt"
    DEFAULT_POLICY_STORE.save_policy(path, _small_policy())
    artifact = torch.load(path, weights_only=False)
    mutate(artifact["model_config"])
    torch.save(artifact, path)

    with pytest.raises(ValueError, match=message):
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
            num_episodes=9, magnet_refresh_interval=1, ramp_up_phase=0.5
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
            num_episodes=9, magnet_refresh_interval=1, ramp_up_phase=0.5
        ),
    )

    trainer.run(start_episode=9)

    assert saved == [(tmp_path / "checkpoint.pt", 9)]


def test_training_checkpoint_round_trip_restores_optimizer_magnet_and_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training.pt"
    store = CheckpointStore()
    policy = build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources())
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
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert restored_optimizer.state
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter, policy.state_dict()[name])
    for name, parameter in restored_magnet.state_dict().items():
        torch.testing.assert_close(parameter, magnet.state_dict()[name])


def test_checkpoint_rejects_global_contract_tampering(tmp_path: Path) -> None:
    path = tmp_path / "policy.pt"
    store = CheckpointStore()
    store.save_policy(path, build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources()))
    artifact = torch.load(path, weights_only=False)
    artifact["global_contract_sha256"] = "0" * 64
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="global_contract_sha256"):
        store.load_policy(path, "cpu")


def test_checkpoint_rejects_weights_only_resume_when_training_is_required(tmp_path: Path) -> None:
    path = tmp_path / "policy.pt"
    store = CheckpointStore()
    store.save_policy(path, build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources()))
    with pytest.raises(ValueError, match="weights-only"):
        store.load_training_state(
            path,
            build_policy(ModelConfig(16, 2, 1, 64), default_runtime_resources()),
            require_training_state=True,
        )


@pytest.mark.network
def test_loopback_port_allocator_returns_distinct_reusable_ports() -> None:
    ports = showdown.allocate_loopback_ports(8)
    assert len(ports) == len(set(ports))
    sockets = [socket.socket() for _ in ports]
    try:
        for port, listener in zip(ports, sockets, strict=True):
            listener.bind(("127.0.0.1", port))
    finally:
        for listener in sockets:
            listener.close()


def test_showdown_server_start_stop_owns_process_log_and_command(
    monkeypatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []

    class FakeProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            commands.append(command)
            self.kwargs = kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        showdown.socket, "create_connection", lambda *args, **kwargs: FakeConnection()
    )
    server = showdown.ShowdownServer(
        9123,
        showdown_root=tmp_path,
        log_path=tmp_path / "nested" / "server.log",
        startup_timeout=1,
    )
    server.start()
    assert commands == [
        [
            "node",
            "--max-old-space-size=1536",
            "pokemon-showdown",
            "start",
            "--no-security",
            "--skip-build",
            "9123",
        ]
    ]
    assert server.websocket_url.endswith(":9123/showdown/websocket")
    assert server.process is not None
    server.stop()
    assert server.process is None
    assert server._log_file is None
    assert (tmp_path / "nested" / "server.log").is_file()


def test_showdown_server_rejects_double_start_and_invalid_port_groups(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeProcess:
        returncode = None

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        showdown.socket, "create_connection", lambda *args, **kwargs: _CheckpointConnection()
    )
    server = showdown.ShowdownServer(9124, showdown_root=tmp_path, startup_timeout=1)
    server.start()
    with pytest.raises(RuntimeError, match="already started"):
        server.start()
    server.stop()

    with pytest.raises(ValueError, match="unique"):
        with showdown.start_showdown_servers(2, showdown_root=tmp_path, ports=(1, 1)):
            pass


class _CheckpointConnection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args


def test_showdown_server_kills_process_when_graceful_stop_times_out(tmp_path: Path) -> None:
    class StuckProcess:
        returncode = None

        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            self.killed = True

        def wait(self, timeout: float | None = None):
            if not self.killed:
                raise subprocess.TimeoutExpired("node", timeout if timeout is not None else 0.0)
            return 0

    process = StuckProcess()
    server = showdown.ShowdownServer(9125, showdown_root=tmp_path, stop_timeout=0.01)
    server.process = cast(Any, process)
    server.stop()
    assert process.killed


def test_showdown_server_rolls_back_after_child_crash(monkeypatch, tmp_path: Path) -> None:
    class CrashedProcess:
        returncode = 17

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    monkeypatch.setattr(showdown.subprocess, "Popen", lambda *args, **kwargs: CrashedProcess())
    server = showdown.ShowdownServer(9126, showdown_root=tmp_path, startup_timeout=0.1)
    with pytest.raises(RuntimeError, match="exited"):
        server.start()
    assert server.process is None
    assert server._log_file is None
