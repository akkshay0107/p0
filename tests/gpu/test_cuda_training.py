"""Opt-in CUDA checks for BC history, precision, and checkpoint recovery."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, canonical_policy_state_dict, compile_policy
from p0.model.policy import MemoryInputs
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.paths import DEFAULT_PATHS
from p0.replays.dataset import ReplayGameChunk
from p0.replays.schema import LabelKind
from p0.teams.corpus_build import write_corpus_manifest
from p0.training.bc import BCTrainer
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig, GlobalConfig, TeamsConfig, TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import ppo_update
from p0.training.ppo_runner import run_training
from p0.training.trajectory import CollectedTrajectory, prepare_trajectory_batches
from p0.training.utils import select_optimization_precision
from tests.team_fixtures import default_test_corpus

pytestmark = pytest.mark.gpu
MODEL_CONFIG = ModelConfig(32, 4, 1, 64)


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda")


def _exact_label_game(length: int, series_id: str) -> ReplayGameChunk:
    """Build a one-game BC chunk whose decisions all have the exact label (7, 8)."""
    action_mask = torch.zeros((length, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 0, 9] = True
    action_mask[:, 1, 8] = True
    action_mask[:, 1, 10] = True
    return ReplayGameChunk(
        series_id=series_id,
        game_number=1,
        player=0,
        canonical_player=0,
        observations=StructuredObservation.empty_batch(length),
        action_mask=action_mask,
        mask_provenance=torch.ones(length, dtype=torch.long),
        label_kind=torch.full((length,), int(LabelKind.EXACT), dtype=torch.long),
        label_confidence=torch.ones(length),
        loss_mask=torch.ones(length),
        decision_type=torch.ones(length, dtype=torch.long),
        exact_action=torch.tensor([(7, 8)] * length, dtype=torch.long),
        candidate_values=torch.tensor([(7, 8)] * length, dtype=torch.long),
        candidate_offsets=torch.arange(length + 1, dtype=torch.long),
        outcome=torch.ones(length),
        final_observation=StructuredObservation.empty_batch(1),
        is_series_end=True,
        outcome_valid=True,
    )


class TestCudaTraining:
    @pytest.mark.heavy
    @pytest.mark.parametrize("dynamic", (False, True), ids=("static", "dynamic"))
    def test_compiled_policy_matches_eager_values_and_gradients(
        self, cuda_device: torch.device, dynamic: bool
    ) -> None:
        torch.manual_seed(37)
        eager = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        compiled = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        compiled.load_state_dict(eager.state_dict())
        compiled = compile_policy(compiled, dynamic=dynamic)
        observations = StructuredObservation.empty_batch(2).to(cuda_device)
        mask = torch.ones((2, 2, eager.act_size), device=cuda_device, dtype=torch.bool)
        actions = torch.tensor([[1, 2], [1, 2]], device=cuda_device)

        def _step(mod):
            enc = mod.encode(observations, mask)
            mem = MemoryInputs.empty(2, mod.d_model, cuda_device, enc.tokens.dtype)
            res = mod.evaluate(mod.prepare(enc, mem), mask, actions)
            (-res.log_probs.float().mean() + res.value.float().square().mean()).backward()
            return res

        eager_result = _step(eager)
        compiled_result = _step(compiled)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            compiled_result.log_probs, eager_result.log_probs, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(compiled_result.value, eager_result.value, rtol=1e-3, atol=1e-3)
        for eager_parameter, compiled_parameter in zip(
            eager.parameters(), compiled.parameters(), strict=True
        ):
            assert eager_parameter.grad is not None
            assert compiled_parameter.grad is not None
            torch.testing.assert_close(
                compiled_parameter.grad, eager_parameter.grad, rtol=1e-2, atol=1e-3
            )

    def test_rejected_minibatch_does_not_break_cuda_checkpoint_resume(
        self, cuda_device: torch.device, tmp_path: Path
    ) -> None:
        torch.manual_seed(31)
        policy = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        collected = [
            CollectedTrajectory(
                observations=StructuredObservation.empty_batch(1),
                action_masks=torch.ones((1, 2, policy.act_size), dtype=torch.bool),
                actions=torch.tensor([[1, 2]]),
                log_probs=torch.tensor([float("nan") if index == 0 else 0.0]),
                values=torch.zeros(1),
                rewards=torch.ones(1),
                dones=torch.ones(1),
                length=1,
                bootstrap_value=0.0,
                series_history=(),
            )
            for index in range(2)
        ]
        batches = prepare_trajectory_batches(collected, cuda_device, gamma=0.99, gae_lambda=0.95)
        config = TrainingConfig(
            num_episodes=20,
            n_envs=1,
            rollout_steps=1,
            batch_size=1,
            minibatch_size=1,
            ppo_epochs=1,
            target_kl=1.0e9,
            enable_optim=True,
        )
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        precision = select_optimization_precision(True, cuda_device)
        scaler = torch.amp.GradScaler("cuda", enabled=precision.grad_scaler)
        stats = ppo_update(
            batches,
            policy,
            Magnet(policy),
            optimizer,
            scaler,
            config,
            episode=0,
            alpha=0.0,
        )
        torch.cuda.synchronize()

        assert stats["optimizer_updates"] == 1
        assert all(math.isfinite(value) for value in stats.values())
        assert any(
            not torch.equal(before[name], value) for name, value in policy.state_dict().items()
        )

        checkpoint = tmp_path / "cuda-ppo.pt"
        store = CheckpointStore()
        store.save_training(
            checkpoint,
            1,
            policy,
            optimizer=optimizer,
            scaler=scaler,
            trainer_kind="ppo",
            metadata={},
        )
        restored = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scaler = torch.amp.GradScaler("cuda", enabled=precision.grad_scaler)
        assert (
            store.load_training(
                checkpoint,
                restored,
                trainer_kind="ppo",
                optimizer=restored_optimizer,
                scaler=restored_scaler,
            )
            == 1
        )
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        assert optimizer.state_dict()["state"]
        torch.testing.assert_close(
            restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0
        )
        assert restored_scaler.state_dict() == scaler.state_dict()

        resumed_before = {
            name: value.detach().clone() for name, value in restored.state_dict().items()
        }
        resumed_stats = ppo_update(
            batches[1:],
            restored,
            Magnet(restored),
            restored_optimizer,
            restored_scaler,
            config,
            episode=1,
            alpha=0.0,
        )
        assert resumed_stats["optimizer_updates"] == 1
        assert all(math.isfinite(value) for value in resumed_stats.values())
        assert any(
            not torch.equal(resumed_before[name], value)
            for name, value in restored.state_dict().items()
        )

    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    def test_bc_fragment_crosses_cpu_cuda_boundary_with_gradients(
        self, cuda_device: torch.device, dtype: torch.dtype
    ) -> None:
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA BF16 is unsupported on this device")

        torch.manual_seed(29)
        policy = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        first = _exact_label_game(3, "cuda-fragment")
        first = replace(
            first,
            is_series_end=False,
            label_kind=torch.full((3,), int(LabelKind.UNKNOWN), dtype=torch.long),
            loss_mask=torch.zeros(3),
            candidate_values=torch.empty((0, 2), dtype=torch.long),
            candidate_offsets=torch.zeros(4, dtype=torch.long),
            outcome_valid=False,
        )
        second = replace(_exact_label_game(2, "cuda-fragment"), game_number=2)
        first.observations.numerical.requires_grad_()
        second.observations.numerical.requires_grad_()
        before = {
            name: value.detach().clone() for name, value in policy.series.state_dict().items()
        }
        trainer = BCTrainer(
            policy,
            (first, second),
            BCConfig(
                batch_decisions=2,
                max_chunk_size=1,
                learning_rate=1e-3,
                enable_optim=dtype is not torch.float32,
            ),
            device=cuda_device,
        )
        trainer.precision = select_optimization_precision(
            dtype is not torch.float32, cuda_device, bf16_supported=dtype is torch.bfloat16
        )
        # Keep this transfer/gradient fixture below FP16 overflow; scaler adaptation has its own owner.
        trainer.scaler = torch.amp.GradScaler(
            "cuda", enabled=dtype is torch.float16, init_scale=16.0
        )
        metrics = trainer.train()
        torch.cuda.synchronize()

        assert metrics["games"] == 2
        assert metrics["decisions"] == 5
        assert metrics["updates"] == 1
        assert all(math.isfinite(float(value)) for value in metrics.values())
        # Game one has no own objective and its completed history must be detached.
        prefix_gradient = first.observations.numerical.grad
        assert prefix_gradient is None or not prefix_gradient.any()
        current_gradient = second.observations.numerical.grad
        assert current_gradient is not None
        assert current_gradient.isfinite().all()
        assert current_gradient.abs().sum() > 0
        assert first.observations.numerical.device.type == "cpu"
        assert any(
            not torch.equal(before[name], value)
            for name, value in policy.series.state_dict().items()
        )

    @pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_autocast_update_and_scaler_resume(
        self, cuda_device: torch.device, dtype: torch.dtype, tmp_path: Path
    ) -> None:
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA BF16 is unsupported on this device")

        torch.manual_seed(23)
        policy = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        original = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)
        observations = StructuredObservation.empty_batch(2).to(cuda_device)
        mask = torch.ones((2, 2, policy.act_size), device=cuda_device, dtype=torch.bool)
        actions = torch.tensor([[1, 2], [1, 2]], device=cuda_device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype):
            encoded = policy.encode(observations, mask)
            memory = MemoryInputs.empty(2, policy.d_model, cuda_device, encoded.tokens.dtype)
            result = policy.evaluate(policy.prepare(encoded, memory), mask, actions)
            loss = -result.log_probs.float().mean() + result.value.float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in policy.parameters()
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        assert any(
            not torch.equal(original[name], value) for name, value in policy.state_dict().items()
        )

        checkpoint = tmp_path / "cuda-training.pt"
        store = CheckpointStore()
        store.save_training(
            checkpoint,
            1,
            policy,
            optimizer=optimizer,
            scaler=scaler,
            trainer_kind="ppo",
            metadata={},
        )
        restored = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)
        episode = store.load_training(
            checkpoint,
            restored,
            trainer_kind="ppo",
            optimizer=restored_optimizer,
            scaler=restored_scaler,
        )
        assert episode == 1
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        assert optimizer.state_dict()["state"]
        torch.testing.assert_close(
            restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0
        )
        assert restored_scaler.state_dict() == scaler.state_dict()


class TestCudaTrainingLoops:
    @pytest.mark.heavy
    def test_compiled_bc_trainer_updates_with_selected_precision(
        self, cuda_device: torch.device
    ) -> None:
        torch.manual_seed(41)
        games = (_exact_label_game(5, "cuda-bc-1"), _exact_label_game(4, "cuda-bc-2"))
        policy = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        policy = compile_policy(policy)
        trainer = BCTrainer(
            policy,
            games,
            BCConfig(batch_decisions=2, learning_rate=1e-3, epochs=2, enable_optim=True),
            device=cuda_device,
        )

        metrics = trainer.train()
        torch.cuda.synchronize()

        assert metrics["games"] == 4
        assert metrics["updates"] >= 2
        assert all(math.isfinite(float(value)) for value in metrics.values())
        assert any(
            not torch.equal(before[name], value)
            for name, value in canonical_policy_state_dict(policy).items()
        )

    def test_compiled_cuda_checkpoint_loads_into_eager_cpu_policy(
        self, cuda_device: torch.device, tmp_path: Path
    ) -> None:
        torch.manual_seed(43)
        eager = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        expected = {name: value.detach().cpu() for name, value in eager.state_dict().items()}
        compiled = compile_policy(eager)
        optimizer = torch.optim.AdamW(compiled.parameters(), lr=1e-3)
        store = CheckpointStore()

        policy_path = tmp_path / "compiled-policy.pt"
        training_path = tmp_path / "compiled-training.pt"
        store.save_policy(policy_path, compiled)
        store.save_training(
            training_path,
            1,
            compiled,
            optimizer=optimizer,
            trainer_kind="ppo",
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            metadata={},
        )
        loaded = store.load_policy(policy_path, "cpu")
        restored = build_policy(MODEL_CONFIG, default_runtime_resources())
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        episode = store.load_training(
            training_path,
            restored,
            trainer_kind="ppo",
            optimizer=restored_optimizer,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
        )

        assert episode == 1
        for policy in (loaded, restored):
            assert policy.device.type == "cpu"
            assert set(policy.state_dict()) == set(expected)
            for name, value in policy.state_dict().items():
                torch.testing.assert_close(value, expected[name])

    @pytest.mark.heavy
    @pytest.mark.integration
    def test_optimized_ppo_runner_trains_and_resumes_on_cuda(
        self, cuda_device: torch.device, tmp_path: Path
    ) -> None:
        team = tmp_path / "teams"
        write_corpus_manifest(default_test_corpus(), team)
        store = CheckpointStore()
        initial = tmp_path / "initial.pt"
        store.save_policy(
            initial,
            build_policy(MODEL_CONFIG, default_runtime_resources()),
            metadata={"gamma": 0.99, "value_target_semantics": "discounted_terminal_outcome.v1"},
        )
        paths = replace(
            DEFAULT_PATHS,
            initial_policy_checkpoint=initial,
            checkpoint_path=tmp_path / "checkpoints" / "ppo.pt",
            runs_dir=tmp_path / "runs",
        )
        training = TrainingConfig(
            num_episodes=4,
            n_envs=2,
            rollout_steps=8,
            batch_size=256,
            minibatch_size=256,
            ppo_epochs=1,
            enable_optim=True,
            ramp_up_phase=0.34,
            magnet_refresh_interval=1,
        )
        config = GlobalConfig(
            training=training, paths=paths, teams=TeamsConfig(all=team, reduced=team)
        )

        run_training(
            config,
            cancel_requested=lambda: (
                paths.checkpoint_path.is_file() and store.load_episode(paths.checkpoint_path) >= 3
            ),
        )

        saved = store.read(paths.checkpoint_path)
        assert store.load_episode(saved) == 3
        metrics_path = paths.runs_dir / "ppo_training" / "metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert [record["step"] for record in metrics] == [1, 2, 3]
        assert all(record["trajectory_count"] > 0 for record in metrics)
        assert all(
            math.isfinite(float(value))
            for record in metrics
            for value in record.values()
            if isinstance(value, (int, float))
        )

        resumed = replace(
            config,
            paths=replace(
                paths, initial_policy_checkpoint=None, resume_checkpoint=paths.checkpoint_path
            ),
        )
        run_training(resumed)
        final = store.read(paths.checkpoint_path)
        assert store.load_episode(final) == 4
        resumed_metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert [record["step"] for record in resumed_metrics] == [1, 2, 3, 4]
        assert resumed_metrics[:3] == metrics
        assert resumed_metrics[-1]["optimizer_updates"] > 0
        assert any(
            not torch.equal(value, final.artifact["model_state_dict"][name])
            for name, value in saved.artifact["model_state_dict"].items()
        )
