"""Opt-in CUDA checks for BC history, precision, and checkpoint recovery."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from p0.battle.series import SeriesPerspectiveKey
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.policy import MemoryInputs
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training._bc_batch import BCGameWindow
from p0.training._bc_history import prepare_series_context
from p0.training.checkpoint import CheckpointStore
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import ppo_update
from p0.training.series_history import SeriesHistoryStore
from p0.training.trajectory import CollectedTrajectory, prepare_trajectory_batches
from p0.training.utils import select_optimization_precision

pytestmark = pytest.mark.gpu
MODEL_CONFIG = ModelConfig(32, 4, 1, 64)


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda")


class TestCudaTraining:
    @pytest.mark.heavy
    def test_compiled_policy_matches_eager_values_and_gradients(
        self, cuda_device: torch.device
    ) -> None:
        torch.manual_seed(37)
        eager = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        compiled = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        compiled.load_state_dict(eager.state_dict())
        compiled = compile_policy(compiled, dynamic=False)
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
            cancel_requested=lambda: False,
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
            checkpoint, 1, policy, optimizer=optimizer, scaler=scaler, trainer_kind="ppo"
        )
        restored = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scaler = torch.amp.GradScaler("cuda", enabled=precision.grad_scaler)
        assert (
            store.load_training(
                checkpoint,
                restored,
                optimizer=restored_optimizer,
                scaler=restored_scaler,
                expected_trainer_kind="ppo",
                require_training_state=True,
            )
            == 1
        )
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    @pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    def test_bc_fragment_crosses_cpu_cuda_boundary_with_gradients(
        self, cuda_device: torch.device, dtype: torch.dtype
    ) -> None:
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA BF16 is unsupported on this device")

        policy = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        key = SeriesPerspectiveKey("cuda-fragment", 0)
        store = SeriesHistoryStore(d_model=policy.d_model)
        store.append(
            key,
            1,
            torch.full((2, policy.d_model), 0.25),
            is_game_end=False,
            is_series_end=False,
        )
        current = torch.randn((2, policy.d_model), device=cuda_device, dtype=dtype)
        current.requires_grad_()
        windows = (
            BCGameWindow(key, 1, 0, 1, True, False, 1),
            BCGameWindow(key, 2, 1, 2, False, False, 2),
        )

        with torch.amp.autocast(
            "cuda",
            enabled=dtype is not torch.float32,
            dtype=dtype if dtype is not torch.float32 else torch.float16,
        ):
            context, mask = prepare_series_context(
                store, windows, (current[0:1], current[1:2]), current, policy.series
            )
            loss = context[1].float().square().sum()
        loss.backward()
        torch.cuda.synchronize()

        assert context.device == cuda_device
        assert mask.device == cuda_device
        assert context.isfinite().all()
        assert current.grad is not None
        assert current.grad[0].isfinite().all()
        assert current.grad[0].abs().sum() > 0
        assert current.grad[1].abs().sum() == 0
        assert store.planning_state(key)[2][0].device.type == "cpu"

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
        )
        restored = build_policy(MODEL_CONFIG, default_runtime_resources()).to(cuda_device)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scaler = torch.amp.GradScaler("cuda", enabled=dtype is torch.float16)
        episode = store.load_training(
            checkpoint,
            restored,
            optimizer=restored_optimizer,
            scaler=restored_scaler,
            expected_trainer_kind="ppo",
            require_training_state=True,
        )
        assert episode == 1
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
        assert restored_scaler.state_dict() == scaler.state_dict()
