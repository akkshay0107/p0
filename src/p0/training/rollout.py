"""Typed all-self-play rollout collection over the fixed memory channel."""

from collections.abc import Mapping

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.policy import MemoryInputs, PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.training.config import TrainingConfig
from p0.training.series_history import SeriesHistoryStore
from p0.training.trajectory import (
    CollectedTrajectory,
    PreparedTrajectory,
    TrajectoryStorage,
    prepare_trajectory_batches,
)
from p0.training.utils import select_optimization_precision
from p0.training.vector_env import ThreadVecEnv

ACT_SIZE = FORMAT.action_size
MAX_TRAJECTORY_STEPS = 200
SERIES_HISTORY_FLUSH_STEPS = 48

__all__ = [
    "BattleMemoryBuffer",
    "RolloutCollector",
]


def _terminal_action_mask(
    info: Mapping[str, object], key: str, device: torch.device
) -> torch.Tensor:
    """Return one validated two-slot mask saved before a vector-env reset."""
    raw_mask = info.get(key)
    if raw_mask is None:
        raise RuntimeError(f"Truncated rollout info is missing {key}")

    mask = torch.as_tensor(raw_mask, device=device, dtype=torch.bool)
    if mask.shape != (2, ACT_SIZE):
        raise ValueError(f"Expected {key} to have shape (2, {ACT_SIZE}), got {tuple(mask.shape)}")
    return mask


class BattleMemoryBuffer:
    """Fixed-capacity per-battle history stored on the policy device."""

    def __init__(
        self,
        n_envs: int,
        d_model: int,
        max_steps: int = MAX_TRAJECTORY_STEPS,
        device: torch.device | str = "cpu",
    ) -> None:
        if n_envs <= 0 or d_model <= 0 or max_steps <= 0:
            raise ValueError("History dimensions must be positive")

        self.tokens = torch.zeros(
            (n_envs, max_steps, d_model),
            device=device,
            dtype=torch.float32,
        )
        self.step_counts = torch.zeros(n_envs, dtype=torch.long)
        self._history_offsets = torch.arange(-HISTORY_WINDOW, 0, device=device)
        self.d_model = d_model
        self.max_steps = max_steps

    def append(self, env_ids: torch.Tensor, history_tokens: torch.Tensor) -> None:
        if history_tokens.shape != (env_ids.numel(), self.d_model):
            raise ValueError("history token batch does not match selected environments")
        if history_tokens.device != self.tokens.device:
            raise ValueError("history tokens must already be on the history buffer device")

        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        steps_cpu = self.step_counts[env_ids_cpu]
        if bool(torch.any(steps_cpu >= self.max_steps).item()):
            raise OverflowError(f"Battle history exceeded {self.max_steps} decisions")

        # The policy returns the pre-memory local summary here, not the
        # post-memory cls readout. Store detached snapshots so this rollout
        # cache does not connect autograd graphs across environment steps.
        env_ids_device = env_ids_cpu.to(self.tokens.device)
        steps_device = steps_cpu.to(self.tokens.device)
        self.tokens[env_ids_device, steps_device] = history_tokens.detach().to(torch.float32)
        self.step_counts[env_ids_cpu] += 1

    def reset(self, env_id: int) -> None:
        """Reset one game's history."""
        self.step_counts[env_id] = 0

    def clear(self) -> None:
        """Reset all per-battle histories at a checkpoint boundary."""
        self.step_counts.zero_()

    def inputs(
        self,
        env_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        env_ids_device = env_ids_cpu.to(self.tokens.device)
        lengths = self.step_counts[env_ids_cpu].to(self.tokens.device)
        indices = lengths.unsqueeze(1) + self._history_offsets.unsqueeze(0)
        mask = indices >= 0
        history = self.tokens[
            env_ids_device.unsqueeze(1),
            indices.clamp_min(0),
        ].masked_fill(~mask.unsqueeze(-1), 0.0)
        return (
            history.to(device=device, dtype=dtype),
            mask.to(device),
        )

    def full_values(self, env_id: int) -> torch.Tensor | None:
        """Return one complete device-resident history before it is reset."""
        length = int(self.step_counts[env_id].item())
        if not length:
            return None
        return self.tokens[env_id, :length].unsqueeze(0)


class RolloutCollector:
    """Own fixed-window memory and active trajectory storage for all self-play."""

    def __init__(
        self,
        vector_env: ThreadVecEnv,
        policy: PolicyNet,
        config: TrainingConfig,
        *,
        max_trajectory_steps: int = MAX_TRAJECTORY_STEPS,
    ) -> None:
        self.vector_env = vector_env
        self.policy = policy
        self.config = config
        self.completed_trajectories: list[CollectedTrajectory] = []
        self.first = TrajectoryStorage.allocate(config.n_envs, max_trajectory_steps)
        self.second = TrajectoryStorage.allocate(config.n_envs, max_trajectory_steps)
        self.memory1 = BattleMemoryBuffer(
            config.n_envs,
            policy.d_model,
            max_steps=max_trajectory_steps,
            device=policy.device,
        )
        self.memory2 = BattleMemoryBuffer(
            config.n_envs,
            policy.d_model,
            max_steps=max_trajectory_steps,
            device=policy.device,
        )
        self.series_store1 = SeriesTokenStore(policy.d_model)
        self.series_store2 = SeriesTokenStore(policy.d_model)
        self.series_history1 = SeriesHistoryStore(policy.d_model)
        self.series_history2 = SeriesHistoryStore(policy.d_model)
        self._pending_history1: list[torch.Tensor] = []
        self._pending_history2: list[torch.Tensor] = []

    @torch.inference_mode()
    def collect(self) -> None:
        """Collect one all-self-play rollout using explicit fixed-window memory."""
        vec_env = self.vector_env
        policy = self.policy
        completed_trajectories = self.completed_trajectories
        config = self.config
        trajectories1 = self.first
        trajectories2 = self.second
        memory1 = self.memory1
        memory2 = self.memory2
        series_store1 = self.series_store1
        series_store2 = self.series_store2
        series_history1 = self.series_history1
        series_history2 = self.series_history2
        pending_history1 = self._pending_history1
        pending_history2 = self._pending_history2

        n_envs = vec_env.n_envs
        device = policy.device
        idx_all = torch.arange(n_envs)
        masks1 = vec_env.last_masks1
        masks2 = vec_env.last_masks2

        for _ in range(config.rollout_steps):
            obs1_gpu = vec_env.get_batched_obs1(device)
            mask1_gpu = torch.from_numpy(masks1).to(device, non_blocking=True)
            obs2_gpu = vec_env.get_batched_obs2(device)
            mask2_gpu = torch.from_numpy(masks2).to(device, non_blocking=True)

            current_obs = StructuredObservation.cat([obs1_gpu, obs2_gpu])
            current_mask = torch.cat([mask1_gpu, mask2_gpu])

            infos = vec_env.last_infos
            assert infos is not None

            # Positional results must describe the corresponding environment.
            if len(infos) != n_envs:
                raise ValueError(f"Expected {n_envs} environment infos, got {len(infos)}")
            series_ids = [str(info["series_id"]) for info in infos]
            series_tokens1, series_mask1 = series_store1.get_tokens(series_ids, device)
            series_tokens2, series_mask2 = series_store2.get_tokens(series_ids, device)
            current_series_tokens = torch.cat([series_tokens1, series_tokens2], dim=0)
            current_series_mask = torch.cat([series_mask1, series_mask2], dim=0)

            memory1_inputs = memory1.inputs(idx_all, device, torch.float32)
            memory2_inputs = memory2.inputs(idx_all, device, torch.float32)
            current_memory = MemoryInputs(
                series_tokens=current_series_tokens,
                series_mask=current_series_mask,
                history_tokens=torch.cat([memory1_inputs[0], memory2_inputs[0]], dim=0),
                history_mask=torch.cat([memory1_inputs[1], memory2_inputs[1]], dim=0),
            )

            precision = select_optimization_precision(config.enable_optim, device)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=precision.autocast,
                dtype=precision.dtype,
            ):
                current_out = policy.act(
                    policy.prepare(policy.encode(current_obs, current_mask), current_memory),
                    current_mask,
                )

            memory1.append(idx_all, current_out.history_token[:n_envs])
            memory2.append(idx_all, current_out.history_token[n_envs:])
            pending_history1.append(current_out.history_token[:n_envs].detach())
            pending_history2.append(current_out.history_token[n_envs:].detach())

            actions_cpu = current_out.actions.to(device="cpu", dtype=torch.long)
            log_probs_cpu = current_out.log_probs.to(device="cpu", dtype=torch.float32)
            values_cpu = current_out.value.to(device="cpu", dtype=torch.float32)

            actions1_cpu = actions_cpu[:n_envs]
            actions2_cpu = actions_cpu[n_envs:]
            log_probs1_cpu = log_probs_cpu[:n_envs]
            log_probs2_cpu = log_probs_cpu[n_envs:]
            values1_cpu = values_cpu[:n_envs]
            values2_cpu = values_cpu[n_envs:]

            obs1_cpu = vec_env.obs1_buffers
            obs2_cpu = vec_env.obs2_buffers

            s1 = trajectories1.record(
                idx_all,
                obs1_cpu,
                actions1_cpu,
                log_probs1_cpu,
                values1_cpu,
                torch.from_numpy(masks1).to(torch.bool),
            )
            s2 = trajectories2.record(
                idx_all,
                obs2_cpu,
                actions2_cpu,
                log_probs2_cpu,
                values2_cpu,
                torch.from_numpy(masks2).to(torch.bool),
            )

            env_actions = [
                {
                    vec_env.envs[i].agent1.username: actions1_cpu[i].numpy(),
                    vec_env.envs[i].agent2.username: actions2_cpu[i].numpy(),
                }
                for i in range(n_envs)
            ]

            next_masks1, next_masks2, rewards1, rewards2, done_status, infos = vec_env.step(
                env_actions
            )

            # RL non-terminal masking only cares if the episode naturally terminated.
            # Status 1 is Terminated, Status 2 is Truncated.
            game_done = (done_status == 1).astype(np.float32)

            trajectories1.rewards[idx_all, s1] = torch.from_numpy(rewards1)
            trajectories1.dones[idx_all, s1] = torch.from_numpy(game_done)
            trajectories2.rewards[idx_all, s2] = torch.from_numpy(rewards2)
            trajectories2.dones[idx_all, s2] = torch.from_numpy(game_done)

            history_snapshots1: list[tuple[torch.Tensor, ...]] = [()] * n_envs
            history_snapshots2: list[tuple[torch.Tensor, ...]] = [()] * n_envs
            should_flush_history = len(pending_history1) >= SERIES_HISTORY_FLUSH_STEPS or bool(
                np.any(done_status)
            )
            for i in range(n_envs):
                history_key = str(series_ids[i])
                if done_status[i]:
                    history_snapshots1[i] = series_history1.snapshot(history_key)
                    history_snapshots2[i] = series_history2.snapshot(history_key)

            if should_flush_history:
                pending_values1 = torch.stack(pending_history1).to(
                    device="cpu", dtype=torch.float32
                )
                pending_values2 = torch.stack(pending_history2).to(
                    device="cpu", dtype=torch.float32
                )
                for i in range(n_envs):
                    history_key = str(series_ids[i])
                    is_game_end = bool(done_status[i])
                    is_series_end = is_game_end and bool(infos[i].get("series_complete"))
                    game_number1 = series_history1.next_game_number(history_key)
                    game_number2 = series_history2.next_game_number(history_key)
                    series_history1.append(
                        history_key,
                        game_number1,
                        pending_values1[:, i],
                        is_game_end=is_game_end,
                        is_series_end=is_series_end,
                    )
                    series_history2.append(
                        history_key,
                        game_number2,
                        pending_values2[:, i],
                        is_game_end=is_game_end,
                        is_series_end=is_series_end,
                    )
                pending_history1.clear()
                pending_history2.clear()

            for i in range(n_envs):
                if not done_status[i]:
                    continue

                info = infos[i]
                bootstrap_value1 = 0.0
                bootstrap_value2 = 0.0

                if done_status[i] == 2:
                    # truncated case
                    term1: StructuredObservation = info.get("terminal_observation1")  # type: ignore
                    term2: StructuredObservation = info.get("terminal_observation2")  # type: ignore

                    term1 = term1.unsqueeze(0).to(device)
                    term2 = term2.unsqueeze(0).to(device)
                    t_obs = StructuredObservation.cat([term1, term2])
                    terminal_mask = torch.stack(
                        (
                            _terminal_action_mask(info, "terminal_action_mask1", device),
                            _terminal_action_mask(info, "terminal_action_mask2", device),
                        ),
                        dim=0,
                    )

                    idx_tensor = torch.tensor([i], dtype=torch.long)
                    mem1 = memory1.inputs(idx_tensor, device, torch.float32)
                    mem2 = memory2.inputs(idx_tensor, device, torch.float32)

                    s_tok1, s_mask1 = series_store1.get_tokens([str(series_ids[i])], device)
                    s_tok2, s_mask2 = series_store2.get_tokens([str(series_ids[i])], device)
                    t_s_tok = torch.cat([s_tok1, s_tok2], dim=0)
                    t_s_mask = torch.cat([s_mask1, s_mask2], dim=0)

                    with (
                        torch.no_grad(),
                        torch.amp.autocast(
                            device_type=device.type,
                            enabled=precision.autocast,
                            dtype=precision.dtype,
                        ),
                    ):
                        t_memory = MemoryInputs(
                            series_tokens=t_s_tok,
                            series_mask=t_s_mask,
                            history_tokens=torch.cat([mem1[0], mem2[0]], dim=0),
                            history_mask=torch.cat([mem1[1], mem2[1]], dim=0),
                        )
                        t_out = policy.act(
                            policy.prepare(policy.encode(t_obs, terminal_mask), t_memory),
                            terminal_mask,
                        )
                        bootstrap_value1 = t_out.value[0].item()
                        bootstrap_value2 = t_out.value[1].item()

                history_key = str(series_ids[i])
                history_snapshot1 = history_snapshots1[i]
                history_snapshot2 = history_snapshots2[i]

                # The artificial truncation value sees only the current game. Commit
                # its summary after inference so the game is not represented twice.
                val1 = memory1.full_values(i)
                if val1 is not None:
                    with torch.no_grad():
                        val1_mask = torch.ones(val1.shape[:2], dtype=torch.bool, device=val1.device)
                        new_tok1 = policy.series(val1, val1_mask)[0]
                    series_store1.append(str(series_ids[i]), new_tok1)

                val2 = memory2.full_values(i)
                if val2 is not None:
                    with torch.no_grad():
                        val2_mask = torch.ones(val2.shape[:2], dtype=torch.bool, device=val2.device)
                        new_tok2 = policy.series(val2, val2_mask)[0]
                    series_store2.append(str(series_ids[i]), new_tok2)

                if info.get("series_complete"):
                    series_store1.drop(series_ids[i])
                    series_store2.drop(series_ids[i])
                    series_history1.drop(history_key)
                    series_history2.drop(history_key)

                completed_trajectories.append(
                    trajectories1.complete(i, bootstrap_value1, history_snapshot1)
                )
                completed_trajectories.append(
                    trajectories2.complete(i, bootstrap_value2, history_snapshot2)
                )

                memory1.reset(i)
                memory2.reset(i)

            masks1 = next_masks1
            masks2 = next_masks2

    def reset_completed(self) -> None:
        self.completed_trajectories.clear()

    def prepare_for_checkpoint(self) -> None:
        """Restart at a clean battle boundary before persisting PPO state."""
        self.completed_trajectories.clear()
        self.first.step_counts.zero_()
        self.second.step_counts.zero_()
        self.memory1.clear()
        self.memory2.clear()
        self._pending_history1.clear()
        self._pending_history2.clear()
        self.series_history1.discard_active_games()
        self.series_history2.discard_active_games()
        for env in self.vector_env.envs:
            env.prepare_for_checkpoint()
        self.vector_env.reset()

    def training_state(self) -> dict[str, object]:
        """Capture cross-game context retained by the collector."""
        return {
            "series_store1": self.series_store1.training_state(),
            "series_store2": self.series_store2.training_state(),
            "series_history1": self.series_history1.training_state(),
            "series_history2": self.series_history2.training_state(),
        }

    def restore_training_state(self, state: Mapping[str, object]) -> None:
        """Restore cross-game context captured at an episode boundary."""
        store1 = state.get("series_store1")
        store2 = state.get("series_store2")
        if not isinstance(store1, Mapping) or not isinstance(store2, Mapping):
            raise ValueError("Invalid PPO collector series training state")
        self.series_store1.restore_training_state(store1)
        self.series_store2.restore_training_state(store2)
        self._pending_history1.clear()
        self._pending_history2.clear()
        history1 = state.get("series_history1")
        history2 = state.get("series_history2")
        if not isinstance(history1, Mapping) or not isinstance(history2, Mapping):
            raise ValueError("Invalid PPO collector history training state")
        self.series_history1.restore_training_state(history1)
        self.series_history2.restore_training_state(history2)

    def get_batches(self, device: torch.device) -> list[PreparedTrajectory]:
        return prepare_trajectory_batches(
            self.completed_trajectories,
            device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
