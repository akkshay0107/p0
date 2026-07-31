"""Typed all-self-play rollout collection over the fixed memory channel."""

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.cls_reducer import pack_history_tokens
from p0.model.policy import PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.training.config import TrainingConfig
from p0.training.trajectory import (
    TrajectoryBatch,
    TrajectoryStorage,
    prepare_trajectory_batches,
)
from p0.training.utils import amp_enabled
from p0.training.vector_env import ThreadVecEnv

ACT_SIZE = FORMAT.action_size
MAX_TRAJECTORY_STEPS = 200

__all__ = [
    "BattleMemoryBuffer",
    "RolloutCollector",
    "collect_rollouts",
]


class BattleMemoryBuffer:
    """Explicit per-battle immutable local-token storage."""

    def __init__(self, n_envs: int, d_model: int):
        self.tokens: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
        self.d_model = d_model

    def append(self, env_ids: torch.Tensor, history_tokens: torch.Tensor) -> None:
        if history_tokens.shape != (env_ids.numel(), self.d_model):
            raise ValueError("history token batch does not match selected environments")
        for env_id, token in zip(env_ids.tolist(), history_tokens, strict=True):
            entries = self.tokens[env_id]
            entries.append(token.detach().to(device="cpu", dtype=torch.float32))
            if len(entries) > HISTORY_WINDOW:
                del entries[0]

    def reset(self, env_id: int) -> None:
        """Reset one game's history."""
        self.tokens[env_id].clear()

    def inputs(
        self,
        env_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = []
        masks = []
        ages = []
        for env_id in env_ids.tolist():
            current = self.tokens[env_id]
            values = (
                torch.stack(current).to(device=device, dtype=dtype).unsqueeze(0)
                if current
                else torch.zeros((1, 0, self.d_model), device=device, dtype=dtype)
            )
            packed, mask, age = pack_history_tokens(values)
            history.append(packed[0])
            masks.append(mask[0])
            ages.append(age[0])
        return (
            torch.stack(history),
            torch.stack(masks),
            torch.stack(ages),
        )


@torch.inference_mode()
def collect_rollouts(
    vec_env: ThreadVecEnv,
    policy: PolicyNet,
    completed_trajectories: list[TrajectoryBatch],
    config: TrainingConfig,
    trajectories1: TrajectoryStorage,
    trajectories2: TrajectoryStorage,
    memory1: BattleMemoryBuffer,
    memory2: BattleMemoryBuffer,
    series_store1: SeriesTokenStore,
    series_store2: SeriesTokenStore,
) -> None:
    """Collect one all-self-play rollout using explicit fixed-window memory.

    Arguments:
        vec_env: Batched self-play environments.
        policy: Policy used for both player seats.
        completed_trajectories: Destination for completed trajectories.
        config: Rollout length and device optimization settings.
        trajectories1: Active trajectory storage for the first seat.
        trajectories2: Active trajectory storage for the second seat.
        memory1: Per-battle history for the first seat.
        memory2: Per-battle history for the second seat.
        series_store1: Persistent cross-episode tokens for the first seat.
        series_store2: Persistent cross-episode tokens for the second seat.

    Returns:
        None.
    """
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

        series_ids = [str(info["series_id"]) for info in infos if info]
        series_tokens1, series_mask1 = series_store1.get_tokens(series_ids, device)
        series_tokens2, series_mask2 = series_store2.get_tokens(series_ids, device)
        current_series_tokens = torch.cat([series_tokens1, series_tokens2], dim=0)
        current_series_mask = torch.cat([series_mask1, series_mask2], dim=0)

        memory1_inputs = memory1.inputs(idx_all, device, torch.float32)
        memory2_inputs = memory2.inputs(idx_all, device, torch.float32)
        current_memory = tuple(
            torch.cat([first, second], dim=0)
            for first, second in zip(memory1_inputs, memory2_inputs, strict=True)
        )

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled(config, device)):
            current_out = policy.act_obs(
                current_obs,
                current_mask,
                current_series_tokens,
                current_series_mask,
                *current_memory,
            )

        memory1.append(idx_all, current_out.history_token[:n_envs])
        memory2.append(idx_all, current_out.history_token[n_envs:])

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
            series_tokens1.cpu(),
            series_mask1.cpu(),
        )
        s2 = trajectories2.record(
            idx_all,
            obs2_cpu,
            actions2_cpu,
            log_probs2_cpu,
            values2_cpu,
            torch.from_numpy(masks2).to(torch.bool),
            series_tokens2.cpu(),
            series_mask2.cpu(),
        )

        env_actions = [
            {
                vec_env.envs[i].agent1.username: actions1_cpu[i].numpy(),
                vec_env.envs[i].agent2.username: actions2_cpu[i].numpy(),
            }
            for i in range(n_envs)
        ]

        next_masks1, next_masks2, rewards1, rewards2, dones, _ = vec_env.step(env_actions)
        trajectories1.rewards[idx_all, s1] = torch.from_numpy(rewards1)
        trajectories1.dones[idx_all, s1] = torch.from_numpy(dones.astype(np.float32))
        trajectories2.rewards[idx_all, s2] = torch.from_numpy(rewards2)
        trajectories2.dones[idx_all, s2] = torch.from_numpy(dones.astype(np.float32))

        for i in range(n_envs):
            if not dones[i]:
                continue

            hist1 = memory1.tokens[i]
            if hist1:
                val1 = torch.stack(hist1).unsqueeze(0).to(device)
                with torch.no_grad():
                    new_tok1 = policy.series.resample_single_game(val1)[0]
                series_store1.append(str(series_ids[i]), new_tok1)

            hist2 = memory2.tokens[i]
            if hist2:
                val2 = torch.stack(hist2).unsqueeze(0).to(device)
                with torch.no_grad():
                    new_tok2 = policy.series.resample_single_game(val2)[0]
                series_store2.append(str(series_ids[i]), new_tok2)

            if infos[i] and infos[i].get("series_complete"):
                series_store1.drop(str(series_ids[i]))
                series_store2.drop(str(series_ids[i]))

            completed_trajectories.append(trajectories1.complete(i))
            completed_trajectories.append(trajectories2.complete(i))

            memory1.reset(i)
            memory2.reset(i)

        masks1 = next_masks1
        masks2 = next_masks2


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
        self.completed_trajectories: list[TrajectoryBatch] = []
        self.first = TrajectoryStorage.allocate(config.n_envs, max_trajectory_steps, policy.d_model)
        self.second = TrajectoryStorage.allocate(config.n_envs, max_trajectory_steps, policy.d_model)
        self.memory1 = BattleMemoryBuffer(config.n_envs, policy.d_model)
        self.memory2 = BattleMemoryBuffer(config.n_envs, policy.d_model)
        self.series_store1 = SeriesTokenStore(policy.d_model)
        self.series_store2 = SeriesTokenStore(policy.d_model)

    def collect(self) -> None:
        collect_rollouts(
            self.vector_env,
            self.policy,
            self.completed_trajectories,
            self.config,
            self.first,
            self.second,
            self.memory1,
            self.memory2,
            self.series_store1,
            self.series_store2,
        )

    def reset_completed(self) -> None:
        self.completed_trajectories.clear()

    def get_batches(self, device: torch.device) -> list[TrajectoryBatch]:
        return prepare_trajectory_batches(
            self.completed_trajectories,
            device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
