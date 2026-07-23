"""Typed all-self-play rollout collection over the fixed memory channel."""

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW, SERIES_SLOTS
from p0.model.cls_reducer import pack_history_tokens
from p0.model.policy import PolicyNet
from p0.model.structured_observation import StructuredObservation
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
    "RolloutBuffer",
    "BattleMemoryBuffer",
    "RolloutCollector",
    "collect_rollouts",
]


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectories: list[TrajectoryBatch] = []

    def add_episode(self, episode: TrajectoryBatch):
        self.trajectories.append(episode)

    def get_batches(self, device: torch.device, config: TrainingConfig):
        return prepare_trajectory_batches(
            self.trajectories,
            device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        batch_size = env_ids.numel()
        return (
            torch.zeros(
                (batch_size, SERIES_SLOTS, self.d_model),
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                (batch_size, SERIES_SLOTS),
                device=device,
                dtype=torch.bool,
            ),
            torch.stack(history),
            torch.stack(masks),
            torch.stack(ages),
        )


@torch.inference_mode()
def collect_rollouts(
    vec_env: ThreadVecEnv,
    policy: PolicyNet,
    buffer: RolloutBuffer,
    config: TrainingConfig,
    trajectories1: TrajectoryStorage,
    trajectories2: TrajectoryStorage,
    memory1: BattleMemoryBuffer,
    memory2: BattleMemoryBuffer,
) -> None:
    """Collect one all-self-play rollout using explicit fixed-window memory.

    Arguments:
        vec_env: Batched self-play environments.
        policy: Policy used for both player seats.
        buffer: Destination for completed trajectories.
        config: Rollout length and device optimization settings.
        trajectories1: Active trajectory storage for the first seat.
        trajectories2: Active trajectory storage for the second seat.
        memory1: Per-battle history for the first seat.
        memory2: Per-battle history for the second seat.

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
        memory1_inputs = memory1.inputs(idx_all, device, torch.float32)
        memory2_inputs = memory2.inputs(idx_all, device, torch.float32)
        current_memory = tuple(
            torch.cat([first, second], dim=0)
            for first, second in zip(memory1_inputs, memory2_inputs, strict=True)
        )
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled(config, device)):
            current_out = policy.act_obs(current_obs, current_mask, *current_memory)

        actions1 = current_out.actions[:n_envs]
        actions2 = current_out.actions[n_envs:]
        log_probs1 = current_out.log_probs[:n_envs]
        log_probs2 = current_out.log_probs[n_envs:]
        values1 = current_out.value[:n_envs]
        values2 = current_out.value[n_envs:]
        memory1.append(idx_all, current_out.history_token[:n_envs])
        memory2.append(idx_all, current_out.history_token[n_envs:])

        actions1_cpu = actions1.to(device="cpu", dtype=torch.long)
        actions2_cpu = actions2.to(device="cpu", dtype=torch.long)
        log_probs1_cpu = log_probs1.to(device="cpu", dtype=torch.float32)
        log_probs2_cpu = log_probs2.to(device="cpu", dtype=torch.float32)
        values1_cpu = values1.to(device="cpu", dtype=torch.float32)
        values2_cpu = values2.to(device="cpu", dtype=torch.float32)
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

        next_masks1, next_masks2, rewards1, rewards2, dones, _ = vec_env.step(env_actions)
        trajectories1.rewards[idx_all, s1] = torch.from_numpy(rewards1)
        trajectories1.dones[idx_all, s1] = torch.from_numpy(dones.astype(np.float32))
        trajectories2.rewards[idx_all, s2] = torch.from_numpy(rewards2)
        trajectories2.dones[idx_all, s2] = torch.from_numpy(dones.astype(np.float32))

        for i in range(n_envs):
            if not dones[i]:
                continue
            buffer.add_episode(trajectories1.complete(i))
            buffer.add_episode(trajectories2.complete(i))
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
        self.buffer = RolloutBuffer()
        self.first = TrajectoryStorage.allocate(config.n_envs, max_trajectory_steps)
        self.second = TrajectoryStorage.allocate(config.n_envs, max_trajectory_steps)
        self.memory1 = BattleMemoryBuffer(config.n_envs, policy.d_model)
        self.memory2 = BattleMemoryBuffer(config.n_envs, policy.d_model)

    def collect(self) -> None:
        collect_rollouts(
            self.vector_env,
            self.policy,
            self.buffer,
            self.config,
            self.first,
            self.second,
            self.memory1,
            self.memory2,
        )

    def reset_completed(self) -> None:
        self.buffer.reset()
