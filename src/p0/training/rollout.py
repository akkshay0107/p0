"""Typed rollout collection over injected runtime and league services."""

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW, SERIES_SLOTS
from p0.model.cls_reducer import pack_history_tokens
from p0.model.policy import PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.training.config import TrainingConfig
from p0.training.league.league import OpponentPool
from p0.training.league.selection import (
    EnvPartition,
    assign_pool_opponents,
    build_partition,
)
from p0.training.trajectory import (
    TrajectoryBatch,
    TrajectoryStorage,
    prepare_trajectory_batches,
)
from p0.training.vector_env import ThreadVecEnv

ACT_SIZE = FORMAT.action_size
MAX_TRAJECTORY_STEPS = 200

__all__ = [
    "EnvPartition",
    "RolloutBuffer",
    "BattleMemoryBuffer",
    "assign_pool_opponents",
    "build_partition",
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
        self.series_tokens: list[torch.Tensor | None] = [None for _ in range(n_envs)]
        self.series_masks: list[torch.Tensor | None] = [None for _ in range(n_envs)]
        self.d_model = d_model

    def append(self, env_ids: torch.Tensor, history_tokens: torch.Tensor) -> None:
        if history_tokens.shape != (env_ids.numel(), self.d_model):
            raise ValueError("history token batch does not match selected environments")
        for env_id, token in zip(env_ids.tolist(), history_tokens, strict=True):
            entries = self.tokens[env_id]
            entries.append(token.detach().to(torch.float32))
            if len(entries) > HISTORY_WINDOW:
                del entries[0]

    def reset(self, env_id: int) -> None:
        """Reset one game's history while preserving any active series context."""
        self.tokens[env_id].clear()

    def set_series(
        self,
        env_ids: torch.Tensor,
        series_tokens: torch.Tensor,
        series_mask: torch.Tensor,
    ) -> None:
        if series_tokens.shape != (env_ids.numel(), SERIES_SLOTS, self.d_model):
            raise ValueError("series token batch does not match the fixed two-slot contract")
        if series_mask.shape != (env_ids.numel(), SERIES_SLOTS):
            raise ValueError("series mask batch does not match the fixed two-slot contract")
        for env_id, tokens, mask in zip(env_ids.tolist(), series_tokens, series_mask, strict=True):
            self.series_tokens[env_id] = tokens.detach().to(torch.float32).clone()
            self.series_masks[env_id] = mask.detach().bool().clone()

    def clear_series(self, env_id: int) -> None:
        self.series_tokens[env_id] = None
        self.series_masks[env_id] = None

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
        series = []
        series_masks = []
        for env_id in env_ids.tolist():
            stored_tokens = self.series_tokens[env_id]
            stored_mask = self.series_masks[env_id]
            if stored_tokens is None or stored_mask is None:
                series.append(torch.zeros((SERIES_SLOTS, self.d_model), device=device, dtype=dtype))
                series_masks.append(torch.zeros((SERIES_SLOTS,), device=device, dtype=torch.bool))
            else:
                series.append(stored_tokens.to(device=device, dtype=dtype))
                series_masks.append(stored_mask.to(device=device))
        return (
            torch.stack(series),
            torch.stack(series_masks),
            torch.stack(history),
            torch.stack(masks),
            torch.stack(ages),
        )


@torch.inference_mode()
def collect_rollouts(
    vec_env: ThreadVecEnv,
    policy: PolicyNet,
    buffer: RolloutBuffer,
    pool: OpponentPool,
    config: TrainingConfig,
    active_pool_policies: dict[str, PolicyNet],
    trajectories1: TrajectoryStorage,
    trajectories2: TrajectoryStorage,
    memory1: BattleMemoryBuffer,
    memory2: BattleMemoryBuffer,
    partition: EnvPartition,
) -> tuple[int, int]:
    """
    Collect one rollout with statically assigned self-play and pool-play environments.
    Pool snapshots change only after that environment completes a battle.
    """
    n_envs = vec_env.n_envs
    device = policy.device

    masks1 = vec_env.last_masks1
    masks2 = vec_env.last_masks2

    pool_wins = 0
    pool_total = 0

    step_counts2 = trajectories2.step_counts

    idx_all = torch.arange(n_envs)
    self_idx_cpu = partition.self_mask_cpu.nonzero().squeeze(-1)
    next_pool_opponents: dict[int, str] = {}
    if partition.pool_idx.numel() > 0:
        roster = pool.sample_many(config.n_pool_opponents)
        assignments = assign_pool_opponents(partition.pool_idx.numel(), roster)
        next_pool_opponents = dict(zip(partition.pool_idx.tolist(), assignments, strict=True))

    for opponent_id, _ in partition.pool_groups():
        if opponent_id not in active_pool_policies:
            active_pool_policies[opponent_id] = pool.load_policy(opponent_id, str(device))
    for opponent_id in set(next_pool_opponents.values()):
        if opponent_id not in active_pool_policies:
            active_pool_policies[opponent_id] = pool.load_policy(opponent_id, str(device))

    for _ in range(config.rollout_steps):
        obs1_gpu = vec_env.get_batched_obs1(device)
        mask1_gpu = torch.from_numpy(masks1).to(device, non_blocking=True)
        obs2_gpu = vec_env.get_batched_obs2(device)
        mask2_gpu = torch.from_numpy(masks2).to(device, non_blocking=True)

        current_obs = StructuredObservation.cat(
            [obs1_gpu, obs2_gpu[partition.self_idx]],
        )
        current_mask = torch.cat([mask1_gpu, mask2_gpu[partition.self_idx]])
        memory1_inputs = memory1.inputs(torch.arange(n_envs), device, torch.float32)
        memory2_self_inputs = memory2.inputs(partition.self_idx, device, torch.float32)
        current_memory = tuple(
            torch.cat([first, second], dim=0)
            for first, second in zip(memory1_inputs, memory2_self_inputs, strict=True)
        )
        with torch.amp.autocast(device_type=device.type, enabled=config.enable_optim):
            current_out = policy.act_obs(current_obs, current_mask, *current_memory)

        actions1 = current_out.actions[:n_envs]
        log_probs1 = current_out.log_probs[:n_envs]
        values1 = current_out.value[:n_envs]

        actions2 = torch.zeros_like(actions1)
        log_probs2 = torch.zeros_like(log_probs1)
        values2 = torch.zeros_like(values1)

        if partition.self_idx.numel() > 0:
            self_slice = slice(n_envs, None)
            actions2[partition.self_idx] = current_out.actions[self_slice]
            log_probs2[partition.self_idx] = current_out.log_probs[self_slice]
            values2[partition.self_idx] = current_out.value[self_slice]
            memory2.append(partition.self_idx, current_out.history_token[self_slice])

        memory1.append(torch.arange(n_envs), current_out.history_token[:n_envs])

        for opponent_id, group_idx in partition.pool_groups():
            with torch.amp.autocast(device_type=device.type, enabled=config.enable_optim):
                group_out = active_pool_policies[opponent_id].act_obs(
                    obs2_gpu[group_idx],
                    mask2_gpu[group_idx],
                    *memory2.inputs(group_idx, device, torch.float32),
                )
            actions2[group_idx] = group_out.actions
            log_probs2[group_idx] = group_out.log_probs
            values2[group_idx] = group_out.value
            memory2.append(group_idx, group_out.history_token)

        # store actions in traj before step to avoid overwrite
        # and needing to clone, instead of with step results
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
            memory1_inputs[0].to(device="cpu"),
            memory1_inputs[1].to(device="cpu"),
        )

        has_self_play = self_idx_cpu.numel() > 0
        s2 = step_counts2[self_idx_cpu]
        if has_self_play:
            s2 = trajectories2.record(
                self_idx_cpu,
                obs2_cpu[self_idx_cpu],
                actions2_cpu[self_idx_cpu],
                log_probs2_cpu[self_idx_cpu],
                values2_cpu[self_idx_cpu],
                torch.from_numpy(masks2)[self_idx_cpu].to(torch.bool),
                memory2_self_inputs[0].to(device="cpu"),
                memory2_self_inputs[1].to(device="cpu"),
            )

        env_actions = [
            {
                vec_env.envs[i].agent1.username: actions1_cpu[i].numpy(),
                vec_env.envs[i].agent2.username: actions2_cpu[i].numpy(),
            }
            for i in range(n_envs)
        ]

        # The D2H action copies above synchronize the H2D observation reads. The env
        # workers may only overwrite their shared observation rows after that point.
        next_masks1, next_masks2, rewards1, rewards2, dones, infos = vec_env.step(env_actions)

        # store step results
        trajectories1.rewards[idx_all, s1] = torch.from_numpy(rewards1)
        trajectories1.dones[idx_all, s1] = torch.from_numpy(dones.astype(np.float32))

        if has_self_play:
            trajectories2.rewards[self_idx_cpu, s2] = torch.from_numpy(
                rewards2[partition.self_mask_cpu.numpy()]
            )
            trajectories2.dones[self_idx_cpu, s2] = torch.from_numpy(
                dones[partition.self_mask_cpu.numpy()].astype(np.float32)
            )

        for i in range(n_envs):
            if dones[i]:
                buffer.add_episode(trajectories1.complete(i))

                if partition.self_mask_cpu[i]:
                    buffer.add_episode(trajectories2.complete(i))
                else:
                    step_counts2[i] = 0

                # ties only happen on truncation; exclude them from win-rate stats
                opponent_id = partition.opponent_ids[i]
                if opponent_id != "self" and rewards1[i] != rewards2[i]:
                    won = bool(rewards1[i] > rewards2[i])
                    pool_wins += int(won)
                    pool_total += 1
                    pool.update_win_rate(opponent_id, int(won))

                memory1.reset(i)
                memory2.reset(i)
                if opponent_id != "self":
                    memory2.clear_series(i)
                    partition.opponent_ids[i] = next_pool_opponents[i]

        masks1 = next_masks1
        masks2 = next_masks2
    # cleanup inactive policies
    active_ids = {opponent_id for opponent_id, _ in partition.pool_groups()}
    keys_to_remove = [k for k in active_pool_policies.keys() if k not in active_ids]
    for k in keys_to_remove:
        del active_pool_policies[k]

    return pool_wins, pool_total


class RolloutCollector:
    """Own loaded opponents, battle memory, and active trajectory storage."""

    def __init__(
        self,
        vector_env: ThreadVecEnv,
        policy: PolicyNet,
        league: OpponentPool,
        config: TrainingConfig,
        *,
        max_trajectory_steps: int = 200,
    ) -> None:
        self.vector_env = vector_env
        self.policy = policy
        self.league = league
        self.config = config
        self.buffer = RolloutBuffer()
        self.active_policies: dict[str, PolicyNet] = {}
        self.first = TrajectoryStorage.allocate(
            config.n_envs, max_trajectory_steps, d_model=policy.d_model
        )
        self.second = TrajectoryStorage.allocate(
            config.n_envs, max_trajectory_steps, d_model=policy.d_model
        )
        self.partition: EnvPartition = build_partition(config, league, policy.device)
        self.memory1 = BattleMemoryBuffer(config.n_envs, policy.d_model)
        self.memory2 = BattleMemoryBuffer(config.n_envs, policy.d_model)
        for opponent_id, _ in self.partition.pool_groups():
            self.active_policies[opponent_id] = league.load_policy(opponent_id, str(policy.device))

    def collect(self) -> tuple[int, int]:
        stats = collect_rollouts(
            self.vector_env,
            self.policy,
            self.buffer,
            self.league,
            self.config,
            self.active_policies,
            self.first,
            self.second,
            self.memory1,
            self.memory2,
            self.partition,
        )
        return stats

    def reset_completed(self) -> None:
        self.buffer.reset()
