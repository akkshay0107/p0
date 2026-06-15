from dataclasses import dataclass

import numpy as np
import torch

from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)
from src.train.config import PPOConfig
from src.train.opponent_pool import OpponentPool
from src.train.vec_env import ThreadVecEnv

MAX_TRAJECTORY_STEPS = 200


def assign_pool_opponents(n_envs: int, roster: list[str]) -> list[str]:
    if not roster:
        raise ValueError("Pool opponent roster must not be empty.")
    return [roster[env_id % len(roster)] for env_id in range(n_envs)]


@dataclass(slots=True)
class EnvPartition:
    self_idx: torch.Tensor
    pool_idx: torch.Tensor
    opponent_ids: list[str]
    self_mask_cpu: torch.Tensor

    def pool_groups(self) -> tuple[tuple[str, torch.Tensor], ...]:
        grouped_indices: dict[str, list[int]] = {}
        for env_id in self.pool_idx.tolist():
            opponent_id = self.opponent_ids[env_id]
            grouped_indices.setdefault(opponent_id, []).append(env_id)
        return tuple(
            (
                opponent_id,
                torch.tensor(indices, device=self.pool_idx.device),
            )
            for opponent_id, indices in grouped_indices.items()
        )


def build_partition(
    config: PPOConfig,
    pool: OpponentPool,
    device: torch.device,
) -> EnvPartition:
    n_self = config.n_envs if len(pool) == 0 else config.n_self_envs
    self_idx = torch.arange(n_self, device=device)
    pool_idx = torch.arange(n_self, config.n_envs, device=device)
    self_mask_cpu = torch.arange(config.n_envs) < n_self

    opponent_ids = ["self"] * config.n_envs
    if pool_idx.numel() > 0:
        roster = pool.sample_many(config.n_pool_opponents)
        assignments = assign_pool_opponents(pool_idx.numel(), roster)
        for env_id, opponent_id in zip(pool_idx.tolist(), assignments, strict=True):
            opponent_ids[env_id] = opponent_id

    return EnvPartition(
        self_idx=self_idx,
        pool_idx=pool_idx,
        opponent_ids=opponent_ids,
        self_mask_cpu=self_mask_cpu,
    )


def create_trajectory_buffers(
    n_envs,
    max_steps=MAX_TRAJECTORY_STEPS,
    device="cpu",
):
    return {
        "step_counts": torch.zeros(n_envs, dtype=torch.long, device=device),
        "categorical": torch.zeros(
            (n_envs, max_steps, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long, device=device
        ),
        "numerical": torch.zeros(
            (n_envs, max_steps, SEQUENCE_LENGTH, NUMERICAL_WIDTH),
            dtype=torch.float32,
            device=device,
        ),
        "token_type_ids": torch.zeros(
            (n_envs, max_steps, SEQUENCE_LENGTH), dtype=torch.long, device=device
        ),
        "side_ids": torch.zeros(
            (n_envs, max_steps, SEQUENCE_LENGTH), dtype=torch.long, device=device
        ),
        "slot_ids": torch.zeros(
            (n_envs, max_steps, SEQUENCE_LENGTH), dtype=torch.long, device=device
        ),
        "events_cat": torch.zeros(
            (n_envs, max_steps, EVENT_COUNT, EVENT_CATEGORICAL_WIDTH),
            dtype=torch.long,
            device=device,
        ),
        "events_num": torch.zeros(
            (n_envs, max_steps, EVENT_COUNT, EVENT_NUMERICAL_WIDTH),
            dtype=torch.float32,
            device=device,
        ),
        "events_side_ids": torch.zeros(
            (n_envs, max_steps, EVENT_COUNT),
            dtype=torch.long,
            device=device,
        ),
        "events_slot_ids": torch.zeros(
            (n_envs, max_steps, EVENT_COUNT),
            dtype=torch.long,
            device=device,
        ),
        "actions": torch.zeros((n_envs, max_steps, 2), dtype=torch.long, device=device),
        "log_probs": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "values": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "rewards": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "dones": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "action_masks": torch.zeros(
            (n_envs, max_steps, 2, ACT_SIZE), dtype=torch.bool, device=device
        ),
    }


def compute_gae_batch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Compute GAE for a padded batch with one reverse scan over time."""
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching padded shapes.")
    if rewards.dim() != 2 or lengths.shape != (rewards.size(0),):
        raise ValueError(
            "Expected padded tensors shaped (batch, time) and lengths shaped (batch,)."
        )

    batch_size, max_steps = rewards.shape
    lengths = lengths.to(rewards.device)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(max_steps)):
        active = t < lengths
        if t + 1 < max_steps:
            next_value = torch.where(t + 1 < lengths, values[:, t + 1], 0.0)
        else:
            next_value = torch.zeros_like(gae)
        nonterminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_value * nonterminal - values[:, t]
        gae = torch.where(
            active,
            delta + gamma * gae_lambda * nonterminal * gae,
            0.0,
        )
        advantages[:, t] = gae

    return advantages


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectories: list[dict] = []

    def add_episode(self, episode: dict):
        if not episode or "length" not in episode or episode["length"] == 0:
            return
        self.trajectories.append(episode)

    def get_batches(self, device: torch.device, config: PPOConfig):
        all_episodes = []
        all_advantages = []
        if not self.trajectories:
            return all_episodes

        rewards_padded = torch.nn.utils.rnn.pad_sequence(
            [ep["rewards"] for ep in self.trajectories],
            batch_first=True,
        )
        values_padded = torch.nn.utils.rnn.pad_sequence(
            [ep["values"] for ep in self.trajectories],
            batch_first=True,
        )
        dones_padded = torch.nn.utils.rnn.pad_sequence(
            [ep["dones"] for ep in self.trajectories],
            batch_first=True,
        )
        lengths = torch.tensor([ep["length"] for ep in self.trajectories])
        advantages_padded = compute_gae_batch(
            rewards_padded,
            values_padded,
            dones_padded,
            lengths,
            config.gamma,
            config.gae_lambda,
        )

        for episode_idx, ep in enumerate(self.trajectories):
            # Ep data is on CPU
            values = ep["values"]
            T = ep["length"]

            adv = advantages_padded[episode_idx, :T]
            ret = adv + values

            episode_data = {
                "obs": ep["obs"].to(device),
                "actions": ep["actions"].to(device),
                "log_probs": ep["log_probs"].to(device),
                "action_masks": ep["action_masks"].to(device),
                "values": values.to(device),
                "advantages": adv,
                "returns": ret.to(device),
                "length": T,
            }
            all_episodes.append(episode_data)
            all_advantages.append(episode_data["advantages"])

        if all_advantages:
            flat_adv = torch.cat(all_advantages, dim=0)
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std().clamp_min(1e-8)
            for ep in all_episodes:
                ep["advantages"] = ((ep["advantages"] - adv_mean) / adv_std).to(device)

        return all_episodes


@torch.inference_mode()
def collect_rollouts(
    vec_env: ThreadVecEnv,
    policy: PolicyNet,
    buffer: RolloutBuffer,
    pool: OpponentPool,
    config: PPOConfig,
    active_pool_policies: dict[str, PolicyNet],
    trajectories1: dict,
    trajectories2: dict,
    state1: torch.Tensor,
    state2: torch.Tensor,
    partition: EnvPartition,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
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

    step_counts1 = trajectories1["step_counts"]
    step_counts2 = trajectories2["step_counts"]

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
        current_state = torch.cat([state1, state2[partition.self_idx]])
        with torch.amp.autocast(device_type=device.type, enabled=config.enable_optim):
            current_out = policy.act_obs(current_obs, current_mask, current_state)

        actions1 = current_out.actions[:n_envs]
        log_probs1 = current_out.log_probs[:n_envs]
        values1 = current_out.value[:n_envs]
        next_state1 = current_out.state[:n_envs].to(torch.float32)

        actions2 = torch.zeros_like(actions1)
        log_probs2 = torch.zeros_like(log_probs1)
        values2 = torch.zeros_like(values1)
        next_state2 = state2.clone()

        if partition.self_idx.numel() > 0:
            self_slice = slice(n_envs, None)
            actions2[partition.self_idx] = current_out.actions[self_slice]
            log_probs2[partition.self_idx] = current_out.log_probs[self_slice]
            values2[partition.self_idx] = current_out.value[self_slice]
            next_state2[partition.self_idx] = current_out.state[self_slice].to(torch.float32)

        for opponent_id, group_idx in partition.pool_groups():
            with torch.amp.autocast(device_type=device.type, enabled=config.enable_optim):
                group_out = active_pool_policies[opponent_id].act_obs(
                    obs2_gpu[group_idx],
                    mask2_gpu[group_idx],
                    state2[group_idx],
                )
            actions2[group_idx] = group_out.actions
            log_probs2[group_idx] = group_out.log_probs
            values2[group_idx] = group_out.value
            next_state2[group_idx] = group_out.state.to(torch.float32)

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

        s1 = step_counts1
        trajectories1["categorical"][idx_all, s1] = obs1_cpu.categorical
        trajectories1["numerical"][idx_all, s1] = obs1_cpu.numerical
        trajectories1["token_type_ids"][idx_all, s1] = obs1_cpu.token_type_ids
        trajectories1["side_ids"][idx_all, s1] = obs1_cpu.side_ids
        trajectories1["slot_ids"][idx_all, s1] = obs1_cpu.slot_ids
        trajectories1["events_cat"][idx_all, s1] = obs1_cpu.events_cat
        trajectories1["events_num"][idx_all, s1] = obs1_cpu.events_num
        trajectories1["events_side_ids"][idx_all, s1] = obs1_cpu.events_side_ids
        trajectories1["events_slot_ids"][idx_all, s1] = obs1_cpu.events_slot_ids
        trajectories1["actions"][idx_all, s1] = actions1_cpu
        trajectories1["log_probs"][idx_all, s1] = log_probs1_cpu
        trajectories1["values"][idx_all, s1] = values1_cpu
        trajectories1["action_masks"][idx_all, s1] = torch.from_numpy(masks1).to(torch.bool)
        step_counts1 += 1

        has_self_play = self_idx_cpu.numel() > 0
        s2 = step_counts2[self_idx_cpu]
        if has_self_play:
            trajectories2["categorical"][self_idx_cpu, s2] = obs2_cpu.categorical[self_idx_cpu]
            trajectories2["numerical"][self_idx_cpu, s2] = obs2_cpu.numerical[self_idx_cpu]
            trajectories2["token_type_ids"][self_idx_cpu, s2] = obs2_cpu.token_type_ids[
                self_idx_cpu
            ]
            trajectories2["side_ids"][self_idx_cpu, s2] = obs2_cpu.side_ids[self_idx_cpu]
            trajectories2["slot_ids"][self_idx_cpu, s2] = obs2_cpu.slot_ids[self_idx_cpu]
            trajectories2["events_cat"][self_idx_cpu, s2] = obs2_cpu.events_cat[self_idx_cpu]
            trajectories2["events_num"][self_idx_cpu, s2] = obs2_cpu.events_num[self_idx_cpu]
            trajectories2["events_side_ids"][self_idx_cpu, s2] = obs2_cpu.events_side_ids[
                self_idx_cpu
            ]
            trajectories2["events_slot_ids"][self_idx_cpu, s2] = obs2_cpu.events_slot_ids[
                self_idx_cpu
            ]
            trajectories2["actions"][self_idx_cpu, s2] = actions2_cpu[self_idx_cpu]
            trajectories2["log_probs"][self_idx_cpu, s2] = log_probs2_cpu[self_idx_cpu]
            trajectories2["values"][self_idx_cpu, s2] = values2_cpu[self_idx_cpu]
            trajectories2["action_masks"][self_idx_cpu, s2] = torch.from_numpy(masks2)[
                self_idx_cpu
            ].to(torch.bool)
            step_counts2[self_idx_cpu] += 1

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
        trajectories1["rewards"][idx_all, s1] = torch.from_numpy(rewards1)
        trajectories1["dones"][idx_all, s1] = torch.from_numpy(dones.astype(np.float32))

        if has_self_play:
            trajectories2["rewards"][self_idx_cpu, s2] = torch.from_numpy(
                rewards2[partition.self_mask_cpu.numpy()]
            )
            trajectories2["dones"][self_idx_cpu, s2] = torch.from_numpy(
                dones[partition.self_mask_cpu.numpy()].astype(np.float32)
            )

        for i in range(n_envs):
            if dones[i]:
                # fetch episode through step count
                length1 = step_counts1[i].item()
                if length1 > 0:
                    ep1 = {
                        "obs": StructuredObservation(
                            token_type_ids=trajectories1["token_type_ids"][i, :length1].clone(),
                            side_ids=trajectories1["side_ids"][i, :length1].clone(),
                            slot_ids=trajectories1["slot_ids"][i, :length1].clone(),
                            categorical=trajectories1["categorical"][i, :length1].clone(),
                            numerical=trajectories1["numerical"][i, :length1].clone(),
                            events_cat=trajectories1["events_cat"][i, :length1].clone(),
                            events_num=trajectories1["events_num"][i, :length1].clone(),
                            events_side_ids=trajectories1["events_side_ids"][i, :length1].clone(),
                            events_slot_ids=trajectories1["events_slot_ids"][i, :length1].clone(),
                        ),
                        "actions": trajectories1["actions"][i, :length1].clone(),
                        "log_probs": trajectories1["log_probs"][i, :length1].clone(),
                        "values": trajectories1["values"][i, :length1].clone(),
                        "rewards": trajectories1["rewards"][i, :length1].clone(),
                        "dones": trajectories1["dones"][i, :length1].clone(),
                        "action_masks": trajectories1["action_masks"][i, :length1].clone(),
                        "length": length1,
                    }
                    buffer.add_episode(ep1)
                step_counts1[i] = 0

                if partition.self_mask_cpu[i]:
                    length2 = step_counts2[i].item()
                    if length2 > 0:
                        ep2 = {
                            "obs": StructuredObservation(
                                token_type_ids=trajectories2["token_type_ids"][i, :length2].clone(),
                                side_ids=trajectories2["side_ids"][i, :length2].clone(),
                                slot_ids=trajectories2["slot_ids"][i, :length2].clone(),
                                categorical=trajectories2["categorical"][i, :length2].clone(),
                                numerical=trajectories2["numerical"][i, :length2].clone(),
                                events_cat=trajectories2["events_cat"][i, :length2].clone(),
                                events_num=trajectories2["events_num"][i, :length2].clone(),
                                events_side_ids=trajectories2["events_side_ids"][
                                    i, :length2
                                ].clone(),
                                events_slot_ids=trajectories2["events_slot_ids"][
                                    i, :length2
                                ].clone(),
                            ),
                            "actions": trajectories2["actions"][i, :length2].clone(),
                            "log_probs": trajectories2["log_probs"][i, :length2].clone(),
                            "values": trajectories2["values"][i, :length2].clone(),
                            "rewards": trajectories2["rewards"][i, :length2].clone(),
                            "dones": trajectories2["dones"][i, :length2].clone(),
                            "action_masks": trajectories2["action_masks"][i, :length2].clone(),
                            "length": length2,
                        }
                        buffer.add_episode(ep2)
                step_counts2[i] = 0

                # ties only happen on truncation; exclude them from win-rate stats
                opponent_id = partition.opponent_ids[i]
                if opponent_id != "self" and rewards1[i] != rewards2[i]:
                    won = bool(rewards1[i] > rewards2[i])
                    pool_wins += int(won)
                    pool_total += 1
                    pool.update_win_rate(opponent_id, int(won))

                # reset to each policy's learned initial state (hg_init is no
                # longer zeros); the BPTT recompute starts episodes from
                # initial_state, so the rollout reset must match it
                next_state1[i : i + 1] = policy.initial_state(1)
                if opponent_id != "self":
                    partition.opponent_ids[i] = next_pool_opponents[i]
                    next_state2[i : i + 1] = active_pool_policies[
                        next_pool_opponents[i]
                    ].initial_state(1)
                else:
                    next_state2[i : i + 1] = policy.initial_state(1)

        masks1 = next_masks1
        masks2 = next_masks2
        state1 = next_state1
        state2 = next_state2

    # cleanup inactive policies
    active_ids = {opponent_id for opponent_id, _ in partition.pool_groups()}
    keys_to_remove = [k for k in active_pool_policies.keys() if k not in active_ids]
    for k in keys_to_remove:
        del active_pool_policies[k]

    return (pool_wins, pool_total), state1, state2
