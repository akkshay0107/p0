import numpy as np
import torch

from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)
from src.train.config import PPOConfig
from src.train.opponent_pool import OpponentPool
from src.train.vec_env import ThreadVecEnv


def assign_pool_opponents(n_envs: int, roster: list[str]) -> list[str]:
    if not roster:
        raise ValueError("Pool opponent roster must not be empty.")
    return [roster[env_id % len(roster)] for env_id in range(n_envs)]


def create_trajectory_buffers(n_envs, max_steps=100, device="cpu"):
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
        "actions": torch.zeros((n_envs, max_steps, 2), dtype=torch.long, device=device),
        "log_probs": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "values": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "rewards": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "dones": torch.zeros((n_envs, max_steps), dtype=torch.float32, device=device),
        "action_masks": torch.zeros(
            (n_envs, max_steps, 2, ACT_SIZE), dtype=torch.bool, device=device
        ),
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation.
    Accepts 1-D tensors on any device. Returns an advantage tensor on the same device.
    """
    T = rewards.size(0)
    adv = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        adv[t] = gae
    return adv


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectories: list[dict] = []

    _FIELDS = (
        "obs",
        "actions",
        "log_probs",
        "values",
        "rewards",
        "dones",
        "action_masks",
    )

    def add_episode(self, episode: dict):
        if not episode or "length" not in episode or episode["length"] == 0:
            return
        self.trajectories.append(episode)

    def get_batches(self, device: torch.device, config: PPOConfig):
        all_episodes = []
        all_advantages = []

        for ep in self.trajectories:
            # Ep data is on CPU
            rewards = ep["rewards"]
            values = ep["values"]
            dones = ep["dones"]
            T = ep["length"]

            adv = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
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
    env_opponents: list[str],
    target_mode: str,
):
    """
    Collects rollouts using the vectorized environment defined in the train module.
    Self-play and pool-play are collected in separate phases across all environments.

    Each pool-play phase samples a bounded policy roster and assigns every environment
    a fixed roster member. Policies are swapped lazily when games end to prevent episodes
    or recurrent states from being split across different policies.
    """
    n_envs = vec_env.n_envs
    device = policy.device

    masks1 = vec_env.last_masks1
    masks2 = vec_env.last_masks2

    pool_wins = 0
    pool_total = 0

    steps = config.self_play_steps if target_mode == "self_play" else config.pool_play_steps

    step_counts1 = trajectories1["step_counts"]
    step_counts2 = trajectories2["step_counts"]

    idx_all = torch.arange(n_envs)
    pool_roster: list[str] = []
    thread_pool_opponents: list[str] = []

    if target_mode != "self_play" and len(pool) > 0:
        pool_roster = pool.sample_many(config.n_pool_opponents)
        thread_pool_opponents = assign_pool_opponents(n_envs, pool_roster)
        for opponent_id in pool_roster:
            if opponent_id not in active_pool_policies:
                active_pool_policies[opponent_id] = pool.load_policy(opponent_id, str(device))

    for step in range(steps):
        obs1_gpu = vec_env.get_batched_obs1(device)
        mask1_gpu = torch.from_numpy(masks1).to(device, non_blocking=True)
        obs2_gpu = vec_env.get_batched_obs2(device)
        mask2_gpu = torch.from_numpy(masks2).to(device, non_blocking=True)

        out1 = policy.act_obs(obs1_gpu, mask1_gpu, state1)
        log_probs1 = out1.log_probs
        actions1 = out1.actions
        values1 = out1.value
        next_state1 = out1.state

        is_all_self_play = all(opp == "self" for opp in env_opponents)
        # self play fast path
        if is_all_self_play:
            out2 = policy.act_obs(obs2_gpu, mask2_gpu, state2)
            log_probs2 = out2.log_probs
            actions2 = out2.actions
            values2 = out2.value
            next_state2 = out2.state
        else:
            actions2 = torch.zeros_like(actions1)
            log_probs2 = torch.zeros_like(log_probs1)
            values2 = torch.zeros_like(values1)
            next_state2 = state2.clone()

            opp_groups = {}
            for i, opp_id in enumerate(env_opponents):
                opp_groups.setdefault(opp_id, []).append(i)

            for opp_id, env_indices in opp_groups.items():
                idx_tensor = torch.tensor(env_indices, device=device)
                group_obs2 = obs2_gpu[idx_tensor]
                group_mask2 = mask2_gpu[idx_tensor]
                group_state2 = state2[idx_tensor]

                active_policy = policy if opp_id == "self" else active_pool_policies[opp_id]
                group_out = active_policy.act_obs(group_obs2, group_mask2, group_state2)

                actions2[idx_tensor] = group_out.actions
                log_probs2[idx_tensor] = group_out.log_probs
                values2[idx_tensor] = group_out.value
                next_state2[idx_tensor] = group_out.state

        # store actions in traj before step to avoid overwrite
        # and needing to clone, instead of with step results
        actions1_cpu = actions1.to("cpu")
        actions2_cpu = actions2.to("cpu")
        log_probs1_cpu = log_probs1.to("cpu")
        log_probs2_cpu = log_probs2.to("cpu")
        values1_cpu = values1.to("cpu")
        values2_cpu = values2.to("cpu")
        obs1_cpu = vec_env.obs1_buffers
        obs2_cpu = vec_env.obs2_buffers

        s1 = step_counts1
        trajectories1["categorical"][idx_all, s1] = obs1_cpu.categorical
        trajectories1["numerical"][idx_all, s1] = obs1_cpu.numerical
        trajectories1["token_type_ids"][idx_all, s1] = obs1_cpu.token_type_ids
        trajectories1["side_ids"][idx_all, s1] = obs1_cpu.side_ids
        trajectories1["slot_ids"][idx_all, s1] = obs1_cpu.slot_ids
        trajectories1["actions"][idx_all, s1] = actions1_cpu
        trajectories1["log_probs"][idx_all, s1] = log_probs1_cpu
        trajectories1["values"][idx_all, s1] = values1_cpu
        trajectories1["action_masks"][idx_all, s1] = torch.from_numpy(masks1).to(torch.bool)
        step_counts1 += 1

        self_play_mask = torch.tensor([opp == "self" for opp in env_opponents], dtype=torch.bool)
        has_self_play = bool(self_play_mask.any())
        sp_idx = self_play_mask.nonzero().squeeze(-1)
        s2 = step_counts2[sp_idx]
        if has_self_play:
            trajectories2["categorical"][sp_idx, s2] = obs2_cpu.categorical[sp_idx]
            trajectories2["numerical"][sp_idx, s2] = obs2_cpu.numerical[sp_idx]
            trajectories2["token_type_ids"][sp_idx, s2] = obs2_cpu.token_type_ids[sp_idx]
            trajectories2["side_ids"][sp_idx, s2] = obs2_cpu.side_ids[sp_idx]
            trajectories2["slot_ids"][sp_idx, s2] = obs2_cpu.slot_ids[sp_idx]
            trajectories2["actions"][sp_idx, s2] = actions2_cpu[sp_idx]
            trajectories2["log_probs"][sp_idx, s2] = log_probs2_cpu[sp_idx]
            trajectories2["values"][sp_idx, s2] = values2_cpu[sp_idx]
            trajectories2["action_masks"][sp_idx, s2] = torch.from_numpy(masks2)[sp_idx].to(
                torch.bool
            )
            step_counts2[sp_idx] += 1

        env_actions = [
            {
                vec_env.envs[i].agent1.username: actions1_cpu[i].numpy(),
                vec_env.envs[i].agent2.username: actions2_cpu[i].numpy(),
            }
            for i in range(n_envs)
        ]

        next_masks1, next_masks2, rewards1, rewards2, dones, infos = vec_env.step(env_actions)

        # store step results
        trajectories1["rewards"][idx_all, s1] = torch.from_numpy(rewards1)
        trajectories1["dones"][idx_all, s1] = torch.from_numpy(dones.astype(np.float32))

        if has_self_play:
            trajectories2["rewards"][sp_idx, s2] = torch.from_numpy(
                rewards2[self_play_mask.numpy()]
            )
            trajectories2["dones"][sp_idx, s2] = torch.from_numpy(
                dones[self_play_mask.numpy()].astype(np.float32)
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

                if env_opponents[i] == "self":
                    length2 = step_counts2[i].item()
                    if length2 > 0:
                        ep2 = {
                            "obs": StructuredObservation(
                                token_type_ids=trajectories2["token_type_ids"][i, :length2].clone(),
                                side_ids=trajectories2["side_ids"][i, :length2].clone(),
                                slot_ids=trajectories2["slot_ids"][i, :length2].clone(),
                                categorical=trajectories2["categorical"][i, :length2].clone(),
                                numerical=trajectories2["numerical"][i, :length2].clone(),
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
                if env_opponents[i] != "self" and rewards1[i] != rewards2[i]:
                    won = bool(rewards1[i] > rewards2[i])
                    pool_wins += int(won)
                    pool_total += 1
                    pool.update_win_rate(env_opponents[i], won)

                next_state1[i : i + 1] = 0
                next_state2[i : i + 1] = 0

                # lazy swap
                if target_mode == "self_play" or len(pool) == 0:
                    env_opponents[i] = "self"
                else:
                    new_opp_id = thread_pool_opponents[i]
                    env_opponents[i] = new_opp_id

        masks1 = next_masks1
        masks2 = next_masks2
        state1 = next_state1
        state2 = next_state2

    # cleanup inactive policies
    active_ids = set(env_opponents)
    keys_to_remove = [k for k in active_pool_policies.keys() if k not in active_ids]
    for k in keys_to_remove:
        del active_pool_policies[k]

    return pool_wins / pool_total if pool_total > 0 else 0.0, state1, state2
