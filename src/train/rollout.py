import numpy as np
import torch

from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.structured_observation import CATEGORICAL_WIDTH, NUMERICAL_WIDTH, SEQUENCE_LENGTH, StructuredObservation
from src.train.config import PPOConfig
from src.train.opponent_pool import OpponentPool
from src.train.vec_env import ThreadVecEnv


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
        "is_team_preview": torch.zeros((n_envs, max_steps), dtype=torch.bool, device=device),
    }


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    T = len(rewards)
    adv = np.zeros_like(rewards)
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
        "is_team_preview",
    )

    def add_episode(self, episode: dict):
        if not episode or "length" not in episode or episode["length"] == 0:
            return
        self.trajectories.append(episode)

    def get_batches(self, device: torch.device, config: PPOConfig):
        all_episodes = []
        all_advantages = []

        for ep in self.trajectories:
            rewards = ep["rewards"]
            values = ep["values"]
            dones = ep["dones"]
            T = ep["length"]

            adv_np = compute_gae(
                rewards.numpy(), values.numpy(), dones.numpy(), config.gamma, config.gae_lambda
            )
            adv = torch.from_numpy(adv_np).to(device)
            values_dev = values.to(device)
            ret = adv + values_dev

            episode_data = {
                "obs": ep["obs"].to(device),
                "actions": ep["actions"].to(device),
                "log_probs": ep["log_probs"].to(device),
                "action_masks": ep["action_masks"].to(device),
                "values": values_dev,
                "advantages": adv,
                "returns": ret,
                "is_team_preview": ep["is_team_preview"].to(device),
                "length": T,
            }
            all_episodes.append(episode_data)
            all_advantages.append(episode_data["advantages"])

        if all_advantages:
            flat_adv = torch.cat(all_advantages, dim=0)
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std().clamp_min(1e-8)
            for ep in all_episodes:
                ep["advantages"] = (ep["advantages"] - adv_mean) / adv_std

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
    state1: tuple[torch.Tensor, torch.Tensor],
    state2: tuple[torch.Tensor, torch.Tensor],
    env_opponents: list[str],
    target_mode: str,
):
    """
    Collects rollouts using the vectorized environment defined in the train module.
    Environments (one per thread) are split into two groups - self play / pop. based training.

    Pool policy is swapped lazily as soon as its game is done to prevent episodes being
    split across different policies in the pool. The current policy receives updates as
    usual through boundaries that split games.
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

    for step in range(steps):
        obs1 = vec_env.get_batched_obs1(device)
        mask1_t = torch.from_numpy(masks1).to(device, non_blocking=True)
        obs2_batched = vec_env.get_batched_obs2(device)
        mask2_t = torch.from_numpy(masks2).to(device, non_blocking=True)

        _, log_probs1, actions1, values1, next_state1 = policy(
            obs1,
            state1,
            mask1_t,
            sample_actions=True,
        )

        actions2 = torch.zeros_like(actions1)
        log_probs2 = torch.zeros_like(log_probs1)
        values2 = torch.zeros_like(values1)

        new_s2_h = state2[0].clone()
        new_s2_c = state2[1].clone()

        # group by env_opponents string
        opp_groups = {}
        for i, opp_id in enumerate(env_opponents):
            if opp_id not in opp_groups:
                opp_groups[opp_id] = []
            opp_groups[opp_id].append(i)

        for opp_id, env_indices in opp_groups.items():
            idx_tensor = torch.tensor(env_indices, device=device)
            group_obs2 = obs2_batched[idx_tensor]
            group_mask2 = mask2_t[idx_tensor]
            group_state2 = (state2[0][idx_tensor], state2[1][idx_tensor])

            if opp_id == "self":
                active_policy = policy
            else:
                active_policy = active_pool_policies[opp_id]

            _, g_log_probs, g_actions, g_values, g_next_state = active_policy(
                group_obs2, group_state2, group_mask2, sample_actions=True
            )

            actions2[idx_tensor] = g_actions
            log_probs2[idx_tensor] = g_log_probs
            values2[idx_tensor] = g_values
            new_s2_h[idx_tensor] = g_next_state[0]
            new_s2_c[idx_tensor] = g_next_state[1]

        next_state2 = (new_s2_h, new_s2_c)

        env_actions = [
            {
                vec_env.envs[i].agent1.username: actions1[i].cpu().numpy(),
                vec_env.envs[i].agent2.username: actions2[i].cpu().numpy(),
            }
            for i in range(n_envs)
        ]

        obs1_cpu = obs1.cpu()
        obs2_cpu = obs2_batched.cpu()
        is_tp1s = np.array([vec_env.envs[i].battle1.teampreview for i in range(n_envs)], dtype=bool)
        is_tp2s = np.array([vec_env.envs[i].battle2.teampreview for i in range(n_envs)], dtype=bool)

        next_masks1, next_masks2, rewards1, rewards2, dones, infos = vec_env.step(env_actions)

        # batch insert for first trajectory
        s1 = step_counts1
        trajectories1["categorical"][idx_all, s1] = obs1_cpu.categorical
        trajectories1["numerical"][idx_all, s1] = obs1_cpu.numerical
        trajectories1["token_type_ids"][idx_all, s1] = obs1_cpu.token_type_ids
        trajectories1["side_ids"][idx_all, s1] = obs1_cpu.side_ids
        trajectories1["slot_ids"][idx_all, s1] = obs1_cpu.slot_ids
        trajectories1["actions"][idx_all, s1] = actions1.cpu()
        trajectories1["log_probs"][idx_all, s1] = log_probs1.cpu()
        trajectories1["values"][idx_all, s1] = values1.cpu()
        trajectories1["rewards"][idx_all, s1] = torch.tensor(rewards1, dtype=torch.float32)
        trajectories1["dones"][idx_all, s1] = torch.tensor(dones, dtype=torch.float32)
        trajectories1["action_masks"][idx_all, s1] = mask1_t.cpu().bool()
        trajectories1["is_team_preview"][idx_all, s1] = torch.tensor(is_tp1s, dtype=torch.bool)
        step_counts1 += 1

        # batch insert for traj 2 (only in self play)
        self_play_mask = torch.tensor([env_opponents[i] == "self" for i in range(n_envs)])
        if self_play_mask.any():
            sp_idx = idx_all[self_play_mask]
            s2 = step_counts2[sp_idx]
            trajectories2["categorical"][sp_idx, s2] = obs2_cpu.categorical[sp_idx]
            trajectories2["numerical"][sp_idx, s2] = obs2_cpu.numerical[sp_idx]
            trajectories2["token_type_ids"][sp_idx, s2] = obs2_cpu.token_type_ids[sp_idx]
            trajectories2["side_ids"][sp_idx, s2] = obs2_cpu.side_ids[sp_idx]
            trajectories2["slot_ids"][sp_idx, s2] = obs2_cpu.slot_ids[sp_idx]
            trajectories2["actions"][sp_idx, s2] = actions2[sp_idx].cpu()
            trajectories2["log_probs"][sp_idx, s2] = log_probs2[sp_idx].cpu()
            trajectories2["values"][sp_idx, s2] = values2[sp_idx].cpu()
            trajectories2["rewards"][sp_idx, s2] = torch.tensor(rewards2, dtype=torch.float32)[
                sp_idx
            ]
            trajectories2["dones"][sp_idx, s2] = torch.tensor(dones, dtype=torch.float32)[sp_idx]
            trajectories2["action_masks"][sp_idx, s2] = mask2_t[sp_idx].cpu().bool()
            trajectories2["is_team_preview"][sp_idx, s2] = torch.tensor(is_tp2s, dtype=torch.bool)[
                sp_idx
            ]
            step_counts2[sp_idx] += 1

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
                        "is_team_preview": trajectories1["is_team_preview"][i, :length1].clone(),
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
                            "is_team_preview": trajectories2["is_team_preview"][i, :length2].clone(),
                            "length": length2,
                        }
                        buffer.add_episode(ep2)
                step_counts2[i] = 0

                if env_opponents[i] != "self":
                    won = bool(rewards1[i] > rewards2[i])
                    pool_wins += int(won)
                    pool_total += 1
                    pool.update_win_rate(env_opponents[i], won)

                next_state1[0][i : i + 1] = 0
                next_state1[1][i : i + 1] = 0
                next_state2[0][i : i + 1] = 0
                next_state2[1][i : i + 1] = 0

                # lazy swap
                if target_mode == "self_play" or len(pool) == 0:
                    env_opponents[i] = "self"
                else:
                    new_opp_id = pool.sample_id()
                    env_opponents[i] = new_opp_id
                    if new_opp_id not in active_pool_policies:
                        active_pool_policies[new_opp_id] = pool.load_policy(new_opp_id, str(device))

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
