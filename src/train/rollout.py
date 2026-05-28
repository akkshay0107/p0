import torch

from src.model.policy import PolicyNet
from src.train.config import PPOConfig
from src.train.opponent_pool import OpponentPool
from src.train.vec_env import ThreadVecEnv


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

    def add_episode(self, trajectory: list[dict]):
        if not trajectory:
            return
        episode = {}
        for f in self._FIELDS:
            if f == "obs":
                keys = trajectory[0]["obs"].keys()
                episode["obs"] = {
                    k: torch.cat([step["obs"][k] for step in trajectory], dim=0) for k in keys
                }
            else:
                episode[f] = torch.cat([step[f] for step in trajectory], dim=0)
        episode["length"] = len(trajectory)
        self.trajectories.append(episode)

    def get_batches(self, device: torch.device, config: PPOConfig):
        all_episodes = []
        all_advantages = []

        for ep in self.trajectories:
            rewards = ep["rewards"].float()
            values = ep["values"].float()
            dones = ep["dones"].float()
            T = ep["length"]

            adv = torch.zeros_like(rewards)
            gae = torch.zeros(1, dtype=torch.float32)

            for t in reversed(range(T)):
                next_value = values[t + 1] if t + 1 < T else torch.zeros_like(values[t])
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + config.gamma * next_value * nonterminal - values[t]
                gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
                adv[t] = gae

            ret = adv + values

            episode_data = {
                "obs": {k: v.to(device) for k, v in ep["obs"].items()},
                "actions": ep["actions"].to(device),
                "log_probs": ep["log_probs"].to(device),
                "action_masks": ep["action_masks"].to(device),
                "values": values.to(device),
                "advantages": adv.to(device),
                "returns": ret.to(device),
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
    trajectories1: list[list[dict]],
    trajectories2: list[list[dict]],
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

    for step in range(steps):
        obs1_dict = vec_env.get_batched_obs1(device)
        mask1_t = torch.from_numpy(masks1).to(device, non_blocking=True)
        obs2_batched = vec_env.get_batched_obs2(device)
        mask2_t = torch.from_numpy(masks2).to(device, non_blocking=True)

        _, log_probs1, actions1, values1, next_state1 = policy(
            obs1_dict,
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
            group_obs2 = {k: v[idx_tensor] for k, v in obs2_batched.items()}
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

        obs1_cpu_dict = {k: v.cpu().clone() for k, v in obs1_dict.items()}

        next_masks1, next_masks2, rewards1, rewards2, dones, is_tp1s, is_tp2s, infos = vec_env.step(
            env_actions
        )

        for i in range(n_envs):
            single_obs1 = {k: v[i].unsqueeze(0) for k, v in obs1_cpu_dict.items()}
            single_obs2 = {k: obs2_batched[k][i].unsqueeze(0).cpu() for k in single_obs1.keys()}

            trajectories1[i].append(
                {
                    "obs": single_obs1,
                    "actions": actions1[i : i + 1].cpu(),
                    "log_probs": log_probs1[i : i + 1].cpu(),
                    "values": values1[i : i + 1].cpu(),
                    "rewards": torch.tensor([rewards1[i]], dtype=torch.float32),
                    "dones": torch.tensor([dones[i]], dtype=torch.float32),
                    "action_masks": mask1_t[i : i + 1].cpu(),
                    "is_team_preview": torch.tensor([is_tp1s[i]], dtype=torch.bool),
                }
            )

            if env_opponents[i] == "self":
                trajectories2[i].append(
                    {
                        "obs": single_obs2,
                        "actions": actions2[i : i + 1].cpu(),
                        "log_probs": log_probs2[i : i + 1].cpu(),
                        "values": values2[i : i + 1].cpu(),
                        "rewards": torch.tensor([rewards2[i]], dtype=torch.float32),
                        "dones": torch.tensor([dones[i]], dtype=torch.float32),
                        "action_masks": mask2_t[i : i + 1].cpu(),
                        "is_team_preview": torch.tensor([is_tp2s[i]], dtype=torch.bool),
                    }
                )

            if dones[i]:
                buffer.add_episode(trajectories1[i])
                if env_opponents[i] == "self":
                    buffer.add_episode(trajectories2[i])
                elif env_opponents[i] != "self":
                    won = bool(rewards1[i] > rewards2[i])
                    pool_wins += int(won)
                    pool_total += 1
                    pool.update_win_rate(env_opponents[i], won)

                trajectories1[i] = []
                trajectories2[i] = []

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
