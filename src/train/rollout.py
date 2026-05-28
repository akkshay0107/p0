import torch

from src.train.config import PPOConfig


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trajectories: list[dict[str, torch.Tensor]] = []

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
        episode = {f: torch.cat([step[f] for step in trajectory], dim=0) for f in self._FIELDS}
        episode["length"] = len(trajectory)  # type: ignore
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
                "obs": ep["obs"].to(device),
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


# TODO: move function that collects rollouts here
