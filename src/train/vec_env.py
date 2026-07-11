from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.env import SimEnv
from src.format_config import FORMAT
from src.model.structured_observation import (
    StructuredObservation,
)

ACT_SIZE = FORMAT.action_size


class ThreadVecEnv:
    def __init__(self, envs: list[SimEnv]):
        self.envs = envs
        self.n_envs = len(envs)
        self.executor = ThreadPoolExecutor(max_workers=self.n_envs)

        # check if pinned memory can be used
        self.use_pinned = torch.cuda.is_available()

        # Pre-allocate shared observation buffers for agent1 and agent2
        self.obs1_buffers = self._create_buffers()
        self.obs2_buffers = self._create_buffers()
        for env_id, env in enumerate(self.envs):
            env.set_observation_targets(
                self.obs1_buffers[env_id],
                self.obs2_buffers[env_id],
            )

        self.last_masks1 = None
        self.last_masks2 = None

    def _create_buffers(self):
        return StructuredObservation.empty_batch(self.n_envs, pin_memory=self.use_pinned)

    def _reset_env(self, env_id: int, env: SimEnv):
        obs, info = env.reset()

        agent1 = env.agent1.username
        agent2 = env.agent2.username

        mask1 = np.reshape(obs[agent1]["action_mask"], (2, ACT_SIZE))
        mask2 = np.reshape(obs[agent2]["action_mask"], (2, ACT_SIZE)) if agent2 in obs else None

        return env_id, mask1, mask2, info

    def reset(self):
        futures = [self.executor.submit(self._reset_env, i, env) for i, env in enumerate(self.envs)]

        results = [f.result() for f in futures]
        results.sort(key=lambda x: x[0])

        masks1 = np.stack([r[1] for r in results])
        masks2 = np.stack([r[2] for r in results]) if results[0][2] is not None else None  # type: ignore
        infos = [r[3] for r in results]

        self.last_masks1 = masks1
        self.last_masks2 = masks2
        return masks1, masks2, infos

    def _step_env(self, env_id: int, env: SimEnv, action: dict):
        next_obs, rewards, terminated, truncated, info = env.step(action)

        agent1 = env.agent1.username
        agent2 = env.agent2.username

        mask1 = np.reshape(next_obs[agent1]["action_mask"], (2, ACT_SIZE))
        mask2 = np.reshape(next_obs[agent2]["action_mask"], (2, ACT_SIZE))

        done = bool(
            terminated[agent1] or truncated[agent1] or terminated[agent2] or truncated[agent2]
        )
        reward1 = rewards[agent1]
        reward2 = rewards[agent2] if agent2 in rewards else 0.0

        if done:
            _, mask1, mask2, _ = self._reset_env(env_id, env)
            return env_id, mask1, mask2, reward1, reward2, done, info

        return env_id, mask1, mask2, reward1, reward2, done, info

    def step(self, actions: list[dict]):
        futures = [
            self.executor.submit(self._step_env, i, env, actions[i])
            for i, env in enumerate(self.envs)
        ]

        results = [f.result() for f in futures]
        results.sort(key=lambda x: x[0])  # guarantee order

        masks1 = np.stack([r[1] for r in results])
        masks2 = np.stack([r[2] for r in results]) if results[0][2] is not None else None  # type: ignore
        rewards1 = np.array([r[3] for r in results], dtype=np.float32)
        rewards2 = np.array([r[4] for r in results], dtype=np.float32)
        dones = np.array([r[5] for r in results], dtype=bool)
        infos = [r[6] for r in results]

        self.last_masks1 = masks1
        self.last_masks2 = masks2
        return masks1, masks2, rewards1, rewards2, dones, infos

    def get_batched_obs1(self, device: torch.device):
        return self.obs1_buffers.to(device, non_blocking=True)

    def get_batched_obs2(self, device: torch.device):
        return self.obs2_buffers.to(device, non_blocking=True)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
