from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.env import SimEnv
from src.lookups import ACT_SIZE
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)


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

        self.last_masks1 = None
        self.last_masks2 = None

    def _create_buffers(self):
        return {
            "token_type_ids": torch.zeros(
                (self.n_envs, SEQUENCE_LENGTH), dtype=torch.long, pin_memory=self.use_pinned
            ),
            "side_ids": torch.zeros(
                (self.n_envs, SEQUENCE_LENGTH), dtype=torch.long, pin_memory=self.use_pinned
            ),
            "slot_ids": torch.zeros(
                (self.n_envs, SEQUENCE_LENGTH), dtype=torch.long, pin_memory=self.use_pinned
            ),
            "categorical": torch.zeros(
                (self.n_envs, SEQUENCE_LENGTH, CATEGORICAL_WIDTH),
                dtype=torch.long,
                pin_memory=self.use_pinned,
            ),
            "numerical": torch.zeros(
                (self.n_envs, SEQUENCE_LENGTH, NUMERICAL_WIDTH),
                dtype=torch.float32,
                pin_memory=self.use_pinned,
            ),
        }

    def _write_obs(
        self, env_id: int, obs1: StructuredObservation, obs2: StructuredObservation | None
    ):
        if obs1 is not None:
            self.obs1_buffers["token_type_ids"][env_id].copy_(obs1.token_type_ids)
            self.obs1_buffers["side_ids"][env_id].copy_(obs1.side_ids)
            self.obs1_buffers["slot_ids"][env_id].copy_(obs1.slot_ids)
            self.obs1_buffers["categorical"][env_id].copy_(obs1.categorical)
            self.obs1_buffers["numerical"][env_id].copy_(obs1.numerical)

        if obs2 is not None:
            self.obs2_buffers["token_type_ids"][env_id].copy_(obs2.token_type_ids)
            self.obs2_buffers["side_ids"][env_id].copy_(obs2.side_ids)
            self.obs2_buffers["slot_ids"][env_id].copy_(obs2.slot_ids)
            self.obs2_buffers["categorical"][env_id].copy_(obs2.categorical)
            self.obs2_buffers["numerical"][env_id].copy_(obs2.numerical)

    def _reset_env(self, env_id: int, env: SimEnv):
        obs, info = env.reset()

        agent1 = env.agent1.username
        agent2 = env.agent2.username

        obs1 = obs[agent1]["observation"]
        obs2 = obs[agent2]["observation"]

        mask1 = np.reshape(obs[agent1]["action_mask"], (2, ACT_SIZE))
        mask2 = np.reshape(obs[agent2]["action_mask"], (2, ACT_SIZE)) if agent2 in obs else None

        self._write_obs(env_id, obs1, obs2)
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

        obs1 = next_obs[agent1]["observation"]
        obs2 = next_obs[agent2]["observation"]

        mask1 = np.reshape(next_obs[agent1]["action_mask"], (2, ACT_SIZE))
        mask2 = np.reshape(next_obs[agent2]["action_mask"], (2, ACT_SIZE))

        done = bool(
            terminated[agent1] or truncated[agent1] or terminated[agent2] or truncated[agent2]
        )
        reward1 = rewards[agent1]
        reward2 = rewards[agent2] if agent2 in rewards else 0.0

        is_tp1 = env.battle1.teampreview if env.battle1 is not None else False
        is_tp2 = env.battle2.teampreview if env.battle2 is not None else False

        if done:
            _, mask1, mask2, _ = self._reset_env(env_id, env)
            return env_id, mask1, mask2, reward1, reward2, done, is_tp1, is_tp2, info

        self._write_obs(env_id, obs1, obs2)
        return env_id, mask1, mask2, reward1, reward2, done, is_tp1, is_tp2, info

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
        is_tp1s = np.array([r[6] for r in results], dtype=bool)
        is_tp2s = np.array([r[7] for r in results], dtype=bool)
        infos = [r[8] for r in results]

        self.last_masks1 = masks1
        self.last_masks2 = masks2
        return masks1, masks2, rewards1, rewards2, dones, is_tp1s, is_tp2s, infos

    def get_batched_obs1(self, device: torch.device):
        return {k: v.to(device, non_blocking=True) for k, v in self.obs1_buffers.items()}

    def get_batched_obs2(self, device: torch.device):
        return {k: v.to(device, non_blocking=True) for k, v in self.obs2_buffers.items()}

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
