import logging
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.structured_observation import (
    StructuredObservation,
)
from p0.runtime.env import SimEnv

ACT_SIZE = FORMAT.action_size
LOGGER = logging.getLogger(__name__)


class ThreadVecEnv:
    def __init__(self, envs: list[SimEnv]):
        self.envs = envs
        self.n_envs = len(envs)

        use_pinned = torch.cuda.is_available()
        self.obs1_buffers = StructuredObservation.empty_batch(self.n_envs, pin_memory=use_pinned)
        self.obs2_buffers = StructuredObservation.empty_batch(self.n_envs, pin_memory=use_pinned)
        for env_id, env in enumerate(self.envs):
            env.set_observation_targets(
                self.obs1_buffers[env_id],
                self.obs2_buffers[env_id],
            )
        self.executor = ThreadPoolExecutor(max_workers=self.n_envs)

        self.last_masks1: np.ndarray | None = None
        self.last_masks2: np.ndarray | None = None
        self.last_infos: list[dict[str, object]] | None = None

    def _reset_env(
        self, env_id: int, env: SimEnv
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        obs, raw_info = env.reset()
        info: dict[str, object] = dict(raw_info)
        info["series_id"] = env.series_id

        agent1 = env.agent1.username
        agent2 = env.agent2.username

        mask1 = np.reshape(obs[agent1]["action_mask"], (2, ACT_SIZE))
        mask2 = np.reshape(obs[agent2]["action_mask"], (2, ACT_SIZE))

        return mask1, mask2, info

    def reset(self) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
        results = list(self.executor.map(self._reset_env, range(self.n_envs), self.envs))

        masks1 = np.stack([r[0] for r in results])
        masks2 = np.stack([r[1] for r in results])
        infos = [r[2] for r in results]

        self.last_masks1 = masks1
        self.last_masks2 = masks2
        self.last_infos = infos
        return masks1, masks2, infos

    def _step_env(self, env_id: int, env: SimEnv, action: dict[str, np.ndarray]):
        next_obs, rewards, terminated, truncated, raw_info = env.step(action)
        info: dict[str, object] = dict(raw_info)

        agent1 = env.agent1.username
        agent2 = env.agent2.username

        mask1 = np.reshape(next_obs[agent1]["action_mask"], (2, ACT_SIZE))
        mask2 = np.reshape(next_obs[agent2]["action_mask"], (2, ACT_SIZE))

        is_truncated = truncated[agent1] or truncated[agent2]
        is_terminated = terminated[agent1] or terminated[agent2]
        done_status = 2 if is_truncated else (1 if is_terminated else 0)

        reward1 = rewards[agent1]
        reward2 = rewards[agent2] if agent2 in rewards else 0.0

        if done_status > 0:
            series_complete = max(env.series_scores) >= 2 or env.series_games_played >= 3
            terminal_obs1 = None
            terminal_obs2 = None
            terminal_mask1 = None
            terminal_mask2 = None
            if is_truncated:
                terminal_obs1 = self.obs1_buffers[env_id].clone()
                terminal_obs2 = self.obs2_buffers[env_id].clone()
                # Preserve the mask paired with the terminal observations before the
                # automatic reset replaces the current observation state.
                terminal_mask1 = mask1.copy()
                terminal_mask2 = mask2.copy()

            mask1, mask2, _ = self._reset_env(env_id, env)
            info["series_id"] = env.series_id
            info["series_complete"] = series_complete
            info["terminal_observation1"] = terminal_obs1
            info["terminal_observation2"] = terminal_obs2
            info["terminal_action_mask1"] = terminal_mask1
            info["terminal_action_mask2"] = terminal_mask2
            return mask1, mask2, reward1, reward2, done_status, info

        info["series_id"] = env.series_id
        info["series_complete"] = False
        return mask1, mask2, reward1, reward2, done_status, info

    def step(self, actions: list[dict[str, np.ndarray]]):
        if len(actions) != self.n_envs:
            raise ValueError("Number of actions must match the number of environments")
        results = list(self.executor.map(self._step_env, range(self.n_envs), self.envs, actions))

        masks1 = np.stack([r[0] for r in results])
        masks2 = np.stack([r[1] for r in results])
        rewards1 = np.array([r[2] for r in results], dtype=np.float32)
        rewards2 = np.array([r[3] for r in results], dtype=np.float32)
        # 0 running, 1 terminated, 2 truncated. A boolean array here would fold
        # truncation into termination and silently disable bootstrapping.
        done_status = np.array([r[4] for r in results], dtype=np.int64)
        infos = [r[5] for r in results]

        self.last_masks1 = masks1
        self.last_masks2 = masks2
        self.last_infos = infos
        return masks1, masks2, rewards1, rewards2, done_status, infos

    def get_batched_obs1(self, device: torch.device) -> StructuredObservation:
        return self.obs1_buffers.to(device, non_blocking=True)

    def get_batched_obs2(self, device: torch.device) -> StructuredObservation:
        return self.obs2_buffers.to(device, non_blocking=True)

    def training_state(self) -> tuple[dict[str, object], ...]:
        """Capture each simulation's RNG and active-series state."""
        return tuple(env.training_state() for env in self.envs)

    def restore_training_state(self, states: Sequence[Mapping[str, object]]) -> None:
        """Restore simulation state captured by the training_state method."""
        if len(states) != self.n_envs:
            raise ValueError("Vector environment state count does not match environment count")
        if any(not isinstance(state, Mapping) for state in states):
            raise ValueError("Vector environment states must be mappings")
        for env, state in zip(self.envs, states, strict=True):
            env.restore_training_state(state)

    def shutdown(self) -> None:
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            for env in self.envs:
                try:
                    env.close()
                except Exception:
                    # Continue closing the remaining independent environments.
                    LOGGER.exception("Failed to close a simulation environment cleanly")
