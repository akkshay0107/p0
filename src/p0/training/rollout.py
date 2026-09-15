"""Self-play rollout collection with history memory."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.policy import MemoryInputs, PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.training.config import TrainingConfig
from p0.training.series_history import SeriesHistoryStore
from p0.training.trajectory import (
    CollectedTrajectory,
    PreparedTrajectory,
    TrajectoryStorage,
    prepare_trajectory_batches,
)
from p0.training.utils import OptimizationPrecision, select_optimization_precision
from p0.training.vector_env import ThreadVecEnv

ACT_SIZE = FORMAT.action_size
MAX_TRAJECTORY_STEPS = 200

__all__ = ["RolloutCollector"]


def _terminal_action_mask(
    info: Mapping[str, object], key: str, device: torch.device
) -> torch.Tensor:
    """Return one validated two-slot mask saved before a vector-env reset."""
    raw_mask = info.get(key)
    if raw_mask is None:
        raise RuntimeError(f"Truncated rollout info is missing {key}")

    mask = torch.as_tensor(raw_mask, device=device, dtype=torch.bool)
    if mask.shape != (2, ACT_SIZE):
        raise ValueError(f"Expected {key} to have shape (2, {ACT_SIZE}), got {tuple(mask.shape)}")
    return mask


@dataclass(frozen=True, slots=True)
class _SeatRollout:
    trajectories: TrajectoryStorage
    series_tokens: SeriesTokenStore
    series_history: SeriesHistoryStore


class RolloutCollector:
    """Collects self-play rollouts and manages episode history."""

    def __init__(
        self,
        vector_env: ThreadVecEnv,
        policy: PolicyNet,
        config: TrainingConfig,
        *,
        max_trajectory_steps: int = MAX_TRAJECTORY_STEPS,
    ) -> None:
        if vector_env.n_envs != config.n_envs:
            raise ValueError("Rollout environment count does not match training configuration")
        self.vector_env = vector_env
        self.policy = policy
        self.config = config
        self.completed_trajectories: list[CollectedTrajectory] = []
        self._seats = tuple(
            _SeatRollout(
                TrajectoryStorage.allocate(
                    config.n_envs,
                    max_trajectory_steps,
                    policy.d_model,
                    policy.device,
                ),
                SeriesTokenStore(policy.d_model),
                SeriesHistoryStore(policy.d_model),
            )
            for _ in range(2)
        )

    @torch.inference_mode()
    def collect(self, cancel_requested: Callable[[], bool] = lambda: False) -> None:
        """
        Collect self-play trajectory steps across all environments.

        Arguments:
            cancel_requested: Callback checked before each environment step.

        Returns:
            None.
        """
        vec_env = self.vector_env
        policy = self.policy
        n_envs = vec_env.n_envs
        device = policy.device
        idx_all = torch.arange(n_envs)
        masks1 = vec_env.last_masks1
        masks2 = vec_env.last_masks2
        if masks1 is None or masks2 is None:
            raise RuntimeError("The vector environment must be reset before rollout collection")

        precision = select_optimization_precision(self.config.enable_optim, device)
        for _ in range(self.config.rollout_steps):
            if cancel_requested():
                break
            masks = masks1, masks2
            current_obs = StructuredObservation.cat(
                [vec_env.get_batched_obs1(device), vec_env.get_batched_obs2(device)]
            )
            mask_tensors = tuple(
                torch.from_numpy(mask).to(device, non_blocking=True) for mask in masks
            )
            current_mask = torch.cat(mask_tensors)

            infos = vec_env.last_infos
            if infos is None:
                raise RuntimeError("The vector environment did not publish rollout metadata")
            if len(infos) != n_envs:
                raise ValueError(f"Expected {n_envs} environment infos, got {len(infos)}")
            series_ids = [str(info["series_id"]) for info in infos]
            series_inputs = [
                seat.series_tokens.get_tokens(series_ids, device) for seat in self._seats
            ]
            history_inputs = [
                seat.trajectories.history_inputs(idx_all, device, torch.float32)
                for seat in self._seats
            ]
            current_memory = MemoryInputs(
                series_tokens=torch.cat([value[0] for value in series_inputs]),
                series_mask=torch.cat([value[1] for value in series_inputs]),
                history_tokens=torch.cat([value[0] for value in history_inputs]),
                history_mask=torch.cat([value[1] for value in history_inputs]),
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=precision.autocast,
                dtype=precision.dtype,
            ):
                current_out = policy.act(
                    policy.prepare(policy.encode(current_obs, current_mask), current_memory),
                    current_mask,
                )

            history_tokens = current_out.history_token.chunk(2)
            actions = current_out.actions.to(device="cpu", dtype=torch.long).chunk(2)
            log_probs = current_out.log_probs.to(device="cpu", dtype=torch.float32).chunk(2)
            values = current_out.value.to(device="cpu", dtype=torch.float32).chunk(2)
            observations = vec_env.obs1_buffers, vec_env.obs2_buffers
            steps = tuple(
                seat.trajectories.record(
                    idx_all,
                    observation,
                    action,
                    log_prob,
                    value,
                    torch.from_numpy(mask).to(torch.bool),
                    history_token,
                )
                for seat, observation, action, log_prob, value, mask, history_token in zip(
                    self._seats,
                    observations,
                    actions,
                    log_probs,
                    values,
                    masks,
                    history_tokens,
                    strict=True,
                )
            )

            env_actions = [
                {
                    vec_env.envs[i].agent1.username: actions[0][i].numpy(),
                    vec_env.envs[i].agent2.username: actions[1][i].numpy(),
                }
                for i in range(n_envs)
            ]

            next_masks1, next_masks2, rewards1, rewards2, done_status, infos = vec_env.step(
                env_actions
            )

            # RL non-terminal masking only cares if the episode naturally terminated.
            # Status 1 is Terminated, Status 2 is Truncated.
            game_done = (done_status == 1).astype(np.float32)
            for seat, rewards, step_indices in zip(
                self._seats, (rewards1, rewards2), steps, strict=True
            ):
                seat.trajectories.rewards[idx_all, step_indices] = torch.from_numpy(rewards)
                seat.trajectories.dones[idx_all, step_indices] = torch.from_numpy(game_done)

            for i in range(n_envs):
                if done_status[i]:
                    self._finish_game(
                        i,
                        int(done_status[i]),
                        infos[i],
                        series_ids[i],
                        precision,
                    )

            masks1 = next_masks1
            masks2 = next_masks2

    def _finish_game(
        self,
        env_id: int,
        done_status: int,
        info: Mapping[str, object],
        series_id: str,
        precision: OptimizationPrecision,
    ) -> None:
        """
        Complete both seat trajectories and retain context for the next game.

        Arguments:
            env_id: Environment index.
            done_status: 1 if game terminated naturally, 2 if truncated.
            info: Terminal metadata captured before the automatic reset.
            series_id: Identifier of the series that produced the game.
            precision: Selected inference autocast precision.

        Returns:
            None.
        """
        policy = self.policy
        device = policy.device
        snapshots = tuple(seat.series_history.snapshot(series_id) for seat in self._seats)
        bootstrap_values = (0.0, 0.0)
        if done_status == 2:
            terminal_observations: list[StructuredObservation] = []
            for key in ("terminal_observation1", "terminal_observation2"):
                observation = info.get(key)
                if not isinstance(observation, StructuredObservation):
                    raise RuntimeError(f"Truncated rollout info is missing {key}")
                terminal_observations.append(observation.unsqueeze(0).to(device))
            terminal_obs = StructuredObservation.cat(terminal_observations)
            terminal_mask = torch.stack(
                tuple(
                    _terminal_action_mask(info, f"terminal_action_mask{seat_index}", device)
                    for seat_index in (1, 2)
                )
            )
            env_index = torch.tensor([env_id], dtype=torch.long)
            history_inputs = [
                seat.trajectories.history_inputs(env_index, device, torch.float32)
                for seat in self._seats
            ]
            series_inputs = [
                seat.series_tokens.get_tokens([series_id], device) for seat in self._seats
            ]
            memory = MemoryInputs(
                series_tokens=torch.cat([value[0] for value in series_inputs]),
                series_mask=torch.cat([value[1] for value in series_inputs]),
                history_tokens=torch.cat([value[0] for value in history_inputs]),
                history_mask=torch.cat([value[1] for value in history_inputs]),
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=precision.autocast,
                dtype=precision.dtype,
            ):
                values = policy.act(
                    policy.prepare(policy.encode(terminal_obs, terminal_mask), memory),
                    terminal_mask,
                ).value
            bootstrap_values = float(values[0].item()), float(values[1].item())

        histories = tuple(seat.trajectories.full_history(env_id) for seat in self._seats)
        if any(history is None for history in histories):
            raise RuntimeError("A completed game has no policy history")
        complete_histories = cast(tuple[torch.Tensor, ...], histories)

        if info.get("series_complete"):
            for seat in self._seats:
                seat.series_tokens.drop(series_id)
                seat.series_history.drop(series_id)
        else:
            game_history = torch.cat(complete_histories)
            game_mask = torch.ones(
                game_history.shape[:2], dtype=torch.bool, device=game_history.device
            )
            summary_tokens = policy.series(game_history, game_mask)
            for seat, history, summary in zip(
                self._seats, complete_histories, summary_tokens, strict=True
            ):
                seat.series_history.append(
                    series_id,
                    seat.series_history.next_game_number(series_id),
                    history[0],
                    is_game_end=True,
                    is_series_end=False,
                )
                seat.series_tokens.append(series_id, summary)

        for seat, bootstrap, snapshot in zip(self._seats, bootstrap_values, snapshots, strict=True):
            self.completed_trajectories.append(
                seat.trajectories.complete(env_id, bootstrap, snapshot)
            )

    def reset_completed(self) -> None:
        self.completed_trajectories.clear()

    def prepare_for_checkpoint(self) -> None:
        """Restart at a clean battle boundary before persisting PPO state."""
        self.completed_trajectories.clear()
        for seat in self._seats:
            seat.trajectories.step_counts.zero_()
            seat.series_history.discard_active_games()
        for env in self.vector_env.envs:
            env.prepare_for_checkpoint()
        self.vector_env.reset()

    def training_state(self) -> dict[str, object]:
        """Capture cross-game context retained by the collector."""
        state: dict[str, object] = {}
        for index, seat in enumerate(self._seats, start=1):
            state[f"series_store{index}"] = seat.series_tokens.training_state()
            state[f"series_history{index}"] = seat.series_history.training_state()
        return state

    def restore_training_state(self, state: Mapping[str, object]) -> None:
        """Restore cross-game context captured at an episode boundary."""
        for index, seat in enumerate(self._seats, start=1):
            tokens = state.get(f"series_store{index}")
            history = state.get(f"series_history{index}")
            if not isinstance(tokens, Mapping) or not isinstance(history, Mapping):
                raise ValueError("Invalid PPO collector training state")
            seat.series_tokens.restore_training_state(tokens)
            seat.series_history.restore_training_state(history)

    def get_batches(self, device: torch.device) -> list[PreparedTrajectory]:
        return prepare_trajectory_batches(
            self.completed_trajectories,
            device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
