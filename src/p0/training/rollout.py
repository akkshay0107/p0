"""Self-play rollout collection with history memory."""

from collections.abc import Callable
from typing import NamedTuple, cast

import numpy as np
import torch

from p0.format_config import FORMAT
from p0.model.policy import MemoryInputs, PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.model.token_store import SeriesTokenStore
from p0.training.config import PPOConfig
from p0.training.series_history import SeriesHistoryStore
from p0.training.trajectory import (
    CollectedTrajectory,
    PreparedTrajectory,
    TrajectoryStorage,
    prepare_trajectory_batches,
)
from p0.training.utils import OptimizationPrecision, select_optimization_precision
from p0.training.vector_env import FinishedGame, ThreadVecEnv

ACT_SIZE = FORMAT.action_size
MAX_TRAJECTORY_STEPS = 200

__all__ = ["RolloutCollector"]


def _terminal_action_mask(raw_mask: np.ndarray, device: torch.device) -> torch.Tensor:
    """Return one validated two-slot mask saved before a vector-env reset."""
    mask = torch.as_tensor(raw_mask, device=device, dtype=torch.bool)
    if mask.shape != (2, ACT_SIZE):
        raise ValueError(
            f"Expected a terminal action mask of shape (2, {ACT_SIZE}), got {tuple(mask.shape)}"
        )
    return mask


class _SeatRollout(NamedTuple):
    trajectories: TrajectoryStorage
    series_tokens: SeriesTokenStore
    series_history: SeriesHistoryStore


class RolloutCollector:
    """Collects self-play rollouts and manages episode history."""

    def __init__(
        self,
        vector_env: ThreadVecEnv,
        policy: PolicyNet,
        config: PPOConfig,
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
            to_move = torch.tensor([cast(tuple[bool, bool], info["to_move"]) for info in infos])
            retry = torch.tensor([cast(tuple[bool, bool], info["retry"]) for info in infos])
            for seat_index, seat in enumerate(self._seats):
                # A seat whose choice was rejected decides the same request again; its
                # new decision replaces the rejected one in history and trajectory.
                seat.trajectories.step_counts[retry[:, seat_index]] -= 1
            current_memory = self._memory_inputs(idx_all, series_ids, device)

            with precision.autocast_context(device):
                current_out = policy.act(
                    policy.prepare(policy.encode(current_obs, current_mask), current_memory),
                    current_mask,
                )

            history_tokens = current_out.history_token.chunk(2)
            actions = current_out.actions.to(device="cpu", dtype=torch.long).chunk(2)
            log_probs = current_out.log_probs.to(device="cpu", dtype=torch.float32).chunk(2)
            values = current_out.value.to(device="cpu", dtype=torch.float32).chunk(2)
            observations = vec_env.obs1_buffers, vec_env.obs2_buffers
            # Both seats are always evaluated so the batch shape stays fixed, but a
            # seat is recorded only when it acts. A waiting seat's forced double
            # pass is never sent, and live play and replays give it no decision.
            for seat_index, seat in enumerate(self._seats):
                acting = to_move[:, seat_index]
                env_ids = idx_all[acting]
                seat.trajectories.record(
                    env_ids,
                    observations[seat_index][env_ids],
                    actions[seat_index][env_ids],
                    log_probs[seat_index][env_ids],
                    values[seat_index][env_ids],
                    torch.from_numpy(masks[seat_index]).to(torch.bool)[env_ids],
                    history_tokens[seat_index][acting.to(device)],
                )

            env_actions = [
                {
                    vec_env.envs[i].agent1.username: actions[0][i].numpy(),
                    vec_env.envs[i].agent2.username: actions[1][i].numpy(),
                }
                for i in range(n_envs)
            ]

            masks1, masks2, rewards1, rewards2, done_status, _, finished = vec_env.step(env_actions)

            # RL non-terminal masking only cares if the episode naturally terminated.
            # Status 1 is Terminated, Status 2 is Truncated.
            game_done = (done_status == 1).astype(np.float32)
            for seat, rewards in zip(self._seats, (rewards1, rewards2), strict=True):
                # Rewards are terminal-only, so each step's outcome belongs to the
                # seat's latest recorded decision even when it did not act this step.
                last_steps = seat.trajectories.step_counts - 1
                if (last_steps < 0).any():
                    raise RuntimeError(
                        "A seat reached an environment step with no recorded decision"
                    )
                seat.trajectories.rewards[idx_all, last_steps] = torch.from_numpy(rewards)
                seat.trajectories.dones[idx_all, last_steps] = torch.from_numpy(game_done)

            for i, game in enumerate(finished):
                if game is not None:
                    self._finish_game(i, game, precision)

    def _memory_inputs(
        self, env_ids: torch.Tensor, series_ids: list[str], device: torch.device
    ) -> MemoryInputs:
        """Stack both seats' series and battle-history memory, seat 1 rows first."""
        series_inputs = [seat.series_tokens.get_tokens(series_ids, device) for seat in self._seats]
        history_inputs = [
            seat.trajectories.history_inputs(env_ids, device, torch.float32) for seat in self._seats
        ]
        return MemoryInputs(
            series_tokens=torch.cat([value[0] for value in series_inputs]),
            series_mask=torch.cat([value[1] for value in series_inputs]),
            history_tokens=torch.cat([value[0] for value in history_inputs]),
            history_mask=torch.cat([value[1] for value in history_inputs]),
        )

    def _finish_game(
        self,
        env_id: int,
        game: FinishedGame,
        precision: OptimizationPrecision,
    ) -> None:
        """
        Complete both seat trajectories and retain context for the next game.

        Arguments:
            env_id: Environment index.
            game: The finished game, captured before the automatic reset.
            precision: Selected inference autocast precision.

        Returns:
            None.
        """
        policy = self.policy
        device = policy.device
        series_id = game.series_id
        done_status = game.done_status
        snapshots = tuple(seat.series_history.snapshot(series_id) for seat in self._seats)
        terminal_obs = StructuredObservation.cat(
            [
                observation.unsqueeze(0).to(device)
                for observation in (game.observation1, game.observation2)
            ]
        )
        terminal_mask = torch.stack(
            tuple(
                _terminal_action_mask(mask, device)
                for mask in (game.action_mask1, game.action_mask2)
            )
        )
        with precision.autocast_context(device):
            terminal_encoded = policy.encode(terminal_obs, terminal_mask)

        bootstrap_values = (0.0, 0.0)
        if done_status == 2:
            memory = self._memory_inputs(
                torch.tensor([env_id], dtype=torch.long), [series_id], device
            )
            with precision.autocast_context(device):
                values = policy.act(
                    policy.prepare(terminal_encoded, memory),
                    terminal_mask,
                ).value
            bootstrap_values = float(values[0].item()), float(values[1].item())

        histories = tuple(seat.trajectories.full_history(env_id) for seat in self._seats)
        if any(history is None for history in histories):
            raise RuntimeError("A completed game has no policy history")
        complete_histories = cast(tuple[torch.Tensor, ...], histories)
        if done_status == 1:
            # The final exchange arrives after the last decision; its board summary
            # joins the series history but is never a trajectory row.
            complete_histories = tuple(
                torch.cat((history, final.to(history).view(1, 1, -1)), dim=1)
                for history, final in zip(
                    complete_histories, terminal_encoded.local_history_token, strict=True
                )
            )

        if game.series_complete:
            for seat in self._seats:
                seat.series_tokens.drop(series_id)
                seat.series_history.drop(series_id)
        else:
            # Seats record different decision counts, so pad to a shared length.
            lengths = torch.tensor([history.size(1) for history in complete_histories])
            game_history = torch.nn.utils.rnn.pad_sequence(
                [history[0] for history in complete_histories], batch_first=True
            )
            game_mask = torch.arange(game_history.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
            summary_tokens = policy.series(game_history, game_mask.to(game_history.device))
            for seat, history, summary in zip(
                self._seats, complete_histories, summary_tokens, strict=True
            ):
                seat.series_history.append(
                    series_id,
                    seat.series_history.next_game_number(series_id),
                    history[0],
                    is_series_end=False,
                )
                seat.series_tokens.append(series_id, summary)

        for seat, bootstrap, snapshot in zip(self._seats, bootstrap_values, snapshots, strict=True):
            self.completed_trajectories.append(
                seat.trajectories.complete(env_id, bootstrap, snapshot)
            )

    def reset_completed(self) -> None:
        self.completed_trajectories.clear()

    def get_batches(self, device: torch.device) -> list[PreparedTrajectory]:
        return prepare_trajectory_batches(
            self.completed_trajectories,
            device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
