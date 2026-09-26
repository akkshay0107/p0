from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest
from poke_env import AccountConfiguration

from p0.battle.actions import PASS_ACTION
from p0.model.observation_builder import ObservationBuilder
from p0.runtime import poke_env_patches
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.teams.source import FixedTeamSource
from p0.training.checkpoint import CheckpointStore
from p0.training.config import TrainingConfig
from p0.training.rollout import RolloutCollector
from p0.training.trajectory import CollectedTrajectory
from p0.training.vector_env import ThreadVecEnv
from tests.team_fixtures import DEFAULT_TEST_TEAM


@pytest.mark.heavy
class TestPpoSimulation:
    @pytest.mark.integration
    def test_two_simulated_bo3_series_run_concurrently(self, showdown_server, model_policy) -> None:
        """Run two real local PPO simulations concurrently through ThreadVecEnv."""
        server_port = urlparse(showdown_server.websocket_url).port
        if server_port is None:
            raise ValueError("The integration server configuration has no websocket port")

        agent_sources = [FixedTeamSource(DEFAULT_TEST_TEAM) for _ in range(2)]
        opponent_sources = [FixedTeamSource(DEFAULT_TEST_TEAM) for _ in range(2)]
        envs: list[SimEnv] = []
        vector_env: ThreadVecEnv | None = None
        completed_series: set[int] = set()
        completed_series_ids: set[str] = set()

        poke_env_patches.install()
        try:
            for index, (agent_source, opponent_source) in enumerate(
                zip(agent_sources, opponent_sources, strict=True)
            ):
                envs.append(
                    build_sim_env(
                        account_configuration1=AccountConfiguration(f"PPOAgent{index}", None),
                        account_configuration2=AccountConfiguration(f"PPOOpponent{index}", None),
                        server_port=server_port,
                        agent_team_source=agent_source,
                        opponent_team_source=opponent_source,
                        observation_builder=ObservationBuilder(model_policy.resources),
                        agent_seed=index * 2,
                        opponent_seed=index * 2 + 1,
                    )
                )

            vector_env = ThreadVecEnv(envs)
            initial_state = envs[0].training_state()
            assert isinstance(initial_state["agent_team"], str)
            assert isinstance(initial_state["opponent_team"], str)
            vector_env.reset()
            collector = RolloutCollector(
                vector_env,
                model_policy,
                TrainingConfig(n_envs=2, rollout_steps=1),
            )

            uneven_game = False
            previous_games: dict[int, tuple[str, tuple[CollectedTrajectory, ...]]] = {}
            final_entry_checks = 0
            for _ in range(1_200):
                active_series_ids = tuple(
                    str(info["series_id"]) for info in (vector_env.last_infos or ())
                )
                completed_before = len(collector.completed_trajectories)
                collector.collect()
                infos = vector_env.last_infos
                if infos is None:
                    raise AssertionError("ThreadVecEnv did not publish battle metadata")
                # Games finishing in one step are completed in environment order,
                # seat 1 then seat 2.
                new_trajectories = collector.completed_trajectories[completed_before:]
                finished = [
                    index for index, info in enumerate(infos) if "terminal_observation1" in info
                ]
                for index, seats in zip(
                    finished,
                    zip(new_trajectories[::2], new_trajectories[1::2], strict=True),
                    strict=True,
                ):
                    previous = previous_games.get(index)
                    if previous is not None and previous[0] == active_series_ids[index]:
                        for prior, current in zip(previous[1], seats, strict=True):
                            # A finished game adds its final board after the recorded
                            # decisions; a truncated game adds nothing.
                            final_entries = int(prior.dones[-1].item())
                            assert current.series_history[-1].size(0) == (
                                prior.length + final_entries
                            )
                            final_entry_checks += final_entries
                    previous_games[index] = (active_series_ids[index], seats)
                for index, info in enumerate(infos):
                    if info.get("series_complete"):
                        completed_series.add(index)
                        completed_series_ids.add(active_series_ids[index])
                # Each game appends seat 1 then seat 2. Opponent-only replacements give
                # the seats different decision counts, which game completion must accept;
                # keep playing until at least one such game has finished.
                trajectories = collector.completed_trajectories
                uneven_game = any(
                    first.length != second.length
                    for first, second in zip(trajectories[::2], trajectories[1::2], strict=True)
                )
                if completed_series == {0, 1} and uneven_game:
                    break

            assert completed_series == {0, 1}
            assert uneven_game
            assert final_entry_checks > 0
            trajectories = collector.completed_trajectories
            assert len(trajectories) >= 8
            for trajectory in trajectories:
                # A waiting seat's double-pass step is not a decision and is never recorded.
                pass_only = trajectory.action_masks[..., PASS_ACTION] & (
                    trajectory.action_masks.sum(-1) == 1
                )
                assert not pass_only.all(dim=-1).any()
                assert not trajectory.rewards[:-1].any()
            collector_state = collector.training_state()
            store1 = collector_state["series_store1"]
            store2 = collector_state["series_store2"]
            history1 = collector_state["series_history1"]
            history2 = collector_state["series_history2"]
            assert isinstance(store1, dict) and isinstance(store2, dict)
            assert isinstance(history1, dict) and isinstance(history2, dict)
            assert not completed_series_ids.intersection(store1)
            assert not completed_series_ids.intersection(store2)
            assert not completed_series_ids.intersection(history1)
            assert not completed_series_ids.intersection(history2)
        finally:
            if vector_env is not None:
                vector_env.shutdown()
            poke_env_patches.uninstall_for_tests()


@pytest.mark.heavy
class TestThirdGameCheckpointResume:
    @pytest.mark.integration
    def test_checkpoint_during_third_game_resumes_the_same_series(
        self, showdown_server, model_policy, tmp_path: Path
    ) -> None:
        """A checkpoint taken during game 3 must not start a new series on save or resume."""
        server_port = urlparse(showdown_server.websocket_url).port
        if server_port is None:
            raise ValueError("The integration server configuration has no websocket port")
        # Resume with different team sources, so matching teams can only come from restore.
        members = DEFAULT_TEST_TEAM.strip().split("\n\n")
        other_team = "\n\n".join(reversed(members))
        store = CheckpointStore()
        checkpoint = tmp_path / "third_game.pt"
        vector_env: ThreadVecEnv | None = None

        poke_env_patches.install()
        try:
            env = build_sim_env(
                account_configuration1=AccountConfiguration("ResumeAgent0", None),
                account_configuration2=AccountConfiguration("ResumeOpponent0", None),
                server_port=server_port,
                agent_team_source=FixedTeamSource(DEFAULT_TEST_TEAM),
                opponent_team_source=FixedTeamSource(DEFAULT_TEST_TEAM),
                observation_builder=ObservationBuilder(model_policy.resources),
            )
            vector_env = ThreadVecEnv([env])
            vector_env.reset()
            collector = RolloutCollector(
                vector_env, model_policy, TrainingConfig(n_envs=1, rollout_steps=1)
            )
            for _ in range(3_000):
                if env.series_games_played == 3:
                    break
                collector.collect()
            assert env.series_games_played == 3
            active = env.training_state()

            collector.prepare_for_checkpoint()
            store.save_policy(
                checkpoint,
                model_policy,
                metadata={
                    "environment_state": vector_env.training_state(),
                    "collector_state": collector.training_state(),
                },
            )
            vector_env.shutdown()
            vector_env = None

            metadata = store.load_metadata(store.read(checkpoint))
            resumed_env = build_sim_env(
                account_configuration1=AccountConfiguration("ResumeAgent1", None),
                account_configuration2=AccountConfiguration("ResumeOpponent1", None),
                server_port=server_port,
                agent_team_source=FixedTeamSource(other_team),
                opponent_team_source=FixedTeamSource(other_team),
                observation_builder=ObservationBuilder(model_policy.resources),
            )
            vector_env = ThreadVecEnv([resumed_env])
            resumed_collector = RolloutCollector(
                vector_env, model_policy, TrainingConfig(n_envs=1, rollout_steps=1)
            )
            vector_env.restore_training_state(metadata["environment_state"])
            resumed_collector.restore_training_state(metadata["collector_state"])
            _, _, infos = vector_env.reset()

            resumed = resumed_env.training_state()
            assert infos[0]["series_id"] == active["series_id"]
            assert resumed["series_id"] == active["series_id"]
            assert resumed["series_scores"] == active["series_scores"]
            assert resumed["series_games_played"] == 3
            assert resumed["agent_team"] == active["agent_team"]
            assert resumed["opponent_team"] == active["opponent_team"]
            collector_state = resumed_collector.training_state()
            for index in (1, 2):
                history = collector_state[f"series_history{index}"]
                assert isinstance(history, dict)
                series = history[active["series_id"]]
                assert [number for number, _ in series["completed_games"]] == [1, 2]
                assert series["active_game_number"] is None

            # The restored third game ends the series.
            for _ in range(3_000):
                resumed_collector.collect()
                infos = vector_env.last_infos
                assert infos is not None
                if "terminal_observation1" in infos[0]:
                    break
            assert infos[0]["series_complete"] is True
            assert resumed_env.series_id != active["series_id"]
            assert resumed_env.series_games_played == 1
        finally:
            if vector_env is not None:
                vector_env.shutdown()
            poke_env_patches.uninstall_for_tests()
