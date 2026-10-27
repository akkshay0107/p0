"""Live rollout memory refresh across a policy update."""

from copy import deepcopy
from urllib.parse import urlparse

import pytest
import torch
from poke_env import AccountConfiguration

from p0.model.architecture_contract import SERIES_TOKENS_PER_GAME
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import MemoryInputs
from p0.runtime.composition import build_sim_env
from p0.training.config import PPOConfig
from p0.training.rollout import MAX_TRAJECTORY_STEPS, RolloutCollector
from p0.training.vector_env import ThreadVecEnv
from tests.team_fixtures import default_test_corpus


@pytest.mark.heavy
@pytest.mark.integration
class TestRolloutCollector:
    @torch.inference_mode()
    def test_refresh_matches_fresh_scoring_and_preserves_behavior_data(
        self, showdown_server, model_policy
    ) -> None:
        """
        Compare resumed decisions with fresh memory and preserve recorded behavior.

        Arguments:
            showdown_server: Local Showdown server configuration.
            model_policy: Small policy on the selected test device.

        Returns:
            None.
        """
        torch.manual_seed(7)
        server_port = urlparse(showdown_server.websocket_url).port
        assert server_port is not None
        env = build_sim_env(
            account_configuration1=AccountConfiguration("RefreshAgent", None),
            account_configuration2=AccountConfiguration("RefreshOpponent", None),
            server_port=server_port,
            agent_team_source=default_test_corpus(),
            opponent_team_source=default_test_corpus(),
            observation_builder=ObservationBuilder(model_policy.resources),
        )
        vector_env = ThreadVecEnv([env])
        try:
            vector_env.reset()
            collector = RolloutCollector(
                vector_env, model_policy, PPOConfig(n_envs=1, rollout_steps=1, enable_optim=False)
            )
            collector.refresh_memory()  # Empty caches need no encoder or resampler work.
            for _ in range(MAX_TRAJECTORY_STEPS):
                collector.collect()
                if collector.completed_trajectories:
                    break
            assert len(collector.completed_trajectories) == 2
            completed_before = deepcopy(collector.completed_trajectories)

            collector.collect()  # Record game two's team preview under the old weights.
            assert vector_env.last_infos is not None
            assert vector_env.last_infos[0]["to_move"] == (True, True)
            old_policy = deepcopy(model_policy)
            for module in (model_policy.local_mixer, model_policy.series):
                for parameter in module.parameters():
                    parameter.add_(0.1 * torch.randn_like(parameter))
            collector.refresh_memory()
            collector.refresh_memory()  # Replacing summaries must not append extra games.

            for _ in range(MAX_TRAJECTORY_STEPS):
                collector.collect()
                if len(collector.completed_trajectories) >= 4:
                    break
            assert len(collector.completed_trajectories) == 4
            device = model_policy.device
            for prior, unchanged, current in zip(
                completed_before,
                collector.completed_trajectories[:2],
                collector.completed_trajectories[2:],
                strict=True,
            ):
                for before, after in zip(
                    (prior.actions, prior.log_probs, prior.values, prior.rewards),
                    (unchanged.actions, unchanged.log_probs, unchanged.values, unchanged.rewards),
                    strict=True,
                ):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                (history,) = current.series_history
                old_local = old_policy.encode(
                    prior.observations.to(device), prior.action_masks.to(device)
                ).local_history_token
                torch.testing.assert_close(history[: prior.length].to(device), old_local)

                history = history.unsqueeze(0).to(device)
                history_mask = torch.ones(history.shape[:2], device=device, dtype=torch.bool)
                fresh = MemoryInputs.empty(1, model_policy.d_model, device, torch.float32)
                stale = MemoryInputs.empty(1, model_policy.d_model, device, torch.float32)
                fresh.series_tokens[:, :SERIES_TOKENS_PER_GAME] = model_policy.series(
                    history, history_mask
                )
                stale.series_tokens[:, :SERIES_TOKENS_PER_GAME] = old_policy.series(
                    history, history_mask
                )
                fresh.series_mask[:, :SERIES_TOKENS_PER_GAME] = True
                stale.series_mask[:, :SERIES_TOKENS_PER_GAME] = True
                observations = current.observations[:2].to(device)
                masks = current.action_masks[:2].to(device)
                actions = current.actions[:2].to(device)
                old_encoded = old_policy.encode(observations, masks)
                new_encoded = model_policy.encode(observations, masks)

                # The first decision's original log probability and value survive refresh.
                original = old_policy.evaluate(
                    old_policy.prepare(old_encoded[:1], stale), masks[:1], actions[:1]
                )
                torch.testing.assert_close(current.log_probs[:1].to(device), original.log_probs)
                torch.testing.assert_close(current.values[:1].to(device), original.value)
                fresh.history_tokens[:, -1] = new_encoded.local_history_token[0]
                stale.history_tokens[:, -1] = old_encoded.local_history_token[0]
                fresh.history_mask[:, -1] = stale.history_mask[:, -1] = True
                expected = model_policy.evaluate(
                    model_policy.prepare(new_encoded[1:2], fresh), masks[1:2], actions[1:2]
                )
                without_refresh = model_policy.evaluate(
                    model_policy.prepare(new_encoded[1:2], stale), masks[1:2], actions[1:2]
                )
                torch.testing.assert_close(current.log_probs[1:2].to(device), expected.log_probs)
                torch.testing.assert_close(current.values[1:2].to(device), expected.value)
                assert not torch.isclose(
                    expected.log_probs, without_refresh.log_probs, rtol=0, atol=1e-6
                ).all()
        finally:
            vector_env.shutdown()
