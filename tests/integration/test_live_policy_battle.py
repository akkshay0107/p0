from collections import Counter

import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.format_config import FORMAT
from p0.model.policy import MemoryInputs
from p0.model.structured_observation import StructuredObservation
from tests.integration.helpers import capture_showdown_decisions, integration_count


@pytest.mark.heavy
class TestLivePolicyBattle:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_observed_showdown_orders_round_trip_to_recorded_actions(
        self, showdown_server
    ) -> None:
        """Verify that decisions captured from Showdown always fall within legal joint actions."""
        decisions = await capture_showdown_decisions(
            showdown_server,
            game_count=integration_count("P0_INTEGRATION_ACTION_GAMES", 2),
        )
        assert decisions

        for decision in decisions:
            assert decision.chosen_action in decision.legal_joint_actions
            assert decision.chosen_order == decision.regenerated_order
            assert decision.legal_joint_actions
            assert all(
                0 <= action < ACT_SIZE for pair in decision.legal_joint_actions for action in pair
            )
            assert all(
                0 <= action < ACT_SIZE for actions in decision.legal_actions for action in actions
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", (1, 8))
    async def test_policy_handles_showdown_captured_batches(
        self,
        showdown_server,
        model_policy,
        model_device: torch.device,
        batch_size: int,
    ) -> None:
        """
        Verify the full forward pipeline (encode -> prepare -> act -> evaluate) on live battle batches.

        Tests that:
        1. Sampled actions strictly respect the action mask (invalid actions have probability 0).
        2. Sampled action pairs are valid double battle combinations in legal_joint_actions.
        3. Log-probabilities, state values, and entropies are finite.
        4. Evaluated logits are finite for legal actions and -inf for masked actions.
        """
        decisions = await capture_showdown_decisions(showdown_server, game_count=1)
        assert decisions
        # Tile captured single-turn decisions to construct the requested batch size
        selected = tuple(decisions[index % len(decisions)] for index in range(batch_size))
        cpu_observation = StructuredObservation.stack(
            [decision.observation for decision in selected]
        )
        observation = cpu_observation.to(model_device)
        for original, transferred in zip(
            cpu_observation.tensors(), observation.tensors(), strict=True
        ):
            assert transferred.device == model_device
            assert transferred.dtype == original.dtype
            torch.testing.assert_close(transferred.cpu(), original, rtol=0, atol=0)
        # Construct 3D boolean action mask: [batch, 2_slots, ACT_SIZE]
        action_mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=model_device)
        for index, decision in enumerate(selected):
            for position, actions in enumerate(decision.legal_actions):
                action_mask[index, position, list(actions)] = True
        memory = MemoryInputs.empty(
            batch_size,
            model_policy.d_model,
            model_policy.device,
            next(model_policy.parameters()).dtype,
        )

        with torch.inference_mode():
            encoded = model_policy.encode(observation, action_mask)
            prepared = model_policy.prepare(encoded, memory)
            acted = model_policy.act(prepared, action_mask)
            evaluated = model_policy.evaluate(prepared, action_mask, acted.actions)

        assert acted.actions.shape == (batch_size, 2)
        assert torch.all((acted.actions >= 0) & (acted.actions < FORMAT.action_size))
        # Gather boolean mask values at the chosen action indices to ensure every chosen action is legal
        assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
        assert torch.isfinite(acted.log_probs).all()
        assert torch.isfinite(acted.value).all()
        assert torch.isfinite(evaluated.log_probs).all()
        assert torch.isfinite(evaluated.entropy).all()
        assert torch.isfinite(evaluated.value).all()
        # Confirm sampled joint action tuples are valid double battle combinations
        for index, decision in enumerate(selected):
            actions = tuple(int(value) for value in acted.actions[index].tolist())
            assert actions in decision.legal_joint_actions

        # Check that chosen action logits are finite and masked logits are either finite or -inf
        selected_logits = evaluated.logits.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1)
        assert torch.isfinite(selected_logits).all()
        assert torch.logical_or(
            torch.isfinite(evaluated.logits), torch.isneginf(evaluated.logits)
        ).all()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_capture_captures_decisions_across_repeated_games(
        self, showdown_server
    ) -> None:
        """Verify live battle decision capture operates reliably across concurrent multi-game matches."""
        game_count = max(2, integration_count("P0_INTEGRATION_SELF_PLAY_GAMES", 2))
        decisions = await capture_showdown_decisions(
            showdown_server,
            game_count=game_count,
            max_concurrent_battles=2,
        )
        assert decisions
        by_game = Counter(decision.battle_tag for decision in decisions)
        assert len(by_game) == game_count
        assert all(count > 0 for count in by_game.values())
        assert all(decision.legal_joint_actions for decision in decisions)
        assert all(decision.observation.numerical.isfinite().all() for decision in decisions)
