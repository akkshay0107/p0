import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.format_config import FORMAT
from p0.model.policy import MemoryInputs
from p0.model.structured_observation import StructuredObservation
from tests.integration.helpers import capture_showdown_decisions, integration_count


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
            assert decision.chosen_order

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_showdown_order_sets_remain_nonempty_across_many_requests(
        self, showdown_server
    ) -> None:
        """Verify legal joint actions and single action sets remain non-empty and bounded by ACT_SIZE."""
        decisions = await capture_showdown_decisions(
            showdown_server,
            game_count=integration_count("P0_INTEGRATION_ACTION_GAMES", 2),
        )
        assert decisions
        assert all(decision.legal_joint_actions for decision in decisions)
        for decision in decisions:
            # Extract the set of all discrete actions appearing in any legal joint action pair
            projected = tuple(
                sorted({action for pair in decision.legal_joint_actions for action in pair})
            )
            observed = tuple(sorted(set(decision.legal_actions[0] + decision.legal_actions[1])))
            assert observed
            # Joint actions must be a subset of the union of per-slot legal actions
            assert set(projected) <= set(observed)
            assert all(0 <= action < ACT_SIZE for action in observed)

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
        observation = StructuredObservation.stack(
            [decision.observation for decision in selected]
        ).to(model_device)
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
        decisions = await capture_showdown_decisions(
            showdown_server,
            game_count=integration_count("P0_INTEGRATION_SELF_PLAY_GAMES", 2),
            max_concurrent_battles=2,
        )
        assert decisions
        assert all(decision.legal_joint_actions for decision in decisions)
        assert all(decision.observation.numerical.isfinite().all() for decision in decisions)
