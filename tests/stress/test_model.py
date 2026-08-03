from __future__ import annotations

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.structured_observation import StructuredObservation
from tests.stress._helpers import capture_showdown_decisions, stress_batch_sizes

ACT_SIZE = FORMAT.action_size


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", stress_batch_sizes())
async def test_policy_handles_showdown_captured_batches(
    showdown_server,
    stress_policy,
    stress_device: torch.device,
    batch_size: int,
) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=1,
    )
    assert decisions
    selected = tuple(decisions[index % len(decisions)] for index in range(batch_size))
    observation = StructuredObservation.stack([decision.observation for decision in selected]).to(
        stress_device
    )
    action_mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=stress_device)
    for index, decision in enumerate(selected):
        for position, actions in enumerate(decision.legal_actions):
            action_mask[index, position, list(actions)] = True
    memory = stress_policy.empty_memory(batch_size)

    with torch.inference_mode():
        acted = stress_policy.act_obs(observation, action_mask, *memory)
        evaluated = stress_policy.evaluate_obs(
            observation,
            action_mask,
            acted.actions,
            *memory,
        )

    assert acted.actions.shape == (batch_size, 2)
    assert torch.all((acted.actions >= 0) & (acted.actions < ACT_SIZE))
    assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
    assert torch.isfinite(acted.log_probs).all()
    assert torch.isfinite(acted.value).all()
    assert torch.isfinite(evaluated.log_probs).all()
    assert torch.isfinite(evaluated.entropy).all()
    assert torch.isfinite(evaluated.value).all()
    for index, decision in enumerate(selected):
        actions = tuple(int(value) for value in acted.actions[index].tolist())
        assert actions in decision.legal_joint_actions

    assert torch.isfinite(evaluated.logits.masked_select(action_mask)).all()
