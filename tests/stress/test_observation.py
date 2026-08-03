from __future__ import annotations

import pytest
import torch

from p0.model.structured_observation import StructuredObservation
from tests.stress._helpers import capture_showdown_decisions, stress_count


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_live_showdown_observations_remain_valid(showdown_server) -> None:
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=stress_count("P0_STRESS_OBSERVATION_GAMES", 2),
    )
    assert decisions

    for decision in decisions:
        observation = decision.observation
        observation.validate_overflow_contract()
        assert all(torch.isfinite(tensor).all() for tensor in observation.tensors())
        assert all(tensor.device.type == "cpu" for tensor in observation.tensors())


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.asyncio
async def test_live_observations_transfer_to_each_available_model_device(
    showdown_server,
    stress_policy,
    stress_device,
) -> None:
    decisions = await capture_showdown_decisions(showdown_server, game_count=1)
    assert decisions
    observation = StructuredObservation.stack([decisions[0].observation]).to(stress_device)
    assert all(tensor.device == stress_device for tensor in observation.tensors())
