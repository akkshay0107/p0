from __future__ import annotations

import pytest
import torch

from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    EVENT_COUNT,
    MAX_EFFECTS,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    StructuredObservation,
)
from p0.replays.protocol import parse_replay_payload
from p0.replays.reconstruct import reconstruct_both
from tests.stress._helpers import capture_showdown_decisions, stress_count
from tests.stress.replay_fixtures import golden_replay_payload


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


@pytest.mark.stress
def test_observation_overflow_contract_holds_at_capacity_boundaries() -> None:
    observation = StructuredObservation.empty_batch(3)
    observation.numerical[0, 0, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS - 1
    observation.numerical[1, 0, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS
    observation.numerical[2, 0, NUM_IDX_EFFECT_COUNT] = MAX_EFFECTS + 2
    observation.numerical[2, 0, NUM_IDX_EFFECT_OVERFLOW] = 2

    observation.events_cat[0, : EVENT_COUNT - 1, 0] = 1
    observation.events_cat[1, :, 0] = 1
    observation.events_cat[2, :, 0] = 1
    observation.events_metadata[:] = torch.tensor(
        ((EVENT_COUNT - 1, 0), (EVENT_COUNT, 0), (EVENT_COUNT + 3, 3)),
        dtype=torch.float32,
    )

    observation.validate_overflow_contract()
    assert observation.overflow_totals() == (2, 3)


@pytest.mark.stress
def test_reconstructed_observations_clear_reused_buffer_state() -> None:
    document = parse_replay_payload(golden_replay_payload("buffer-reuse"))
    perspective = reconstruct_both(document)[0]
    assert len(perspective.snapshots) >= 2

    builder = ObservationBuilder(default_runtime_resources())
    output = StructuredObservation.empty_batch(1)[0]
    for snapshot in perspective.snapshots:
        snapshot.view.events = list(snapshot.events)
        output.token_type_ids.fill_(99)
        output.categorical.fill_(99)
        output.numerical.fill_(99.0)
        output.events_cat.fill_(99)
        output.events_num.fill_(99.0)
        output.events_side_ids.fill_(99)
        output.events_slot_ids.fill_(99)
        output.events_metadata.fill_(99.0)

        builder.build_into(snapshot.view, output)
        output.validate(batch_rank=0)
        output.validate_overflow_contract()
        assert all(torch.isfinite(tensor).all() for tensor in output.tensors())
        assert output.events_metadata[0].item() == len(snapshot.events)
        assert not torch.any(output.events_cat[len(snapshot.events) :, :])
        assert not torch.any(output.events_num[len(snapshot.events) :, :])
