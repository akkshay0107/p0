import asyncio

import pytest
import torch
from poke_env.player import RandomPlayer

from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)
from p0.model.tokenizer import tokenizer
from tests.stress._helpers import capture_showdown_decisions, stress_count
from tests.unit.test_battle import from_battle


@pytest.mark.integration
@pytest.mark.asyncio
async def test_observation_builder_live(showdown_server, battle_format, sample_team):
    """Exercise observation construction against the real local Showdown protocol."""
    captured_battles = []
    captured_errors = []

    class CapturePlayer(RandomPlayer):
        def teampreview(self, battle):
            try:
                captured_battles.append((battle.teampreview, from_battle(battle, tokenizer)))
            except Exception as exc:
                captured_errors.append(f"teampreview: {exc}")
            return super().teampreview(battle)

        def choose_move(self, battle):
            try:
                captured_battles.append((battle.teampreview, from_battle(battle, tokenizer)))
            except Exception as exc:
                captured_errors.append(f"choose_move: {exc}")
            return super().choose_move(battle)

    p1 = CapturePlayer(
        battle_format=battle_format,
        server_configuration=showdown_server,
        team=sample_team,
        max_concurrent_battles=1,
    )
    p2 = RandomPlayer(
        battle_format=battle_format,
        server_configuration=showdown_server,
        team=sample_team,
        max_concurrent_battles=1,
    )

    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=15.0)
    except asyncio.TimeoutError:
        pytest.fail(f"Battle timed out. Internal errors: {captured_errors}")
    except Exception as exc:
        pytest.fail(f"Battle failed with exception: {exc}. Internal errors: {captured_errors}")

    assert not captured_errors
    assert captured_battles
    assert any(is_teampreview for is_teampreview, _ in captured_battles)
    assert any(not is_teampreview for is_teampreview, _ in captured_battles)
    for _, obs in captured_battles:
        assert isinstance(obs, StructuredObservation)
        assert obs.categorical.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
        assert obs.numerical.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)


@pytest.mark.integration
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
@pytest.mark.asyncio
async def test_live_observations_transfer_to_each_available_model_device(
    showdown_server,
    model_device,
) -> None:
    decisions = await capture_showdown_decisions(showdown_server, game_count=1)
    assert decisions
    observation = StructuredObservation.stack([decisions[0].observation]).to(model_device)
    assert all(tensor.device == model_device for tensor in observation.tensors())
