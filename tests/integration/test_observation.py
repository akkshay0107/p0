import asyncio
from typing import Any

import pytest
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer

from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view
from tests.integration.helpers import capture_showdown_decisions, integration_count


def _build_double_observation(
    battle: AbstractBattle, builder: ObservationBuilder
) -> StructuredObservation:
    """Safely cast and featurize a DoubleBattle instance into a StructuredObservation."""
    if not isinstance(battle, DoubleBattle):
        raise TypeError(f"Expected DoubleBattle, got {type(battle).__name__}")
    return builder.build(battle_view(battle))


class _ObservationCapturePlayer(RandomPlayer):
    """Test harness player that records observations and caught errors during teampreview and move choices."""

    def __init__(
        self,
        *,
        observation_builder: ObservationBuilder,
        captured_battles: list[tuple[bool, StructuredObservation]],
        captured_errors: list[str],
        **kwargs: Any,
    ) -> None:
        self.observation_builder = observation_builder
        self.captured_battles = captured_battles
        self.captured_errors = captured_errors
        super().__init__(**kwargs)

    def _capture_observation(self, battle: AbstractBattle, callback: str) -> None:
        try:
            self.captured_battles.append(
                (
                    battle.teampreview,
                    _build_double_observation(battle, self.observation_builder),
                )
            )
        except Exception as exc:
            self.captured_errors.append(f"{callback}: {exc}")

    def teampreview(self, battle: AbstractBattle):
        self._capture_observation(battle, "teampreview")
        return super().teampreview(battle)

    def choose_move(self, battle: AbstractBattle):
        self._capture_observation(battle, "choose_move")
        return super().choose_move(battle)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_observation_builder_live(showdown_server, battle_format, sample_team) -> None:
    """Exercise observation construction against the real local Showdown protocol.

    Verifies that the ObservationBuilder succeeds on real wire state across both
    the initial Team Preview phase and mid-battle turn phases without raising exceptions,
    producing tensors that strictly adhere to the expected sequence and feature width dimensions.
    """
    captured_battles = []
    captured_errors = []
    builder = ObservationBuilder(default_runtime_resources())
    p1 = _ObservationCapturePlayer(
        account_configuration=AccountConfiguration("ObservationA", None),
        battle_format=battle_format,
        server_configuration=showdown_server,
        team=sample_team,
        max_concurrent_battles=1,
        observation_builder=builder,
        captured_battles=captured_battles,
        captured_errors=captured_errors,
    )
    p2 = RandomPlayer(
        account_configuration=AccountConfiguration("ObservationB", None),
        battle_format=battle_format,
        server_configuration=showdown_server,
        team=sample_team,
        max_concurrent_battles=1,
    )

    poke_env_patches.install()
    try:
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=15.0)
        except asyncio.TimeoutError:
            pytest.fail(f"Battle timed out. Internal errors: {captured_errors}")
        except Exception as exc:
            pytest.fail(f"Battle failed with exception: {exc}. Internal errors: {captured_errors}")
    finally:
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
        poke_env_patches.uninstall_for_tests()

    # Ensure no builder exceptions occurred during any game turn
    assert not captured_errors
    assert captured_battles
    # Confirm both team-preview and active-battle states were observed
    assert any(is_teampreview for is_teampreview, _ in captured_battles)
    assert any(not is_teampreview for is_teampreview, _ in captured_battles)
    # Check fixed tensor shapes for Transformer embedding layers
    for _, obs in captured_battles:
        assert isinstance(obs, StructuredObservation)
        assert obs.categorical.shape == (SEQUENCE_LENGTH, CATEGORICAL_WIDTH)
        assert obs.numerical.shape == (SEQUENCE_LENGTH, NUMERICAL_WIDTH)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_showdown_observations_remain_valid(showdown_server) -> None:
    """Verify live Showdown observations satisfy numerical validity and vocabulary bounds.

    Captures turns over multiple live games to verify:
    1. Categorical token IDs do not violate vocabulary bounds or overflow contracts.
    2. Numerical features are all finite (no NaN or infinite float values).
    3. Tensors are initialized cleanly on CPU.
    """
    decisions = await capture_showdown_decisions(
        showdown_server,
        game_count=integration_count("P0_INTEGRATION_OBSERVATION_GAMES", 2),
    )
    assert decisions

    for decision in decisions:
        observation = decision.observation
        # Validate that categorical indices fall strictly within tokenizer vocabulary sizes
        observation.validate_overflow_contract()
        # Verify no NaN or Inf values in numerical feature tensors
        assert all(torch.isfinite(tensor).all() for tensor in observation.tensors())
        assert all(tensor.device.type == "cpu" for tensor in observation.tensors())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_observations_transfer_to_each_available_model_device(
    showdown_server,
    model_device,
) -> None:
    """Verify batch-stacked live observations transfer correctly to target devices (CPU/CUDA)."""
    decisions = await capture_showdown_decisions(showdown_server, game_count=1)
    assert decisions
    # Stack single-turn observations into a batched representation and transfer to device under test
    observation = StructuredObservation.stack([decisions[0].observation]).to(model_device)
    assert all(tensor.device == model_device for tensor in observation.tensors())
