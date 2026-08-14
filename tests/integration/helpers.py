"""Shared live-battle capture helpers for integration tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import order_to_action, single_order_to_action
from p0.runtime.poke_env_battle_adapter import battle_view


@dataclass(frozen=True, slots=True)
class GroundTruthDecision:
    """One observation and order set captured from a live Showdown request.
    
    Contains both the high-level featurized StructuredObservation and the low-level
    valid/chosen Showdown protocol order strings alongside their mapped discrete action indices.
    """

    observation: StructuredObservation
    # Tuple of valid discrete action IDs per position slot: (slot_0_actions, slot_1_actions)
    legal_actions: tuple[tuple[int, ...], tuple[int, ...]]
    # Tuple of valid Showdown command strings per position slot: (slot_0_orders, slot_1_orders)
    legal_orders: tuple[tuple[str, ...], tuple[str, ...]]
    # Valid pairwise combinations of (slot_0_action, slot_1_action) for double battles
    legal_joint_actions: tuple[tuple[int, int], ...]
    chosen_action: tuple[int, int]
    chosen_order: str


class _GroundTruthPlayer(RandomPlayer):
    """Instrumented RandomPlayer that records observations and action mappings for every turn."""

    def __init__(self, *, observation_builder: ObservationBuilder, **kwargs: Any):
        self.observation_builder = observation_builder
        self.decisions: list[GroundTruthDecision] = []
        super().__init__(**kwargs)

    def choose_move(self, battle: Any):
        # Extract per-slot valid SingleBattleOrders from poke-env's battle state
        legal_orders: tuple[tuple[SingleBattleOrder, ...], tuple[SingleBattleOrder, ...]] = (
            tuple(battle.valid_orders[0]),
            tuple(battle.valid_orders[1]),
        )
        # Convert each single order to its corresponding discrete model action integer for each slot.
        # fake=True enables hypothetical action mapping without altering internal battle state.
        legal_actions = (
            tuple(
                sorted(
                    {
                        int(single_order_to_action(order, battle, fake=True, position=0))
                        for order in legal_orders[0]
                    }
                )
            ),
            tuple(
                sorted(
                    {
                        int(single_order_to_action(order, battle, fake=True, position=1))
                        for order in legal_orders[1]
                    }
                )
            ),
        )
        # Generate valid joint action pairs by taking the product of valid individual orders
        # compatible with Double Battle rules (e.g. avoiding invalid double target conflicts).
        legal_joint_actions = tuple(
            (
                int(single_order_to_action(order.first_order, battle, fake=True, position=0)),
                int(single_order_to_action(order.second_order, battle, fake=True, position=1)),
            )
            for order in DoubleBattleOrder.join_orders(list(legal_orders[0]), list(legal_orders[1]))
        )
        order = super().choose_move(battle)
        action = order_to_action(order, battle)
        self.decisions.append(
            GroundTruthDecision(
                # Snapshot the observation on CPU so it is safe across devices and threads
                observation=self.observation_builder.build(battle_view(battle)).cpu(),
                legal_actions=legal_actions,
                legal_orders=(
                    tuple(sorted(str(order) for order in legal_orders[0])),
                    tuple(sorted(str(order) for order in legal_orders[1])),
                ),
                legal_joint_actions=legal_joint_actions,
                chosen_action=(int(action[0]), int(action[1])),
                chosen_order=str(order),
            )
        )
        return order


async def capture_showdown_decisions(
    server_configuration, *, game_count: int = 1, max_concurrent_battles: int = 1
):
    """Capture live requests, observations, valid orders, and chosen orders from a Showdown server.
    
    Installs necessary poke-env runtime patches during the battle session and ensures
    websockets and monkey patches are cleanly uninstalled/cleaned up on completion or error.
    """
    builder = ObservationBuilder(default_runtime_resources())
    player_a = _GroundTruthPlayer(
        account_configuration=AccountConfiguration("GroundTruthA", None),
        battle_format=FORMAT.battle_format,
        server_configuration=server_configuration,
        team=DEFAULT_TEST_TEAM,
        accept_open_team_sheet=True,
        max_concurrent_battles=max_concurrent_battles,
        observation_builder=builder,
    )
    player_b = RandomPlayer(
        account_configuration=AccountConfiguration("GroundTruthB", None),
        battle_format=FORMAT.battle_format,
        server_configuration=server_configuration,
        team=DEFAULT_TEST_TEAM,
        accept_open_team_sheet=True,
        max_concurrent_battles=max_concurrent_battles,
    )
    poke_env_patches.install()
    try:
        await player_a.battle_against(player_b, n_battles=game_count)
    finally:
        await player_a.ps_client.stop_listening()
        await player_b.ps_client.stop_listening()
        poke_env_patches.uninstall_for_tests()
    return tuple(player_a.decisions)


def integration_count(name: str, default: int) -> int:
    """Read a positive integration workload count from the environment."""
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least one, got {value}")
    return value
