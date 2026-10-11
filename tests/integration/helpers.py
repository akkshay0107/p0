"""Shared live-battle capture helpers for integration tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from poke_env import AccountConfiguration
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

from p0.battle.actions import (
    FORCED_ACTION,
    MEGA_FORCED_ACTION,
    TEAM_SIZE,
    ActionKind,
    SlotAction,
    encode_action,
    encode_team_pair,
)
from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_action_adapter import action_to_order
from p0.runtime.poke_env_battle_adapter import action_move_slots, battle_view
from p0.runtime.poke_env_patches import locked_choice
from tests.team_fixtures import DEFAULT_TEST_TEAM


def canonical_team_actions(
    selection: tuple[int, ...], team_size: int = TEAM_SIZE
) -> tuple[int, int]:
    """Derive ordered lead and back pair preview actions from a team order."""
    defaults = tuple(range(team_size))
    values = tuple(index for index in selection if 0 <= index < team_size)
    values += tuple(index for index in defaults if index not in values)

    lead = values[:2]
    back = values[2:4]

    if len(set((*lead, *back))) != 4:
        raise ValueError("Team preview must select four distinct members")

    return (
        encode_team_pair(lead[0], lead[1], team_size),
        encode_team_pair(back[0], back[1], team_size),
    )


def order_to_action(order: BattleOrder, battle: DoubleBattle) -> npt.NDArray[np.int64]:
    """Encode a poke-env BattleOrder back into a policy joint action vector."""
    if battle.teampreview:
        digits = tuple(int(char) - 1 for char in order.message[6:] if char.isdigit())
        return np.asarray(canonical_team_actions(digits, len(battle.team)), dtype=np.int64)

    if isinstance(order, DefaultBattleOrder):
        return np.array([-2, -2], dtype=np.int64)

    if isinstance(order, ForfeitBattleOrder):
        return np.array([-1, -1], dtype=np.int64)

    if not isinstance(order, DoubleBattleOrder):
        raise TypeError(f"Expected a doubles order, got {type(order).__name__}")

    if not DoubleBattleOrder.join_orders([order.first_order], [order.second_order]):
        raise ValueError(f"Orders in {order} are incompatible")

    return np.array(
        [
            single_order_to_action(order.first_order, battle, False, 0),
            single_order_to_action(order.second_order, battle, False, 1),
        ],
        dtype=np.int64,
    )


def single_order_to_action(
    order: SingleBattleOrder, battle: DoubleBattle, fake: bool, position: int
) -> np.int64:
    """Encode a SingleBattleOrder into a single slot action integer."""
    if isinstance(order.order, str):
        if isinstance(order, DefaultBattleOrder):
            return np.int64(-2)
        if not isinstance(order, PassBattleOrder):
            raise ValueError(f"Unsupported string order {order}")
        return np.int64(0)

    if not fake:
        if str(order) not in map(str, battle.valid_orders[position]):
            raise ValueError(f"Order {order} is not valid for position {position}")

    if isinstance(order.order, Pokemon):
        species = tuple(pokemon.base_species for pokemon in battle.team.values())
        return np.int64(species.index(order.order.base_species) + 1)

    active = battle.active_pokemon[position]
    if active is None:
        raise ValueError(f"Move order targets empty position {position}")

    if battle_view(battle).decision.slots[position].forced_move:
        return np.int64(MEGA_FORCED_ACTION if order.mega else FORCED_ACTION)

    move_slot = next(
        i for i, move in enumerate(action_move_slots(battle, position)) if move.id == order.order.id
    )
    return np.int64(
        encode_action(
            SlotAction(
                ActionKind.MOVE, move_slot=move_slot, target=order.move_target, mega=order.mega
            )
        )
    )


@dataclass(frozen=True, slots=True)
class GroundTruthDecision:
    """
    One observation and order set captured from a live Showdown request.

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
    regenerated_order: str
    battle_tag: str


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
                # A locked slot is sent by position, so compare the commands as sent.
                chosen_order=locked_choice(battle, order.message),
                regenerated_order=locked_choice(battle, action_to_order(action, battle).message),
                battle_tag=battle.battle_tag,
            )
        )
        return order


async def capture_showdown_decisions(
    server_configuration, *, game_count: int = 1, max_concurrent_battles: int = 1
):
    """
    Capture live requests, observations, valid orders, and chosen orders from a Showdown server.

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
        assert player_a.n_finished_battles == game_count
        assert player_b.n_finished_battles == game_count
        assert len(player_a.battles) == game_count
        assert all(battle.finished for battle in player_a.battles.values())
    finally:
        await player_a.ps_client.stop_listening()
        await player_b.ps_client.stop_listening()
    return tuple(player_a.decisions)


def integration_count(name: str, default: int) -> int:
    """Read a positive integration workload count from the environment."""
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least one, got {value}")
    return value
