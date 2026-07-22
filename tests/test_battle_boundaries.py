from __future__ import annotations

import logging

import numpy as np
import torch
from poke_env.battle import DoubleBattle, Pokemon

from p0.battle.actions import (
    ACT_SIZE,
    decode_action,
    decode_team_pair,
    encode_action,
    encode_team_pair,
    team_selection,
)
from p0.battle.events import EventTypeId, RawBattleEvent, parse_events
from p0.battle.legality import (
    DecisionView,
    SlotDecision,
    action_mask,
    legal_actions,
    second_action_mask,
)
from p0.battle.views import FixtureBattleView
from p0.format_config import ACTION_CONTRACT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import PolicyNet
from p0.model.resources import default_runtime_resources
from p0.runtime import poke_env_patches
from p0.runtime.poke_env_battle_adapter import battle_view, decision_view


def test_action_contract_round_trips_ids_and_describes_canonical_ranges() -> None:
    assert [encode_action(decode_action(action)) for action in range(ACT_SIZE)] == list(
        range(ACT_SIZE)
    )
    assert ACTION_CONTRACT["action_count"] == ACT_SIZE
    ranges = ACTION_CONTRACT["ranges"]
    assert [(entry["start"], entry["end"]) for entry in ranges] == [
        (0, 1),
        (1, 7),
        (7, 27),
        (27, 47),
        (47, 48),
        (48, 49),
    ]
    assert [decode_action(index).kind.name.lower() for index in (0, 1, 7, 27, 47, 48)] == [
        "pass",
        "switch",
        "move",
        "move",
        "forced_move",
        "forced_move",
    ]
    actions = {
        encode_team_pair(first, second) for first in range(6) for second in range(first + 1, 6)
    }
    assert len(actions) == 15
    assert all(decode_team_pair(action)[0] < decode_team_pair(action)[1] for action in actions)
    assert team_selection(1, 8)[:4] == (0, 1, 2, 3)


def test_scalar_joint_constraints_match_policy_vectorization() -> None:
    view = DecisionView(
        slots=(
            SlotDecision(
                switch_slots=(2, 3),
                move_targets=((-2, 1, 2), (0,), (), (1,)),
                can_mega=True,
            ),
            SlotDecision(
                switch_slots=(2, 4),
                move_targets=((-1, 1), (2,), (0,), ()),
                can_mega=True,
            ),
        )
    )
    base = torch.from_numpy(action_mask(view)).unsqueeze(0)
    resources = default_runtime_resources()
    policy = build_policy(ModelConfig(32, 2, 1, 128), resources)
    for first in legal_actions(view, 0):
        logits = torch.zeros((1, 2, ACT_SIZE))
        masked = policy.actor._apply_sequential_masks(
            logits,
            torch.tensor([first]),
            base,
            torch.tensor([False]),
        )
        actual = torch.isfinite(masked[0, 1]).numpy()
        np.testing.assert_array_equal(actual, second_action_mask(view, first))


def test_event_parser_import_does_not_install_poke_env_patches() -> None:
    poke_env_patches.uninstall_for_tests()
    originals = (
        DoubleBattle.parse_message,
        Pokemon.switch_out,
        logging.Handler.handle,
    )
    __import__("p0.battle.events")
    assert originals == (
        DoubleBattle.parse_message,
        Pokemon.switch_out,
        logging.Handler.handle,
    )
    assert not poke_env_patches.is_installed()


def test_protocol_parser_accepts_an_injected_resource_resolver() -> None:
    class Resolver:
        def id_for(self, table: str, name: str | None) -> int:
            return 17 if table == "moves" and name == "Thunderbolt" else 0

        def effect_id_for(self, table: str, name: str | None) -> int:
            return 0

        def resolve(self, table: str, name: str | None) -> tuple[int, str]:
            resolved = self.id_for(table, name)
            return resolved, "known" if resolved else "oov"

    events = parse_events(
        [RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"))],
        Resolver(),
    )
    assert events[0].event_type is EventTypeId.MOVE
    assert events[0].move_id == 17


def test_patch_installation_is_idempotent_reversible_and_logger_scoped() -> None:
    poke_env_patches.uninstall_for_tests()
    original = DoubleBattle.parse_message
    poke_env_patches.install()
    installed = DoubleBattle.parse_message
    poke_env_patches.install()
    assert DoubleBattle.parse_message is installed
    assert installed is not original
    poke_env_patches.uninstall_for_tests()
    assert DoubleBattle.parse_message is original

    target = logging.getLogger("test.poke-env")
    other = logging.getLogger("test.other")
    poke_env_patches.install(target)
    record = logging.LogRecord("test", logging.WARNING, "", 0, "is active, but it's not", (), None)
    assert not target.filter(record)
    assert other.filter(record)
    poke_env_patches.uninstall_for_tests()


def test_live_adapter_and_pure_fixture_build_identical_observations() -> None:
    battle = DoubleBattle("view", "player", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    ally = Pokemon(gen=9, species="charizard")
    opponent = Pokemon(gen=9, species="venusaur")
    ally._active = True
    opponent._active = True
    battle._team = {"p1: Charizard": ally}
    battle._opponent_team = {"p2: Venusaur": opponent}
    battle._active_pokemon = {"p1a": ally}
    battle._opponent_active_pokemon = {"p2a": opponent}

    fixture = FixtureBattleView(
        team=battle.team,
        opponent_team=battle.opponent_team,
        active_pokemon=battle.active_pokemon,
        opponent_active_pokemon=battle.opponent_active_pokemon,
        available_moves=battle.available_moves,
        available_switches=battle.available_switches,
        can_mega_evolve=battle.can_mega_evolve,
        force_switch=battle.force_switch,
        trapped=battle.trapped,
        maybe_trapped=battle.maybe_trapped,
        teampreview=battle.teampreview,
        player_role=battle.player_role,
        wait=battle._wait,
        weather=battle.weather,
        fields=battle.fields,
        side_conditions=battle.side_conditions,
        opponent_side_conditions=battle.opponent_side_conditions,
        turn=battle.turn,
        used_mega_evolve=battle.used_mega_evolve,
        opponent_used_mega_evolve=battle.opponent_used_mega_evolve,
        decision=decision_view(battle),
    )
    builder = ObservationBuilder(default_runtime_resources())
    live = builder.build(battle_view(battle))
    pure = builder.build(fixture)
    for name in live._FIELD_NAMES:
        torch.testing.assert_close(getattr(live, name), getattr(pure, name))


def test_factory_shares_resources_and_preserves_state_dict_layout() -> None:
    resources = default_runtime_resources()
    config = ModelConfig(32, 2, 1, 128)
    direct = PolicyNet(config, resources)
    policy = build_policy(config, resources)
    builder = ObservationBuilder(resources=resources)
    assert policy.resources is policy.encoder.resources is builder.resources is resources
    assert policy.config == config == ModelConfig.from_dict(config.to_dict())
    assert direct.state_dict().keys() == policy.state_dict().keys()
