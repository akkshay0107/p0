from __future__ import annotations

import logging
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player.battle_order import PassBattleOrder
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder import TeambuilderPokemon
from poke_env.teambuilder.teambuilder import Teambuilder

from p0.battle.events import SpatialActionType
from p0.battle.views import FixtureBattleView
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.runtime import poke_env_patches, showdown
from p0.runtime.live_event_capture import capture_message
from p0.runtime.poke_env_action_adapter import (
    action_to_order,
    action_to_single_order,
    order_to_action,
    single_order_to_action,
)
from p0.runtime.poke_env_battle_adapter import battle_view, current_battle_view, decision_view


def test_event_parser_import_does_not_install_poke_env_patches() -> None:
    """Verify that importing event parsing modules does not implicitly mutate poke-env global state."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, p0.battle.events as events, p0.runtime.poke_env_patches as patches; "
                "sys.exit(0 if not patches.is_installed() else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_patch_installation_is_idempotent_reversible_and_logger_scoped() -> None:
    """Verify poke-env monkey patches can be installed repeatedly, uninstalled cleanly, and filter logs."""
    poke_env_patches.uninstall_for_tests()
    original = DoubleBattle.parse_message
    original_stop = PSClient.stop_listening
    poke_env_patches.install()
    installed = DoubleBattle.parse_message
    # Installing again should be a no-op idempotent call
    poke_env_patches.install()
    assert DoubleBattle.parse_message is installed
    assert installed is not original
    assert PSClient.stop_listening is not original_stop
    # Uninstalling must restore original method references
    poke_env_patches.uninstall_for_tests()
    assert DoubleBattle.parse_message is original
    assert PSClient.stop_listening is original_stop

    # Verify log filtering only suppresses targeted warning messages on the designated logger
    target = logging.getLogger("test.poke-env")
    other = logging.getLogger("test.other")
    poke_env_patches.install(target)
    record = logging.LogRecord("test", logging.WARNING, "", 0, "is active, but it's not", (), None)
    assert not target.filter(record)
    assert other.filter(record)
    poke_env_patches.uninstall_for_tests()


def test_live_adapter_and_pure_fixture_build_identical_observations() -> None:
    """Verify ObservationBuilder produces bitwise identical tensors from live poke-env adapter and pure fixture views."""
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
    # Validate every structured tensor attribute matches identically across live and fixture views
    for name in live._FIELD_NAMES:
        torch.testing.assert_close(getattr(live, name), getattr(pure, name))


def test_transformed_pokemon_view_supports_poke_env_target_resolution() -> None:
    """Verify a transformed active wrapper satisfies poke-env's target resolver contract."""
    battle = DoubleBattle("transform-targets", "player", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    transformed = Pokemon(gen=9, species="ditto")
    target = Pokemon(gen=9, species="charizard")
    transformed._active = True
    target._active = True
    move = Move("heatwave", gen=9)
    target._moves[move.id] = move
    battle._team = {"p1: Ditto": transformed}
    battle._opponent_team = {"p2: Charizard": target}
    battle._active_pokemon = {"p1a": transformed}
    battle._opponent_active_pokemon = {"p2a": target}
    battle._available_moves = [[move], []]
    setattr(battle, "_p0_transform_targets", {id(transformed): target})

    decision = decision_view(battle)

    # This calls the pinned poke-env DoubleBattle.get_possible_showdown_targets,
    # which reads the extra runtime fields absent from the old wrapper.
    assert decision.slots[0].move_targets == ((0,),)


def test_live_spatial_turn_capture() -> None:
    """Verify capture_message correctly parses live websocket messages into spatial records."""
    battle = DoubleBattle("events", "player", logging.getLogger(__name__), 9)
    battle._player_role = "p1"
    capture_message(battle, ["", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard"])

    view = battle_view(battle)
    records = view.spatial_turn
    assert len(records) == 4
    assert records[0].action_type == int(SpatialActionType.MOVE)

    capture_message(battle, ["", "turn", "2"])
    records_turn2 = view.spatial_turn
    assert records_turn2[0].action_type == int(SpatialActionType.NONE)


class _AdapterBattleState:
    pass


def _adapter_battle(*, teampreview: bool = False, forced: bool = False) -> DoubleBattle:
    """Helper creating a lightweight DoubleBattle mock for testing action and decision adapters."""
    active = SimpleNamespace(
        moves={"tackle": SimpleNamespace(id="tackle")},
        fainted=False,
        base_species="Pikachu",
    )
    team = {
        f"p1: Species-{index}": SimpleNamespace(base_species=f"Species-{index}")
        for index in range(6)
    }
    battle = _AdapterBattleState()
    for name, value in {
        "player_username": "player",
        "battle_tag": "runtime-battle",
        "teampreview": teampreview,
        "team": team,
        "active_pokemon": [active, None],
        "opponent_active_pokemon": [None, None],
        "available_moves": [[SimpleNamespace(id="struggle" if forced else "tackle")], []],
        "available_switches": [[], []],
        "valid_orders": [[], []],
        "can_mega_evolve": [False, False],
        "force_switch": [False, False],
        "trapped": [False, False],
        "maybe_trapped": [False, False],
        "_wait": False,
        "player_role": "p1",
        "opponent_team": {},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "turn": 1,
        "used_mega_evolve": False,
        "opponent_used_mega_evolve": False,
        "get_possible_showdown_targets": lambda move, pokemon: [0],
    }.items():
        setattr(battle, name, value)
    return cast(DoubleBattle, battle)


def test_runtime_action_adapters_round_trip_control_move_and_preview_orders() -> None:
    """Verify bidirectional roundtrip conversion between discrete model actions and Showdown order strings."""
    battle = _adapter_battle()
    # Test standard move actions
    for action in (0, 7):
        order = action_to_single_order(action, battle, fake=True, position=0)
        assert int(single_order_to_action(order, battle, fake=True, position=0)) == action
    # Test forced action (action ID 48) when forced move like struggle is required
    forced_battle = _adapter_battle(forced=True)
    forced_order = action_to_single_order(48, forced_battle, fake=True, position=0)
    assert int(single_order_to_action(forced_order, forced_battle, fake=True, position=0)) == 48
    # Test special control actions: -2 (pass/wait), -1 (default)
    assert np.array_equal(
        order_to_action(action_to_order(np.array([-2, -2]), battle), battle), [-2, -2]
    )
    assert np.array_equal(
        order_to_action(action_to_order(np.array([-1, -1]), battle), battle), [-1, -1]
    )
    # Test team preview lead selection order mapping
    preview = _adapter_battle(teampreview=True)
    selected = np.array([1, 15], dtype=np.int64)
    preview_order = action_to_order(selected, preview)
    assert np.array_equal(order_to_action(preview_order, preview), selected)


def test_action_to_order_composes_two_slot_decisions() -> None:
    """Verify action_to_order composes joint double-battle commands across slot 0 and slot 1."""
    battle = _adapter_battle()
    active_b = SimpleNamespace(
        moves={"tackle": SimpleNamespace(id="tackle")},
        fainted=False,
        base_species="Raichu",
    )
    cast(Any, battle).active_pokemon = [cast(Any, battle).active_pokemon[0], active_b]
    cast(Any, battle).available_moves = [
        [SimpleNamespace(id="tackle")],
        [SimpleNamespace(id="tackle")],
    ]
    # Action (0, 7) corresponds to Move 1 Target 1 on Slot 0, Move 2 Target 4 on Slot 1
    actions = np.array([0, 7], dtype=np.int64)
    order = action_to_order(actions, battle, fake=True)
    recovered = order_to_action(order, battle, fake=True)
    assert np.array_equal(recovered, actions)


def test_battle_view_cache_refreshes_decisions_without_replacing_facade() -> None:
    """Verify battle_view retains facade instance identity while updating dynamic decision states when battle mutates."""
    battle = _adapter_battle()
    first = current_battle_view(battle)
    first_decision = first.decision
    assert first is current_battle_view(battle)
    assert first_decision is first.decision
    # Mutate wait state to simulate server request transition
    battle._wait = True
    refreshed = battle_view(battle)
    # Same view wrapper instance is reused, but its decision snapshot is refreshed
    assert refreshed is first
    assert refreshed.decision is not first_decision
    assert refreshed.decision.wait is True
    assert decision_view(battle) == refreshed.decision


def test_runtime_action_adapters_reject_invalid_orders_in_strict_mode() -> None:
    """Verify action conversion in strict mode rejects out-of-bounds actions and invalid order objects."""
    battle = _adapter_battle()
    with pytest.raises(ValueError):
        action_to_single_order(26, battle, fake=False, position=0)
    with pytest.raises((TypeError, ValueError)):
        order_to_action(cast(Any, SimpleNamespace()), battle, strict=True)

    valid_order = PassBattleOrder()
    mock_battle = cast(
        DoubleBattle,
        SimpleNamespace(
            player_username="player",
            battle_tag="battle",
            valid_orders=([valid_order], []),
        ),
    )
    assert str(action_to_single_order(0, mock_battle, fake=False, position=0)) == str(valid_order)
    mock_battle.valid_orders[0].clear()
    with pytest.raises(ValueError, match="not in action space"):
        action_to_single_order(0, mock_battle, fake=False, position=0)


def test_recharge_is_encoded_as_forced_move() -> None:
    """Verify that recharge turns map to forced action ID 48."""
    battle = _adapter_battle()
    cast(Any, battle).available_moves = [[SimpleNamespace(id="recharge")], []]
    order = action_to_single_order(48, battle, fake=True, position=0)
    assert int(single_order_to_action(order, battle, fake=True, position=0)) == 48


_EV_LESS_SHEET = "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|||||50|"
_EV_BEARING_SHEET = (
    "Incineroar||sitrusberry|intimidate|fakeout,partingshot|Impish|252,0,0,0,4,0||||50|"
)


def _teambuilder_mon(packed: str) -> TeambuilderPokemon:
    """Parse single Pokémon record from packed Showdown team string."""
    return Teambuilder.parse_packed_team(packed)[0]


def test_poke_env_drops_the_open_team_sheet_nature_without_the_patch() -> None:
    """Demonstrate upstream poke-env bug: unpatched poke-env drops natures on OTS 0-EV Pokémon."""
    assert not poke_env_patches.is_installed()
    # Unpatched poke-env drops nature when EV string is empty
    assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None
    # But retains nature when explicit EV values are present
    assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_BEARING_SHEET)).nature == "impish"


def test_nature_patch_restores_the_open_team_sheet_nature() -> None:
    """Verify our poke_env_patches monkeypatch restores nature parsing on 0-EV Open Team Sheet Pokémon."""
    poke_env_patches.install()
    try:
        mon = Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET))
        assert mon.nature == "impish"
        no_nature = _teambuilder_mon("Incineroar||sitrusberry|intimidate|fakeout||||||50|")
        assert Pokemon(gen=9, teambuilder=no_nature).nature is None
    finally:
        poke_env_patches.uninstall_for_tests()

    # Confirm clean uninstallation leaves poke-env in original unpatched state
    assert Pokemon(gen=9, teambuilder=_teambuilder_mon(_EV_LESS_SHEET)).nature is None


def test_showdown_group_rolls_back_servers_when_later_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify start_showdown_servers gracefully shuts down already-started instances if a subsequent server fails."""
    events = []

    class FakeServer:
        def __init__(self, port, **kwargs):
            self.port = port

        def __enter__(self):
            events.append(("start", self.port))
            if self.port == 2:
                raise RuntimeError("failed")
            return self

        def __exit__(self, *args):
            events.append(("stop", self.port))

    monkeypatch.setattr(showdown, "ShowdownServer", FakeServer)
    monkeypatch.setattr(showdown, "build_showdown", lambda root: None)
    with pytest.raises(RuntimeError, match="failed"):
        with showdown.start_showdown_servers(2, ports=(1, 2)):
            pass
    # Server 1 must be stopped after Server 2 start raises exception
    assert events == [("start", 1), ("start", 2), ("stop", 1)]


def test_showdown_build_failure_has_bounded_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify build_showdown error output truncates excessively large stderr outputs to avoid flooding logs."""

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="x" * 5000)

    monkeypatch.setattr(showdown.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as error:
        showdown.build_showdown(tmp_path)
    # Output length should be bounded (< 4200 characters)
    assert len(str(error.value)) < 4200


@pytest.mark.network
def test_loopback_port_allocator_returns_distinct_reusable_ports() -> None:
    """Verify allocate_loopback_ports binds and frees distinct available TCP loopback ports."""
    ports = showdown.allocate_loopback_ports(8)
    assert len(ports) == len(set(ports))
    sockets = [socket.socket() for _ in ports]
    try:
        # Verify that all allocated ports are immediately bindable by caller
        for port, listener in zip(ports, sockets, strict=True):
            listener.bind(("127.0.0.1", port))
    finally:
        for listener in sockets:
            listener.close()


def test_showdown_server_start_stop_owns_process_log_and_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify ShowdownServer launches Node with proper CLI arguments, manages log files, and terminates cleanly."""
    commands: list[list[str]] = []

    class FakeProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            commands.append(command)
            self.kwargs = kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        showdown.socket, "create_connection", lambda *args, **kwargs: FakeConnection()
    )
    server = showdown.ShowdownServer(
        9123,
        showdown_root=tmp_path,
        log_path=tmp_path / "nested" / "server.log",
        startup_timeout=1,
    )
    server.start()
    # Check CLI command shape passed to Popen
    assert commands == [
        [
            "node",
            "--max-old-space-size=1536",
            "pokemon-showdown",
            "start",
            "--no-security",
            "--skip-build",
            "9123",
        ]
    ]
    assert server.websocket_url.endswith(":9123/showdown/websocket")
    assert server.process is not None
    server.stop()
    assert server.process is None
    assert server._log_file is None
    assert (tmp_path / "nested" / "server.log").is_file()


def test_showdown_server_rejects_double_start_and_invalid_port_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify ShowdownServer rejects start() if already running and checks port uniqueness."""

    class FakeProcess:
        returncode = None

        def __init__(self, *args, **kwargs):
            del args, kwargs

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        showdown.socket, "create_connection", lambda *args, **kwargs: FakeConnection()
    )
    server = showdown.ShowdownServer(9124, showdown_root=tmp_path, startup_timeout=1)
    server.start()
    with pytest.raises(RuntimeError, match="already started"):
        server.start()
    server.stop()

    # Reject duplicate port numbers in server group allocation
    with pytest.raises(ValueError, match="unique"):
        with showdown.start_showdown_servers(2, showdown_root=tmp_path, ports=(1, 1)):
            pass


def test_showdown_server_kills_process_when_graceful_stop_times_out(tmp_path: Path) -> None:
    """Verify ShowdownServer escalates to SIGKILL if the process fails to terminate gracefully within stop_timeout."""

    class StuckProcess:
        returncode = None

        def __init__(self):
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def kill(self):
            self.killed = True

        def wait(self, timeout: float | None = None):
            if not self.killed:
                raise subprocess.TimeoutExpired("node", timeout if timeout is not None else 0.0)
            return 0

    process = StuckProcess()
    server = showdown.ShowdownServer(9125, showdown_root=tmp_path, stop_timeout=0.01)
    server.process = cast(Any, process)
    server.stop()
    assert process.killed


def test_showdown_server_rolls_back_after_child_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify ShowdownServer cleans up resources and raises RuntimeError if child process crashes during startup."""

    class CrashedProcess:
        returncode = 17

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    monkeypatch.setattr(showdown.subprocess, "Popen", lambda *args, **kwargs: CrashedProcess())
    server = showdown.ShowdownServer(9126, showdown_root=tmp_path, startup_timeout=0.1)
    with pytest.raises(RuntimeError, match="exited"):
        server.start()
    assert server.process is None
    assert server._log_file is None


def test_showdown_server_preserves_custom_flags_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify ShowdownServer accepts custom port numbers and log file destinations."""
    captured_command: list[list[str]] = []

    class FakeProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            captured_command.append(command)

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout: float | None = None):
            del timeout
            return 0

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

    monkeypatch.setattr(showdown.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        showdown.socket, "create_connection", lambda *args, **kwargs: FakeConnection()
    )

    custom_log = tmp_path / "custom" / "showdown.log"
    server = showdown.ShowdownServer(
        9999,
        showdown_root=tmp_path,
        log_path=custom_log,
        startup_timeout=1,
    )
    server.start()
    assert server.port == 9999
    assert str(custom_log.parent) in str(server.log_path)
    assert captured_command[0][-1] == "9999"
    server.stop()
