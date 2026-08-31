"""Unavoidable poke-env compatibility patches."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any, cast

from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.environment.env import _EnvPlayer
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

from p0.runtime.live_event_capture import capture_message

_ORIGINAL_WAIT_FOR_LOGIN = PSClient.wait_for_login
_ORIGINAL_STOP_LISTENING = PSClient.stop_listening
_ORIGINAL_HANDLE_MESSAGE = PSClient._handle_message
_ORIGINAL_SEND_MESSAGE = PSClient.send_message
_ORIGINAL_PARSE_MESSAGE = DoubleBattle.parse_message
_ORIGINAL_FORME_CHANGE = Pokemon.forme_change
_ORIGINAL_UPDATE_FROM_TEAMBUILDER = Pokemon._update_from_teambuilder
_installed = False
_capture_protocol_lines = False
_filtered_loggers: list[logging.Logger] = []


class _InactivePokemonFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.msg and "is active, but it's not" in str(record.msg))


_INACTIVE_POKEMON_FILTER = _InactivePokemonFilter()


class _TeamPreviewEnvPlayer(_EnvPlayer):
    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ) -> None:
        if battle.teampreview:
            await self.battle_queue.async_put(battle)
            order = await self.order_queue.async_get()
            await self.ps_client.send_message(order.message, battle.battle_tag)
            return

        await super()._handle_battle_request(battle, maybe_default_order)


def enable_environment_team_preview(player: _EnvPlayer) -> None:
    """
    Enable policy-selected preview for poke-env's private environment player.

    poke-env 0.15 constructs _EnvPlayer internally and exposes no player
    factory. The instance-local class replacement is therefore isolated here.
    """
    player.__class__ = _TeamPreviewEnvPlayer


def enable_forced_open_team_sheet(player: Any) -> None:
    """Configure a poke-env player for a format with server-forced open sheets."""
    try:
        client = player.ps_client
    except AttributeError as exc:
        raise TypeError("player must expose a poke-env ps_client") from exc
    setattr(client, "_p0_force_open_team_sheet", True)


async def _wait_for_login(self: PSClient, checking_interval: float = 0.1, wait_for: int = 30):
    start = perf_counter()
    while perf_counter() - start < wait_for:
        await asyncio.sleep(checking_interval)
        if self.logged_in.is_set():
            return

    assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."


def _parse_message(self: DoubleBattle, split_message: list[str]):
    # Best-of rooms send this UI-only notification to the child battle room.
    # It is not a battle event and poke-env 0.15 raises NotImplementedError for it.
    if len(split_message) >= 2 and split_message[1] in {"tempnotify", "tempnotifyoff"}:
        return None
    capture_message(
        self,
        split_message,
        capture_protocol_line=_capture_protocol_lines,
    )
    return _ORIGINAL_PARSE_MESSAGE(self, split_message)


async def _handle_message(self: PSClient, message: str):
    # Showdown sends the Bo3 parent room as >game-bestof..., while poke-env only
    # understands >battle rooms and otherwise indexes a non-existent protocol field.
    if message.startswith(">game-"):
        if "I'm ready!</button>" in message:
            parent_room = message.split("\n", 1)[0][1:]
            await self.send_message(f"/msgroom {parent_room},/confirmready")
        return None
    return await _ORIGINAL_HANDLE_MESSAGE(self, message)


async def _send_message(self: PSClient, message: str, room: str = "", message_2=None):
    # A Bo3 format with Force Open Team Sheets has no accept/reject negotiation.
    # The pinned client unconditionally sends one of those commands for every VGC
    # battle, which Showdown rejects before the first request is processed.
    if getattr(self, "_p0_force_open_team_sheet", False) and message in {
        "/acceptopenteamsheets",
        "/rejectopenteamsheets",
    }:
        return None
    return await _ORIGINAL_SEND_MESSAGE(self, message, room, message_2)


def _forme_change(self: Pokemon, species: str) -> None:
    """Preserve the observed battle form in addition to its changed dex data."""
    normalized_species = species.split(",", 1)[0]
    self._update_from_pokedex(normalized_species, store_species=True)


def _update_from_teambuilder(self: Pokemon, tb: TeambuilderPokemon) -> None:
    """
    Keep the open-team-sheet nature that poke-env drops for EV-less formats.

    poke-env 0.15 assigns nature only inside if not all(e == 0 for e in tb.evs),
    so a sheet that declares a nature but no EVs loses it. Champions spends Stat
    Points rather than EVs, so every opponent sheet parses with all-zero EVs and
    the revealed nature is discarded - which is exactly the nature stat imputation
    keys on. Upstream deleted the gate in PR #920, merged 2026-05-31 but unreleased
    as of 0.15.0, so this restores that behaviour without moving off the pin.
    """
    _ORIGINAL_UPDATE_FROM_TEAMBUILDER(self, tb)

    if self._nature is None and tb.nature is not None:
        self._nature = tb.nature.lower()


async def _stop_listening_cleanly(self: PSClient) -> None:
    """
    Close a client and drain poke-env's listener/message-handler tasks.

    poke-env 0.15 closes the websocket from stop_listening but does not
    wait for the listener future or the message-handler tasks it creates on its
    dedicated event loop. Those tasks otherwise survive until the loop is
    closed, producing pending-task warnings during integration-test cleanup.
    """
    await _ORIGINAL_STOP_LISTENING(self)

    listening_future = getattr(self, "_listening_coroutine", None)
    if listening_future is not None:
        await asyncio.wrap_future(listening_future)

    async def cancel_active_tasks() -> None:
        tasks = tuple(cast(set[asyncio.Task[Any]], getattr(self, "_active_tasks", set())))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(cancel_active_tasks(), self.loop))


def install(
    logger: logging.Logger | None = None,
    *,
    capture_protocol_lines: bool = False,
) -> None:
    """
    Install compatibility patches for the pinned poke-env release.

    Arguments:
      logger: Optional logger that receives the inactive-Pokémon filter.
      capture_protocol_lines: Retain parsed protocol lines for replay verification.

    Returns:
      None
    """
    global _capture_protocol_lines, _installed
    target = logger or logging.getLogger("poke_env")

    if target not in _filtered_loggers:
        target.addFilter(_INACTIVE_POKEMON_FILTER)
        _filtered_loggers.append(target)

    if _installed:
        _capture_protocol_lines = _capture_protocol_lines or capture_protocol_lines
        return

    _capture_protocol_lines = capture_protocol_lines
    PSClient.wait_for_login = _wait_for_login
    PSClient.stop_listening = _stop_listening_cleanly
    PSClient._handle_message = _handle_message
    PSClient.send_message = _send_message
    DoubleBattle.parse_message = _parse_message
    Pokemon.forme_change = _forme_change
    Pokemon._update_from_teambuilder = _update_from_teambuilder
    _installed = True


def uninstall_for_tests() -> None:
    """Uninstall monkey patches for unit test isolation."""
    global _capture_protocol_lines, _installed
    _capture_protocol_lines = False
    for logger in _filtered_loggers:
        logger.removeFilter(_INACTIVE_POKEMON_FILTER)

    _filtered_loggers.clear()
    if _installed:
        PSClient.wait_for_login = _ORIGINAL_WAIT_FOR_LOGIN
        PSClient.stop_listening = _ORIGINAL_STOP_LISTENING
        PSClient._handle_message = _ORIGINAL_HANDLE_MESSAGE
        PSClient.send_message = _ORIGINAL_SEND_MESSAGE
        DoubleBattle.parse_message = _ORIGINAL_PARSE_MESSAGE
        Pokemon.forme_change = _ORIGINAL_FORME_CHANGE
        Pokemon._update_from_teambuilder = _ORIGINAL_UPDATE_FROM_TEAMBUILDER
        _installed = False


def is_installed() -> bool:
    """Return whether poke-env monkey patches are active."""
    return _installed
