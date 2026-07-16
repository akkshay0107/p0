"""Unavoidable poke-env patches."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.environment.env import _EnvPlayer
from poke_env.ps_client.ps_client import PSClient

from p0.runtime.live_event_capture import capture_message

_ORIGINAL_WAIT_FOR_LOGIN = PSClient.wait_for_login
_ORIGINAL_PARSE_MESSAGE = DoubleBattle.parse_message
_installed = False
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
    """Enable policy-selected preview for poke-env's private environment player.

    poke-env 0.15 constructs ``_EnvPlayer`` internally and exposes no player
    factory. The instance-local class replacement is therefore isolated here.
    """
    player.__class__ = _TeamPreviewEnvPlayer


async def _wait_for_login(self: PSClient, checking_interval: float = 0.1, wait_for: int = 30):
    start = perf_counter()
    while perf_counter() - start < wait_for:
        await asyncio.sleep(checking_interval)
        if self.logged_in.is_set():
            return
    assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."


def _parse_message(self: DoubleBattle, split_message: list[str]):
    capture_message(self, split_message)
    return _ORIGINAL_PARSE_MESSAGE(self, split_message)


def install(logger: logging.Logger | None = None) -> None:
    global _installed
    target = logger or logging.getLogger("poke_env")
    if target not in _filtered_loggers:
        target.addFilter(_INACTIVE_POKEMON_FILTER)
        _filtered_loggers.append(target)
    if _installed:
        return
    PSClient.wait_for_login = _wait_for_login
    DoubleBattle.parse_message = _parse_message
    _installed = True


def uninstall_for_tests() -> None:
    global _installed
    for logger in _filtered_loggers:
        logger.removeFilter(_INACTIVE_POKEMON_FILTER)
    _filtered_loggers.clear()
    if _installed:
        PSClient.wait_for_login = _ORIGINAL_WAIT_FOR_LOGIN
        DoubleBattle.parse_message = _ORIGINAL_PARSE_MESSAGE
        _installed = False


def is_installed() -> bool:
    return _installed
