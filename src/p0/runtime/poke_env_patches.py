"""Explicit, idempotent ownership of unavoidable poke-env patches."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from poke_env.battle import DoubleBattle, Pokemon
from poke_env.ps_client.ps_client import PSClient

from p0.runtime.live_event_capture import capture_message, clear_last_move

_ORIGINAL_WAIT_FOR_LOGIN = PSClient.wait_for_login
_ORIGINAL_HANDLER_HANDLE = logging.Handler.handle
_ORIGINAL_SWITCH_OUT = Pokemon.switch_out
_ORIGINAL_PARSE_MESSAGE = DoubleBattle.parse_message
_installed = False


async def _wait_for_login(self: PSClient, checking_interval: float = 0.1, wait_for: int = 30):
    start = perf_counter()
    while perf_counter() - start < wait_for:
        await asyncio.sleep(checking_interval)
        if self.logged_in.is_set():
            return
    assert self.logged_in.is_set(), f"Expected {self.username} to be logged in."


def _handler_handle(self: logging.Handler, record: logging.LogRecord):
    if record.msg and "is active, but it's not" in str(record.msg):
        return False
    return _ORIGINAL_HANDLER_HANDLE(self, record)


def _switch_out(self: Pokemon, fields: Any):
    clear_last_move(self)
    return _ORIGINAL_SWITCH_OUT(self, fields)


def _parse_message(self: DoubleBattle, split_message: list[str]):
    capture_message(self, split_message)
    return _ORIGINAL_PARSE_MESSAGE(self, split_message)


def install() -> None:
    global _installed
    if _installed:
        return
    PSClient.wait_for_login = _wait_for_login
    logging.Handler.handle = _handler_handle
    Pokemon.switch_out = _switch_out
    DoubleBattle.parse_message = _parse_message
    _installed = True


def uninstall() -> None:
    global _installed
    if not _installed:
        return
    PSClient.wait_for_login = _ORIGINAL_WAIT_FOR_LOGIN
    logging.Handler.handle = _ORIGINAL_HANDLER_HANDLE
    Pokemon.switch_out = _ORIGINAL_SWITCH_OUT
    DoubleBattle.parse_message = _ORIGINAL_PARSE_MESSAGE
    _installed = False


def is_installed() -> bool:
    return _installed
