"""Unavoidable poke-env compatibility patches."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, cast

from poke_env.battle import AbstractBattle, DoubleBattle, Effect, Pokemon, PokemonType
from poke_env.data import to_id_str
from poke_env.environment.env import _EnvPlayer
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

from p0.replays.reconstruction.contract import COPYABLE_VOLATILES
from p0.runtime.live_event_capture import capture_message, transform_target_reference

_ORIGINAL_WAIT_FOR_LOGIN = PSClient.wait_for_login
_ORIGINAL_STOP_LISTENING = PSClient.stop_listening
_ORIGINAL_HANDLE_MESSAGE = PSClient._handle_message
_ORIGINAL_SEND_MESSAGE = PSClient.send_message
_ORIGINAL_PARSE_MESSAGE = DoubleBattle.parse_message
_ORIGINAL_GET_POKEMON = DoubleBattle.get_pokemon
_ORIGINAL_FORME_CHANGE = Pokemon.forme_change
_ORIGINAL_UPDATE_FROM_TEAMBUILDER = Pokemon._update_from_teambuilder
_ORIGINAL_START_EFFECT = Pokemon.start_effect
_ORIGINAL_COPY_BOOSTS = Pokemon.copy_boosts
_installed = False
_capture_protocol_lines = False
_filtered_loggers: set[logging.Logger] = set()

_CRITICAL_COPY_EFFECTS = (
    Effect.DRAGON_CHEER,
    Effect.FOCUS_ENERGY,
    Effect.G_MAX_CHI_STRIKE,
    Effect.LASER_FOCUS,
)
_MAX_G_MAX_CHI_STRIKE_LAYERS = 3
_LEPPA_PP_RECOVERY = 10
_RIPEN_LEPPA_PP_RECOVERY = 20
_PARENT_RESULTS_CREATION_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _ParentResults:
    lock: threading.Lock = field(default_factory=threading.Lock)
    completed_rooms: dict[str, str | None] = field(default_factory=dict)
    waiters: list[tuple[int, Future[None]]] = field(default_factory=list)

    def record(self, room: str, winner: str | None) -> None:
        with self.lock:
            if room not in self.completed_rooms:
                self.completed_rooms[room] = winner
                for expected, waiter in self.waiters:
                    if len(self.completed_rooms) >= expected and not waiter.done():
                        waiter.set_result(None)
                self.waiters[:] = [w for w in self.waiters if len(self.completed_rooms) < w[0]]

    def wait_for(self, expected: int) -> Future[None]:
        waiter: Future[None] = Future()
        with self.lock:
            if len(self.completed_rooms) >= expected:
                waiter.set_result(None)
            else:
                self.waiters.append((expected, waiter))
        return waiter

    def discard(self, waiter: Future[None]) -> None:
        with self.lock:
            self.waiters[:] = [w for w in self.waiters if w[1] is not waiter]

    def result_at(self, index: int) -> tuple[str, str | None]:
        with self.lock:
            return tuple(self.completed_rooms.items())[index]


def _parent_results(client: PSClient) -> _ParentResults:
    with _PARENT_RESULTS_CREATION_LOCK:
        if not hasattr(client, "_p0_parent_results"):
            client._p0_parent_results = _ParentResults()  # type: ignore[attr-defined]
        return client._p0_parent_results  # type: ignore[attr-defined]


async def wait_for_parent_results(client: PSClient, expected: int) -> None:
    """Wait until the server has published the requested number of Bo3 results."""
    if expected < 0:
        raise ValueError("Expected parent result count must be non-negative")
    tracker = _parent_results(client)
    waiter = tracker.wait_for(expected)
    try:
        await asyncio.wrap_future(waiter)
    finally:
        tracker.discard(waiter)


async def wait_for_parent_result(client: PSClient, expected: int) -> tuple[str, str | None]:
    """Return the server's result for the requested Bo3 series."""
    if expected <= 0:
        raise ValueError("Expected parent result count must be positive")
    await wait_for_parent_results(client, expected)
    return _parent_results(client).result_at(expected - 1)


class _InactivePokemonFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.msg and "is active, but it's not" in str(record.msg))


_INACTIVE_POKEMON_FILTER = _InactivePokemonFilter()


class _TeamPreviewEnvPlayer(_EnvPlayer):
    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ) -> None:
        if battle.teampreview:
            # The inherited method also tracks the battle needed by reset/forfeit.
            order = await self._choose_move(battle)
            await self.ps_client.send_message(order.message, battle.battle_tag)
            return

        await super()._handle_battle_request(battle, maybe_default_order)


def enable_environment_team_preview(player: _EnvPlayer) -> None:
    """Enable policy-selected preview on poke-env's private environment player."""
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


def _start_effect(self: Pokemon, effect_str: str, details: Any = None) -> None:
    """Track the counters and metadata needed by Showdown's copied volatiles."""
    _ORIGINAL_START_EFFECT(self, effect_str, details)
    effect = Effect.from_showdown_message(effect_str)
    if effect == Effect.G_MAX_CHI_STRIKE:
        self.effects[effect] = min(
            self.effects.get(effect, 0) + 1,
            _MAX_G_MAX_CHI_STRIKE_LAYERS,
        )
    elif effect == Effect.DRAGON_CHEER:
        self.effects[effect] = int(PokemonType.DRAGON in self.types)


def _copy_boosts(self: Pokemon, mon: Pokemon) -> None:
    """Copy boosts and Showdown's critical-stage volatiles from the donor."""
    self.boosts = dict(mon.boosts)
    receiver_effects = self.effects
    donor_effects = mon.effects
    for effect in _CRITICAL_COPY_EFFECTS:
        receiver_effects.pop(effect, None)
        if effect in donor_effects:
            receiver_effects[effect] = donor_effects[effect]


def _restore_leppa_pp(battle: DoubleBattle, event: list[str]) -> None:
    target = battle.get_pokemon(event[2])
    move_id = to_id_str(event[4])
    move = target.moves.get(move_id)
    if move is None:
        raise KeyError(f"Leppa Berry activation names unknown move: {event[4]}")

    restoration = _RIPEN_LEPPA_PP_RECOVERY if target.ability == "ripen" else _LEPPA_PP_RECOVERY
    move._current_pp = min(move._current_pp + restoration, move.max_pp)


def _form_baselines(battle: DoubleBattle) -> dict[Pokemon, str]:
    """Track Showdown's baseSpecies, which poke-env does not retain across forms."""
    baselines = getattr(battle, "_p0_form_baselines", None)
    if baselines is None:
        baselines = {}
        battle._p0_form_baselines = baselines  # type: ignore[attr-defined]
    return baselines


def _parse_message(self: DoubleBattle, split_message: list[str]):
    event_type = split_message[1] if len(split_message) >= 2 else None
    # Best-of rooms send this UI-only notification to the child battle room.
    # It is not a battle event and poke-env 0.15 raises NotImplementedError for it.
    if event_type in {"tempnotify", "tempnotifyoff"}:
        return None
    outgoing = None
    fainted = None
    original_size = None
    baselines = _form_baselines(self)
    if event_type in {"switch", "drag", "replace"} and len(split_message) >= 3:
        identifier = split_message[2]
        role = identifier[:2]
        slot = identifier[2:3]
        if role in {"p1", "p2"} and slot in {"a", "b"}:
            active = (
                self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
            )
            outgoing = active[0 if slot == "a" else 1]
            if outgoing is not None:
                captured = getattr(self, "_p0_transform_targets", {}).get(id(outgoing))
                if captured is not None:
                    original_size = (captured.original_height, captured.original_weight)
            if event_type in {"switch", "drag"}:
                active_by_slot = (
                    self._active_pokemon
                    if role == self.player_role
                    else self._opponent_active_pokemon
                )
                incoming = _ORIGINAL_GET_POKEMON(self, identifier, details=split_message[3])
                other_slot = f"{role}{'b' if slot == 'a' else 'a'}"
                if active_by_slot.get(other_slot) is incoming:
                    # In Illusion, two active slots may show the same species. poke-env
                    # would assign the same Pokemon object to both slots; clone it for the partner.
                    duplicate = deepcopy(incoming)
                    active_by_slot[other_slot] = duplicate
                    aliases = getattr(self, "_p0_duplicate_active", None)
                    if aliases is None:
                        aliases = {}
                        self._p0_duplicate_active = aliases  # type: ignore[attr-defined]
                    aliases[other_slot] = duplicate
    elif event_type == "faint" and len(split_message) >= 3:
        fainted = self.get_pokemon(split_message[2])
        captured = getattr(self, "_p0_transform_targets", {}).get(id(fainted))
        if captured is not None:
            original_size = (captured.original_height, captured.original_weight)
    capture_message(
        self,
        split_message,
        capture_protocol_line=_capture_protocol_lines,
    )
    transferred_boosts: dict[str, int] | None = None
    transferred_effects: dict[Effect, int] = {}
    if event_type == "replace" and outgoing is not None:
        transferred_boosts = dict(outgoing.boosts)
        transferred_effects = dict(outgoing.effects)
    if event_type == "switch" and len(split_message) >= 6:
        causes = tuple(to_id_str(part.removeprefix("[from]")) for part in split_message[5:])
        baton_pass = "batonpass" in causes or "movebatonpass" in causes
        shed_tail = "moveshedtail" in causes or "shedtail" in causes
        if baton_pass or shed_tail:
            if outgoing is not None:
                if baton_pass:
                    transferred_boosts = dict(outgoing.boosts)
                    transferred_effects = {
                        effect: value
                        for effect, value in outgoing.effects.items()
                        if to_id_str(effect.name) in COPYABLE_VOLATILES
                    }
                elif Effect.SUBSTITUTE in outgoing.effects:
                    transferred_effects = {Effect.SUBSTITUTE: outgoing.effects[Effect.SUBSTITUTE]}
    if event_type == "-copyboost" and len(split_message) >= 4:
        self._replay_data.append(split_message[:])
        receiver = self.get_pokemon(split_message[2])
        donor = self.get_pokemon(split_message[3])
        receiver.copy_boosts(donor)
        return None
    if (
        len(split_message) >= 5
        and event_type == "-activate"
        and split_message[3] == "item: Leppa Berry"
    ):
        self._replay_data.append(split_message[:])
        _restore_leppa_pp(self, split_message)
        return None
    if event_type == "-transform" and len(split_message) >= 4:
        try:
            base = self.get_pokemon(split_message[2])
            target_reference = transform_target_reference(self, base, split_message[3])
        except (AssertionError, IndexError, KeyError, ValueError):
            target_reference = split_message[3]
        if target_reference != split_message[3]:
            split_message = [*split_message[:3], target_reference, *split_message[4:]]
    if event_type == "-formechange" and len(split_message) >= 4:
        current = self.get_pokemon(split_message[2])
        baselines.setdefault(current, current.species)
    result = _ORIGINAL_PARSE_MESSAGE(self, split_message)
    if event_type in {"switch", "drag", "replace"} and len(split_message) >= 3:
        getattr(self, "_p0_duplicate_active", {}).pop(split_message[2][:3], None)
    if fainted is not None:
        # Showdown clears volatiles on faint; poke-env only clears a subset.
        fainted.switch_out(self.fields)
    if event_type in {"switch", "drag", "replace", "detailschange"} and len(split_message) >= 4:
        current = self.get_pokemon(split_message[2])
        baselines[current] = current.species
    if event_type == "-start" and len(split_message) >= 5 and split_message[3] == "typeadd":
        target = self.get_pokemon(split_message[2])
        added_type = PokemonType.from_name(split_message[4])
        current_types = target.types
        if added_type not in current_types:
            target._temporary_types = [*current_types, added_type]
    cleared = fainted
    if cleared is None and event_type in {"switch", "drag"}:
        cleared = outgoing
    if cleared is not None:
        if original_size is not None:
            cleared._heightm, cleared._weightkg = original_size
        base_form = baselines.get(cleared)
        if base_form is not None and cleared.species != base_form:
            cleared.forme_change(base_form)
    if transferred_boosts is not None or transferred_effects:
        incoming = self.get_pokemon(split_message[2])
        if transferred_boosts is not None:
            incoming.boosts = transferred_boosts
        incoming.effects.update(transferred_effects)
    return result


def _get_pokemon(
    self: DoubleBattle,
    identifier: str,
    force_self_team: bool = False,
    details: str = "",
    request: dict[str, Any] | None = None,
) -> Pokemon:
    """Resolve a duplicated Illusion display by its active slot."""
    if not force_self_team and len(identifier) >= 5 and identifier[3:5] == ": ":
        alias = getattr(self, "_p0_duplicate_active", {}).get(identifier[:3])
        if alias is not None and alias.identifies_as(identifier[5:]):
            return alias
    return _ORIGINAL_GET_POKEMON(self, identifier, force_self_team, details, request)


async def _handle_message(self: PSClient, message: str):
    # Showdown sends the Bo3 parent room as >game-bestof..., while poke-env only
    # understands >battle rooms and otherwise indexes a non-existent protocol field.
    if message.startswith(">game-"):
        parent_room, _, body = message.partition("\n")
        parent_room = parent_room[1:]
        if "I'm ready!</button>" in message:
            await self.send_message(f"/msgroom {parent_room},/confirmready")
        for line in body.splitlines():
            if line.startswith("|win|"):
                _parent_results(self).record(parent_room, line.removeprefix("|win|"))
            elif line == "|tie":
                _parent_results(self).record(parent_room, None)
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
    """Restore open-team-sheet natures for EV-less teams in poke-env 0.15."""
    _ORIGINAL_UPDATE_FROM_TEAMBUILDER(self, tb)

    if self._nature is None and tb.nature is not None:
        self._nature = tb.nature.lower()


async def _cancel_active_tasks(tasks: tuple[asyncio.Task[Any], ...]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _stop_listening_cleanly(self: PSClient) -> None:
    """Close a client and drain its listener and message-handler tasks."""
    await _ORIGINAL_STOP_LISTENING(self)

    listening_future = getattr(self, "_listening_coroutine", None)
    if listening_future is not None:
        await asyncio.wrap_future(listening_future)

    tasks = tuple(cast(set[asyncio.Task[Any]], getattr(self, "_active_tasks", set())))
    if tasks:
        await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(_cancel_active_tasks(tasks), self.loop)
        )


def install(
    logger: logging.Logger | None = None,
    *,
    capture_protocol_lines: bool = False,
) -> None:
    """Install compatibility patches for the pinned poke-env release."""
    global _capture_protocol_lines, _installed
    target = logger if logger is not None else logging.getLogger("poke_env")

    if target not in _filtered_loggers:
        target.addFilter(_INACTIVE_POKEMON_FILTER)
        _filtered_loggers.add(target)

    if _installed:
        _capture_protocol_lines = _capture_protocol_lines or capture_protocol_lines
        return

    _capture_protocol_lines = capture_protocol_lines
    PSClient.wait_for_login = _wait_for_login
    PSClient.stop_listening = _stop_listening_cleanly
    PSClient._handle_message = _handle_message
    PSClient.send_message = _send_message
    DoubleBattle.parse_message = _parse_message
    DoubleBattle.get_pokemon = _get_pokemon
    Pokemon.forme_change = _forme_change
    Pokemon._update_from_teambuilder = _update_from_teambuilder
    Pokemon.start_effect = _start_effect
    Pokemon.copy_boosts = _copy_boosts
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
        DoubleBattle.get_pokemon = _ORIGINAL_GET_POKEMON
        Pokemon.forme_change = _ORIGINAL_FORME_CHANGE
        Pokemon._update_from_teambuilder = _ORIGINAL_UPDATE_FROM_TEAMBUILDER
        Pokemon.start_effect = _ORIGINAL_START_EFFECT
        Pokemon.copy_boosts = _ORIGINAL_COPY_BOOSTS
        _installed = False


def is_installed() -> bool:
    """Return whether poke-env monkey patches are active."""
    return _installed
