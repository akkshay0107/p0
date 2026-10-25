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

from poke_env.battle import (
    AbstractBattle,
    DoubleBattle,
    Effect,
    Move,
    Pokemon,
    PokemonType,
    SideCondition,
)
from poke_env.data import GenData, to_id_str
from poke_env.environment.env import _EnvPlayer
from poke_env.ps_client.ps_client import PSClient
from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon

from p0.model.resources import default_runtime_resources
from p0.replays.reconstruction.contract import COPYABLE_VOLATILES
from p0.runtime.live_event_capture import capture_message
from p0.runtime.poke_env_battle_adapter import locked_slots

_ORIGINAL_STOP_LISTENING = PSClient.stop_listening
_ORIGINAL_HANDLE_MESSAGE = PSClient._handle_message
_ORIGINAL_SEND_MESSAGE = PSClient.send_message
_ORIGINAL_PARSE_MESSAGE = DoubleBattle.parse_message
_ORIGINAL_GET_POKEMON = DoubleBattle.get_pokemon
_ORIGINAL_PARSE_REQUEST = DoubleBattle.parse_request
_ORIGINAL_UPDATE_FROM_TEAMBUILDER = Pokemon._update_from_teambuilder
_ORIGINAL_START_EFFECT = Pokemon.start_effect
_ORIGINAL_MAX_PP = Move.max_pp


def _install_champions_data() -> None:
    """Synchronize poke-env's read-only mechanics with the active Champions dex."""
    dex = default_runtime_resources().dex
    gen_data = GenData.from_gen(9)
    gen_data.pokedex = {
        **gen_data.pokedex,
        **{
            entry["id"]: {
                **gen_data.pokedex.get(entry["id"], {}),
                **deepcopy(entry),
                "species": entry["baseSpecies"],
            }
            for entry in dex["species"]
        },
    }
    move_fields = frozenset(dex["source"]["moveFields"])
    gen_data.moves = {
        **gen_data.moves,
        **{
            entry["id"]: {
                **{
                    key: value
                    for key, value in gen_data.moves.get(entry["id"], {}).items()
                    if key not in move_fields
                },
                **deepcopy(entry),
            }
            for entry in dex["moves"]
        },
    }
    Move.should_be_stored.cache_clear()


def _max_pp(self: Move) -> int:
    """Use Champions PP limits, including moves copied by Transform."""
    if self.gen != 9:
        return _ORIGINAL_MAX_PP.__get__(self, Move)

    base_pp = self.entry["pp"]
    maximum = base_pp if self.entry.get("noPPBoosts") else (base_pp // 5 + 1) * 4
    return min(5, maximum) if self._from_transform else maximum


def _pressure_on(self: DoubleBattle, pokemon: str, move: str, target: str | None) -> int:
    """Count each opposing Pressure target for moves that hit both opponents."""
    data = GenData.from_gen(self.gen).moves.get(Move.retrieve_id(move))
    if data is None:
        return 0

    if data["target"] in {"all", "allAdjacent", "allAdjacentFoes"} or data["flags"].get(
        "mustpressure"
    ):
        targets = (
            self.opponent_active_pokemon if pokemon[:2] == self.player_role else self.active_pokemon
        )
    elif data["target"] in {"any", "normal", "randomNormal", "scripted"}:
        targets = [self._get_target_mon(pokemon, data["target"], target)]
    else:
        return 0

    pressure_targets = [
        mon
        for mon in targets
        if mon is not None
        and mon.ability == "pressure"
        and not mon.fainted
        and Effect.GASTRO_ACID not in mon.effects
    ]
    if not pressure_targets:
        return 0

    gas_active = any(
        mon is not None
        and mon.ability == "neutralizinggas"
        and not mon.transformed
        and Effect.GASTRO_ACID not in mon.effects
        for mon in (*self.active_pokemon, *self.opponent_active_pokemon)
    )
    return (
        sum(mon.item == "abilityshield" for mon in pressure_targets)
        if gas_active
        else len(pressure_targets)
    )


def _use_move(self: Move, pressure: bool | int = False, overridden: bool = False) -> None:
    """Spend one PP plus the count supplied by the battle's Pressure check."""
    self._current_pp = max(self._current_pp - 1 - pressure + overridden, 0)


_CRITICAL_COPY_EFFECTS = (Effect.DRAGON_CHEER, Effect.FOCUS_ENERGY)
_LEPPA_PP_RECOVERY = 10
_RIPEN_LEPPA_PP_RECOVERY = 20
# Showdown emits these through -singleturn, but they protect the whole side until the turn ends.
_SIDE_GUARDS = {
    "quickguard": SideCondition.QUICK_GUARD,
    "wideguard": SideCondition.WIDE_GUARD,
}
_PARENT_RESULTS_CREATION_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _ParentResults:
    lock: threading.Lock = field(default_factory=threading.Lock)
    completed_rooms: dict[str, str | None] = field(default_factory=dict)
    child_parents: dict[str, str] = field(default_factory=dict)
    waiters: list[tuple[int, Future[None]]] = field(default_factory=list)

    def record(self, room: str, winner: str | None) -> None:
        with self.lock:
            if room not in self.completed_rooms:
                self.completed_rooms[room] = winner
                for expected, waiter in self.waiters:
                    if len(self.completed_rooms) >= expected and not waiter.done():
                        waiter.set_result(None)
                self.waiters[:] = [w for w in self.waiters if len(self.completed_rooms) < w[0]]

    def link_child(self, child: str, parent: str) -> None:
        with self.lock:
            self.child_parents[child] = parent

    def parent_for(self, child: str) -> str | None:
        with self.lock:
            return self.child_parents.get(child)

    def finished(self, room: str) -> bool:
        with self.lock:
            return room in self.completed_rooms

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


def parent_room_for_battle(client: PSClient, battle_tag: str) -> str | None:
    """Return the parent room named by the server for a child battle, when present."""
    return _parent_results(client).parent_for(battle_tag)


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
    """Record whether Dragon Cheer reached a Dragon type, which a copy must keep."""
    _ORIGINAL_START_EFFECT(self, effect_str, details)
    if Effect.from_showdown_message(effect_str) == Effect.DRAGON_CHEER:
        self.effects[Effect.DRAGON_CHEER] = int(PokemonType.DRAGON in self.types)


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


def _promote_duplicate_partner(battle: DoubleBattle, slot: str, leaving: Pokemon) -> None:
    """Make the still-active display the team entry when its same-species twin leaves."""
    role = slot[:2]
    active = (
        battle._active_pokemon if role == battle.player_role else battle._opponent_active_pokemon
    )
    partner = active.get(f"{role}{'b' if slot[2] == 'a' else 'a'}")
    # One display may have changed form since entering (Morpeko's Hunger Switch).
    if partner is None or partner.base_species != leaving.base_species:
        return

    _replace_team_entry(battle, role, leaving, partner)
    # Keep the real Pokemon: it returns to its team entry when the display is revealed
    # as the Illusion user, when it comes back beside the display, or, once it has
    # fainted, when the Illusion user is revealed or switches in as itself.
    _battle_state(battle, "_p0_hidden_twins")[partner] = leaving


def _restore_real_twins(battle: DoubleBattle, role: str, revealed: Pokemon) -> None:
    """When the Illusion user shows itself, give disguise copies' team entries back."""
    twins = _battle_state(battle, "_p0_hidden_twins")
    team = battle._team if role == battle.player_role else battle._opponent_team
    for key, member in team.items():
        twin = twins.get(member)
        # The revealed display was the disguise. A twin that fainted is real as well,
        # because the Illusion user is alive at its reveal.
        if twin is not None and (member is revealed or twin.fainted):
            del twins[member]
            team[key] = twin


def _replace_team_entry(battle: DoubleBattle, role: str, old: Pokemon, new: Pokemon) -> None:
    team = battle._team if role == battle.player_role else battle._opponent_team
    for key, member in team.items():
        if member is old:
            team[key] = new
            return


def _faint_illusion_user(battle: DoubleBattle, role: str) -> None:
    """Move a faint shown on a disguise display to the Illusion user."""
    team = battle._team if role == battle.player_role else battle._opponent_team
    for member in team.values():
        if member.ability == "illusion":
            member.faint()
            return


def _start_roost(battle: DoubleBattle, mon: Pokemon) -> None:
    """Remove Flying until the end of the turn, as Showdown's Roost does."""
    if PokemonType.FLYING not in mon.types:
        return

    previous = mon._temporary_types
    original_types = list(mon.types)
    roosted = [type_ for type_ in original_types if type_ is not PokemonType.FLYING]
    mon._temporary_types = roosted or [PokemonType.NORMAL]
    _battle_state(battle, "_p0_roosting")[mon] = (
        previous,
        mon._temporary_types,
        original_types,
    )


def _end_roost(battle: DoubleBattle) -> None:
    roosting = _battle_state(battle, "_p0_roosting")
    for mon, (previous, roosted, _) in roosting.items():
        mon.effects.pop(Effect.ROOST, None)
        # A later type change or switch-out replaced the Roost types; keep that result.
        if mon._temporary_types is roosted:
            mon._temporary_types = previous
    roosting.clear()


def _types_before_roost(battle: DoubleBattle, mon: Pokemon) -> list[PokemonType]:
    """Return the types with any Flying that an active Roost removed put back."""
    roosting = _battle_state(battle, "_p0_roosting").get(mon)
    if roosting is not None and mon._temporary_types is roosting[1]:
        return list(roosting[2])
    return list(mon.types)


def _battle_state(battle: DoubleBattle, name: str) -> dict[Any, Any]:
    """Return a patch-owned mapping stored on the battle, creating it on first use."""
    return vars(battle).setdefault(name, {})


def _parse_message(self: DoubleBattle, split_message: list[str]):
    """
    Apply one protocol message with the state fixes poke-env's parser lacks.

    Arguments:
      self: battle receiving the message.
      split_message: protocol line split on "|", with an empty first field.

    Returns:
      Whatever poke-env's own parser returns, or None for a message handled here.
    """
    event_type = split_message[1] if len(split_message) >= 2 else None
    # Best-of rooms send this UI-only notification to the child battle room.
    # It is not a battle event and poke-env 0.15 raises NotImplementedError for it.
    if event_type in {"tempnotify", "tempnotifyoff"}:
        return None
    outgoing = None
    fainted = None
    original_size = None
    baselines: dict[Pokemon, str] = _battle_state(self, "_p0_form_baselines")
    if event_type in {"switch", "drag", "replace"} and len(split_message) >= 3:
        identifier = split_message[2]
        role = identifier[:2]
        slot = identifier[2:3]
        if role in {"p1", "p2"} and slot in {"a", "b"}:
            active = (
                self.active_pokemon if role == self.player_role else self.opponent_active_pokemon
            )
            active_by_slot = (
                self._active_pokemon if role == self.player_role else self._opponent_active_pokemon
            )
            # A revealed Illusion user may already be at 0 HP, which poke-env's active list hides.
            outgoing = (
                active_by_slot.get(identifier[:3])
                if event_type == "replace"
                else active[0 if slot == "a" else 1]
            )
            if outgoing is not None:
                captured = _battle_state(self, "_p0_transform_targets").get(id(outgoing))
                if captured is not None:
                    original_size = (captured.original_height, captured.original_weight)
            if event_type in {"switch", "drag"}:
                incoming = _ORIGINAL_GET_POKEMON(self, identifier, details=split_message[3])
                if incoming.fainted:
                    # A fainted Pokemon returns only after a revive heal clears its faint,
                    # so a disguised Illusion user fainted.
                    _faint_illusion_user(self, role)
                other_slot = f"{role}{'b' if slot == 'a' else 'a'}"
                hidden_twin = _battle_state(self, "_p0_hidden_twins").get(incoming)
                if active_by_slot.get(other_slot) is incoming and hidden_twin is not None:
                    # The display holding this team entry stays; the real Pokemon it took
                    # the entry from comes back to it.
                    del _battle_state(self, "_p0_hidden_twins")[incoming]
                    _replace_team_entry(self, role, incoming, hidden_twin)
                elif active_by_slot.get(other_slot) is incoming:
                    # In Illusion, two active slots may show the same species. poke-env
                    # would assign the same Pokemon object to both slots; clone it for the partner.
                    duplicate = deepcopy(incoming)
                    active_by_slot[other_slot] = duplicate
                    targets = _battle_state(self, "_p0_transform_targets")
                    shown_types = incoming._temporary_types
                    # The partner keeps the shown volatile state; the incoming Pokemon enters fresh.
                    incoming.switch_out(self.fields)
                    added_types = _battle_state(self, "_p0_added_types")
                    if incoming in added_types:
                        added_types[duplicate] = added_types.pop(incoming)
                    if id(incoming) in targets:
                        # A disguise copies the transformed look, so both displays show it.
                        targets[id(duplicate)] = targets[id(incoming)]
                        incoming._temporary_types = shown_types
    elif event_type == "faint" and len(split_message) >= 3:
        fainted = self.get_pokemon(split_message[2])
        captured = _battle_state(self, "_p0_transform_targets").get(id(fainted))
        if captured is not None:
            original_size = (captured.original_height, captured.original_weight)
    capture_message(self, split_message)
    transferred_boosts: dict[str, int] | None = None
    transferred_effects: dict[Effect, int] = {}
    revealed_status = None
    if event_type == "replace" and outgoing is not None:
        revealed = self.get_pokemon(split_message[2], details=split_message[3])
        if revealed is not outgoing:
            # poke-env clears the display when it reveals, so its state moves first.
            _move_active_state(outgoing, revealed, _battle_state(self, "_p0_added_types"))
        revealed_status = outgoing.status
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
    if event_type == "-singleturn" and len(split_message) >= 4:
        guard = _SIDE_GUARDS.get(to_id_str(split_message[3].removeprefix("move: ")))
        if guard is not None:
            self._replay_data.append(split_message[:])
            own = split_message[2][:2] == self.player_role
            (self.side_conditions if own else self.opponent_side_conditions)[guard] = self.turn
            return None
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
    if event_type == "-formechange" and len(split_message) >= 4:
        current = self.get_pokemon(split_message[2])
        baselines.setdefault(current, current.species)
    round_user = None
    if event_type == "move" and split_message[-1] == "[from] move: Round":
        # Round moves later Round users up and names itself as the cause. poke-env reads
        # that as a move Round called, so it neither reveals nor spends the user's Round.
        round_user = self.get_pokemon(split_message[2])
        round_user._add_move("round")
    result = _ORIGINAL_PARSE_MESSAGE(self, split_message)
    if event_type == "-endability" and len(split_message) >= 3:
        # The pinned Champions protocol emits this for the gastroacid volatile,
        # shared by Gastro Acid and Core Enforcer. The named ability is retained.
        self.get_pokemon(split_message[2]).start_effect("Gastro Acid")
    if round_user is not None:
        round_user.moves["round"].use()
    if event_type == "-transform" and len(split_message) >= 4:
        # Transform copies changed types, retaining Flying when the target is using Roost.
        target = self.get_pokemon(split_message[3])
        self.get_pokemon(split_message[2])._temporary_types = _types_before_roost(self, target)
    if event_type == "-singleturn" and len(split_message) >= 4:
        if to_id_str(split_message[3].removeprefix("move: ")) == "roost":
            _start_roost(self, self.get_pokemon(split_message[2]))
    if event_type in {"upkeep", "turn"}:
        # Roost and the side guards last until the end of the turn.
        _end_roost(self)
        for guard in _SIDE_GUARDS.values():
            self.side_conditions.pop(guard, None)
            self.opponent_side_conditions.pop(guard, None)
    leaving = fainted if outgoing is None else outgoing
    if leaving is not None:
        _promote_duplicate_partner(self, split_message[2][:3], leaving)
    if event_type == "replace" and outgoing is not None:
        _restore_real_twins(self, split_message[2][:2], outgoing)
    elif event_type in {"switch", "drag"} and len(split_message) >= 3:
        shown = self.get_pokemon(split_message[2])
        if shown.ability == "illusion":
            # The Illusion user shows itself, so it is alive and no fainted twin was it.
            _restore_real_twins(self, split_message[2][:2], shown)
    if fainted is not None:
        # Showdown clears volatiles on faint; poke-env only clears a subset.
        fainted.switch_out(self.fields)
    if event_type == "-start" and len(split_message) >= 5 and split_message[3] == "typeadd":
        target = self.get_pokemon(split_message[2])
        added_type = PokemonType.from_name(split_message[4])
        current_types = _types_before_roost(self, target)
        if added_type not in current_types:
            target._temporary_types = [*current_types, added_type]
            if Effect.ROOST in target.effects:
                _start_roost(self, target)
        # Kept apart from the types: a disguise can already show the added type.
        _battle_state(self, "_p0_added_types")[target] = added_type
    if event_type == "-start" and len(split_message) >= 4 and split_message[3] == "typechange":
        # A type change replaces an added type too.
        target = self.get_pokemon(split_message[2])
        _battle_state(self, "_p0_added_types").pop(target, None)
        if Effect.ROOST in target.effects:
            _start_roost(self, target)
    cleared = fainted
    if cleared is None and event_type in {"switch", "drag"}:
        cleared = outgoing
    if cleared is not None:
        _battle_state(self, "_p0_added_types").pop(cleared, None)
        if original_size is not None:
            cleared._heightm, cleared._weightkg = original_size
        base_form = baselines.get(cleared)
        if base_form is not None and cleared.species != base_form:
            cleared.forme_change(base_form)
    # Record the incoming baseline after the reset: in a same-species Illusion switch the
    # outgoing and incoming Pokemon share one object.
    if event_type in {"switch", "drag", "replace", "detailschange"} and len(split_message) >= 4:
        current = self.get_pokemon(split_message[2])
        shown_species = split_message[3].split(",", 1)[0]
        shown_id = to_id_str(shown_species)
        species_bases = default_runtime_resources().species_bases
        if (
            event_type in {"switch", "drag"}
            and current.species != shown_id
            and species_bases.get(shown_id, shown_id) == current.base_species
        ):
            # poke-env applies switch details only to a new Pokemon. A team entry that
            # passed to a disguise's copy can hold a stale form (Mimikyu before Busted).
            # Another species here is a disguise of a transformed Pokemon, not a form.
            current.forme_change(shown_species)
        baselines[current] = current.species
    if transferred_boosts is not None or transferred_effects:
        incoming = self.get_pokemon(split_message[2])
        if transferred_boosts is not None:
            incoming.boosts = transferred_boosts
        incoming.effects.update(transferred_effects)
    if event_type == "replace" and outgoing is not None:
        # poke-env copies the status to the revealed Pokemon, then clears it when copying HP.
        self.get_pokemon(split_message[2]).status = revealed_status
    return result


def _get_pokemon(
    self: DoubleBattle,
    identifier: str,
    force_self_team: bool = False,
    details: str = "",
    request: dict[str, Any] | None = None,
) -> Pokemon:
    """Resolve active protocol references by slot, and off-field references by nickname."""
    if not force_self_team and identifier[3:5] == ": " and not details and request is None:
        # The owner's request already names its disguised Illusion user. Later
        # public lines still use the copied name, but always mean this active slot.
        active = (
            self._active_pokemon
            if identifier[:2] == self.player_role
            else self._opponent_active_pokemon
        )
        member = active.get(identifier[:3])
        if member is not None:
            return member
    if not details and identifier[2:4] == ": ":
        # An off-field Pokemon is named without a slot and with no details. Its default
        # nickname is the base species, while the team sheet keys it by the full species.
        # An Illusion user that never showed itself has had no switch line to link the two.
        details = identifier[4:]
    return _ORIGINAL_GET_POKEMON(self, identifier, force_self_team, details, request)


def _place_requested_actives(battle: DoubleBattle, side_pokemon: list[dict[str, Any]]) -> None:
    """
    Put the request's active Pokemon in their slots.

    poke-env detects a disguised own Illusion user from the request but leaves the
    disguise display in its slot, which then reads as empty. Until the request, the
    display was the team entry of the benched Pokemon it copies, so the state the
    disguise gained moves to the Illusion user and the benched Pokemon is reset.
    """
    # Showdown lists the active Pokemon first, in slot order.
    for position, pokemon in zip("ab", side_pokemon):
        slot = f"{pokemon['ident'][:2]}{position}"
        shown = battle._active_pokemon.get(slot)
        member = battle._team.get(pokemon["ident"])
        if not pokemon["active"] or shown is None or member is None or shown is member:
            continue
        if shown.base_species == member.base_species:
            # The clone made for two same-species displays holds this member's state.
            battle._team[pokemon["ident"]] = shown
            continue

        _move_active_state(shown, member, _battle_state(battle, "_p0_added_types"))
        # The request that follows sets the benched Pokemon's HP and status.
        shown.switch_out(battle.fields)
        battle._active_pokemon[slot] = member


def _move_active_state(
    shown: Pokemon, member: Pokemon, added_types: dict[Pokemon, PokemonType]
) -> None:
    """Give the Illusion user the battle state its disguise display gained on the field."""
    member.boosts = dict(shown.boosts)
    member.effects.update(shown.effects)
    member._active_turns = shown._active_turns
    member._protect_counter = shown._protect_counter
    member._must_recharge = shown._must_recharge
    member._preparing_move = shown._preparing_move
    member._preparing_target = shown._preparing_target
    if Effect.TYPECHANGE in shown.effects:
        # A type change replaces the types, so the display's types are the user's.
        member._temporary_types = list(shown._temporary_types)
    added = added_types.pop(shown, None)
    if added is not None:
        # An added type (Forest's Curse, Trick-or-Treat) joins the user's own types.
        added_types[member] = added
        if added not in member.types:
            member._temporary_types = [*member.types, added]


def _parse_request(
    self: DoubleBattle, request: dict[str, Any], strict_battle_tracking: bool = False
) -> None:
    preview = request.get("teamPreview", False)
    if not preview:
        _place_requested_actives(self, request["side"]["pokemon"])
    _ORIGINAL_PARSE_REQUEST(self, request, strict_battle_tracking)
    if not preview:
        # Normal requests carry authoritative PP, including copied and disabled moves.
        # Locked, recharge and Struggle entries omit PP and retain event-tracked values.
        active_requests: list[dict[str, Any]] = request.get("active", [])
        for pokemon, active in zip(self.active_pokemon, active_requests):
            if pokemon is None:
                continue
            for move_request in active["moves"]:
                move = pokemon.moves.get(move_request["id"])
                if move is not None and "pp" in move_request:
                    move._current_pp = move_request["pp"]
        # poke-env takes Showdown's active flag (the first two party positions) as true
        # even in the team preview request. A disguised Illusion user that leads from
        # one of them is then never switched in, so it would never count as seen.
        for pokemon in request["side"]["pokemon"]:
            if pokemon["active"]:
                self._team[pokemon["ident"]]._revealed = True


def locked_choice(battle: DoubleBattle, message: str) -> str:
    """
    Send each locked slot's move as "move 1" in a choice command.

    Showdown reads a move named by id in a locked request as single-target and rejects
    it without one, then ignores the target it was given. poke-env names moves by id
    and only works around this for recharge.
    """
    if not message.startswith("/choose "):
        return message
    choices = message.removeprefix("/choose ").split(", ")
    for position, locked in enumerate(locked_slots(battle)[: len(choices)]):
        if locked and choices[position].startswith("move "):
            choices[position] = "move 1"
    return "/choose " + ", ".join(choices)


async def _handle_message(self: PSClient, message: str):
    # Showdown sends the Bo3 parent room as >game-bestof..., while poke-env only
    # understands >battle rooms and otherwise indexes a non-existent protocol field.
    if message.startswith(">game-"):
        parent_room, _, body = message.partition("\n")
        parent_room = parent_room[1:]
        tracker = _parent_results(self)
        player = getattr(self._on_battle_message, "__self__", None)
        on_result = getattr(player, "_series_finished_callback", None)
        for line in body.splitlines():
            if line.startswith(("|uhtml|game", "|uhtmlchange|game")):
                child = line.partition('href="/')[2].partition('"')[0]
                if child.startswith("battle-"):
                    tracker.link_child(child, parent_room)
            if line.startswith("|win|"):
                winner = line.removeprefix("|win|")
            elif line == "|tie":
                winner = None
            else:
                continue
            if not tracker.finished(parent_room):
                if on_result is not None:
                    on_result(parent_room)
                tracker.record(parent_room, winner)
        if "I'm ready!</button>" in message and not tracker.finished(parent_room):
            await self.send_message(f"/msgroom {parent_room},/confirmready")
        return None
    if message.startswith(">battle-") and "|bestof|" in message:
        battle_room, _, body = message.partition("\n")
        for line in body.splitlines():
            if line.startswith(("|uhtml|bestof|", "|uhtmlchange|bestof|")):
                parent = line.partition('href="/')[2].partition('"')[0]
                if parent.startswith("game-"):
                    _parent_results(self).link_child(battle_room[1:], parent)
    return await _ORIGINAL_HANDLE_MESSAGE(self, message)


async def _send_message(self: PSClient, message: str, room: str = "", message_2=None):
    # A Bo3 format with Force Open Team Sheets has no accept/reject negotiation.
    # The pinned client sends one of those commands for every VGC battle. Showdown
    # answers with an error line, which poke-env logs at CRITICAL once per game.
    if getattr(self, "_p0_force_open_team_sheet", False) and message in {
        "/acceptopenteamsheets",
        "/rejectopenteamsheets",
    }:
        return None
    # The client holds its player's bound battle callback, which leads to the room's battle.
    player = getattr(self._on_battle_message, "__self__", None)
    battle = getattr(player, "_battles", {}).get(room)
    if battle is not None:
        message = locked_choice(battle, message)
    return await _ORIGINAL_SEND_MESSAGE(self, message, room, message_2)


def _forme_change(self: Pokemon, species: str) -> None:
    """Preserve the observed battle form; like Showdown's setSpecies, it resets type changes."""
    normalized_species = species.split(",", 1)[0]
    self._update_from_pokedex(normalized_species, store_species=True)
    self._temporary_types = []


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


# Every patched attribute with its replacement.
_PATCHES = (
    (Move, "max_pp", property(_max_pp)),
    (Move, "use", _use_move),
    (DoubleBattle, "_pressure_on", _pressure_on),
    (PSClient, "wait_for_login", _wait_for_login),
    (PSClient, "stop_listening", _stop_listening_cleanly),
    (PSClient, "_handle_message", _handle_message),
    (PSClient, "send_message", _send_message),
    (DoubleBattle, "parse_message", _parse_message),
    (DoubleBattle, "get_pokemon", _get_pokemon),
    (DoubleBattle, "parse_request", _parse_request),
    (Pokemon, "forme_change", _forme_change),
    (Pokemon, "_update_from_teambuilder", _update_from_teambuilder),
    (Pokemon, "start_effect", _start_effect),
    (Pokemon, "copy_boosts", _copy_boosts),
)


def install(logger: logging.Logger | None = None) -> None:
    """Install compatibility patches for the pinned poke-env release."""
    target = logger if logger is not None else logging.getLogger("poke_env")
    # addFilter ignores a filter the logger already has.
    target.addFilter(_INACTIVE_POKEMON_FILTER)

    if PSClient.wait_for_login is _wait_for_login:
        return

    _install_champions_data()
    for owner, name, patched in _PATCHES:
        setattr(owner, name, patched)
