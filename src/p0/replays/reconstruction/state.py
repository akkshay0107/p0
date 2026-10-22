"""Owned battle state and immutable snapshots for replay reconstruction v2."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import cache, partial
from typing import Any, NamedTuple

from p0.model.resources import default_runtime_resources
from p0.replays.identity import ReplayMemberId, ReplaySide, normalize_showdown_id
from p0.replays.reconstruction.classification import (
    CLASSIFICATION_REGISTRY,
    EventClassification,
)
from p0.replays.reconstruction.contract import (
    BREAKING_PROTECT_MOVES,
    COPYABLE_VOLATILES,
    DELAYED_MOVES,
    SLOT_CONDITION_MOVES,
    STALLING_MOVES,
    TERRAINS,
)
from p0.replays.reconstruction.diagnostics import (
    ReplayEventDiagnostic,
    ReplayEventParseError,
    ReplayRejectionCategory,
)
from p0.replays.reconstruction.events import EffectReference
from p0.replays.reconstruction.resolution import (
    ResolvedPokemonRefArgument,
    ResolvedProtocolEvent,
)
from p0.replays.schema import OTSData, OTSMember

_STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
_BOOST_NAMES = (*_STAT_NAMES[1:], "accuracy", "evasion")
_STATUS_NAMES = frozenset({"brn", "frz", "par", "psn", "slp", "tox"})
_STACKING_SIDE_CONDITIONS = frozenset({"spikes", "toxicspikes"})
# These two are emitted through -singleturn but are side conditions in the
# simulator. Other ally-side conditions remain governed by the generated dex.
_SIDE_GUARDS = frozenset({"quickguard", "wideguard"})
_BREAKABLE_MEMBER_PROTECT_EFFECTS = frozenset(
    {"banefulbunker", "kingsshield", "protect", "spikyshield"}
)
# Moves and effects that raise the user's consecutive-protection counter.
_PROTECT_COUNTER_MOVES = STALLING_MOVES | _SIDE_GUARDS
_CANT_SUPPRESS_ABILITIES = frozenset(
    {
        "battlebond",
        "disguise",
        "gulpmissile",
        "iceface",
        "shieldsdown",
        "stancechange",
        "zerotohero",
    }
)
# Showdown names these effects with their current count (perish3 is Perish Song).
# Any other count fails effect validation.
_DYNAMIC_EFFECT_IDS = {
    **{f"perish{count}": "perishsong" for count in range(4)},
    **{f"stockpile{layer}": "stockpile" for layer in range(1, 4)},
    **{f"fallen{count}": "supremeoverlord" for count in range(1, 6)},
}
_DYNAMIC_EFFECTS = frozenset(_DYNAMIC_EFFECT_IDS.values())


@dataclass(frozen=True, slots=True)
class AbilityState:
    """Original, form-native, and temporary ability components."""

    base: str
    forme: str | None = None
    temporary: str | None = None

    @property
    def current(self) -> str:
        """Return the effective named ability without modelling suppression."""
        return self.temporary or self.forme or self.base


class DelayedMoveState(NamedTuple):
    """A validated Future Sight condition attached to a side slot."""

    source_member_id: ReplayMemberId
    target_side: ReplaySide
    target_slot: int
    move_id: str
    scheduled_turn: int
    announced: bool


class SlotConditionState(NamedTuple):
    """A validated Wish or Healing Wish condition attached to a side slot."""

    source_member_id: ReplayMemberId
    target_side: ReplaySide
    target_slot: int
    move_id: str
    created_turn: int
    expiration_turn: int | None


class MoveState(NamedTuple):
    """Immutable move-slot state used by base members and Transform overlays."""

    move_id: str
    name: str
    move_type: str
    category: str
    target: str
    current_pp: int
    max_pp: int


class TransformSnapshot(NamedTuple):
    """Copied values that override the transforming member's native state."""

    species: str
    weight: float
    non_hp_base_stats: tuple[tuple[str, int], ...]
    ability: str
    moves: tuple[MoveState, ...]


class ReplayPokemonState(NamedTuple):
    """One stable roster member at a specific public replay cursor."""

    member_id: ReplayMemberId
    nickname: str
    original_species: str
    current_form: str
    nature: str
    level: int
    hp_fraction: float | None
    status: str | None
    item: str | None
    ability: AbilityState
    current_types: tuple[str, ...]
    base_stats: tuple[tuple[str, int], ...]
    weight: float
    # The move slots in use: a Transform's copied moves while it lasts.
    moves: tuple[MoveState, ...]
    boosts: tuple[tuple[str, int], ...]
    effects: tuple[tuple[str, int], ...]
    transform: TransformSnapshot | None
    # Seen on the field: under its own name, or as the Pokemon a disguise copied. Never reset.
    revealed: bool
    selected: bool | None
    fainted: bool
    active_turns: int
    status_counter: int
    protect_counter: int
    preparing: str | None
    last_move: str | None


class PublicSideState(NamedTuple):
    """Causal team entries and active displays, including same-species copies."""

    members: tuple[ReplayPokemonState, ...]
    active: tuple[ReplayPokemonState | None, ReplayPokemonState | None]


class ReplaySideState(NamedTuple):
    """Immutable side state and active-slot bindings."""

    side: ReplaySide
    active: tuple[ReplayMemberId | None, ReplayMemberId | None]
    conditions: tuple[tuple[str, int], ...]
    used_mega: bool


@dataclass(frozen=True, slots=True)
class ReplayBattleState:
    """Deeply immutable public battle state after one protocol line."""

    replay_id: str
    line_index: int
    turn: int
    members: tuple[ReplayPokemonState, ...]
    sides: tuple[ReplaySideState, ReplaySideState]
    weather: tuple[tuple[str, int], ...]
    fields: tuple[tuple[str, int], ...]
    delayed_moves: tuple[DelayedMoveState, ...] = ()
    slot_conditions: tuple[SlotConditionState, ...] = ()
    team_sizes: tuple[int, int] = (4, 4)
    public_sides: tuple[PublicSideState | None, PublicSideState | None] = (None, None)

    def member(self, member_id: ReplayMemberId) -> ReplayPokemonState:
        """Return one member; members are stored in side order, then roster order."""
        return self.members[member_id.side.side_index * 6 + member_id.roster_index]


@dataclass(frozen=True, slots=True)
class ReconstructedReplayState:
    """Selected line-indexed snapshots or a whole-replay rejection."""

    replay_id: str
    snapshots: tuple[ReplayBattleState, ...]
    diagnostics: tuple[ReplayEventDiagnostic, ...] = ()
    # The HP fraction a -damage or -heal line's member had before that line, by line index.
    hp_before: Mapping[int, float | None] = field(default_factory=dict)

    def require_accepted(self) -> tuple[ReplayBattleState, ...]:
        """Return snapshots or raise the structured whole-replay rejection."""
        if self.diagnostics:
            raise ReplayEventParseError(self.diagnostics)
        return self.snapshots


class _SpeciesData(NamedTuple):
    species: str
    base_species: str
    types: tuple[str, ...]
    base_stats: tuple[tuple[str, int], ...]
    weight: float
    native_ability: str


@dataclass(slots=True)
class _MutablePokemon:
    member_id: ReplayMemberId
    nickname: str
    original_species: str
    current_form: str
    persistent_form: str
    nature: str
    level: int
    hp_fraction: float | None
    status: str | None
    item: str | None
    ability: AbilityState
    current_types: tuple[str, ...]
    base_stats: tuple[tuple[str, int], ...]
    weight: float
    moves: dict[str, MoveState]
    boosts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_BOOST_NAMES, 0))
    effects: dict[str, int] = field(default_factory=dict)
    effect_variants: dict[str, str] = field(default_factory=dict)
    single_turn_effects: set[str] = field(default_factory=set)
    single_move_effects: set[str] = field(default_factory=set)
    transform: TransformSnapshot | None = None
    revealed: bool = False
    selected: bool | None = None
    fainted: bool = False
    active_turns: int = 0
    status_counter: int = 0
    protect_counter: int = 0
    preparing: str | None = None
    last_move: str | None = None
    added_type: str | None = None
    underlying_types: tuple[str, ...] = ()

    def move_snapshots(self) -> tuple[MoveState, ...]:
        """Return effective move slots without exposing mutable move objects."""
        if self.transform is not None:
            return self.transform.moves
        return tuple(self.moves.values())

    def snapshot(self) -> ReplayPokemonState:
        return ReplayPokemonState(
            member_id=self.member_id,
            nickname=self.nickname,
            original_species=self.original_species,
            current_form=self.current_form,
            nature=self.nature,
            level=self.level,
            hp_fraction=self.hp_fraction,
            status=self.status,
            item=self.item,
            ability=self.ability,
            current_types=self.current_types,
            base_stats=self.base_stats,
            weight=self.weight,
            moves=self.move_snapshots(),
            boosts=tuple((name, self.boosts[name]) for name in _BOOST_NAMES),
            effects=tuple(sorted(self.effects.items())),
            transform=self.transform,
            revealed=self.revealed,
            selected=self.selected,
            fainted=self.fainted,
            active_turns=self.active_turns,
            status_counter=self.status_counter,
            protect_counter=self.protect_counter,
            preparing=self.preparing,
            last_move=self.last_move,
        )


class _StateTransitionError(ValueError):
    category = ReplayRejectionCategory.INVALID_INPUT_CONTRACT


class _UnsupportedMechanic(_StateTransitionError):
    """A public mechanic whose state is not supported."""

    category = ReplayRejectionCategory.UNSUPPORTED_EVENT


class _StateReducer:
    """Single owner of mutable replay state and all supported transitions."""

    def __init__(self, replay_id: str, ots: tuple[OTSData, OTSData]) -> None:
        self.replay_id = replay_id
        self.turn = 0
        self._species, self._moves, self._legal_effects = dex_indexes()
        self._members = {
            member.member_id: self._member_from_ots(member)
            for sheet in sorted(ots, key=lambda sheet: sheet.side.side_index)
            for member in sheet.members
        }
        self.active: dict[tuple[ReplaySide, int], ReplayMemberId] = {}
        self.team_sizes = {sheet.side: min(4, len(sheet.members)) for sheet in ots}
        self.side_conditions = {ReplaySide.P1: {}, ReplaySide.P2: {}}
        self.side_single_turn_effects = {ReplaySide.P1: set(), ReplaySide.P2: set()}
        self.used_mega = {ReplaySide.P1: False, ReplaySide.P2: False}
        self.weather: dict[str, int] = {}
        self.fields: dict[str, int] = {}
        self.delayed_moves: dict[tuple[ReplaySide, int], DelayedMoveState] = {}
        # The slot each fainted member left; a move can still target that empty slot.
        self._fainted_slots: dict[ReplayMemberId, int] = {}
        # Items that ended while their member held a different item, until the turn ends.
        self._unheld_item_ends: dict[ReplayMemberId, str] = {}
        self.slot_conditions: dict[tuple[ReplaySide, int, str], SlotConditionState] = {}
        self.hp_before: dict[int, float | None] = {}
        # Most members do not change between snapshots. Projection reuses a member's
        # view while its snapshot is the same object.
        self._member_snapshots: dict[ReplayMemberId, ReplayPokemonState] = {}
        # Members handed out by _member since the last snapshot; all of them before the first.
        self._touched: set[ReplayMemberId] = set(self._members)

    def _member_from_ots(self, member: OTSMember) -> _MutablePokemon:
        species = self._species_data(member.species)
        if not member.ability:
            raise _StateTransitionError(
                f"OTS member {member.member_id!r} has no recoverable base ability"
            )
        moves = {move.move_id: move for move in (move_state(name) for name in member.moves)}
        return _MutablePokemon(
            member_id=member.member_id,
            nickname=member.nickname,
            original_species=member.species,
            current_form=member.species,
            persistent_form=member.species,
            nature=member.nature,
            level=member.level,
            hp_fraction=None,
            status=None,
            item=member.item or None,
            ability=AbilityState(member.ability),
            current_types=species.types,
            underlying_types=species.types,
            base_stats=species.base_stats,
            weight=species.weight,
            moves=moves,
        )

    def _species_data(self, name: str) -> _SpeciesData:
        try:
            return self._species[normalize_showdown_id(name)]
        except KeyError as exc:
            raise _StateTransitionError(f"species {name!r} is absent from the pinned dex") from exc

    def apply(self, resolved: ResolvedProtocolEvent) -> None:
        handler = _HANDLERS.get(resolved.event.tag)
        if handler is not None:
            handler(self, resolved)

    def _handle_turn(self, resolved: ResolvedProtocolEvent) -> None:
        turn = int(resolved.event.arguments[0])
        self._unheld_item_ends.clear()
        if turn <= self.turn:
            raise _StateTransitionError(
                f"turn number {turn} does not advance current turn {self.turn}"
            )
        self._expire_member_single_turn_effects()
        self._expire_side_single_turn_effects()
        self.turn = turn
        for member_id in self.active.values():
            member = self._member(member_id)
            member.active_turns += 1
            if member.status == "tox":
                member.status_counter += 1

    def _handle_upkeep(self, resolved: ResolvedProtocolEvent) -> None:
        self._expire_member_single_turn_effects()
        self._expire_side_single_turn_effects()
        self._expire_slot_conditions()

    def _handle_teamsize(self, resolved: ResolvedProtocolEvent) -> None:
        side = ReplaySide(resolved.event.arguments[0])
        self.team_sizes[side] = int(resolved.event.arguments[1])

    def _expire_member_single_turn_effects(self) -> None:
        for member_id in self.active.values():
            member = self._member(member_id)
            for effect in member.single_turn_effects:
                member.effects.pop(effect, None)
                if effect == "roost":
                    _restore_underlying_types(member)
            member.single_turn_effects.clear()

    def _expire_side_single_turn_effects(self) -> None:
        for side, effects in self.side_single_turn_effects.items():
            for effect in effects:
                self.side_conditions[side].pop(effect, None)
            effects.clear()

    def _switch(self, resolved: ResolvedProtocolEvent) -> None:
        reference = _resolved_reference(resolved, 0)
        member_id = _required_member(reference)
        slot = _required_slot(reference)
        key = (member_id.side, slot)
        outgoing_id = self.active.get(key)
        if outgoing_id is not None and outgoing_id != member_id:
            outgoing = self._member(outgoing_id)
            incoming = self._member(member_id)
            cause = resolved.event.cause
            if cause is not None and cause.normalized == "batonpass":
                incoming.boosts = dict(outgoing.boosts)
                for effect, created_turn in outgoing.effects.items():
                    if self._is_copyable_volatile(effect):
                        incoming.effects[effect] = created_turn
                incoming.effect_variants.update(
                    (effect, variant)
                    for effect, variant in outgoing.effect_variants.items()
                    if effect in incoming.effects
                )
                incoming.single_turn_effects.update(
                    effect for effect in outgoing.single_turn_effects if effect in incoming.effects
                )
                incoming.single_move_effects.update(
                    effect for effect in outgoing.single_move_effects if effect in incoming.effects
                )
                if (
                    "gastroacid" in incoming.effects
                    and normalize_showdown_id(incoming.ability.current) in _CANT_SUPPRESS_ABILITIES
                ):
                    incoming.effects.pop("gastroacid")
            elif cause is not None and cause.normalized == "shedtail":
                if "substitute" in outgoing.effects:
                    incoming.effects["substitute"] = self.turn
            self._clear_battle_state(outgoing)

        member = self._member(member_id)
        details_species = _details_species(resolved.event.arguments[1])
        details_data = self._species_data(details_species)
        original_data = self._species_data(member.original_species)
        disguised = normalize_showdown_id(details_data.base_species) != normalize_showdown_id(
            original_data.base_species
        )
        copied = self._copied_member(member_id, details_species) if disguised else None
        if (disguised and copied is None) or (
            normalize_showdown_id(member.ability.current) == "illusion"
            and self._transformed_into(member_id.side, details_data.base_species)
        ):
            # The disguise copies a transformed teammate, whose look neither side tracks.
            # It can even show the Illusion user's own species (a Ditto copying Zoroark).
            raise _UnsupportedMechanic("the disguise copies a transformed Pokemon")
        if copied is not None:
            if normalize_showdown_id(member.ability.current) != "illusion":
                raise _StateTransitionError(
                    "disguised switch resolved to a member without Illusion"
                )
            member.effects["illusion"] = self.turn
            # Both players see the copied Pokemon's name enter the field.
            self._member(copied).revealed = True
        else:
            self._set_form(member, details_species, persistent=True)
            member.effects.pop("illusion", None)
            member.revealed = True
        hp_fraction, status = _parse_hp_status(resolved.event.arguments[2])
        member.hp_fraction = hp_fraction
        member.status = status
        if status != "slp":
            member.status_counter = 0
        member.fainted = hp_fraction == 0.0
        member.selected = True
        member.active_turns = 0
        self.active[key] = member_id

    def _handle_replace(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        revealed_species = _details_species(resolved.event.arguments[1])
        revealed_data = self._species_data(revealed_species)
        original_data = self._species_data(member.original_species)
        if normalize_showdown_id(revealed_data.base_species) != normalize_showdown_id(
            original_data.base_species
        ):
            raise _StateTransitionError("Illusion reveal does not match the active member")
        if "illusion" not in member.effects:
            raise _StateTransitionError("replace requires an active Illusion history")

        member.revealed = True
        member.effects.pop("illusion")

    def _handle_swap(self, resolved: ResolvedProtocolEvent) -> None:
        reference = _resolved_reference(resolved, 0)
        member_id = _required_member(reference)
        source_slot = _required_slot(reference)
        target_slot = int(resolved.event.arguments[1])
        source_key = (member_id.side, source_slot)
        target_key = (member_id.side, target_slot)
        target = self.active.get(target_key)
        self.active[target_key] = member_id
        if target is None:
            self.active.pop(source_key)
        else:
            self.active[source_key] = target

    def _handle_faint(self, resolved: ResolvedProtocolEvent) -> None:
        reference = _resolved_reference(resolved, 0)
        member_id = _required_member(reference)
        slot = _required_slot(reference)
        key = (member_id.side, slot)
        member = self._member(member_id)
        member.hp_fraction = 0.0
        member.fainted = True
        member.status = None
        self._clear_battle_state(member)
        self.active.pop(key)
        self._fainted_slots[member_id] = slot

    def _handle_move(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        move_id = normalize_showdown_id(resolved.event.arguments[1])
        move_states = member.move_snapshots()
        if (
            resolved.event.cause is None
            and move_id != "struggle"
            and all(item.move_id != move_id for item in move_states)
        ):
            raise _StateTransitionError(
                f"move {resolved.event.arguments[1]!r} is not in the member's effective move slots"
            )
        if move_id in DELAYED_MOVES:
            # A failed use names no target and schedules nothing.
            if len(resolved.event.arguments) > 2 and resolved.event.arguments[2]:
                self._schedule_delayed_move(resolved, member, move_id)
        elif move_id in SLOT_CONDITION_MOVES:
            self._schedule_slot_condition(resolved, member, move_id)
        elif move_id not in _PROTECT_COUNTER_MOVES:
            member.protect_counter = 0
        cause = resolved.event.cause
        pp_owner: str | None = move_id
        if cause is not None:
            owns_pp = cause.namespace == "move" and any(
                move.move_id == cause.normalized for move in move_states
            )
            pp_owner = cause.normalized if owns_pp else None
        if pp_owner is not None and pp_owner != "struggle":
            if cause is None:
                self._change_move_pp(member, pp_owner, -1)
            # Pressure spends one additional PP for each opposing Pressure
            # target. This expenditure has no dedicated minor protocol line.
            pressure_count = self._pressure_target_count(resolved, member, move_id)
            if pressure_count:
                self._change_move_pp(member, pp_owner, -pressure_count)
        for effect in member.single_move_effects:
            member.effects.pop(effect, None)
        member.single_move_effects.clear()
        member.last_move = move_id
        member.preparing = None

    def _pressure_target_count(
        self,
        resolved: ResolvedProtocolEvent,
        source: _MutablePokemon,
        move_id: str,
    ) -> int:
        """Count active Pressure targets that consume an extra PP."""
        data = self._moves.get(move_id)
        target_class = "" if data is None else str(data.get("target", ""))
        target_ids = {
            reference.member_id
            for reference in resolved.pokemon_refs
            if reference.argument_index != 0 and reference.member_id is not None
        }
        if target_class in {"all", "allAdjacent", "allAdjacentFoes"} or (
            data is not None and data.get("flags", {}).get("mustpressure")
        ):
            target_ids = {
                member_id
                for (side, _), member_id in self.active.items()
                if side is not source.member_id.side
            }
        return sum(
            normalize_showdown_id(self._current_ability(self._member(member_id))) == "pressure"
            for member_id in target_ids
            if member_id.side is not source.member_id.side
        )

    def _schedule_delayed_move(
        self,
        resolved: ResolvedProtocolEvent,
        source: _MutablePokemon,
        move_id: str,
    ) -> None:
        target_ref = _resolved_reference(resolved, 2)
        target_member_id = _required_member(target_ref)
        target_slot = target_ref.pokemon_ref.active_slot
        if target_slot is None:
            # The target fainted earlier this turn. The move still lands on the slot it left.
            target_slot = self._fainted_slots.get(target_member_id)
            occupant = None
        else:
            occupant = target_member_id
        if target_slot is None or self.active.get((target_member_id.side, target_slot)) != occupant:
            raise _StateTransitionError("delayed move target is not in its referenced slot")
        target_key = (target_member_id.side, target_slot)
        if target_key in self.delayed_moves:
            raise _StateTransitionError("delayed move target slot already has a pending effect")

        self.delayed_moves[target_key] = DelayedMoveState(
            source_member_id=source.member_id,
            target_side=target_member_id.side,
            target_slot=target_slot,
            move_id=move_id,
            # Showdown resolves delayed moves at the end of the second turn after use.
            scheduled_turn=self.turn + 2,
            announced=False,
        )

    def _schedule_slot_condition(
        self,
        resolved: ResolvedProtocolEvent,
        source: _MutablePokemon,
        move_id: str,
    ) -> None:
        source_slot = _required_slot(_resolved_reference(resolved, 0))
        key = (source.member_id.side, source_slot, move_id)
        if key in self.slot_conditions:
            return
        self.slot_conditions[key] = SlotConditionState(
            source_member_id=source.member_id,
            target_side=source.member_id.side,
            target_slot=source_slot,
            move_id=move_id,
            created_turn=self.turn,
            expiration_turn=self.turn + 1 if move_id == "wish" else None,
        )

    def _expire_slot_conditions(self) -> None:
        for key, condition in tuple(self.slot_conditions.items()):
            if (
                condition.move_id == "wish"
                and condition.expiration_turn is not None
                and condition.expiration_turn <= self.turn
            ):
                del self.slot_conditions[key]

    def _discard_failed_slot_condition(self, member: _MutablePokemon) -> None:
        if member.last_move not in SLOT_CONDITION_MOVES:
            return
        for key, condition in tuple(self.slot_conditions.items()):
            if (
                condition.source_member_id == member.member_id
                and condition.move_id == member.last_move
                and condition.created_turn == self.turn
            ):
                del self.slot_conditions[key]

    def _handle_delayed_start(
        self,
        resolved: ResolvedProtocolEvent,
        move_id: str,
    ) -> None:
        source = self._member_for(resolved, 0)
        matches = tuple(
            (key, delayed)
            for key, delayed in self.delayed_moves.items()
            if delayed.source_member_id == source.member_id and delayed.move_id == move_id
        )
        pending = tuple((key, delayed) for key, delayed in matches if not delayed.announced)
        if len(pending) != 1:
            if not pending and matches:
                raise _StateTransitionError(f"{move_id} start was emitted more than once")
            raise _StateTransitionError(
                f"{move_id} start has an ambiguous pending move from the referenced caster"
            )
        key, delayed = pending[0]
        self.delayed_moves[key] = delayed._replace(announced=True)

    def _handle_delayed_end(
        self,
        resolved: ResolvedProtocolEvent,
        move_id: str,
    ) -> None:
        target_ref = _resolved_reference(resolved, 0)
        target_member_id = _required_member(target_ref)
        target_slot = _required_slot(target_ref)
        target_key = (target_member_id.side, target_slot)
        if self.active.get(target_key) != target_member_id:
            raise _StateTransitionError("delayed move ended for a non-active target slot")
        try:
            delayed = self.delayed_moves[target_key]
        except KeyError as exc:
            raise _StateTransitionError(
                f"{move_id} ended without a pending target-slot condition"
            ) from exc
        if delayed.move_id != move_id:
            raise _StateTransitionError(
                f"delayed move end {move_id!r} does not match {delayed.move_id!r}"
            )
        if not delayed.announced:
            raise _StateTransitionError("delayed move ended before its start message")
        if self.turn > 0 and self.turn != delayed.scheduled_turn:
            timing = "before" if self.turn < delayed.scheduled_turn else "after"
            raise _StateTransitionError(f"delayed move ended {timing} its scheduled turn")
        del self.delayed_moves[target_key]

    def _handle_hint(self, resolved: ResolvedProtocolEvent) -> None:
        hint = resolved.event.arguments[0]
        marker = " did not hit because the target is "
        move_name, separator, outcome = hint.partition(marker)
        if not separator:
            return

        move_id = normalize_showdown_id(move_name)
        if move_id not in DELAYED_MOVES:
            return

        outcome = outcome.casefold()
        if outcome not in {"the user.", "fainted."}:
            return

        for key, delayed in tuple(self.delayed_moves.items()):
            # The move misses when its user now holds the slot, or when the slot is empty.
            expected = delayed.source_member_id if outcome == "the user." else None
            if (
                delayed.move_id == move_id
                and delayed.scheduled_turn == self.turn
                and self.active.get(key) == expected
            ):
                del self.delayed_moves[key]

    def _handle_cant(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        effect = _required_effect(resolved).normalized
        if effect == "recharge":
            member.effects.pop("mustrecharge", None)
        else:
            self._discard_failed_slot_condition(member)
        if member.status == "slp":
            member.status_counter += 1
        member.protect_counter = 0

    def _handle_minus_heal(self, resolved: ResolvedProtocolEvent) -> None:
        cause = resolved.event.cause
        move_id = cause.normalized if cause is not None and cause.namespace == "move" else None
        condition_key = None
        if move_id is not None and move_id in SLOT_CONDITION_MOVES:
            target_reference = _resolved_reference(resolved, 0)
            target_member_id = _required_member(target_reference)
            target_slot = _required_slot(target_reference)
            condition_key = (target_member_id.side, target_slot, move_id)
            if condition_key not in self.slot_conditions:
                raise _StateTransitionError(
                    f"{move_id} healed a slot without a pending slot condition"
                )
        self._set_hp(resolved, clear_status=move_id == "healingwish")
        if condition_key is not None:
            del self.slot_conditions[condition_key]

    def _handle_minus_sethp(self, resolved: ResolvedProtocolEvent) -> None:
        arguments = resolved.event.arguments
        member = self._member_for(resolved, 0)
        hp_fraction, status = _parse_hp_status(arguments[1])
        member.hp_fraction = hp_fraction
        member.fainted = hp_fraction == 0.0
        if status is not None:
            member.status = status

    def _set_hp(self, resolved: ResolvedProtocolEvent, *, clear_status: bool = False) -> None:
        member = self._member_for(resolved, 0)
        hp_fraction, status = _parse_hp_status(resolved.event.arguments[1])
        if clear_status and status is not None:
            raise _StateTransitionError("Healing Wish heal must not retain a status")
        self.hp_before[resolved.event.line_index] = member.hp_fraction
        member.hp_fraction = hp_fraction
        member.fainted = hp_fraction == 0.0
        if clear_status:
            member.status = None
            member.status_counter = 0
        elif hp_fraction > 0.0 and status is not None:
            member.status = status

    def _handle_minus_status(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        status = normalize_showdown_id(resolved.event.arguments[1])
        if status not in _STATUS_NAMES:
            raise _StateTransitionError(f"unsupported status {status!r}")
        member.status = status
        member.status_counter = 0

    def _handle_minus_curestatus(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        expected = normalize_showdown_id(resolved.event.arguments[1])
        if member.status is not None and member.status != expected:
            raise _StateTransitionError("curestatus does not match the current status")
        member.status = None
        member.status_counter = 0

    def _change_boost(self, resolved: ResolvedProtocolEvent, *, direction: int) -> None:
        member = self._member_for(resolved, 0)
        stat = normalize_showdown_id(resolved.event.arguments[1])
        if stat not in _BOOST_NAMES:
            raise _StateTransitionError(f"unsupported boost stat {stat!r}")
        amount = int(resolved.event.arguments[2])
        value = amount if direction == 0 else member.boosts[stat] + direction * amount
        member.boosts[stat] = max(-6, min(6, value))

    def _handle_minus_clearboost(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0).boosts = dict.fromkeys(_BOOST_NAMES, 0)

    def _handle_minus_clearallboost(self, resolved: ResolvedProtocolEvent) -> None:
        for member_id in self.active.values():
            self._member(member_id).boosts = dict.fromkeys(_BOOST_NAMES, 0)

    def _handle_minus_clearnegativeboost(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.boosts = {name: max(0, value) for name, value in member.boosts.items()}

    def _handle_minus_invertboost(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.boosts = {name: -value for name, value in member.boosts.items()}

    def _handle_minus_copyboost(self, resolved: ResolvedProtocolEvent) -> None:
        target = self._member_for(resolved, 0)
        source = self._member_for(resolved, 1)
        target.boosts = dict(source.boosts)
        for effect in ("focusenergy", "dragoncheer"):
            target.effects.pop(effect, None)
            target.single_turn_effects.discard(effect)
            if effect in source.effects:
                target.effects[effect] = source.effects[effect]

    def _handle_minus_swapboost(self, resolved: ResolvedProtocolEvent) -> None:
        source = self._member_for(resolved, 0)
        target = self._member_for(resolved, 1)
        stats = tuple(
            normalize_showdown_id(value) for value in resolved.event.arguments[2].split(",")
        )
        if any(stat not in _BOOST_NAMES for stat in stats):
            raise _StateTransitionError(f"invalid boost list {resolved.event.arguments[2]!r}")
        for stat in stats:
            source.boosts[stat], target.boosts[stat] = target.boosts[stat], source.boosts[stat]

    def _handle_minus_item(self, resolved: ResolvedProtocolEvent) -> None:
        recipient = self._member_for(resolved, 0)
        received = resolved.event.arguments[1]
        used = self._unheld_item_ends.pop(recipient.member_id, None)
        recipient.item = None if used == normalize_showdown_id(received) else received
        cause = resolved.event.cause
        if cause is None or cause.normalized not in {
            "covet",
            "thief",
            "magician",
            "pickpocket",
        }:
            return
        donors = tuple(
            reference
            for reference in resolved.pokemon_refs
            if reference.argument_index != 0 and reference.member_id is not None
        )
        if len(donors) == 1:
            self._member(_required_member(donors[0])).item = None

    def _handle_minus_enditem(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        if "[weaken]" in resolved.event.arguments:
            # A resist berry is announced twice: once eaten, once for its effect. The
            # holder may already hold a new item from Symbiosis when the second line arrives.
            return
        announced = normalize_showdown_id(resolved.event.arguments[1])
        if member.item is not None and normalize_showdown_id(member.item) != announced:
            if "[eat]" not in resolved.event.arguments and resolved.event.cause is not None:
                raise _StateTransitionError("enditem does not match the member's current item")
            # Two legal cases keep the held item. Cud Chew eats an earlier berry again.
            # An item handed over by a move can be used at once, and the simulator
            # announces its end before the -item line that hands it over.
            self._unheld_item_ends[member.member_id] = announced
            return
        member.item = None

    def _handle_minus_ability(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        ability = resolved.event.arguments[1]
        current_ids = {
            normalize_showdown_id(value)
            for value in (member.ability.base, member.ability.forme, member.ability.temporary)
            if value is not None
        }
        temporary = member.ability.temporary
        if resolved.event.cause is not None or normalize_showdown_id(ability) not in current_ids:
            temporary = ability
        if temporary is not None:
            self._set_temporary_ability(member, temporary)

    def _handle_minus_endability(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.effects.setdefault("gastroacid", self.turn)

    def _change_form(self, resolved: ResolvedProtocolEvent, *, persistent: bool) -> None:
        member = self._member_for(resolved, 0)
        form = _details_species(resolved.event.arguments[1])
        self._set_form(member, form, persistent=persistent)

    def _set_form(self, member: _MutablePokemon, form: str, *, persistent: bool = False) -> None:
        data = self._species_data(form)
        member.current_form = form
        if persistent:
            member.persistent_form = form
        member.underlying_types = data.types
        member.added_type = None
        member.current_types = data.types
        member.base_stats = data.base_stats
        member.weight = data.weight
        forme_ability = (
            None
            if normalize_showdown_id(form) == normalize_showdown_id(member.original_species)
            else data.native_ability
        )
        member.ability = AbilityState(
            member.ability.base,
            forme_ability,
            member.ability.temporary,
        )

    def _handle_minus_mega(self, resolved: ResolvedProtocolEvent) -> None:
        self.used_mega[self._member_for(resolved, 0).member_id.side] = True

    def _handle_minus_transform(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        target = self._member_for(resolved, 1)
        if target is member:
            raise _StateTransitionError("Transform cannot target the transforming member")
        if target.transform is not None:
            raise _StateTransitionError("Transform cannot target a transformed member")
        target_types, target_added_type = target.current_types, target.added_type
        if "roost" in target.effects:
            # Transform copies the types Roost temporarily removed too.
            target_types = target.underlying_types
            if target_added_type is not None and target_added_type not in target_types:
                target_types = (*target_types, target_added_type)
        base_stats = dict(target.base_stats)
        member.transform = TransformSnapshot(
            species=target.current_form,
            weight=target.weight,
            non_hp_base_stats=tuple((name, base_stats[name]) for name in _BOOST_NAMES[:5]),
            ability=self._held_ability(target),
            moves=tuple(
                move._replace(current_pp=min(5, move.max_pp), max_pp=min(5, move.max_pp))
                for move in target.move_snapshots()
            ),
        )
        member.boosts = dict(target.boosts)
        # Transform copies the target's types as the member's own, so later type
        # additions and Roost's end build on them.
        member.underlying_types = tuple(
            type_name for type_name in target_types if type_name != target_added_type
        )
        member.added_type = target_added_type
        member.current_types = target_types

    @staticmethod
    def _apply_roost_types(member: _MutablePokemon) -> None:
        types = tuple(t for t in member.underlying_types if t != "Flying") or ("Normal",)
        member.current_types = (
            tuple(dict.fromkeys((*types, member.added_type))) if member.added_type else types
        )

    def _handle_minus_start(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved)
        dynamic_id = _DYNAMIC_EFFECT_IDS.get(effect.normalized)
        canonical_id = dynamic_id or effect.normalized
        if canonical_id in DELAYED_MOVES:
            self._handle_delayed_start(resolved, canonical_id)
            return

        member = self._member_for(resolved, 0)
        if dynamic_id is None and effect.normalized in _DYNAMIC_EFFECTS:
            raise _StateTransitionError(f"unsupported effect {effect.normalized!r}")
        if (
            canonical_id == "typechange"
            and resolved.event.cause is not None
            and resolved.event.cause.normalized == "reflecttype"
        ):
            references = tuple(
                reference for reference in resolved.pokemon_refs if reference.argument_index != 0
            )
            if len(references) != 1:
                raise _StateTransitionError("Reflect Type requires one annotated target")
            target = self._member(_required_member(references[0]))
            if "illusion" in member.effects or "illusion" in target.effects:
                # Each player sees the types copied from what it believes the disguise is.
                raise _UnsupportedMechanic("Reflect Type involves a disguised Pokemon")
            member.underlying_types = tuple(
                type_name for type_name in target.current_types if type_name != target.added_type
            ) or ("Normal",)
            member.added_type = target.added_type
            member.current_types = target.current_types
            member.effects.setdefault(canonical_id, self.turn)
            return
        if dynamic_id == "perishsong":
            member.effects.setdefault("perishsong", self.turn)
            return
        if canonical_id == "typeadd":
            if len(resolved.event.arguments) < 3:
                raise _StateTransitionError("typeadd start requires a type")
            member.added_type = resolved.event.arguments[2]
            _restore_underlying_types(member)
            member.effects.setdefault(canonical_id, self.turn)
            return
        self._validate_effect(canonical_id, "effect")
        if dynamic_id is not None:
            member.effects.setdefault(canonical_id, self.turn)
            member.effect_variants[canonical_id] = effect.normalized
            return
        if canonical_id == "typechange":
            if len(resolved.event.arguments) < 3:
                raise _StateTransitionError("typechange start requires resulting types")
            member.current_types = _parse_types(resolved.event.arguments[2])
            if resolved.event.cause is None or resolved.event.cause.normalized != "mimicry":
                member.underlying_types = member.current_types
            if "roost" in member.effects:
                self._apply_roost_types(member)
        member.effects.setdefault(canonical_id, self.turn)

    def _handle_minus_end(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved)
        if len(resolved.event.arguments) > 2:
            secondary_effect = normalize_showdown_id(resolved.event.arguments[2])
            if secondary_effect == "partiallytrapped":
                member = self._member_for(resolved, 0)
                trapping_effect = normalize_showdown_id(resolved.event.arguments[1])
                member.effects.pop(trapping_effect, None)
                return
        dynamic_id = _DYNAMIC_EFFECT_IDS.get(effect.normalized)
        canonical_id = dynamic_id or effect.normalized
        if canonical_id in DELAYED_MOVES:
            self._handle_delayed_end(resolved, canonical_id)
            return
        member = self._member_for(resolved, 0)
        self._validate_effect(canonical_id, "effect")
        active_variant = member.effect_variants.get(canonical_id)
        if (
            dynamic_id is not None
            and active_variant is not None
            and active_variant != effect.normalized
        ):
            raise _StateTransitionError(
                f"{canonical_id} ended with variant {effect.normalized!r}, "
                f"but active variant is {active_variant!r}"
            )
        member.effects.pop(canonical_id, None)
        member.effect_variants.pop(canonical_id, None)
        if canonical_id == "typechange":
            member.underlying_types = self._species_data(member.current_form).types
            _restore_underlying_types(member)

    def _handle_minus_singleturn(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved).normalized
        member = self._member_for(resolved, 0)
        if effect in _SIDE_GUARDS:
            side = member.member_id.side
            if effect in self.side_single_turn_effects[side]:
                raise _StateTransitionError(f"side guard {effect!r} started twice")
            self.side_conditions[side][effect] = self.turn
            self.side_single_turn_effects[side].add(effect)
            member.protect_counter += 1
            return

        self._validate_effect(effect, "effect")
        member.effects[effect] = self.turn
        member.single_turn_effects.add(effect)
        if effect == "roost":
            self._apply_roost_types(member)
        if effect in _PROTECT_COUNTER_MOVES:
            member.protect_counter += 1

    def _handle_minus_singlemove(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "effect")
        member.effects[effect] = self.turn
        member.single_move_effects.add(effect)

    def _handle_minus_activate(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved)
        breaking = (
            "[broken]" in resolved.event.arguments or effect.normalized in BREAKING_PROTECT_MOVES
        )
        target = self._member_for(resolved, 0)
        if breaking:
            self._break_protect(target.member_id.side, target)
            return
        move_data = self._moves.get(effect.normalized)
        if (
            effect.namespace == "move"
            and move_data is not None
            and normalize_showdown_id(str(move_data.get("volatileStatus", "")))
            == "partiallytrapped"
        ):
            source = next(
                (
                    reference.member_id
                    for reference in resolved.pokemon_refs
                    if reference.argument_index != 0 and reference.member_id is not None
                ),
                None,
            )
            if source is None:
                raise _StateTransitionError("partial trapping activation requires its source")
            target.effects[effect.normalized] = self.turn
            return
        if effect.normalized in {"symbiosis", "mummy"}:
            references = tuple(
                reference for reference in resolved.pokemon_refs if reference.member_id is not None
            )
            if effect.normalized == "symbiosis" and len(references) >= 2:
                donor = self._member(_required_member(references[0]))
                recipient = self._member(_required_member(references[1]))
                recipient.item = resolved.event.arguments[2]
                donor.item = None
            elif effect.normalized == "mummy" and len(references) >= 2:
                self._set_temporary_ability(self._member(_required_member(references[1])), "Mummy")
            return
        if effect.normalized in {"spite", "eeriespell"}:
            move_id = normalize_showdown_id(resolved.event.arguments[2])
            deduction = int(resolved.event.arguments[3])
            if not any(move.move_id == move_id for move in target.move_snapshots()):
                raise _StateTransitionError("PP draining effect names an unavailable move")
            self._change_move_pp(target, move_id, -deduction)
            return
        if effect.normalized == "leppaberry":
            move_id = normalize_showdown_id(resolved.event.arguments[2])
            if not any(move.move_id == move_id for move in target.move_snapshots()):
                raise _StateTransitionError("Leppa Berry names an unavailable move")
            restoration = (
                20 if normalize_showdown_id(self._current_ability(target)) == "ripen" else 10
            )
            self._change_move_pp(target, move_id, restoration)
            return
        if effect.normalized != "skillswap":
            return
        references = tuple(
            reference for reference in resolved.pokemon_refs if reference.member_id is not None
        )
        if len(references) != 2:
            raise _StateTransitionError("Skill Swap requires source and target references")
        source = self._member(_required_member(references[0]))
        target = self._member(_required_member(references[1]))
        source_ability = self._held_ability(source)
        target_ability = self._held_ability(target)
        arguments = resolved.event.arguments
        if len(arguments) >= 4 and arguments[2] and arguments[3]:
            target_ability = arguments[2]
            source_ability = arguments[3]
        self._set_temporary_ability(source, target_ability)
        self._set_temporary_ability(target, source_ability)

    def _handle_minus_fail(self, resolved: ResolvedProtocolEvent) -> None:
        member = self._member_for(resolved, 0)
        member.protect_counter = 0
        if len(resolved.event.arguments) > 1:
            # With an effect argument the line names the target on which something
            # failed, not a user whose own move failed.
            return
        self._discard_failed_slot_condition(member)
        for key, delayed in tuple(self.delayed_moves.items()):
            if delayed.source_member_id == member.member_id and not delayed.announced:
                del self.delayed_moves[key]

    def _break_protect(self, side: ReplaySide, target: _MutablePokemon) -> None:
        active_guards = self.side_single_turn_effects[side]
        for effect in _BREAKABLE_MEMBER_PROTECT_EFFECTS:
            target.effects.pop(effect, None)
            target.single_turn_effects.discard(effect)
        for effect in active_guards:
            self.side_conditions[side].pop(effect, None)
        active_guards.clear()
        target.protect_counter = 0

    def _handle_minus_prepare(self, resolved: ResolvedProtocolEvent) -> None:
        move_id = normalize_showdown_id(resolved.event.arguments[1])
        if move_id not in self._moves:
            raise _StateTransitionError(f"prepare references unknown move {move_id!r}")
        self._member_for(resolved, 0).preparing = move_id

    def _handle_minus_mustrecharge(self, resolved: ResolvedProtocolEvent) -> None:
        self._member_for(resolved, 0).effects["mustrecharge"] = self.turn

    def _handle_minus_weather(self, resolved: ResolvedProtocolEvent) -> None:
        weather = _required_effect(resolved).normalized
        if weather == "none":
            self.weather.clear()
            return
        self._validate_effect(weather, "weather")
        if "[upkeep]" in resolved.event.arguments[1:]:
            if weather not in self.weather:
                raise _StateTransitionError(f"weather upkeep has no active {weather!r} weather")
            return
        self.weather = {weather: self.turn}

    def _handle_minus_fieldstart(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "field")
        if effect in TERRAINS:
            for terrain in TERRAINS:
                self.fields.pop(terrain, None)
        self.fields[effect] = self.turn

    def _handle_minus_fieldend(self, resolved: ResolvedProtocolEvent) -> None:
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "field")
        if effect not in self.fields:
            raise _StateTransitionError(f"field effect {effect!r} ended before it started")
        self.fields.pop(effect)

    def _handle_minus_sidestart(self, resolved: ResolvedProtocolEvent) -> None:
        side = _resolved_reference(resolved, 0).pokemon_ref.side
        effect = _required_effect(resolved).normalized
        self._validate_effect(effect, "side_condition")
        conditions = self.side_conditions[side]
        if effect in _STACKING_SIDE_CONDITIONS:
            conditions[effect] = conditions.get(effect, 0) + 1
        else:
            conditions.setdefault(effect, self.turn)

    def _handle_minus_sideend(self, resolved: ResolvedProtocolEvent) -> None:
        side = _resolved_reference(resolved, 0).pokemon_ref.side
        effect = _required_effect(resolved).normalized
        if effect not in _SIDE_GUARDS:
            self._validate_effect(effect, "side_condition")
        if effect not in self.side_conditions[side]:
            raise _StateTransitionError(f"side condition {effect!r} ended before it started")
        self.side_conditions[side].pop(effect)
        self.side_single_turn_effects[side].discard(effect)

    def _handle_minus_swapsideconditions(self, resolved: ResolvedProtocolEvent) -> None:
        # Court Change swaps every lasting side condition. Quick Guard and Wide Guard stay.
        kept = {
            side: {effect: conditions.pop(effect) for effect in self.side_single_turn_effects[side]}
            for side, conditions in self.side_conditions.items()
        }
        self.side_conditions = {
            ReplaySide.P1: self.side_conditions[ReplaySide.P2] | kept[ReplaySide.P1],
            ReplaySide.P2: self.side_conditions[ReplaySide.P1] | kept[ReplaySide.P2],
        }

    def _member_for(
        self, resolved: ResolvedProtocolEvent, argument_index: int = 0
    ) -> _MutablePokemon:
        return self._member(_required_member(_resolved_reference(resolved, argument_index)))

    def _validate_effect(self, effect: str, category: str) -> None:
        if category == "effect" and (
            effect in {"illusion", "typechange"} or effect in _DYNAMIC_EFFECTS
        ):
            return
        if effect not in self._legal_effects[category]:
            raise _StateTransitionError(f"unsupported {category.replace('_', ' ')} {effect!r}")

    def _current_ability(self, member: _MutablePokemon) -> str:
        if "gastroacid" in member.effects:
            return ""
        ability = self._held_ability(member)
        if normalize_showdown_id(ability) in {"pressure", "ripen"} and any(
            normalize_showdown_id(self._held_ability(source := self._members[member_id]))
            == "neutralizinggas"
            and not source.fainted
            and source.transform is None
            and "gastroacid" not in source.effects
            for member_id in self.active.values()
        ):
            raise _UnsupportedMechanic(
                "Neutralizing Gas suppression of PP-related abilities is not reconstructed"
            )
        return ability

    @staticmethod
    def _held_ability(member: _MutablePokemon) -> str:
        """Return the named ability, which Skill Swap and Transform copy even while suppressed."""
        return member.transform.ability if member.transform is not None else member.ability.current

    def _is_copyable_volatile(self, effect: str) -> bool:
        if effect in COPYABLE_VOLATILES:
            return True
        move = self._moves.get(effect)
        return bool(
            move is not None
            and normalize_showdown_id(str(move.get("volatileStatus", ""))) in COPYABLE_VOLATILES
        )

    @staticmethod
    def _set_temporary_ability(member: _MutablePokemon, ability: str) -> None:
        if member.transform is not None:
            member.transform = member.transform._replace(ability=ability)
        else:
            member.ability = AbilityState(member.ability.base, member.ability.forme, ability)

    def _clear_battle_state(self, member: _MutablePokemon) -> None:
        self._set_form(member, member.persistent_form, persistent=False)
        member.ability = AbilityState(member.ability.base, member.ability.forme)
        member.boosts = dict.fromkeys(_BOOST_NAMES, 0)
        member.effects.clear()
        member.effect_variants.clear()
        member.single_turn_effects.clear()
        member.single_move_effects.clear()
        member.active_turns = 0
        member.protect_counter = 0
        member.preparing = None
        member.last_move = None
        member.transform = None
        if member.status != "slp":
            member.status_counter = 0

    @staticmethod
    def _change_move_pp(member: _MutablePokemon, move_id: str, amount: int) -> None:
        if member.transform is not None:
            member.transform = member.transform._replace(
                moves=tuple(
                    move._replace(
                        current_pp=max(0, min(move.max_pp, move.current_pp + amount)),
                    )
                    if move.move_id == move_id
                    else move
                    for move in member.transform.moves
                ),
            )
            return
        move = member.moves.get(move_id)
        if move is not None:
            member.moves[move_id] = move._replace(
                current_pp=max(0, min(move.max_pp, move.current_pp + amount))
            )

    def _transformed_into(self, side: ReplaySide, base_species: str) -> bool:
        """Whether an active member of the side is transformed into the given base species."""
        return any(
            member_id.side == side
            and (transform := self._members[member_id].transform) is not None
            and self._species_data(transform.species).base_species == base_species
            for member_id in self.active.values()
        )

    def _copied_member(
        self, member_id: ReplayMemberId, displayed_species: str
    ) -> ReplayMemberId | None:
        """Return the teammate a disguise copies; the species clause makes it unique."""
        displayed = self._species_data(displayed_species).base_species
        return next(
            (
                other_id
                for other_id, other in self._members.items()
                if other_id.side == member_id.side
                and self._species_data(other.original_species).base_species == displayed
            ),
            None,
        )

    def _member(self, member_id: ReplayMemberId) -> _MutablePokemon:
        """Return a member for reading or change. This is the only way to reach one."""
        self._touched.add(member_id)
        return self._members[member_id]

    def snapshot(self, line_index: int) -> ReplayBattleState:
        for member_id in self._touched:
            self._member_snapshots[member_id] = self._members[member_id].snapshot()
        self._touched.clear()
        members = tuple(self._member_snapshots[member_id] for member_id in self._members)
        sides = tuple(
            ReplaySideState(
                side=side,
                active=(self.active.get((side, 0)), self.active.get((side, 1))),
                conditions=tuple(sorted(self.side_conditions[side].items())),
                used_mega=self.used_mega[side],
            )
            for side in (ReplaySide.P1, ReplaySide.P2)
        )
        delayed_moves = tuple(
            self.delayed_moves[key]
            for key in sorted(self.delayed_moves, key=lambda item: (item[0].side_index, item[1]))
        )
        slot_conditions = tuple(
            self.slot_conditions[key]
            for key in sorted(
                self.slot_conditions,
                key=lambda item: (item[0].side_index, item[1], item[2]),
            )
        )
        return ReplayBattleState(
            replay_id=self.replay_id,
            line_index=line_index,
            turn=self.turn,
            members=members,
            sides=(sides[0], sides[1]),
            weather=tuple(sorted(self.weather.items())),
            fields=tuple(sorted(self.fields.items())),
            delayed_moves=delayed_moves,
            slot_conditions=slot_conditions,
            team_sizes=(self.team_sizes[ReplaySide.P1], self.team_sizes[ReplaySide.P2]),
        )


# One row per tag that changes reducer state. Tags without a row leave the state unchanged.
_HANDLERS: dict[str, Callable[[_StateReducer, ResolvedProtocolEvent], None]] = {
    "turn": _StateReducer._handle_turn,
    "upkeep": _StateReducer._handle_upkeep,
    "teamsize": _StateReducer._handle_teamsize,
    "-hint": _StateReducer._handle_hint,
    "switch": _StateReducer._switch,
    "drag": _StateReducer._switch,
    "replace": _StateReducer._handle_replace,
    "swap": _StateReducer._handle_swap,
    "faint": _StateReducer._handle_faint,
    "move": _StateReducer._handle_move,
    "cant": _StateReducer._handle_cant,
    "-damage": _StateReducer._set_hp,
    "-heal": _StateReducer._handle_minus_heal,
    "-sethp": _StateReducer._handle_minus_sethp,
    "-status": _StateReducer._handle_minus_status,
    "-curestatus": _StateReducer._handle_minus_curestatus,
    "-boost": partial(_StateReducer._change_boost, direction=1),
    "-unboost": partial(_StateReducer._change_boost, direction=-1),
    "-setboost": partial(_StateReducer._change_boost, direction=0),
    "-clearboost": _StateReducer._handle_minus_clearboost,
    "-clearallboost": _StateReducer._handle_minus_clearallboost,
    "-clearnegativeboost": _StateReducer._handle_minus_clearnegativeboost,
    "-invertboost": _StateReducer._handle_minus_invertboost,
    "-copyboost": _StateReducer._handle_minus_copyboost,
    "-swapboost": _StateReducer._handle_minus_swapboost,
    "-item": _StateReducer._handle_minus_item,
    "-enditem": _StateReducer._handle_minus_enditem,
    "-ability": _StateReducer._handle_minus_ability,
    "-endability": _StateReducer._handle_minus_endability,
    "detailschange": partial(_StateReducer._change_form, persistent=True),
    "-formechange": partial(_StateReducer._change_form, persistent=False),
    "-mega": _StateReducer._handle_minus_mega,
    "-transform": _StateReducer._handle_minus_transform,
    "-start": _StateReducer._handle_minus_start,
    "-end": _StateReducer._handle_minus_end,
    "-singleturn": _StateReducer._handle_minus_singleturn,
    "-singlemove": _StateReducer._handle_minus_singlemove,
    "-activate": _StateReducer._handle_minus_activate,
    "-prepare": _StateReducer._handle_minus_prepare,
    "-fail": _StateReducer._handle_minus_fail,
    "-mustrecharge": _StateReducer._handle_minus_mustrecharge,
    "-weather": _StateReducer._handle_minus_weather,
    "-fieldstart": _StateReducer._handle_minus_fieldstart,
    "-fieldend": _StateReducer._handle_minus_fieldend,
    "-sidestart": _StateReducer._handle_minus_sidestart,
    "-sideend": _StateReducer._handle_minus_sideend,
    "-swapsideconditions": _StateReducer._handle_minus_swapsideconditions,
}
_STATE_TAGS = {
    tag
    for tag, rule in CLASSIFICATION_REGISTRY.items()
    if rule.classification is EventClassification.PUBLIC_STATE
}
if _HANDLERS.keys() != _STATE_TAGS:
    raise RuntimeError(
        f"Reducer handlers and state-changing protocol tags differ: "
        f"{sorted(_HANDLERS.keys() ^ _STATE_TAGS)}"
    )


def _restore_underlying_types(member: _MutablePokemon) -> None:
    member.current_types = member.underlying_types
    if member.added_type is not None:
        member.current_types = tuple(dict.fromkeys((*member.current_types, member.added_type)))


@cache
def dex_indexes() -> tuple[
    dict[str, _SpeciesData], dict[str, Mapping[str, Any]], dict[str, frozenset[str]]
]:
    """Index the generated dex by species id, by move id, and by legal effect category."""
    dex = default_runtime_resources().dex
    species = {
        entry["id"]: _SpeciesData(
            entry["name"],
            entry["baseSpecies"],
            tuple(entry["types"]),
            tuple((stat, entry["baseStats"][stat]) for stat in _STAT_NAMES),
            float(entry["weightkg"]),
            entry["abilities"]["0"],
        )
        for entry in dex["species"]
    }
    moves = {entry["id"]: entry for entry in dex["moves"]}
    legal_effects = {
        category: frozenset(effects) for category, effects in dex["legalProtocolEffects"].items()
    }
    return species, moves, legal_effects


def move_state(name: str) -> MoveState:
    """Return the full-PP move slot of a move name."""
    move_id = normalize_showdown_id(name)
    try:
        data = dex_indexes()[1][move_id]
    except KeyError as exc:
        raise _StateTransitionError(f"move {name!r} is absent from the pinned dex") from exc
    # The Champions mod uses the integer base-PP quotient before multiplying.
    max_pp = data["pp"] if data.get("noPPBoosts") else (data["pp"] // 5 + 1) * 4
    return MoveState(
        move_id, data["name"], data["type"], data["category"], data["target"], max_pp, max_pp
    )


def _resolved_reference(
    resolved: ResolvedProtocolEvent,
    argument_index: int,
) -> ResolvedPokemonRefArgument:
    try:
        return next(
            reference
            for reference in resolved.pokemon_refs
            if reference.argument_index == argument_index
        )
    except StopIteration as exc:
        raise _StateTransitionError(
            f"event {resolved.event.tag!r} has no resolved reference at argument {argument_index}"
        ) from exc


def _required_member(reference: ResolvedPokemonRefArgument) -> ReplayMemberId:
    if reference.member_id is None:
        raise _StateTransitionError("state transition requires an active member reference")
    return reference.member_id


def _required_slot(reference: ResolvedPokemonRefArgument) -> int:
    slot = reference.pokemon_ref.active_slot
    if slot is None:
        raise _StateTransitionError("state transition requires an active slot")
    return slot


def _required_effect(resolved: ResolvedProtocolEvent) -> EffectReference:
    if resolved.event.effect is None:
        raise _StateTransitionError(f"event {resolved.event.tag!r} requires a normalized effect")
    return resolved.event.effect


def _details_species(details: str) -> str:
    species = details.split(",", 1)[0].strip()
    if not species:
        raise _StateTransitionError("Pokémon details require a species")
    return species


def _parse_hp_status(value: str) -> tuple[float, str | None]:
    parts = value.split()
    if not parts:
        raise _StateTransitionError("HP status must not be empty")
    hp = parts[0]
    if "/" in hp:
        numerator_text, denominator_text = hp.split("/", 1)
        denominator_text = denominator_text.rstrip(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%"
        )
        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError as exc:
            raise _StateTransitionError(f"invalid HP status {value!r}") from exc
        if denominator <= 0 or not 0 <= numerator <= denominator:
            raise _StateTransitionError(f"invalid HP range {value!r}")
        fraction = numerator / denominator
    elif hp == "0":
        fraction = 0.0
    else:
        raise _StateTransitionError(f"invalid HP status {value!r}")
    status = normalize_showdown_id(parts[1]) if len(parts) > 1 else None
    if status == "fnt":
        status = None
    elif status is not None and status not in _STATUS_NAMES:
        raise _StateTransitionError(f"invalid HP status suffix {status!r}")
    return fraction, status


def _parse_types(value: str) -> tuple[str, ...]:
    types = tuple(item.strip() for item in value.split("/") if item.strip())
    if not 1 <= len(types) <= 2:
        raise _StateTransitionError(f"invalid type list {value!r}")
    return types


def reduce_replay_state(
    replay_id: str,
    ots: tuple[OTSData, OTSData],
    events: Iterable[ResolvedProtocolEvent],
    *,
    snapshot_line_indices: Iterable[int] | None = None,
) -> ReconstructedReplayState:
    """Apply all events and snapshot every line or the requested lines."""
    event_tuple = tuple(events)
    requested_lines = None if snapshot_line_indices is None else frozenset(snapshot_line_indices)
    try:
        reducer = _StateReducer(replay_id, ots)
    except _StateTransitionError as exc:
        return ReconstructedReplayState(
            replay_id, (), (event_tuple[0].event.rejection(str(exc), exc.category),)
        )

    # Only Illusion needs a different public identity history. Imported here because
    # public_tracking imports this module.
    from p0.replays.reconstruction.public_tracking import PublicReplayTracker

    illusion_sides = tuple(
        any(normalize_showdown_id(member.ability) == "illusion" for member in sheet.members)
        for sheet in ots
    )
    any_illusion = any(illusion_sides)
    initial = reducer.snapshot(-1).members if any_illusion else ()
    public = tuple(
        PublicReplayTracker(replay_id, ots, initial, perspective=1 - side) if has_illusion else None
        for side, has_illusion in enumerate(illusion_sides)
    )
    snapshots: list[ReplayBattleState] = []
    for resolved in event_tuple:
        try:
            reducer.apply(resolved)
            if resolved.event.tag in _HANDLERS or resolved.event.pokemon_refs:
                for tracker in public:
                    if tracker is not None:
                        tracker.apply(resolved)
            if requested_lines is None or resolved.event.line_index in requested_lines:
                snapshot = reducer.snapshot(resolved.event.line_index)
                if any_illusion:
                    snapshot = replace(
                        snapshot,
                        public_sides=(
                            None if public[0] is None else public[0].snapshot(),
                            None if public[1] is None else public[1].snapshot(),
                        ),
                    )
                snapshots.append(snapshot)
        except _StateTransitionError as exc:
            return ReconstructedReplayState(
                replay_id, (), (resolved.event.rejection(str(exc), exc.category),)
            )
    return ReconstructedReplayState(replay_id, tuple(snapshots), hp_before=reducer.hp_before)


__all__ = [
    "AbilityState",
    "DelayedMoveState",
    "MoveState",
    "ReconstructedReplayState",
    "ReplayBattleState",
    "ReplayPokemonState",
    "ReplaySideState",
    "SlotConditionState",
    "TransformSnapshot",
    "dex_indexes",
    "move_state",
    "reduce_replay_state",
]
