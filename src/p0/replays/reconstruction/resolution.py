"""Replay-wide identity inference and sequential protocol-reference resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple

from p0.replays.identity import ReplayMemberId, ReplaySide, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.diagnostics import (
    ReplayEventDiagnostic,
    ReplayEventParseError,
)
from p0.replays.reconstruction.events import (
    PokemonRefArgument,
    ProtocolEvent,
    parse_replay_events,
)
from p0.replays.reconstruction.identity import ProtocolPokemonReference
from p0.replays.schema import OTSData, OTSMember


@dataclass(frozen=True, slots=True)
class ResolvedPokemonRefArgument:
    """One protocol reference resolved to either a side or a roster member."""

    argument_index: int
    pokemon_ref: ProtocolPokemonReference
    member_id: ReplayMemberId | None

    def __post_init__(self) -> None:
        if self.argument_index < 0:
            raise ValueError("Resolved reference argument index must be nonnegative")
        if self.pokemon_ref.active_slot is None:
            if self.member_id is not None:
                raise ValueError("Side references cannot resolve to roster members")
        elif self.member_id is None or self.member_id.side is not self.pokemon_ref.side:
            raise ValueError("Active-slot references require a member on the same side")


@dataclass(frozen=True, slots=True)
class ResolvedProtocolEvent:
    """A classified event whose Pokémon references have explicit identities."""

    event: ProtocolEvent
    pokemon_refs: tuple[ResolvedPokemonRefArgument, ...]

    def __post_init__(self) -> None:
        if tuple(reference.argument_index for reference in self.pokemon_refs) != tuple(
            reference.argument_index for reference in self.event.pokemon_refs
        ):
            raise ValueError("Resolved references must match the event reference order")


@dataclass(frozen=True, slots=True)
class ResolvedReplayEvents:
    """A complete resolved replay or its whole-replay rejection diagnostic."""

    replay_id: str
    events: tuple[ResolvedProtocolEvent, ...]
    diagnostics: tuple[ReplayEventDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("ResolvedReplayEvents.replay_id must not be empty")
        if self.events and self.diagnostics:
            raise ValueError("Rejected replays cannot retain resolved events")
        if any(event.event.replay_id != self.replay_id for event in self.events):
            raise ValueError("ResolvedReplayEvents cannot contain another replay")

    def require_accepted(self) -> tuple[ResolvedProtocolEvent, ...]:
        """Return resolved events or raise the replay rejection."""
        if self.diagnostics:
            raise ReplayEventParseError(self.diagnostics)
        return self.events


class _HistoryCandidate(NamedTuple):
    actual_member: ReplayMemberId
    displayed_member: ReplayMemberId


class _ActiveHistory(NamedTuple):
    history_id: int
    side: ReplaySide
    entry_event: ProtocolEvent
    candidates: tuple[_HistoryCandidate, ...]
    overlapping_history_ids: frozenset[int]
    predecessor_history_id: int | None
    revealed_member: ReplayMemberId | None
    reveal_event: ProtocolEvent | None
    faint_line_index: int | None


@dataclass(slots=True)
class _HistoryBuilder:
    history_id: int
    side: ReplaySide
    entry_event: ProtocolEvent
    candidates: tuple[_HistoryCandidate, ...]
    overlapping_history_ids: set[int] = field(default_factory=set)
    predecessor_history_id: int | None = None
    revealed_member: ReplayMemberId | None = None
    reveal_event: ProtocolEvent | None = None
    faint_line_index: int | None = None

    def freeze(self) -> _ActiveHistory:
        return _ActiveHistory(
            self.history_id,
            self.side,
            self.entry_event,
            self.candidates,
            frozenset(self.overlapping_history_ids),
            self.predecessor_history_id,
            self.revealed_member,
            self.reveal_event,
            self.faint_line_index,
        )


class _HistoryResolutionError(ValueError):
    def __init__(self, event: ProtocolEvent, reason: str) -> None:
        self.event = event
        super().__init__(reason)


class _HistoryScanner:
    """Collect active histories before assigning stable roster members."""

    def __init__(
        self,
        ots: tuple[OTSData, OTSData],
        species_bases: Mapping[str, str],
    ) -> None:
        self._members = {sheet.side: sheet.members for sheet in ots}
        self._species_bases = species_bases
        self._illusion_members = {
            sheet.side: tuple(
                member.member_id
                for member in sheet.members
                if normalize_showdown_id(member.ability) == "illusion"
            )
            for sheet in ots
        }
        self._active: dict[tuple[ReplaySide, int], int] = {}
        self._histories: list[_HistoryBuilder] = []
        self.team_sizes = {sheet.side: min(4, len(sheet.members)) for sheet in ots}

    def scan(self, events: tuple[ProtocolEvent, ...]) -> tuple[_ActiveHistory, ...]:
        for event in events:
            if event.tag == "teamsize":
                self._set_team_size(event)
            elif event.tag in {"switch", "drag"}:
                self._start_history(event)
            elif event.tag == "replace":
                self._reveal_history(event)
            elif event.tag == "faint":
                self._faint_history(event)
            elif event.tag == "swap":
                self._swap_history(event)
        return tuple(history.freeze() for history in self._histories)

    def _set_team_size(self, event: ProtocolEvent) -> None:
        side = ReplaySide(event.arguments[0])
        size = int(event.arguments[1])
        if not 1 <= size <= len(self._members[side]):
            raise _HistoryResolutionError(event, f"invalid selected team size {size}")
        self.team_sizes[side] = size

    def _start_history(self, event: ProtocolEvent) -> None:
        reference = _event_reference(event)
        slot = _required_active_slot(event, reference)
        side = reference.pokemon_ref.side
        candidates = _history_candidates(
            event,
            reference.pokemon_ref,
            self._members[side],
            self._illusion_members[side],
            self._species_bases,
        )
        if not candidates:
            raise _HistoryResolutionError(
                event,
                f"incoming reference {reference.pokemon_ref.displayed_name!r} resolved to "
                "0 available roster members",
            )

        key = (side, slot)
        predecessor_history_id = self._active.pop(key, None)
        overlaps = {
            history_id
            for (active_side, _), history_id in self._active.items()
            if active_side is side
        }
        history_id = len(self._histories)
        history = _HistoryBuilder(
            history_id,
            side,
            event,
            candidates,
            overlaps,
            predecessor_history_id,
        )
        self._histories.append(history)
        for overlapping_id in overlaps:
            self._histories[overlapping_id].overlapping_history_ids.add(history_id)
        self._active[key] = history_id

    def _reveal_history(self, event: ProtocolEvent) -> None:
        reference = _event_reference(event)
        history = self._active_history(event, reference)
        revealed = _revealed_member(
            event,
            reference.pokemon_ref,
            self._members[history.side],
            self._species_bases,
        )
        if history.revealed_member is not None and history.revealed_member != revealed:
            raise _HistoryResolutionError(event, "Illusion history has conflicting reveals")
        history.revealed_member = revealed
        history.reveal_event = event

    def _faint_history(self, event: ProtocolEvent) -> None:
        reference = _event_reference(event)
        slot = _required_active_slot(event, reference)
        history = self._active_history(event, reference)
        history.faint_line_index = event.line_index
        self._active.pop((history.side, slot))

    def _swap_history(self, event: ProtocolEvent) -> None:
        reference = _event_reference(event)
        source_slot = _required_active_slot(event, reference)
        target_slot = int(event.arguments[1])
        if not 0 <= target_slot < 2:
            raise _HistoryResolutionError(
                event,
                f"swap target slot {target_slot} is outside a doubles battle",
            )

        side = reference.pokemon_ref.side
        source_key = (side, source_slot)
        target_key = (side, target_slot)
        try:
            source_history = self._active[source_key]
        except KeyError as exc:
            raise _HistoryResolutionError(event, "swap source has no active history") from exc
        target_history = self._active.get(target_key)
        self._active[target_key] = source_history
        if target_history is None:
            self._active.pop(source_key)
        else:
            self._active[source_key] = target_history

    def _active_history(
        self,
        event: ProtocolEvent,
        reference: PokemonRefArgument,
    ) -> _HistoryBuilder:
        slot = _required_active_slot(event, reference)
        key = (reference.pokemon_ref.side, slot)
        try:
            return self._histories[self._active[key]]
        except KeyError as exc:
            raise _HistoryResolutionError(
                event,
                f"{reference.pokemon_ref.side.value}{slot} has no active member",
            ) from exc

    def has_illusion_member(self, side: ReplaySide) -> bool:
        return bool(self._illusion_members[side])


class _HistorySolver:
    """Find at most two valid assignments for one side's active histories."""

    def __init__(
        self,
        histories: tuple[_ActiveHistory, ...],
        team_size: int,
    ) -> None:
        self._histories = histories
        self._histories_by_id = {history.history_id: history for history in histories}
        self._team_size = team_size
        self._assignments: dict[int, _HistoryCandidate] = {}
        self._selected_counts: dict[ReplayMemberId, int] = {}
        self.solutions: list[tuple[_HistoryCandidate, ...]] = []

    def solve(self) -> tuple[tuple[_HistoryCandidate, ...], ...]:
        self._search(0)
        return tuple(self.solutions)

    def _search(self, position: int) -> None:
        if len(self.solutions) >= 2:
            return
        if position == len(self._histories):
            self.solutions.append(
                tuple(self._assignments[history.history_id] for history in self._histories)
            )
            return

        history = self._histories[position]
        for candidate in history.candidates:
            if not self._is_valid(history, candidate):
                continue
            selected = frozenset((candidate.actual_member, candidate.displayed_member))
            if self._selected_size_after(selected) > self._team_size:
                continue

            self._assignments[history.history_id] = candidate
            for member_id in selected:
                self._selected_counts[member_id] = self._selected_counts.get(member_id, 0) + 1
            self._search(position + 1)
            for member_id in selected:
                remaining = self._selected_counts[member_id] - 1
                if remaining:
                    self._selected_counts[member_id] = remaining
                else:
                    del self._selected_counts[member_id]
            del self._assignments[history.history_id]

    def _is_valid(self, history: _ActiveHistory, candidate: _HistoryCandidate) -> bool:
        if (
            history.revealed_member is not None
            and candidate.actual_member != history.revealed_member
        ):
            return False
        for previous_id, previous_candidate in self._assignments.items():
            previous = self._histories_by_id[previous_id]
            if (
                previous_id in history.overlapping_history_ids
                or previous_id == history.predecessor_history_id
            ) and previous_candidate.actual_member == candidate.actual_member:
                return False
            if (
                previous.faint_line_index is not None
                and previous.faint_line_index < history.entry_event.line_index
                and previous_candidate.actual_member
                in {candidate.actual_member, candidate.displayed_member}
            ):
                return False
        return True

    def _selected_size_after(self, selected: frozenset[ReplayMemberId]) -> int:
        return len(self._selected_counts) + sum(
            member_id not in self._selected_counts for member_id in selected
        )


def _event_reference(event: ProtocolEvent, argument_index: int = 0) -> PokemonRefArgument:
    try:
        return next(
            reference
            for reference in event.pokemon_refs
            if reference.argument_index == argument_index
        )
    except StopIteration as exc:
        raise _HistoryResolutionError(
            event,
            f"event {event.tag!r} has no Pokémon reference at argument {argument_index}",
        ) from exc


def _required_active_slot(event: ProtocolEvent, reference: PokemonRefArgument) -> int:
    slot = reference.pokemon_ref.active_slot
    if slot is None:
        raise _HistoryResolutionError(event, f"event {event.tag!r} requires an active slot")
    return slot


def _species_base_index(dex: Mapping[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for value in dex.get("species", ()):
        if not isinstance(value, Mapping):
            continue
        name = str(value.get("name", value.get("id", "")))
        species_id = normalize_showdown_id(str(value.get("id", name)))
        base_species = normalize_showdown_id(str(value.get("baseSpecies", name)))
        if species_id and base_species:
            index[species_id] = base_species
            index[normalize_showdown_id(name)] = base_species
    return index


def _base_species_id(species: str, species_bases: Mapping[str, str]) -> str:
    species_id = normalize_showdown_id(species)
    return species_bases.get(species_id, species_id)


def _matching_members(
    event: ProtocolEvent,
    pokemon_ref: ProtocolPokemonReference,
    members: tuple[OTSMember, ...],
    species_bases: Mapping[str, str],
) -> tuple[OTSMember, ...]:
    displayed_name = normalize_showdown_id(pokemon_ref.displayed_name)
    name_matches = tuple(
        member
        for member in members
        if displayed_name
        in {
            normalize_showdown_id(member.nickname),
            normalize_showdown_id(member.species),
        }
    )
    details_species = _base_species_id(event.arguments[1].split(",", 1)[0], species_bases)
    species_matches = tuple(
        member
        for member in members
        if _base_species_id(member.species, species_bases) == details_species
    )
    if not species_matches:
        return name_matches
    if len(species_matches) == 1:
        return species_matches
    narrowed = tuple(member for member in species_matches if member in name_matches)
    return narrowed or species_matches


def _history_candidates(
    event: ProtocolEvent,
    pokemon_ref: ProtocolPokemonReference,
    members: tuple[OTSMember, ...],
    illusion_members: tuple[ReplayMemberId, ...],
    species_bases: Mapping[str, str],
) -> tuple[_HistoryCandidate, ...]:
    displayed_members = _matching_members(event, pokemon_ref, members, species_bases)
    choices = {
        _HistoryCandidate(member.member_id, member.member_id) for member in displayed_members
    }
    choices.update(
        _HistoryCandidate(illusion_member, displayed.member_id)
        for illusion_member in illusion_members
        for displayed in displayed_members
        if illusion_member != displayed.member_id
    )
    return tuple(
        sorted(choices, key=lambda choice: (choice.actual_member, choice.displayed_member))
    )


def _revealed_member(
    event: ProtocolEvent,
    pokemon_ref: ProtocolPokemonReference,
    members: tuple[OTSMember, ...],
    species_bases: Mapping[str, str],
) -> ReplayMemberId:
    candidates = _matching_members(event, pokemon_ref, members, species_bases)
    if len(candidates) != 1:
        raise _HistoryResolutionError(
            event,
            f"Illusion reveal {pokemon_ref.displayed_name!r} resolved to "
            f"{len(candidates)} roster members",
        )
    return candidates[0].member_id


def _resolve_incoming_histories(
    ots: tuple[OTSData, OTSData],
    events: tuple[ProtocolEvent, ...],
    species_bases: Mapping[str, str],
) -> dict[int, ReplayMemberId]:
    scanner = _HistoryScanner(ots, species_bases)
    histories = scanner.scan(events)
    bindings: dict[int, ReplayMemberId] = {}
    for side in (ReplaySide.P1, ReplaySide.P2):
        side_histories = tuple(history for history in histories if history.side is side)
        if not side_histories:
            continue
        solutions = _HistorySolver(side_histories, scanner.team_sizes[side]).solve()
        if not solutions:
            impossible_reveal = next(
                (
                    history.reveal_event
                    for history in side_histories
                    if history.reveal_event is not None
                    and history.revealed_member
                    not in {candidate.actual_member for candidate in history.candidates}
                ),
                None,
            )
            event = (
                side_histories[0].entry_event if impossible_reveal is None else impossible_reveal
            )
            raise _HistoryResolutionError(
                event,
                "unresolved_illusion: active history has no valid roster assignment",
            )
        if len(solutions) > 1:
            first, second = solutions
            differing_index = next(
                index
                for index, choices in enumerate(zip(first, second, strict=True))
                if choices[0] != choices[1]
            )
            history = side_histories[differing_index]
            if not scanner.has_illusion_member(side):
                count = len(history.candidates)
                reason = (
                    f"incoming reference "
                    f"{_event_reference(history.entry_event).pokemon_ref.displayed_name!r} "
                    f"resolved to {count} available roster members"
                )
            else:
                reason = "unresolved_illusion: active history has multiple valid assignments"
            raise _HistoryResolutionError(history.entry_event, reason)

        solution = next(iter(solutions))
        for history, candidate in zip(side_histories, solution, strict=True):
            bindings[history.entry_event.line_index] = candidate.actual_member
    return bindings


class _IdentityResolver:
    def __init__(
        self,
        ots: tuple[OTSData, OTSData],
        incoming_bindings: dict[int, ReplayMemberId],
        species_bases: Mapping[str, str],
    ) -> None:
        self._members = {side_sheet.side: side_sheet.members for side_sheet in ots}
        self._incoming_bindings = incoming_bindings
        self._species_bases = species_bases
        self._active: dict[tuple[ReplaySide, int], ReplayMemberId] = {}

    def resolve(self, event: ProtocolEvent) -> ResolvedProtocolEvent:
        switch_reference = self._switch_reference(event)
        resolved = tuple(
            self._resolve_reference(event, reference, switch_reference)
            for reference in event.pokemon_refs
        )

        if switch_reference is not None:
            incoming = next(
                reference.member_id
                for reference in resolved
                if reference.argument_index == switch_reference.argument_index
            )
            if incoming is None or switch_reference.pokemon_ref.active_slot is None:
                raise ValueError("switch reference did not resolve to an active member")
            occupied_elsewhere = any(
                member_id == incoming
                and key
                != (
                    incoming.side,
                    switch_reference.pokemon_ref.active_slot,
                )
                for key, member_id in self._active.items()
            )
            if occupied_elsewhere:
                raise ValueError("incoming member is already active in another slot")
            self._active[(incoming.side, switch_reference.pokemon_ref.active_slot)] = incoming
        elif event.tag == "replace" and resolved:
            revealed = _revealed_member(
                event,
                resolved[0].pokemon_ref,
                self._members[resolved[0].pokemon_ref.side],
                self._species_bases,
            )
            if resolved[0].member_id != revealed:
                raise ValueError("Illusion reveal does not match the active history assignment")
        elif event.tag == "faint" and resolved:
            fainted = resolved[0]
            slot = fainted.pokemon_ref.active_slot
            if slot is not None:
                self._active.pop((fainted.pokemon_ref.side, slot), None)
        elif event.tag == "swap" and resolved:
            self._apply_swap(event, resolved[0])

        return ResolvedProtocolEvent(event, resolved)

    @staticmethod
    def _switch_reference(event: ProtocolEvent) -> PokemonRefArgument | None:
        if event.tag not in {"switch", "drag"} or not event.pokemon_refs:
            return None
        return event.pokemon_refs[0]

    def _resolve_reference(
        self,
        event: ProtocolEvent,
        reference: PokemonRefArgument,
        switch_reference: PokemonRefArgument | None,
    ) -> ResolvedPokemonRefArgument:
        pokemon_ref = reference.pokemon_ref
        if pokemon_ref.active_slot is None:
            return ResolvedPokemonRefArgument(reference.argument_index, pokemon_ref, None)

        if (
            switch_reference is not None
            and reference.argument_index == switch_reference.argument_index
        ):
            member_id = self._resolve_incoming(event, pokemon_ref)
        else:
            key = (pokemon_ref.side, pokemon_ref.active_slot)
            try:
                member_id = self._active[key]
            except KeyError as exc:
                raise ValueError(
                    f"{pokemon_ref.side.value}{pokemon_ref.active_slot} has no active member"
                ) from exc
        return ResolvedPokemonRefArgument(reference.argument_index, pokemon_ref, member_id)

    def _resolve_incoming(
        self,
        event: ProtocolEvent,
        pokemon_ref: ProtocolPokemonReference,
    ) -> ReplayMemberId:
        try:
            member_id = self._incoming_bindings[event.line_index]
        except KeyError as exc:
            raise ValueError("switch event has no inferred incoming member") from exc
        if member_id.side is not pokemon_ref.side:
            raise ValueError("inferred incoming member belongs to the wrong side")
        return member_id

    def _apply_swap(
        self,
        event: ProtocolEvent,
        reference: ResolvedPokemonRefArgument,
    ) -> None:
        source_slot = reference.pokemon_ref.active_slot
        if source_slot is None:
            raise ValueError("swap requires an active-slot reference")
        target_slot = int(event.arguments[1])
        if not 0 <= target_slot < 2:
            raise ValueError(f"swap target slot {target_slot} is outside a doubles battle")

        side = reference.pokemon_ref.side
        source_key = (side, source_slot)
        target_key = (side, target_slot)
        source_member = self._active[source_key]
        target_member = self._active.get(target_key)
        self._active[target_key] = source_member
        if target_member is None:
            self._active.pop(source_key, None)
        else:
            self._active[source_key] = target_member


def _diagnostic(event: ProtocolEvent, reason: str) -> ReplayEventDiagnostic:
    return ReplayEventDiagnostic(
        replay_id=event.replay_id,
        line_index=event.line_index,
        tag=event.tag,
        normalized_effect="" if event.effect is None else event.effect.normalized,
        normalized_cause="" if event.cause is None else event.cause.normalized,
        raw_line=event.raw_line,
        reason=reason,
    )


def resolve_protocol_events(
    replay_id: str,
    ots: tuple[OTSData, OTSData],
    events: Iterable[ProtocolEvent],
    *,
    dex: Mapping[str, Any] | None = None,
) -> ResolvedReplayEvents:
    """Resolve one accepted event stream without retaining a partial result."""
    event_tuple = tuple(events)
    parser_diagnostics = tuple(
        diagnostic for event in event_tuple if (diagnostic := event.diagnostic) is not None
    )
    if parser_diagnostics:
        return ResolvedReplayEvents(replay_id, (), parser_diagnostics)
    if tuple(sheet.side for sheet in ots) != (ReplaySide.P1, ReplaySide.P2):
        raise ValueError("OTS sheets must be ordered as p1 and p2")
    if any(not sheet.is_complete for sheet in ots):
        if not event_tuple:
            raise ValueError("Identity resolution requires events and complete OTS")
        reason = "identity resolution requires complete OTS for both sides"
        return ResolvedReplayEvents(replay_id, (), (_diagnostic(event_tuple[0], reason),))

    species_bases = _species_base_index({} if dex is None else dex)
    try:
        incoming_bindings = _resolve_incoming_histories(ots, event_tuple, species_bases)
    except _HistoryResolutionError as exc:
        return ResolvedReplayEvents(replay_id, (), (_diagnostic(exc.event, str(exc)),))

    resolver = _IdentityResolver(ots, incoming_bindings, species_bases)
    resolved: list[ResolvedProtocolEvent] = []
    for event in event_tuple:
        try:
            resolved.append(resolver.resolve(event))
        except (KeyError, ValueError) as exc:
            return ResolvedReplayEvents(replay_id, (), (_diagnostic(event, str(exc)),))
    return ResolvedReplayEvents(replay_id, tuple(resolved))


def resolve_replay_events(
    document: ReplayDocument,
    *,
    dex: Mapping[str, Any] | None = None,
) -> ResolvedReplayEvents:
    """Parse and resolve every event in one normalized replay document."""
    if dex is None:
        from p0.model.resources import default_runtime_resources

        dex = default_runtime_resources().dex
    parsed = parse_replay_events(document)
    return resolve_protocol_events(
        document.metadata.replay_id,
        document.ots,
        parsed.events,
        dex=dex,
    )


__all__ = [
    "ResolvedPokemonRefArgument",
    "ResolvedProtocolEvent",
    "ResolvedReplayEvents",
    "resolve_protocol_events",
    "resolve_replay_events",
]
