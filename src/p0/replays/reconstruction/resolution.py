"""Sequential resolution of protocol references to stable roster identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
from p0.replays.schema import OTSData


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


class _IdentityResolver:
    def __init__(self, ots: tuple[OTSData, OTSData]) -> None:
        self._members = {side_sheet.side: side_sheet.members for side_sheet in ots}
        self._active: dict[tuple[ReplaySide, int], ReplayMemberId] = {}

    def resolve(self, event: ProtocolEvent) -> ResolvedProtocolEvent:
        if event.tag == "replace":
            raise ValueError("replace requires Illusion resolution, which is not implemented")

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
            self._active[(incoming.side, switch_reference.pokemon_ref.active_slot)] = incoming
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
        name = normalize_showdown_id(pokemon_ref.displayed_name)
        candidates = [
            member
            for member in self._members[pokemon_ref.side]
            if name
            in {
                normalize_showdown_id(member.nickname),
                normalize_showdown_id(member.species),
            }
        ]

        if len(event.arguments) > 1:
            details_species = normalize_showdown_id(event.arguments[1].split(",", 1)[0])
            narrowed = [
                member
                for member in candidates
                if normalize_showdown_id(member.species) == details_species
            ]
            if narrowed:
                candidates = narrowed

        occupied = set(self._active.values())
        candidates = [member for member in candidates if member.member_id not in occupied]
        if len(candidates) != 1:
            raise ValueError(
                f"incoming reference {pokemon_ref.displayed_name!r} resolved to "
                f"{len(candidates)} available roster members"
            )
        return candidates[0].member_id

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

    resolver = _IdentityResolver(ots)
    resolved: list[ResolvedProtocolEvent] = []
    for event in event_tuple:
        try:
            resolved.append(resolver.resolve(event))
        except (KeyError, ValueError) as exc:
            return ResolvedReplayEvents(replay_id, (), (_diagnostic(event, str(exc)),))
    return ResolvedReplayEvents(replay_id, tuple(resolved))


def resolve_replay_events(document: ReplayDocument) -> ResolvedReplayEvents:
    """Parse and resolve every event in one normalized replay document."""
    parsed = parse_replay_events(document)
    return resolve_protocol_events(document.metadata.replay_id, document.ots, parsed.events)


__all__ = [
    "ResolvedPokemonRefArgument",
    "ResolvedProtocolEvent",
    "ResolvedReplayEvents",
    "resolve_protocol_events",
    "resolve_replay_events",
]
