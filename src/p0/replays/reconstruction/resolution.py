"""Replay-wide identity inference and sequential protocol-reference resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from math import isfinite
from time import monotonic
from typing import Iterable, Mapping, NamedTuple

from p0.model.resources import default_runtime_resources
from p0.replays.identity import ReplayMemberId, ReplaySide, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.contract import REVIVAL_BLESSING
from p0.replays.reconstruction.diagnostics import (
    ReplayEventDiagnostic,
    ReplayEventParseError,
    ReplayRejectionCategory,
)
from p0.replays.reconstruction.events import (
    PokemonRefArgument,
    ProtocolEvent,
    parse_replay_events,
)
from p0.replays.reconstruction.identity import ProtocolPokemonReference
from p0.replays.schema import OTSData, OTSMember

# Tags whose second argument is the Pokémon's HP after the event.
_HP_TAGS = frozenset({"-damage", "-heal", "-sethp"})
# These moves can give a member an item before the protocol announces its use.
_ITEM_TRANSFER_MOVES = frozenset(
    {"bestow", "covet", "fling", "recycle", "switcheroo", "thief", "trick"}
)
_ABILITY_CHANGE_MOVES = frozenset(
    {"doodle", "entrainment", "roleplay", "simplebeam", "skillswap", "worryseed"}
)
_SELF_ABILITY_CAUSES = frozenset(
    {"-immune", "-heal", "-boost", "-setboost", "-curestatus", "-start", "-transform", "move"}
)
_SOURCE_ABILITY_CAUSES = frozenset(
    {"-weather", "-fieldstart", "-status", "-damage", "-item", "-clearboost", "-fail"}
)
# Measured local resolutions take milliseconds; leave ample room for longer public games.
IDENTITY_TIME_LIMIT_SECONDS = 1.0
_DEADLINE_CHECK_INTERVAL = 256


class ResolvedPokemonRefArgument(NamedTuple):
    """One protocol reference resolved to either a side or a roster member."""

    argument_index: int
    pokemon_ref: ProtocolPokemonReference
    member_id: ReplayMemberId | None


class ResolvedProtocolEvent(NamedTuple):
    """A classified event whose Pokémon references have explicit identities."""

    event: ProtocolEvent
    pokemon_refs: tuple[ResolvedPokemonRefArgument, ...]


@dataclass(frozen=True, slots=True)
class ResolvedReplayEvents:
    """A complete resolved replay or its whole-replay rejection diagnostic."""

    replay_id: str
    events: tuple[ResolvedProtocolEvent, ...]
    diagnostics: tuple[ReplayEventDiagnostic, ...] = ()

    def require_accepted(self) -> tuple[ResolvedProtocolEvent, ...]:
        """Return resolved events or raise the replay rejection."""
        if self.diagnostics:
            raise ReplayEventParseError(self.diagnostics)
        return self.events


class _HistoryCandidate(NamedTuple):
    actual_member: ReplayMemberId
    displayed_member: ReplayMemberId


@dataclass(slots=True)
class _ActiveHistory:
    history_id: int
    side: ReplaySide
    entry_event: ProtocolEvent
    candidates: tuple[_HistoryCandidate, ...]
    entry_hp: str
    last_hp: str
    end_line: int
    check_evidence: bool
    revealed_member: ReplayMemberId | None = None
    reveal_event: ProtocolEvent | None = None
    faint_line: int | None = None
    own_moves: set[str] = field(default_factory=set)
    abilities: set[str] = field(default_factory=set)
    items: set[str] = field(default_factory=set)
    transformed: bool = False
    took_hit: bool = False


class _Revive(NamedTuple):
    event: ProtocolEvent
    member_id: ReplayMemberId
    hp: str


class _ResolutionError(Exception):
    def __init__(
        self,
        event: ProtocolEvent,
        reason: str,
        category: ReplayRejectionCategory = ReplayRejectionCategory.INVALID_INPUT_CONTRACT,
    ) -> None:
        self.event = event
        self.category = category
        super().__init__(reason)


def _check_deadline(event: ProtocolEvent, deadline: float) -> None:
    if monotonic() >= deadline:
        raise _ResolutionError(
            event,
            "identity resolution exceeded its time limit",
            ReplayRejectionCategory.TIME_LIMIT_EXCEEDED,
        )


class _HistoryScanner:
    """Bind event references to stints while collecting the evidence that identifies them."""

    def __init__(self, ots: tuple[OTSData, OTSData], species_bases: Mapping[str, str]) -> None:
        self._members = {sheet.side: sheet.members for sheet in ots}
        self._species_bases = species_bases
        self.illusion_members = {
            sheet.side: tuple(
                member.member_id
                for member in sheet.members
                if normalize_showdown_id(member.ability) == "illusion"
            )
            for sheet in ots
        }
        self._active: dict[tuple[ReplaySide, int], int] = {}
        self._last_occupant: dict[tuple[ReplaySide, int], int] = {}
        self.histories: list[_ActiveHistory] = []
        self.references: list[tuple[int | ReplayMemberId | None, ...]] = []
        self.revives: list[_Revive] = []
        self._last_mover_id: int | None = None
        self._member_moves = {
            member.member_id: frozenset(normalize_showdown_id(move) for move in member.moves)
            for sheet in ots
            for member in sheet.members
        }
        self._member_abilities = {
            member.member_id: normalize_showdown_id(member.ability)
            for sheet in ots
            for member in sheet.members
        }
        self._member_items = {
            member.member_id: normalize_showdown_id(member.item)
            for sheet in ots
            for member in sheet.members
        }
        self._changed_abilities: set[ReplaySide] = set()
        self._changed_items = False
        self.team_sizes = {sheet.side: min(4, len(sheet.members)) for sheet in ots}
        self._slotless: dict[tuple[ReplaySide, str], set[ReplayMemberId]] = {}
        for sheet in ots:
            for member in sheet.members:
                for name in _slotless_names(member, species_bases):
                    if name:
                        self._slotless.setdefault((sheet.side, name), set()).add(member.member_id)

    def scan(self, events: tuple[ProtocolEvent, ...], deadline: float) -> None:
        """
        Bind references once and collect each active stint's public evidence.

        Arguments:
          events: parsed public events in chronological order
          deadline: monotonic time at which identity work must stop

        Returns:
          None; histories and reference bindings belong to this scanner
        """
        end_line = events[-1].line_index + 1 if events else 0
        has_illusion = any(self.illusion_members.values())
        for index, event in enumerate(events):
            if index % _DEADLINE_CHECK_INTERVAL == 0:
                _check_deadline(event, deadline)
            if event.tag in {"switch", "drag"}:
                self._start_history(event, end_line)
            self.references.append(tuple(self._bind(event, ref) for ref in event.pokemon_refs))
            if has_illusion:
                self._record_sheet_evidence(event)
            if event.tag == "teamsize":
                side = ReplaySide(event.arguments[0])
                size = int(event.arguments[1])
                if not 1 <= size <= len(self._members[side]):
                    raise _ResolutionError(event, f"invalid selected team size {size}")
                self.team_sizes[side] = size
            elif event.tag == "replace":
                history = self._history(event)
                member = _revealed_member(
                    event,
                    _event_reference(event).pokemon_ref,
                    self._members[history.side],
                    self._species_bases,
                )
                if history.revealed_member is not None and history.revealed_member != member:
                    raise _ResolutionError(
                        event,
                        "Illusion history has conflicting reveals",
                        ReplayRejectionCategory.INCONSISTENT_IDENTITY,
                    )
                history.revealed_member = member
                history.reveal_event = event
            elif event.tag == "faint":
                history = self._history(event)
                history.faint_line = history.end_line = event.line_index
                history.last_hp = "0"
                reference = _event_reference(event)
                key = (history.side, _required_active_slot(event, reference))
                if key not in self._active:
                    raise _ResolutionError(event, "faint source has no active history")
                self._last_occupant[key] = self._active.pop(key)
            elif event.tag == "swap":
                reference = _event_reference(event)
                source = (reference.pokemon_ref.side, _required_active_slot(event, reference))
                target_slot = int(event.arguments[1])
                if not 0 <= target_slot < 2:
                    raise _ResolutionError(
                        event, f"swap target slot {target_slot} is outside a doubles battle"
                    )
                if source not in self._active:
                    raise _ResolutionError(event, "swap source has no active history")
                target = (source[0], target_slot)
                if source == target:
                    continue
                occupant = self._active.pop(target, None)
                self._active[target] = self._active.pop(source)
                if occupant is not None:
                    self._active[source] = occupant
            elif event.tag == "move":
                history = self._history(event)
                self._last_mover_id = history.history_id
                if history.check_evidence and event.cause is None and not history.transformed:
                    move = normalize_showdown_id(event.arguments[1])
                    if move != "struggle":
                        history.own_moves.add(move)
            elif event.tag in _HP_TAGS:
                self._record_hp(event)
            elif event.tag == "-transform":
                self._history(event).transformed = True
        for history in self.histories:
            history.candidates = tuple(
                candidate
                for candidate in history.candidates
                if (
                    history.revealed_member is None
                    or candidate.actual_member == history.revealed_member
                )
                and (
                    not history.check_evidence
                    or (
                        history.own_moves <= self._member_moves[candidate.actual_member]
                        and history.abilities <= {self._member_abilities[candidate.actual_member]}
                        and history.items <= {self._member_items[candidate.actual_member]}
                        and not (
                            history.took_hit
                            and history.revealed_member is None
                            and candidate.actual_member != candidate.displayed_member
                        )
                    )
                )
            )

    def _record_sheet_evidence(self, event: ProtocolEvent) -> None:
        """
        Use native abilities and items while they still match the initial sheet.

        Arguments:
          event: the next public event, with references already bound

        Returns:
          None; possible changes stop later comparisons to initial sheet values
        """
        cause = event.cause
        if (
            event.tag == "move"
            and normalize_showdown_id(event.arguments[1]) in _ABILITY_CHANGE_MOVES
        ) or (
            event.tag == "-activate"
            and event.effect is not None
            and event.effect.normalized in {"skillswap", "mummy", "lingeringaroma"}
        ):
            self._changed_abilities.update((ReplaySide.P1, ReplaySide.P2))
        if (
            event.tag == "move"
            and normalize_showdown_id(event.arguments[1]) in _ITEM_TRANSFER_MOVES
        ):
            self._changed_items = True
        if event.tag == "-item" and (cause is None or cause.normalized != "frisk"):
            self._changed_items = True
        if event.tag in {"detailschange", "-formechange", "-endability"} or (
            event.tag == "-ability" and cause is not None
        ):
            self._changed_abilities.add(event.pokemon_refs[0].pokemon_ref.side)

        ability_reference = None
        ability = ""
        if event.tag == "-ability" and cause is None:
            ability_reference = 0
            ability = normalize_showdown_id(event.arguments[1])
        elif (
            event.effect is not None
            and event.effect.namespace == "ability"
            and event.tag in {"-activate", "-start"}
        ):
            ability_reference = 0
            ability = event.effect.normalized
        elif cause is not None and cause.namespace == "ability":
            if event.tag in _SOURCE_ABILITY_CAUSES and (
                event.tag != "-item" or cause.normalized == "frisk"
            ):
                ability_reference = next(
                    (
                        ref.argument_index
                        for ref in event.pokemon_refs
                        if event.arguments[ref.argument_index].startswith("[of]")
                    ),
                    None,
                )
            elif event.tag in _SELF_ABILITY_CAUSES and not any(
                argument.startswith("[of]") for argument in event.arguments
            ):
                ability_reference = 0
            ability = cause.normalized
        if ability_reference is not None:
            history = self._bound_history(event, ability_reference)
            if history is not None and history.side not in self._changed_abilities:
                history.abilities.add(ability)

        if not self._changed_items:
            item = ""
            item_reference = 0
            if event.tag in {"-item", "-enditem"}:
                item = normalize_showdown_id(event.arguments[1])
            elif (
                cause is not None
                and cause.namespace == "item"
                and event.tag in {"-damage", "-heal", "-boost"}
            ):
                item = cause.normalized
                if event.tag == "-damage":
                    # Helmet and retaliatory berries hurt the attacker; [of] names
                    # their holder. Shell Bell heals its holder and names its victim.
                    item_reference = next(
                        (
                            ref.argument_index
                            for ref in event.pokemon_refs
                            if event.arguments[ref.argument_index].startswith("[of]")
                        ),
                        0,
                    )
            elif (
                event.effect is not None
                and event.effect.namespace == "item"
                and event.tag == "-activate"
            ):
                item = event.effect.normalized
            if item:
                history = self._bound_history(event, item_reference)
                if history is not None:
                    history.items.add(item)
        if event.tag == "-enditem":
            self._changed_items = True
        if event.tag == "-transform":
            self._changed_abilities.add(event.pokemon_refs[0].pokemon_ref.side)

    def _bound_history(self, event: ProtocolEvent, argument_index: int) -> _ActiveHistory | None:
        binding = next(
            (
                binding
                for ref, binding in zip(event.pokemon_refs, self.references[-1], strict=True)
                if ref.argument_index == argument_index
            ),
            None,
        )
        if not isinstance(binding, int):
            return None
        history = self.histories[binding]
        return (
            history
            if history.check_evidence
            and history.revealed_member is None
            and not history.transformed
            else None
        )

    def _start_history(self, event: ProtocolEvent, end_line: int) -> None:
        reference = _event_reference(event)
        side = reference.pokemon_ref.side
        key = (side, _required_active_slot(event, reference))
        candidates = _history_candidates(
            event,
            reference.pokemon_ref,
            self._members[side],
            self.illusion_members[side],
            self._species_bases,
        )
        displayed = {candidate.displayed_member for candidate in candidates}
        if len(displayed) > 1:
            raise _ResolutionError(
                event,
                f"incoming reference {reference.pokemon_ref.displayed_name!r} "
                f"resolved to {len(displayed)} available roster members",
            )
        if not candidates:
            raise _ResolutionError(
                event,
                f"incoming reference {reference.pokemon_ref.displayed_name!r} "
                "resolved to 0 available roster members",
            )
        previous = self._active.get(key)
        if previous is not None:
            self.histories[previous].end_line = event.line_index
        self._last_occupant.pop(key, None)
        hp = _shown_hp(event.arguments[2])
        history = _ActiveHistory(
            len(self.histories), side, event, candidates, hp, hp, end_line, len(candidates) > 1
        )
        self.histories.append(history)
        self._active[key] = history.history_id

    def _bind(
        self, event: ProtocolEvent, reference: PokemonRefArgument
    ) -> int | ReplayMemberId | None:
        ref = reference.pokemon_ref
        if ref.active_slot is not None:
            key = (ref.side, ref.active_slot)
            history = self._active.get(key, self._last_occupant.get(key))
            if history is None:
                raise _ResolutionError(
                    event, f"{ref.side.value}{ref.active_slot} has no active member"
                )
            return history
        if event.tag in {"-sidestart", "-sideend"}:
            return None
        name = normalize_showdown_id(ref.displayed_name)
        matches = self._slotless.get((ref.side, name), set())
        if len(matches) == 1:
            return next(iter(matches))
        if matches:
            raise _ResolutionError(
                event,
                f"side reference {ref.displayed_name!r} resolved to {len(matches)} roster members",
            )
        if name:
            raise _ResolutionError(
                event, f"side reference {ref.displayed_name!r} matched no roster member"
            )
        return None

    def _history(self, event: ProtocolEvent) -> _ActiveHistory:
        reference = self.references[-1][0]
        if not isinstance(reference, int):
            raise _ResolutionError(event, f"event {event.tag!r} requires an active slot")
        return self.histories[reference]

    def _record_hp(self, event: ProtocolEvent) -> None:
        reference = self.references[-1][0]
        hp = _shown_hp(event.arguments[1])
        if isinstance(reference, ReplayMemberId):
            if event.cause is not None and event.cause.normalized == REVIVAL_BLESSING:
                self.revives.append(_Revive(event, reference, hp))
            return
        history = self._history(event)
        history.last_hp = hp
        if (
            history.check_evidence
            and event.tag == "-damage"
            and event.cause is None
            and self._last_mover_id not in {None, history.history_id}
        ):
            history.took_hit = True


class _HistorySolver:
    """Search only unresolved assignments, retaining at most two complete histories."""

    def __init__(
        self,
        histories: tuple[_ActiveHistory, ...],
        revives: tuple[_Revive, ...],
        team_size: int,
        deadline: float,
    ) -> None:
        self._histories = histories
        self._revives = revives
        self._team_size = team_size
        self._deadline = deadline
        self._latest: dict[ReplayMemberId, _ActiveHistory] = {}
        self._revives_before: dict[tuple[int, ReplayMemberId], _Revive] = {}
        latest_revives: dict[ReplayMemberId, _Revive] = {}
        cursor = 0
        for history in histories:
            while (
                cursor < len(revives)
                and revives[cursor].event.line_index < history.entry_event.line_index
            ):
                revive = revives[cursor]
                latest_revives[revive.member_id] = revive
                cursor += 1
            for member, revive in latest_revives.items():
                self._revives_before[history.history_id, member] = revive

    def solve(self) -> tuple[tuple[_HistoryCandidate, ...], ...]:
        if any(not history.candidates for history in self._histories):
            return ()
        _check_deadline(self._histories[0].entry_event, self._deadline)
        if all(len(history.candidates) == 1 for history in self._histories):
            assigned: list[_HistoryCandidate] = []
            selected = 0
            for history in self._histories:
                _check_deadline(history.entry_event, self._deadline)
                candidate = history.candidates[0]
                mask = selected | _candidate_mask(candidate)
                if mask.bit_count() > self._team_size or not self._valid(history, candidate):
                    return ()
                self._latest[candidate.actual_member] = history
                assigned.append(candidate)
                selected = mask
            return (tuple(assigned),) if self._valid_revives(assigned) else ()

        solutions: list[tuple[_HistoryCandidate, ...]] = []
        assigned = []
        previous: list[_ActiveHistory | None] = []
        masks = [0]
        choices = [0] * len(self._histories)
        position = 0
        while position >= 0:
            _check_deadline(
                self._histories[min(position, len(self._histories) - 1)].entry_event, self._deadline
            )
            if position == len(self._histories):
                if self._valid_revives(assigned):
                    solutions.append(tuple(assigned))
                    if len(solutions) == 2:
                        break
            else:
                history = self._histories[position]
                for choice in range(choices[position], len(history.candidates)):
                    choices[position] = choice + 1
                    candidate = history.candidates[choice]
                    mask = masks[-1] | _candidate_mask(candidate)
                    if mask.bit_count() > self._team_size or not self._valid(history, candidate):
                        continue
                    previous.append(self._latest.get(candidate.actual_member))
                    self._latest[candidate.actual_member] = history
                    assigned.append(candidate)
                    masks.append(mask)
                    position += 1
                    if position < len(choices):
                        choices[position] = 0
                    break
                else:
                    position -= 1
                    if position < 0:
                        break
                    self._undo(assigned, previous, masks)
                continue
            position -= 1
            self._undo(assigned, previous, masks)
        return tuple(solutions)

    def _undo(
        self,
        assigned: list[_HistoryCandidate],
        previous: list[_ActiveHistory | None],
        masks: list[int],
    ) -> None:
        member = assigned.pop().actual_member
        last = previous.pop()
        if last is None:
            del self._latest[member]
        else:
            self._latest[member] = last
        masks.pop()

    def _valid(self, history: _ActiveHistory, candidate: _HistoryCandidate) -> bool:
        entry = history.entry_event.line_index
        previous = self._latest.get(candidate.actual_member)
        if previous is not None and previous.end_line >= entry:
            return False
        for member in {candidate.actual_member, candidate.displayed_member}:
            last = self._latest.get(member)
            revive = self._revives_before.get((history.history_id, member))
            if (
                last is not None
                and last.faint_line is not None
                and last.faint_line < entry
                and (revive is None or revive.event.line_index <= last.faint_line)
            ):
                return False
        if not history.check_evidence:
            return True
        revive = self._revives_before.get((history.history_id, candidate.actual_member))
        if revive is not None and (
            previous is None or revive.event.line_index > previous.entry_event.line_index
        ):
            hp = revive.hp
        else:
            hp = None if previous is None else previous.last_hp
        return _is_full_hp(history.entry_hp) if hp is None else hp == history.entry_hp

    def _valid_revives(self, assigned: list[_HistoryCandidate]) -> bool:
        # A named revival proves that this member, rather than its disguise, had fainted.
        if not self._revives:
            return True
        latest: dict[ReplayMemberId, _ActiveHistory] = {}
        cursor = 0
        last_revive: dict[ReplayMemberId, int] = {}
        for revive in self._revives:
            line = revive.event.line_index
            while cursor < len(assigned) and self._histories[cursor].entry_event.line_index < line:
                latest[assigned[cursor].actual_member] = self._histories[cursor]
                cursor += 1
            previous = latest.get(revive.member_id)
            if (
                previous is None
                or previous.faint_line is None
                or previous.faint_line >= line
                or previous.faint_line <= last_revive.get(revive.member_id, -1)
            ):
                return False
            last_revive[revive.member_id] = line
        return True


def _candidate_mask(candidate: _HistoryCandidate) -> int:
    return (1 << candidate.actual_member.roster_index) | (
        1 << candidate.displayed_member.roster_index
    )


def _event_reference(event: ProtocolEvent, argument_index: int = 0) -> PokemonRefArgument:
    try:
        return next(
            reference
            for reference in event.pokemon_refs
            if reference.argument_index == argument_index
        )
    except StopIteration as exc:
        raise _ResolutionError(
            event,
            f"event {event.tag!r} has no Pokémon reference at argument {argument_index}",
        ) from exc


def _shown_hp(condition: str) -> str:
    """Return the HP text of a condition such as 61/100 par, without the status."""
    return condition.split(" ", 1)[0].rstrip("gyr")


def _is_full_hp(shown_hp: str) -> bool:
    hp, _, maximum = shown_hp.partition("/")
    return hp == maximum


def _required_active_slot(event: ProtocolEvent, reference: PokemonRefArgument) -> int:
    slot = reference.pokemon_ref.active_slot
    if slot is None:
        raise _ResolutionError(event, f"event {event.tag!r} requires an active slot")
    return slot


@cache
def _species_base_index() -> Mapping[str, str]:
    """Map each dex species id to the id of its base species."""
    return {
        entry["id"]: normalize_showdown_id(entry["baseSpecies"])
        for entry in default_runtime_resources().dex["species"]
    }


def _base_species_id(species: str, species_bases: Mapping[str, str]) -> str:
    species_id = normalize_showdown_id(species)
    return species_bases.get(species_id, species_id)


def _slotless_names(member: OTSMember, species_bases: Mapping[str, str]) -> set[str]:
    """Return the names a slotless reference can use for a member."""
    # A Pokemon with no nickname is named by its base species: Vivillon-Sandstorm is Vivillon.
    return {
        normalize_showdown_id(member.nickname),
        normalize_showdown_id(member.species),
        _base_species_id(member.species, species_bases),
    }


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
        raise _ResolutionError(
            event,
            f"Illusion reveal {pokemon_ref.displayed_name!r} resolved to "
            f"{len(candidates)} roster members",
        )
    return candidates[0].member_id


def resolve_protocol_events(
    replay_id: str,
    ots: tuple[OTSData, OTSData],
    events: Iterable[ProtocolEvent],
    *,
    time_limit_seconds: float = IDENTITY_TIME_LIMIT_SECONDS,
) -> ResolvedReplayEvents:
    """
    Resolve complete identities, or return one rejection without partial events.

    Arguments:
      replay_id: replay identity for diagnostics
      ots: both players' public team sheets
      events: parsed public events in chronological order
      time_limit_seconds: maximum time for collecting and solving identity evidence

    Returns:
      resolved events or a diagnostic rejecting the entire replay
    """
    if not isfinite(time_limit_seconds) or time_limit_seconds < 0:
        raise ValueError("identity time limit must be finite and nonnegative")
    event_tuple = tuple(events)
    diagnostics = tuple(event.diagnostic for event in event_tuple if event.diagnostic is not None)
    if diagnostics:
        return ResolvedReplayEvents(replay_id, (), diagnostics)
    deadline = monotonic() + time_limit_seconds
    scanner = _HistoryScanner(ots, _species_base_index())
    bindings: dict[int, ReplayMemberId] = {}
    ambiguity: ProtocolEvent | None = None
    try:
        scanner.scan(event_tuple, deadline)
        for side in (ReplaySide.P1, ReplaySide.P2):
            histories = tuple(history for history in scanner.histories if history.side is side)
            if not histories:
                continue
            revives = tuple(revive for revive in scanner.revives if revive.member_id.side is side)
            solutions = _HistorySolver(
                histories, revives, scanner.team_sizes[side], deadline
            ).solve()
            if not solutions:
                event = next(
                    (
                        history.reveal_event
                        for history in histories
                        if history.reveal_event is not None and not history.candidates
                    ),
                    histories[0].entry_event,
                )
                raise _ResolutionError(
                    event,
                    "active history has no valid roster assignment",
                    ReplayRejectionCategory.INCONSISTENT_IDENTITY,
                )
            if len(solutions) > 1:
                first, second = solutions
                index = next(
                    index
                    for index, (left, right) in enumerate(zip(first, second, strict=True))
                    if left != right
                )
                ambiguity = histories[index].entry_event
                continue
            solution = next(iter(solutions))
            for history, candidate in zip(histories, solution, strict=True):
                bindings[history.history_id] = candidate.actual_member
        if ambiguity is not None:
            raise _ResolutionError(
                ambiguity,
                "active history has multiple valid assignments",
                ReplayRejectionCategory.AMBIGUOUS_IDENTITY,
            )
    except _ResolutionError as exc:
        return ResolvedReplayEvents(replay_id, (), (exc.event.rejection(str(exc), exc.category),))
    resolved = tuple(
        ResolvedProtocolEvent(
            event,
            tuple(
                ResolvedPokemonRefArgument(
                    reference.argument_index,
                    reference.pokemon_ref,
                    bindings[binding] if isinstance(binding, int) else binding,
                )
                for reference, binding in zip(event.pokemon_refs, references, strict=True)
            ),
        )
        for event, references in zip(event_tuple, scanner.references, strict=True)
    )
    return ResolvedReplayEvents(replay_id, resolved)


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
