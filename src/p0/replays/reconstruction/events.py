"""Lossless conversion of normalized protocol lines into classified events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from p0.replays.identity import normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.classification import (
    CLASSIFICATION_REGISTRY,
    UNSUPPORTED_PREDICATES,
    UNSUPPORTED_TAGS,
    EventClassification,
    EventRule,
)
from p0.replays.reconstruction.contract import (
    ALL_LEGAL_EFFECT_NAMES,
    ALLOWED_CAUSE_NAMESPACES,
    ALLOWED_EFFECTS_BY_TAG,
    KNOWN_ACTIVATION_EFFECTS,
    LEGAL_EFFECT_IDS,
)
from p0.replays.reconstruction.diagnostics import (
    ReplayEventDiagnostic,
    ReplayEventParseError,
    ReplayRejectionCategory,
)
from p0.replays.reconstruction.identity import (
    ProtocolPokemonReference,
    looks_like_protocol_pokemon_reference,
    parse_protocol_pokemon_reference,
)
from p0.replays.schema import ProtocolLine


@dataclass(frozen=True, slots=True)
class EffectReference:
    """A normalized protocol effect such as a move, item, or ability."""

    namespace: str
    name: str
    normalized: str

    def __post_init__(self) -> None:
        if not self.name or not self.normalized:
            raise ValueError("EffectReference requires a nonempty name")


@dataclass(frozen=True, slots=True)
class PokemonRefArgument:
    """A Pokémon reference together with its position in the event arguments."""

    argument_index: int
    pokemon_ref: ProtocolPokemonReference

    def __post_init__(self) -> None:
        if type(self.argument_index) is not int or self.argument_index < 0:
            raise ValueError("PokemonRefArgument.argument_index must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProtocolEvent:
    """One immutable and explicitly classified replay protocol event."""

    replay_id: str
    line_index: int
    turn: int | None
    tag: str
    arguments: tuple[str, ...]
    raw_line: str
    classification: EventClassification
    pokemon_refs: tuple[PokemonRefArgument, ...] = ()
    effect: EffectReference | None = None
    cause: EffectReference | None = None
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("ProtocolEvent.replay_id must not be empty")
        if type(self.line_index) is not int or self.line_index < 0:
            raise ValueError("ProtocolEvent.line_index must be nonnegative")
        if not self.raw_line.startswith("|"):
            raise ValueError("ProtocolEvent.raw_line must be a protocol line")
        if not isinstance(self.classification, EventClassification):
            raise TypeError("ProtocolEvent.classification must be an EventClassification")
        if self.classification.rejects_replay != bool(self.rejection_reason):
            raise ValueError("Rejecting events require a reason and accepted events forbid one")

    @property
    def diagnostic(self) -> ReplayEventDiagnostic | None:
        """Return a structured whole-replay rejection diagnostic when applicable."""
        if not self.classification.rejects_replay:
            return None
        return ReplayEventDiagnostic(
            replay_id=self.replay_id,
            line_index=self.line_index,
            tag=self.tag,
            normalized_effect="" if self.effect is None else self.effect.normalized,
            normalized_cause="" if self.cause is None else self.cause.normalized,
            raw_line=self.raw_line,
            reason=self.rejection_reason,
            category=(
                ReplayRejectionCategory.UNSUPPORTED_EVENT
                if self.classification is EventClassification.UNSUPPORTED_STATE
                else ReplayRejectionCategory.INVALID_INPUT_CONTRACT
            ),
        )


@dataclass(frozen=True, slots=True)
class ParsedReplayEvents:
    """Complete event parsing result for one replay."""

    replay_id: str
    events: tuple[ProtocolEvent, ...]
    diagnostics: tuple[ReplayEventDiagnostic, ...]

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("ParsedReplayEvents.replay_id must not be empty")
        if any(event.replay_id != self.replay_id for event in self.events):
            raise ValueError("ParsedReplayEvents cannot contain events from another replay")
        expected_diagnostics = tuple(
            diagnostic for event in self.events if (diagnostic := event.diagnostic) is not None
        )
        if self.diagnostics != expected_diagnostics:
            raise ValueError("ParsedReplayEvents.diagnostics must match its rejecting events")

    def require_accepted(self) -> tuple[ProtocolEvent, ...]:
        """Return events or reject the whole replay with all parsing diagnostics."""
        if self.diagnostics:
            raise ReplayEventParseError(self.diagnostics)
        return self.events


def _effect_reference(value: str) -> EffectReference | None:
    value = value.strip()
    if value.startswith("[") and "]" in value:
        value = value.split("]", 1)[1].strip()
    if not value:
        return None

    namespace = ""
    name = value
    if ":" in value:
        namespace_value, name_value = value.split(":", 1)
        namespace = normalize_showdown_id(namespace_value)
        name = name_value.strip()

    normalized = normalize_showdown_id(name)
    if not normalized:
        return None
    return EffectReference(namespace, name, normalized)


def _cause(arguments: tuple[str, ...]) -> EffectReference | None:
    for argument in arguments:
        if argument.startswith("[from] "):
            return _effect_reference(argument)
    return None


def _shape_error(rule: EventRule, arguments: tuple[str, ...]) -> str:
    count = len(arguments)
    if count < rule.minimum_arguments:
        return f"expected at least {rule.minimum_arguments} arguments, received {count}"
    if rule.maximum_arguments is not None and count > rule.maximum_arguments:
        return f"expected at most {rule.maximum_arguments} arguments, received {count}"
    for index in rule.required_nonempty:
        if index >= count or not arguments[index]:
            return f"argument {index} must be present and nonempty"
    return ""


def _semantic_shape_error(tag: str, arguments: tuple[str, ...]) -> str:
    if (
        tag == "move"
        and len(arguments) == 2
        and normalize_showdown_id(arguments[1])
        in {
            "doomdesire",
            "futuresight",
        }
    ):
        return "delayed move requires an explicit target"
    if tag in {"turn", "gen"} and not arguments[0].isdigit():
        return f"{tag} argument must be an integer"
    if tag == "teampreview" and arguments and not arguments[0].isdigit():
        return "teampreview argument must be an integer when present"
    if tag == "teamsize" and (arguments[0] not in {"p1", "p2"} or not arguments[1].isdigit()):
        return "teamsize requires a p1/p2 side and integer size"
    if tag == "swap" and not arguments[1].isdigit():
        return "swap position must be an integer"
    if tag == "-hitcount" and not arguments[1].isdigit():
        return "-hitcount count must be an integer"
    if tag in {"-boost", "-unboost", "-setboost"} and len(arguments) > 2:
        try:
            int(arguments[2])
        except ValueError:
            return f"{tag} amount must be an integer"
    if tag in {"poke", "showteam"} and (not arguments or arguments[0] not in {"p1", "p2"}):
        return f"{tag} requires a p1/p2 side"
    if tag == "player" and (not arguments or arguments[0] not in {"p1", "p2"}):
        return "player requires a p1/p2 side"
    if tag == "-activate" and len(arguments) > 1:
        effect = _effect_reference(arguments[1])
        catalog_kind = (
            {
                "item": "items",
                "ability": "abilities",
            }.get(effect.namespace)
            if effect is not None
            else None
        )
        catalog_effect = (
            catalog_kind is not None
            and effect is not None
            and effect.normalized in LEGAL_EFFECT_IDS[catalog_kind]
        )
        if (
            effect is not None
            and effect.normalized not in KNOWN_ACTIVATION_EFFECTS
            and not catalog_effect
        ):
            return f"unsupported -activate effect {effect.normalized!r}"
        if effect is not None and effect.normalized in {"spite", "eeriespell"}:
            if (
                len(arguments) != 4
                or not arguments[2]
                or not arguments[3].isdigit()
                or not 1 <= int(arguments[3]) <= (4 if effect.normalized == "spite" else 3)
            ):
                return "PP deduction activation requires a named move and bounded positive amount"
        if effect is not None and effect.normalized == "leppaberry":
            if len(arguments) < 4 or not arguments[2] or arguments[3] != "[consumed]":
                return "Leppa activation requires a named move and [consumed] annotation"
    if tag == "-sethp":
        # Showdown's sethp wire shape is exactly one target/HP pair plus
        # annotations. Annotation fields are never additional pairs.
        hp = arguments[1] if len(arguments) > 1 else ""
        if not hp or hp.startswith("[") or "/" not in hp:
            return "-sethp requires one target/HP pair"
        if any(not value.startswith("[") for value in arguments[2:]):
            return "-sethp permits only annotations after the HP value"
    if tag == "-activate":
        effect_index = 1 if arguments and looks_like_protocol_pokemon_reference(arguments[0]) else 0
        if effect_index >= len(arguments) or _effect_reference(arguments[effect_index]) is None:
            return "-activate requires a source-backed effect"
    return ""


def _annotated_pokemon_ref(value: str) -> str | None:
    for prefix in ("[of] ", "[from] "):
        if value.startswith(prefix):
            candidate = value.removeprefix(prefix)
            if looks_like_protocol_pokemon_reference(candidate):
                return candidate
    return None


def _parse_pokemon_refs(
    rule: EventRule,
    arguments: tuple[str, ...],
) -> tuple[tuple[PokemonRefArgument, ...], str]:
    parsed: dict[int, PokemonRefArgument] = {}
    for index in rule.required_pokemon_refs:
        try:
            pokemon_ref = parse_protocol_pokemon_reference(arguments[index])
        except (IndexError, ValueError) as exc:
            return (), f"argument {index} is not a valid required Pokémon reference: {exc}"
        parsed[index] = PokemonRefArgument(index, pokemon_ref)

    for index in rule.optional_pokemon_refs:
        if index >= len(arguments) or not arguments[index]:
            continue
        try:
            pokemon_ref = parse_protocol_pokemon_reference(arguments[index])
        except ValueError as exc:
            return (), f"argument {index} is not a valid optional Pokémon reference: {exc}"
        parsed[index] = PokemonRefArgument(index, pokemon_ref)

    for index, argument in enumerate(arguments):
        if index not in parsed and looks_like_protocol_pokemon_reference(argument):
            try:
                parsed[index] = PokemonRefArgument(
                    index, parse_protocol_pokemon_reference(argument)
                )
            except ValueError as exc:
                return (), f"argument {index} resembles a malformed Pokémon reference: {exc}"
        candidate = _annotated_pokemon_ref(argument)
        if candidate is not None:
            try:
                parsed[index] = PokemonRefArgument(
                    index, parse_protocol_pokemon_reference(candidate)
                )
            except ValueError as exc:
                return (
                    (),
                    f"argument {index} contains a malformed annotated Pokémon reference: {exc}",
                )

    return tuple(parsed[index] for index in sorted(parsed)), ""


def _unsupported_effect(arguments: tuple[str, ...]) -> EffectReference | None:
    for argument in arguments:
        if (
            not argument
            or argument.startswith("[")
            or looks_like_protocol_pokemon_reference(argument)
        ):
            continue
        return _effect_reference(argument)
    return None


def _event_effect(
    tag: str,
    rule: EventRule,
    arguments: tuple[str, ...],
) -> EffectReference | None:
    index = rule.effect_argument
    if index is None or not arguments:
        return None
    if tag == "-activate" and looks_like_protocol_pokemon_reference(arguments[0]):
        index = 1
    if not -len(arguments) <= index < len(arguments):
        return None
    return _effect_reference(arguments[index])


def parse_protocol_event(replay_id: str, line: ProtocolLine) -> ProtocolEvent:
    """Parse and classify one normalized protocol line without changing state."""
    tag = line.parts[1]
    arguments = line.parts[2:]
    # Showdown serializes a missing move target as the literal JSON-ish token
    # ``null``.  It is a protocol sentinel, never a Pokémon reference.
    if tag == "move" and len(arguments) >= 3 and arguments[2] == "null":
        arguments = (*arguments[:2], "", *arguments[3:])
    cause = _cause(arguments)

    if line.raw == "|":
        return ProtocolEvent(
            replay_id,
            line.index,
            line.turn,
            tag,
            arguments,
            line.raw,
            EventClassification.BOUNDARY_SIGNAL,
            cause=cause,
        )

    if tag == "":
        if len(arguments) == 1 and arguments[0]:
            return ProtocolEvent(
                replay_id,
                line.index,
                line.turn,
                tag,
                arguments,
                line.raw,
                EventClassification.NO_STATE_CHANGE,
                cause=cause,
            )
        return ProtocolEvent(
            replay_id,
            line.index,
            line.turn,
            tag,
            arguments,
            line.raw,
            EventClassification.MALFORMED,
            cause=cause,
            rejection_reason="empty protocol tag is only valid for bare separators or text messages",
        )

    rule = CLASSIFICATION_REGISTRY.get(tag)
    if rule is None:
        return ProtocolEvent(
            replay_id,
            line.index,
            line.turn,
            tag,
            arguments,
            line.raw,
            EventClassification.UNSUPPORTED_STATE,
            effect=_unsupported_effect(arguments),
            cause=cause,
            rejection_reason=f"unclassified protocol tag {tag!r}",
        )

    reason = _shape_error(rule, arguments)
    if not reason:
        reason = _semantic_shape_error(tag, arguments)
    pokemon_refs: tuple[PokemonRefArgument, ...] = ()
    if not reason:
        pokemon_refs, reason = _parse_pokemon_refs(rule, arguments)

    effect = _event_effect(tag, rule, arguments)

    if effect is not None and tag in {"-activate", "-singlemove"}:
        allowed = ALLOWED_EFFECTS_BY_TAG.get(tag, frozenset())
        allowed_for_tag = (effect.namespace, effect.normalized) in allowed or (
            "",
            effect.normalized,
        ) in allowed
        globally_legal_activation = (
            tag == "-activate" and effect.normalized in ALL_LEGAL_EFFECT_NAMES
        )
        if not allowed_for_tag and not globally_legal_activation:
            reason = f"unsupported {tag} effect {effect.normalized!r}"
    if cause is not None and cause.namespace and cause.namespace not in ALLOWED_CAUSE_NAMESPACES:
        reason = f"unsupported cause namespace {cause.namespace!r}"
    if cause is not None:
        catalog_kind = {"move": "moves", "item": "items", "ability": "abilities"}.get(
            cause.namespace
        )
        if catalog_kind is not None and cause.normalized not in LEGAL_EFFECT_IDS[catalog_kind]:
            reason = f"unsupported cause {cause.namespace}:{cause.normalized}"
        elif not cause.namespace and cause.normalized not in ALL_LEGAL_EFFECT_NAMES:
            reason = f"unsupported cause {cause.normalized!r}"

    predicate = None
    for candidate in (effect, cause):
        if candidate and (tag, candidate.normalized) in UNSUPPORTED_PREDICATES:
            predicate = candidate
            break

    if not reason and tag in UNSUPPORTED_TAGS:
        reason = f"protocol tag {tag!r} is unsupported by the reconstruction contract"
        classification = EventClassification.UNSUPPORTED_STATE
    elif not reason and predicate is not None:
        reason = f"protocol effect {predicate.normalized!r} on {tag!r} requires unsupported stored-stat state"
        classification = EventClassification.UNSUPPORTED_STATE
    else:
        classification = (
            EventClassification.UNSUPPORTED_STATE
            if reason.startswith("unsupported ")
            else rule.classification
            if not reason
            else EventClassification.MALFORMED
        )
    return ProtocolEvent(
        replay_id,
        line.index,
        line.turn,
        tag,
        arguments,
        line.raw,
        classification,
        pokemon_refs=pokemon_refs,
        effect=effect,
        cause=cause,
        rejection_reason=reason,
    )


def parse_protocol_events(
    replay_id: str,
    lines: Iterable[ProtocolLine],
) -> ParsedReplayEvents:
    """Parse every normalized line and retain all rejection diagnostics."""
    events = tuple(parse_protocol_event(replay_id, line) for line in lines)
    diagnostics = tuple(
        diagnostic for event in events if (diagnostic := event.diagnostic) is not None
    )
    return ParsedReplayEvents(replay_id, events, diagnostics)


def parse_replay_events(document: ReplayDocument) -> ParsedReplayEvents:
    """Parse every line in a normalized replay document exactly once."""
    return parse_protocol_events(document.metadata.replay_id, document.protocol_lines)


__all__ = [
    "EffectReference",
    "PokemonRefArgument",
    "ParsedReplayEvents",
    "ProtocolEvent",
    "parse_protocol_event",
    "parse_protocol_events",
    "parse_replay_events",
]
