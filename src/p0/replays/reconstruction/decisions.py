"""
Decision-window inference and action evidence for replay reconstruction v2.

This module consumes resolved protocol events and immutable state snapshots. It
keeps request-boundary inference separate from action extraction so execution
lines cannot create policy decisions by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, NamedTuple

from p0.battle.actions import (
    FORCED_ACTION,
    MEGA_FORCED_ACTION,
    PASS_ACTION,
    SWITCH_START,
    ActionKind,
    SlotAction,
    encode_action,
    encode_team_pair,
)
from p0.battle.legality import DecisionView, SlotDecision
from p0.model.resources import default_runtime_resources
from p0.replays.evidence import (
    EvidenceRequest,
    ObservedAction,
    extract_action_evidence,
)
from p0.replays.identity import ReplayMemberId, ReplaySide, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.diagnostics import ReplayEventDiagnostic
from p0.replays.reconstruction.events import EventClassification
from p0.replays.reconstruction.resolution import (
    ResolvedPokemonRefArgument,
    ResolvedProtocolEvent,
    resolve_replay_events,
)
from p0.replays.reconstruction.state import (
    MoveState,
    ReconstructedReplayState,
    ReplayBattleState,
    ReplayPokemonState,
    reduce_replay_state,
)
from p0.replays.schema import DecisionRecord, DecisionType, OTSData

_ACTION_TAGS = frozenset({"move", "switch", "cant"})
_PIVOT_CAUSES = frozenset(
    {
        "uturn",
        "flipturn",
        "voltswitch",
        "batonpass",
        "partingshot",
        "chillyreception",
        "shedtail",
    }
)
_FORCED_MOVES = frozenset({"struggle", "recharge"})
_SELF_TARGETS = frozenset(
    {
        "self",
        "all",
        "alladjacent",
        "alladjacentfoes",
        "allies",
        "allyside",
        "allyteam",
        "foeside",
        "randomnormal",
        "scripted",
    }
)
_FOE_TARGETS = frozenset({"adjacentfoe"})
_NORMAL_TARGETS = frozenset({"normal", "any"})
_ALLY_TARGETS = frozenset({"adjacentally"})
_ALLY_OR_SELF_TARGETS = frozenset({"adjacentallyorself"})


def _target_codes(move_target: str, actor_slot: int) -> tuple[int, ...]:
    if move_target in _SELF_TARGETS:
        return (0,)
    if move_target in _ALLY_TARGETS:
        return (-2,) if actor_slot == 0 else (-1,)
    if move_target in _ALLY_OR_SELF_TARGETS:
        return (0, -2) if actor_slot == 0 else (0, -1)
    if move_target in _FOE_TARGETS:
        return (1, 2)
    if move_target in _NORMAL_TARGETS:
        return ((-2,) if actor_slot == 0 else (-1,)) + (1, 2)
    return (0,)


class _MegaRules(NamedTuple):
    species_by_item: Mapping[str, frozenset[str]]
    z_items: frozenset[str]


def _effective_move_target(member: ReplayPokemonState, move: MoveState) -> str:
    if move.move_id == "curse":
        return (
            "any"
            if any(normalize_showdown_id(value) == "ghost" for value in member.current_types)
            else "self"
        )
    return move.target.casefold()


class BoundaryKind(StrEnum):
    """Classification of an update block before policy-row filtering."""

    TEAM_PREVIEW = "team_preview"
    NORMAL_TURN = "normal_turn"
    FORCED_SWITCH = "forced_switch"
    PIVOT_SWITCH = "pivot_switch"
    FORCED_EXECUTION = "forced_execution"
    RESIDUAL_TERMINAL = "residual_terminal"
    WAITING = "waiting"
    UNRECOVERABLE = "unrecoverable"


@dataclass(frozen=True, slots=True)
class DecisionWindow:
    """One ordered protocol update block and its request classification."""

    start_line_index: int
    end_line_index: int
    kind: BoundaryKind
    decision_type: DecisionType

    def __post_init__(self) -> None:
        if self.start_line_index < 0 or self.end_line_index < self.start_line_index:
            raise ValueError("DecisionWindow line indices must be ordered and nonnegative")
        if self.kind in {
            BoundaryKind.TEAM_PREVIEW,
            BoundaryKind.NORMAL_TURN,
            BoundaryKind.FORCED_SWITCH,
            BoundaryKind.PIVOT_SWITCH,
        }:
            if self.decision_type is DecisionType.UNSPECIFIED:
                raise ValueError("Policy windows require a decision type")
        elif self.decision_type is not DecisionType.UNSPECIFIED:
            raise ValueError("Non-policy windows cannot carry a decision type")

    @property
    def is_policy_request(self) -> bool:
        """Return whether this block should produce a policy row."""
        return self.decision_type is not DecisionType.UNSPECIFIED


@dataclass(frozen=True, slots=True)
class DecisionReconstruction:
    """Decision records for one perspective plus all classified windows."""

    replay_id: str
    player: int
    windows: tuple[DecisionWindow, ...]
    decisions: tuple[DecisionRecord, ...]
    diagnostics: tuple[ReplayEventDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.replay_id:
            raise ValueError("DecisionReconstruction.replay_id must not be empty")
        if self.player not in (0, 1):
            raise ValueError("DecisionReconstruction.player must be 0 or 1")
        if any(decision.player != self.player for decision in self.decisions):
            raise ValueError("Decision records must belong to the reconstruction perspective")
        if any(
            left.start_line_index > right.start_line_index
            for left, right in zip(self.windows, self.windows[1:])
        ):
            raise ValueError("Decision windows must be ordered")


def _diagnostic(event: ResolvedProtocolEvent, reason: str) -> ReplayEventDiagnostic:
    parsed = event.event
    return ReplayEventDiagnostic(
        replay_id=parsed.replay_id,
        line_index=parsed.line_index,
        tag=parsed.tag,
        normalized_effect="" if parsed.effect is None else parsed.effect.normalized,
        normalized_cause="" if parsed.cause is None else parsed.cause.normalized,
        raw_line=parsed.raw_line,
        reason=reason,
    )


def _reference(
    event: ResolvedProtocolEvent,
    argument_index: int,
) -> ResolvedPokemonRefArgument | None:
    return next(
        (
            reference
            for reference in event.pokemon_refs
            if reference.argument_index == argument_index
        ),
        None,
    )


def _is_separator(event: ResolvedProtocolEvent) -> bool:
    parsed = event.event
    return parsed.classification is EventClassification.BOUNDARY_SIGNAL and parsed.tag == ""


def _has_policy_action(event: ResolvedProtocolEvent) -> bool:
    parsed = event.event
    if parsed.tag == "cant":
        return True
    if parsed.tag == "move":
        return parsed.cause is None
    if parsed.tag == "switch":
        return parsed.cause is None or parsed.cause.normalized in _PIVOT_CAUSES
    return False


def _has_player_policy_action(
    events: Iterable[ResolvedProtocolEvent],
    perspective: int,
) -> bool:
    return any(
        _has_policy_action(event)
        and (actor := _reference(event, 0)) is not None
        and actor.pokemon_ref.side.side_index == perspective
        for event in events
    )


def _is_pivot(event: ResolvedProtocolEvent) -> bool:
    parsed = event.event
    return (
        parsed.tag == "switch"
        and parsed.cause is not None
        and parsed.cause.normalized in _PIVOT_CAUSES
    )


def _has_action(event: ResolvedProtocolEvent) -> bool:
    return event.event.tag in _ACTION_TAGS


def _update_blocks(
    events: tuple[ResolvedProtocolEvent, ...],
) -> tuple[tuple[int, int], ...]:
    """Split the resolved stream at Showdown's bare update separators."""
    if not events:
        return ()

    blocks: list[tuple[int, int]] = []
    start = 0
    for event in events:
        if not _is_separator(event):
            continue
        if event.event.line_index > start:
            blocks.append((start, event.event.line_index))
        start = event.event.line_index
    if start < len(events):
        blocks.append((start, len(events)))
    return tuple(blocks)


def _classify_block(
    events: tuple[ResolvedProtocolEvent, ...],
    *,
    turn_seen_since_request: bool,
    has_preview: bool,
    policy_request_count: int,
) -> tuple[BoundaryKind, DecisionType, bool]:
    policy_events = tuple(event for event in events if _has_policy_action(event))
    has_pivot = any(_is_pivot(event) for event in policy_events)
    has_turn = any(event.event.tag == "turn" for event in events)

    if not policy_events:
        if any(event.event.tag == "-waiting" for event in events):
            kind = BoundaryKind.WAITING
        elif any(event.event.tag in {"move", "switch", "drag", "replace"} for event in events):
            kind = BoundaryKind.FORCED_EXECUTION
        else:
            kind = BoundaryKind.RESIDUAL_TERMINAL
        return (
            kind,
            DecisionType.UNSPECIFIED,
            turn_seen_since_request or has_turn,
        )

    if has_preview and policy_request_count == 0:
        return BoundaryKind.TEAM_PREVIEW, DecisionType.TEAM_PREVIEW, has_turn
    if has_pivot:
        return BoundaryKind.PIVOT_SWITCH, DecisionType.PIVOT_SWITCH, has_turn
    if turn_seen_since_request:
        return BoundaryKind.NORMAL_TURN, DecisionType.TURN, has_turn
    return BoundaryKind.FORCED_SWITCH, DecisionType.FORCED_SWITCH, has_turn


def infer_decision_windows(
    events: Iterable[ResolvedProtocolEvent],
) -> tuple[DecisionWindow, ...]:
    """Classify every update block without extracting actions or building views."""
    event_tuple = tuple(events)
    if not event_tuple:
        return ()
    if tuple(event.event.line_index for event in event_tuple) != tuple(range(len(event_tuple))):
        raise ValueError("Decision inference requires contiguous resolved event line indices")

    has_separator = any(_is_separator(event) for event in event_tuple)
    has_action = any(_has_action(event) for event in event_tuple)
    if has_action and not has_separator:
        first_action = next(event for event in event_tuple if _has_action(event))
        raise ValueError(
            f"Replay {first_action.event.replay_id!r} has actions but no update separators"
        )

    has_preview = any(event.event.tag == "teampreview" for event in event_tuple)
    turn_seen_since_request = False
    policy_request_count = 0
    windows: list[DecisionWindow] = []
    for start, end in _update_blocks(event_tuple):
        block = event_tuple[start:end]
        kind, decision_type, turn_seen_since_request = _classify_block(
            block,
            turn_seen_since_request=turn_seen_since_request,
            has_preview=has_preview,
            policy_request_count=policy_request_count,
        )
        windows.append(DecisionWindow(start, end, kind, decision_type))
        if decision_type is not DecisionType.UNSPECIFIED:
            policy_request_count += 1
    return tuple(windows)


def _animation_targets(
    events: Iterable[ResolvedProtocolEvent],
) -> dict[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument]:
    targets: dict[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument] = {}
    for event in events:
        if event.event.tag != "-anim":
            continue
        actor = _reference(event, 0)
        target = _reference(event, 2)
        if actor is None or actor.member_id is None or target is None:
            continue
        if len(event.event.arguments) < 2:
            continue
        key = (actor.member_id, normalize_showdown_id(event.event.arguments[1]))
        targets.setdefault(key, target)
    return targets


def _move_states(member: ReplayPokemonState) -> tuple[MoveState, ...]:
    if member.transform is not None:
        return member.transform.moves
    return member.moves


def _target_code(
    actor: ResolvedPokemonRefArgument,
    target: ResolvedPokemonRefArgument | None,
) -> int | None:
    if target is None or target.pokemon_ref.active_slot is None:
        return None
    target_slot = target.pokemon_ref.active_slot
    if actor.pokemon_ref.side is target.pokemon_ref.side:
        return -(target_slot + 1)
    return target_slot + 1


def _observed_move(
    event: ResolvedProtocolEvent,
    state: ReplayBattleState,
    animation_targets: Mapping[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument],
    mega_slots: set[tuple[ReplaySide, int]],
) -> ObservedAction:
    actor = _reference(event, 0)
    if actor is None or actor.member_id is None:
        return ObservedAction(tag="move_slot_or_target_unknown")
    member = state.member(actor.member_id)
    move_id = normalize_showdown_id(event.event.arguments[1])
    forced = move_id in _FORCED_MOVES
    slot = actor.pokemon_ref.active_slot
    mega = slot is not None and (actor.pokemon_ref.side, slot) in mega_slots
    if forced:
        action = MEGA_FORCED_ACTION if mega else FORCED_ACTION
        return ObservedAction(action, tag="forced_move")

    move_states = _move_states(member)
    move_slot = next(
        (index for index, move in enumerate(move_states) if move.move_id == move_id),
        None,
    )
    if move_slot is None:
        return ObservedAction(tag="move_slot_or_target_unknown")

    move_target = _effective_move_target(member, move_states[move_slot])
    target = _reference(event, 2)
    if target is None:
        target = animation_targets.get((actor.member_id, move_id))
        target_tag = "move_anim_target"
    else:
        target_tag = "move"
    allowed_targets = _target_codes(move_target, slot or 0)
    target_code = 0 if move_target in _SELF_TARGETS else _target_code(actor, target)
    if target_code is not None and target_code not in allowed_targets:
        target_code = None
    if target_code is None:
        return ObservedAction(tag="move_slot_or_target_unknown")

    action = encode_action(
        SlotAction(ActionKind.MOVE, move_slot=move_slot, target=target_code, mega=mega)
    )
    if target_code == 0:
        return ObservedAction(action, tag=target_tag)

    alternatives = tuple(
        encode_action(SlotAction(ActionKind.MOVE, move_slot=move_slot, target=value, mega=mega))
        for value in allowed_targets
    )
    return ObservedAction(
        action,
        alternatives=alternatives,
        exact=False,
        tag="execution_target",
    )


def _observed_actions(
    events: tuple[ResolvedProtocolEvent, ...],
    state: ReplayBattleState,
    perspective: int,
    animation_targets: Mapping[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument],
) -> tuple[ObservedAction | None, ObservedAction | None, tuple[str, ...]]:
    observed: list[ObservedAction | None] = [None, None]
    tags: list[str] = []
    mega_slots: set[tuple[ReplaySide, int]] = set()
    generated_slots: set[tuple[ReplaySide, int]] = set()
    for event in events:
        parsed = event.event
        if parsed.tag == "-mega":
            actor = _reference(event, 0)
            if actor is not None and actor.member_id is not None:
                slot = actor.pokemon_ref.active_slot
                if slot is not None:
                    mega_slots.add((actor.pokemon_ref.side, slot))
        elif parsed.tag == "-singleturn" and any(
            normalize_showdown_id(argument).startswith("moveinstruct")
            for argument in parsed.arguments
        ):
            actor = _reference(event, 0)
            if actor is not None and actor.member_id is not None:
                slot = actor.pokemon_ref.active_slot
                if slot is not None:
                    generated_slots.add((actor.pokemon_ref.side, slot))

        if parsed.tag not in _ACTION_TAGS:
            continue
        actor = _reference(event, 0)
        if actor is None or actor.member_id is None or actor.pokemon_ref.active_slot is None:
            continue
        if actor.pokemon_ref.side.side_index != perspective:
            continue
        slot = actor.pokemon_ref.active_slot
        key = (actor.pokemon_ref.side, slot)

        if parsed.tag == "move":
            if key in generated_slots:
                generated_slots.remove(key)
                tags.append("externally_generated_move")
                continue
            if parsed.cause is not None:
                tags.append("externally_generated_move")
                continue
            if observed[slot] is not None:
                observed[slot] = ObservedAction(tag="multiple_moves_same_slot", exact=False)
                tags.append("multiple_moves_same_slot")
                continue
            action = _observed_move(event, state, animation_targets, mega_slots)
            observed[slot] = action
            if action.tag and action.tag != "move":
                tags.append(action.tag)
        elif parsed.tag == "switch" and (
            parsed.cause is None or parsed.cause.normalized in _PIVOT_CAUSES
        ):
            incoming = actor.member_id
            observed[slot] = ObservedAction(
                SWITCH_START + incoming.roster_index,
                tag="switch",
            )
        elif parsed.tag == "cant":
            tags.append("no_observed_order")

    return observed[0], observed[1], tuple(dict.fromkeys(tags))


def _preview_actions(
    events: tuple[ResolvedProtocolEvent, ...],
    ots: OTSData,
    final_state: ReplayBattleState,
    perspective: int,
) -> tuple[ObservedAction | None, ObservedAction | None, tuple[str, ...]]:
    leads = tuple(
        dict.fromkeys(
            reference.member_id
            for event in events
            if event.event.tag == "switch"
            and (reference := _reference(event, 0)) is not None
            and reference.member_id is not None
            and reference.member_id.side.side_index == perspective
            and event.event.cause is None
        )
    )
    if len(leads) != 2:
        return None, None, ("preview_leads_unknown",)

    team_size = len(ots.members)
    lead_indices = tuple(member.roster_index for member in leads)
    if len(set(lead_indices)) != 2:
        return None, None, ("preview_duplicate_lead",)
    first = encode_team_pair(lead_indices[0], lead_indices[1], team_size=team_size)

    selected = tuple(
        member.member_id.roster_index
        for member in final_state.members
        if member.member_id.side.side_index == perspective
        and member.selected is True
        and member.member_id.roster_index not in lead_indices
    )
    if len(selected) == 2:
        alternatives = (
            encode_team_pair(selected[0], selected[1], team_size=team_size),
            encode_team_pair(selected[1], selected[0], team_size=team_size),
        )
        return (
            ObservedAction(first, tag="preview_leads"),
            ObservedAction(
                alternatives=alternatives,
                exact=False,
                tag="preview_reserves_revealed",
            ),
            ("preview_reserves_revealed",),
        )

    alternatives = tuple(
        encode_team_pair(first_index, second_index, team_size=team_size)
        for first_index in range(team_size)
        for second_index in range(team_size)
        if first_index != second_index
        and first_index not in lead_indices
        and second_index not in lead_indices
    )
    return (
        ObservedAction(first, tag="preview_leads"),
        ObservedAction(alternatives=alternatives, exact=False, tag="preview_reserves_unknown"),
        ("preview_reserves_unknown",),
    )


def _mega_rules(dex: Mapping[str, Any]) -> _MegaRules:
    species_by_item: dict[str, set[str]] = {}
    for entry in dex.get("items", ()):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("megaStone"), Mapping):
            continue
        item = normalize_showdown_id(str(entry.get("id", entry.get("name", ""))))
        species_by_item[item] = {
            normalize_showdown_id(str(species)) for species in entry["megaStone"]
        }
    z_items = frozenset(
        normalize_showdown_id(str(entry.get("id", entry.get("name", ""))))
        for entry in dex.get("items", ())
        if isinstance(entry, Mapping) and entry.get("zMove")
    )
    return _MegaRules(
        {item: frozenset(species) for item, species in species_by_item.items()},
        z_items,
    )


def _can_mega(member: ReplayPokemonState, rules: _MegaRules, used: bool) -> bool:
    if used:
        return False
    item = normalize_showdown_id(member.item or "")
    species = normalize_showdown_id(member.current_form)
    return species in rules.species_by_item.get(item, ()) or (
        species == "rayquaza"
        and "dragonascent" in {move.move_id for move in member.moves}
        and item not in rules.z_items
    )


def build_decision_view(
    state: ReplayBattleState,
    ots: OTSData,
    perspective: int,
    window_events: tuple[ResolvedProtocolEvent, ...],
    *,
    preview: bool,
    dex: Mapping[str, Any],
    mega_rules: _MegaRules | None = None,
) -> DecisionView:
    roster_size = len(ots.members)
    if preview:
        return DecisionView(
            slots=(SlotDecision(), SlotDecision()),
            team_preview=True,
            team_size=roster_size,
        )

    own_side = state.sides[perspective]
    rules = _mega_rules(dex) if mega_rules is None else mega_rules
    active = own_side.active
    active_set = frozenset(member_id for member_id in active if member_id is not None)
    selected_count = sum(
        member.selected is True
        for member in state.members
        if member.member_id.side.side_index == perspective
    )
    selection_complete = selected_count >= state.team_sizes[perspective]
    switches = tuple(
        member.member_id.roster_index
        for member in state.members
        if member.member_id.side.side_index == perspective
        and member.member_id not in active_set
        and not member.fainted
        and (not selection_complete or member.selected is True)
    )
    forced_slots = frozenset(
        reference.pokemon_ref.active_slot
        for event in window_events
        if event.event.tag == "move"
        and len(event.event.arguments) >= 2
        and normalize_showdown_id(event.event.arguments[1]) in _FORCED_MOVES
        and (reference := _reference(event, 0)) is not None
        and reference.member_id is not None
        and reference.member_id.side.side_index == perspective
        and reference.pokemon_ref.active_slot is not None
    )
    slots: list[SlotDecision] = []
    for slot, member_id in enumerate(active):
        member = None if member_id is None else state.member(member_id)
        moves = (
            ()
            if member is None
            else tuple(
                _target_codes(_effective_move_target(member, move), slot)
                for move in _move_states(member)
            )
        )
        can_mega = bool(member is not None and _can_mega(member, rules, own_side.used_mega))
        slots.append(
            SlotDecision(
                switch_slots=switches,
                move_targets=moves,
                active=member is not None,
                force_switch=member is None and bool(switches),
                can_mega=can_mega,
                forced_move=slot in forced_slots,
                legality_known=False,
            )
        )
    return DecisionView(
        slots=(slots[0], slots[1]),
        team_size=roster_size,
    )


def _pre_state(
    snapshots: tuple[ReplayBattleState, ...],
    start_line_index: int,
) -> ReplayBattleState | None:
    if start_line_index == 0:
        return None
    return snapshots[start_line_index - 1]


def _decision_for_window(
    window: DecisionWindow,
    events: tuple[ResolvedProtocolEvent, ...],
    snapshots: tuple[ReplayBattleState, ...],
    document: ReplayDocument,
    perspective: int,
    max_candidates: int,
    animation_targets: Mapping[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument],
    dex: Mapping[str, Any],
    mega_rules: _MegaRules,
) -> DecisionRecord | None:
    window_events = events[window.start_line_index : window.end_line_index]
    if window.decision_type in {
        DecisionType.FORCED_SWITCH,
        DecisionType.PIVOT_SWITCH,
    } and not _has_player_policy_action(window_events, perspective):
        return None
    pre_state = _pre_state(snapshots, window.start_line_index)
    if window.kind is BoundaryKind.TEAM_PREVIEW:
        final_state = snapshots[-1]
        observed = _preview_actions(
            window_events, document.ots[perspective], final_state, perspective
        )
        view = build_decision_view(
            final_state,
            document.ots[perspective],
            perspective,
            window_events,
            preview=True,
            dex=dex,
            mega_rules=mega_rules,
        )
        tags = observed[2]
        unknown = not any(action is not None for action in observed[:2])
    else:
        if pre_state is None:
            raise ValueError("Policy decision window has no pre-decision state")
        observed = _observed_actions(window_events, pre_state, perspective, animation_targets)
        view = build_decision_view(
            pre_state,
            document.ots[perspective],
            perspective,
            window_events,
            preview=False,
            dex=dex,
            mega_rules=mega_rules,
        )
        observed_slots = list(observed[:2])
        tags = observed[2]
        if window.decision_type is DecisionType.FORCED_SWITCH and any(
            action is not None for action in observed_slots
        ):
            for slot, action in enumerate(observed_slots):
                if action is None:
                    observed_slots[slot] = ObservedAction(PASS_ACTION, tag="implicit_pass")
            observed = (observed_slots[0], observed_slots[1], tags)
            tags = tuple(dict.fromkeys((*tags, "implicit_pass")))
        unknown = not any(action is not None for action in observed[:2])

    evidence = extract_action_evidence(
        EvidenceRequest(
            view=view,
            slots=(observed[0], observed[1]),
            tags=tags,
            max_candidates=max_candidates,
            unknown=unknown,
        )
    )
    return DecisionRecord(
        decision_index=0,
        player=perspective,
        decision_type=window.decision_type,
        pre_line_index=window.start_line_index,
        post_line_index=window.end_line_index,
        evidence=evidence,
    )


def reconstruct_decisions_from_trace(
    document: ReplayDocument,
    events: Iterable[ResolvedProtocolEvent],
    state: ReconstructedReplayState,
    *,
    perspective: int,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
    windows: tuple[DecisionWindow, ...] | None = None,
) -> DecisionReconstruction:
    """
    Build decisions from an already resolved and reduced replay trace.

    Arguments:
        document: The normalized replay document containing OTS and protocol lines.
        events: The complete resolved event stream for the document.
        state: Accepted immutable state snapshots for the same event stream.
        perspective: The player index to reconstruct, either 0 or 1.
        max_candidates: Maximum joint-action candidates to retain per decision.
        dex: Optional runtime dex used for mega legality metadata.
        windows: Optional shared boundary classification for both perspectives.

    Returns:
        A decision reconstruction containing shared windows and player-owned records.
    """
    if perspective not in (0, 1):
        raise ValueError("perspective must be 0 or 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if state.diagnostics:
        return DecisionReconstruction(
            document.metadata.replay_id,
            perspective,
            (),
            (),
            state.diagnostics,
        )
    event_tuple = tuple(events)
    snapshots = state.require_accepted()
    if len(event_tuple) != len(snapshots):
        raise ValueError("Resolved events and state snapshots must have equal lengths")
    if tuple(event.event.line_index for event in event_tuple) != tuple(
        snapshot.line_index for snapshot in snapshots
    ):
        raise ValueError("Resolved events and state snapshots must have matching line indices")
    if any(event.event.replay_id != document.metadata.replay_id for event in event_tuple):
        raise ValueError("Decision events must belong to the document replay")

    runtime_dex = default_runtime_resources().dex if dex is None else dex
    animation_targets = _animation_targets(event_tuple)
    mega_rules = _mega_rules(runtime_dex)
    if windows is None:
        try:
            windows = infer_decision_windows(event_tuple)
        except ValueError as exc:
            first_event = event_tuple[0]
            diagnostic = _diagnostic(first_event, str(exc))
            return DecisionReconstruction(
                document.metadata.replay_id,
                perspective,
                (),
                (),
                (diagnostic,),
            )

    records: list[DecisionRecord] = []
    for window in windows:
        if not window.is_policy_request:
            continue
        record = _decision_for_window(
            window,
            event_tuple,
            snapshots,
            document,
            perspective,
            max_candidates,
            animation_targets,
            runtime_dex,
            mega_rules,
        )
        if record is not None:
            records.append(record)

    decisions = tuple(
        DecisionRecord(
            decision_index=index,
            player=record.player,
            decision_type=record.decision_type,
            pre_line_index=record.pre_line_index,
            post_line_index=record.post_line_index,
            evidence=record.evidence,
        )
        for index, record in enumerate(records)
    )
    return DecisionReconstruction(
        document.metadata.replay_id,
        perspective,
        windows,
        decisions,
    )


def reconstruct_replay_decisions(
    document: ReplayDocument,
    *,
    perspective: int,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
) -> DecisionReconstruction:
    """Resolve and reconstruct one replay perspective."""
    return _reconstruct_perspectives(
        document,
        (perspective,),
        max_candidates=max_candidates,
        dex=dex,
    )[0]


def reconstruct_replay_decisions_both(
    document: ReplayDocument,
    *,
    max_candidates: int = 256,
    dex: Mapping[str, Any] | None = None,
) -> tuple[DecisionReconstruction, DecisionReconstruction]:
    """Resolve and reduce once, then reconstruct both player perspectives."""
    results = _reconstruct_perspectives(
        document,
        (0, 1),
        max_candidates=max_candidates,
        dex=dex,
    )
    return results[0], results[1]


def _reconstruct_perspectives(
    document: ReplayDocument,
    perspectives: tuple[int, ...],
    *,
    max_candidates: int,
    dex: Mapping[str, Any] | None,
) -> tuple[DecisionReconstruction, ...]:
    if not perspectives or any(perspective not in (0, 1) for perspective in perspectives):
        raise ValueError("perspective must be 0 or 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    runtime_dex = default_runtime_resources().dex if dex is None else dex
    resolved = resolve_replay_events(document, dex=runtime_dex)
    if resolved.diagnostics:
        return tuple(
            DecisionReconstruction(
                document.metadata.replay_id,
                perspective,
                (),
                (),
                resolved.diagnostics,
            )
            for perspective in perspectives
        )
    state = reduce_replay_state(
        document.metadata.replay_id,
        document.ots,
        resolved.events,
        dex=runtime_dex,
    )

    results: list[DecisionReconstruction] = []
    windows: tuple[DecisionWindow, ...] | None = None
    for perspective in perspectives:
        result = reconstruct_decisions_from_trace(
            document,
            resolved.events,
            state,
            perspective=perspective,
            max_candidates=max_candidates,
            dex=runtime_dex,
            windows=windows,
        )
        results.append(result)
        if result.diagnostics:
            return tuple(
                DecisionReconstruction(
                    document.metadata.replay_id,
                    value,
                    (),
                    (),
                    result.diagnostics,
                )
                for value in perspectives
            )
        windows = result.windows
    return tuple(results)


__all__ = [
    "BoundaryKind",
    "DecisionReconstruction",
    "DecisionWindow",
    "build_decision_view",
    "infer_decision_windows",
    "reconstruct_decisions_from_trace",
    "reconstruct_replay_decisions",
    "reconstruct_replay_decisions_both",
]
