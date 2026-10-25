"""Infer replay decisions from resolved events and immutable state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple

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
from p0.replays.identity import ReplayMemberId, normalize_showdown_id
from p0.replays.protocol import ReplayDocument
from p0.replays.reconstruction.contract import PIVOT_MOVES, REVIVAL_BLESSING
from p0.replays.reconstruction.events import EventClassification
from p0.replays.reconstruction.resolution import (
    ResolvedPokemonRefArgument,
    ResolvedProtocolEvent,
)
from p0.replays.reconstruction.state import (
    MoveState,
    ReconstructedReplayState,
    ReplayBattleState,
    ReplayPokemonState,
)
from p0.replays.schema import DecisionRecord, DecisionType, OTSData

_ACTION_TAGS = frozenset({"move", "switch", "cant"})
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
_NORMAL_TARGETS = frozenset({"normal", "any"})


def _opponent_imprison_active(state: ReplayBattleState, perspective: int) -> bool:
    """
    Whether an opposing Imprison hides disabled moves from the player's request.

    The player then submits a move that Showdown replaces with Struggle, so an
    observed Struggle does not show the submitted choice.
    """
    return any(
        member_id is not None
        and any(effect == "imprison" for effect, _ in state.member(member_id).effects)
        for member_id in state.sides[1 - perspective].active
    )


def _target_codes(move_target: str, actor_slot: int) -> tuple[int, ...]:
    ally, own = (-2, -1) if actor_slot == 0 else (-1, -2)
    if move_target == "adjacentally":
        return (ally,)
    if move_target == "adjacentallyorself":
        # Live orders name the user by its own slot, like poke-env: ally first, then self.
        return (ally, own)
    if move_target in _NORMAL_TARGETS:
        return (ally, 1, 2)
    return (0,)


def _effective_move_target(member: ReplayPokemonState, move: MoveState) -> str:
    if move.move_id == "curse":
        return (
            "any"
            if any(normalize_showdown_id(value) == "ghost" for value in member.current_types)
            else "self"
        )
    return move.target.casefold()


@dataclass(frozen=True, slots=True)
class DecisionWindow:
    """One ordered protocol update block; decision_type is UNSPECIFIED when no player chose."""

    start_line_index: int
    end_line_index: int
    decision_type: DecisionType
    between_turns: bool

    @property
    def is_policy_request(self) -> bool:
        """Return whether this block should produce a policy row."""
        return self.decision_type is not DecisionType.UNSPECIFIED


class DecisionReconstruction(NamedTuple):
    """Decision records and their observation-facing requests for one perspective."""

    decisions: tuple[DecisionRecord, ...]
    # The request each decision answered, in the same order.
    views: tuple[DecisionView, ...]


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


def _is_revive(event: ResolvedProtocolEvent) -> bool:
    """A revive heal is the answer to the switch request Revival Blessing opens."""
    parsed = event.event
    return (
        parsed.tag == "-heal"
        and parsed.cause is not None
        and parsed.cause.normalized == REVIVAL_BLESSING
    )


def _reviving_slot(state: ReplayBattleState, perspective: int) -> int:
    """Return the slot of the active Pokemon whose Revival Blessing opened the request."""
    for slot, member_id in enumerate(state.sides[perspective].active):
        if member_id is not None and state.member(member_id).last_move == REVIVAL_BLESSING:
            return slot
    raise ValueError(
        f"Replay {state.replay_id!r} has a revive with no active Revival Blessing user"
    )


def _has_policy_action(event: ResolvedProtocolEvent) -> bool:
    parsed = event.event
    if parsed.tag == "cant" or _is_revive(event):
        return True
    if parsed.tag == "move":
        return parsed.cause is None or _is_locked_move(event)
    if parsed.tag == "switch":
        return parsed.cause is None or parsed.cause.normalized in PIVOT_MOVES
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
        and parsed.cause.normalized in PIVOT_MOVES
    )


def _update_blocks(
    events: tuple[ResolvedProtocolEvent, ...],
) -> tuple[tuple[int, int], ...]:
    """Split the resolved stream at Showdown's bare update separators."""
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
) -> tuple[DecisionType, bool]:
    policy_events = tuple(event for event in events if _has_policy_action(event))
    has_turn = any(event.event.tag == "turn" for event in events)

    if not policy_events:
        return DecisionType.UNSPECIFIED, turn_seen_since_request or has_turn
    if has_preview and policy_request_count == 0:
        return DecisionType.TEAM_PREVIEW, has_turn
    if any(_is_pivot(event) for event in policy_events):
        return DecisionType.PIVOT_SWITCH, has_turn
    if turn_seen_since_request:
        return DecisionType.TURN, has_turn
    return DecisionType.FORCED_SWITCH, has_turn


def infer_decision_windows(
    events: Iterable[ResolvedProtocolEvent],
) -> tuple[DecisionWindow, ...]:
    """Classify every update block without extracting actions or building views."""
    event_tuple = tuple(events)
    if not event_tuple:
        return ()
    first_action = next((event for event in event_tuple if event.event.tag in _ACTION_TAGS), None)
    if first_action is not None and not any(_is_separator(event) for event in event_tuple):
        raise ValueError(
            f"Replay {first_action.event.replay_id!r} has actions but no update separators"
        )

    has_preview = any(event.event.tag == "teampreview" for event in event_tuple)
    turn_seen_since_request = False
    policy_request_count = 0
    between_turns = True
    windows: list[DecisionWindow] = []
    for start, end in _update_blocks(event_tuple):
        block = event_tuple[start:end]
        decision_type, turn_seen_since_request = _classify_block(
            block,
            turn_seen_since_request=turn_seen_since_request,
            has_preview=has_preview,
            policy_request_count=policy_request_count,
        )
        windows.append(DecisionWindow(start, end, decision_type, between_turns))
        for event in reversed(block):
            if event.event.tag in {"turn", "upkeep"}:
                between_turns = event.event.tag == "upkeep"
                break
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
        key = (actor.member_id, normalize_showdown_id(event.event.arguments[1]))
        targets.setdefault(key, target)
    return targets


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


def _is_locked_move(event: ResolvedProtocolEvent) -> bool:
    """A locked Pokemon is given no choice, so its request is a forced move."""
    cause = event.event.cause
    return cause is not None and cause.normalized == "lockedmove"


def _request_slot(
    reference: ResolvedPokemonRefArgument, line_slot: int, state: ReplayBattleState
) -> int:
    """Return the slot the Pokemon held at the request; Ally Switch can move it before it acts."""
    active = state.sides[reference.pokemon_ref.side.side_index].active
    return active.index(reference.member_id) if reference.member_id in active else line_slot


def _observed_move(
    event: ResolvedProtocolEvent,
    state: ReplayBattleState,
    animation_targets: Mapping[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument],
    mega_slots: set[int],
    slot: int,
) -> ObservedAction:
    actor = _reference(event, 0)
    if actor is None or actor.member_id is None:
        return ObservedAction(tag="move_slot_or_target_unknown")
    member = state.member(actor.member_id)
    move_id = normalize_showdown_id(event.event.arguments[1])
    mega = slot in mega_slots
    if move_id == "struggle":
        action = MEGA_FORCED_ACTION if mega else FORCED_ACTION
        return ObservedAction(action, tag="forced_move")

    move_slot = next(
        (index for index, move in enumerate(member.moves) if move.move_id == move_id),
        None,
    )
    if move_slot is None:
        return ObservedAction(tag="move_slot_or_target_unknown")

    move_target = _effective_move_target(member, member.moves[move_slot])
    target = _reference(event, 2)
    if target is None:
        target = animation_targets.get((actor.member_id, move_id))
        target_tag = "move_anim_target"
    else:
        target_tag = "move"
    allowed_targets = _target_codes(move_target, slot)
    target_code = 0 if move_target in _SELF_TARGETS else _target_code(actor, target)
    if target_code is not None and target_code < 0 and slot != actor.pokemon_ref.active_slot:
        # The line names own-side slots after an Ally Switch swap; the order named them before.
        target_code = -3 - target_code
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
    mega_slots: set[int] = set()
    generated_slots: set[int] = set()
    encored_before_action: set[int] = set()
    hidden_disables = _opponent_imprison_active(state, perspective)
    for event in events:
        parsed = event.event
        actor = _reference(event, 0)
        if (
            actor is None
            or actor.member_id is None
            or actor.pokemon_ref.side.side_index != perspective
        ):
            continue
        if _is_revive(event):
            # Only the user's slot answers a revive request. Later lines resume the turn.
            observed[_reviving_slot(state, perspective)] = ObservedAction(
                SWITCH_START + actor.member_id.roster_index,
                tag="revive",
            )
            break
        slot = actor.pokemon_ref.active_slot
        if slot is None:
            continue
        if (
            parsed.tag == "-start"
            and normalize_showdown_id(parsed.arguments[1]) == "encore"
            and observed[slot] is None
        ):
            encored_before_action.add(slot)
        if parsed.tag == "-mega" or (
            parsed.tag == "-singleturn"
            and any(
                normalize_showdown_id(argument).startswith("moveinstruct")
                for argument in parsed.arguments
            )
        ):
            marked_slots = mega_slots if parsed.tag == "-mega" else generated_slots
            marked_slots.add(slot)

        if parsed.tag == "move":
            if slot in encored_before_action:
                tags.append("mid_turn_encore_override")
                continue
            if hidden_disables and normalize_showdown_id(parsed.arguments[1]) == "struggle":
                tags.append("hidden_disable_struggle")
                continue
            if slot in generated_slots:
                generated_slots.remove(slot)
                tags.append("externally_generated_move")
                continue
            slot = _request_slot(actor, slot, state)
            if _is_locked_move(event):
                observed[slot] = ObservedAction(FORCED_ACTION, tag="forced_move")
                tags.append("forced_move")
                continue
            if parsed.cause is not None:
                tags.append("externally_generated_move")
                continue
            if observed[slot] is not None:
                observed[slot] = ObservedAction(tag="multiple_moves_same_slot", exact=False)
                tags.append("multiple_moves_same_slot")
                continue
            action = _observed_move(event, state, animation_targets, mega_slots, slot)
            observed[slot] = action
            if action.tag and action.tag != "move":
                tags.append(action.tag)
        elif parsed.tag == "switch" and (
            parsed.cause is None or parsed.cause.normalized in PIVOT_MOVES
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


def _can_mega(member: ReplayPokemonState, used: bool) -> bool:
    item = normalize_showdown_id(member.item or "")
    species = normalize_showdown_id(member.current_form)
    return not used and species in default_runtime_resources().mega_species_by_item.get(item, ())


def build_decision_view(
    state: ReplayBattleState,
    ots: OTSData,
    perspective: int,
    window_events: tuple[ResolvedProtocolEvent, ...],
    *,
    preview: bool,
    replacement: bool = False,
    between_turns: bool = True,
) -> DecisionView:
    """
    Build the action choices visible before a replay decision.

    Arguments:
      state: Public battle state before the request.
      ots: Open team sheet for the acting player.
      perspective: Acting side index.
      window_events: Events resolving this request.
      preview: Whether the request chooses a team.
      replacement: Whether the request replaces active Pokemon instead of choosing moves.
      between_turns: Whether upkeep has completed and the next turn has not started.

    Returns:
      The two slots' moves, switches, and forced actions.
    """
    roster_size = len(ots.members)
    if preview:
        return DecisionView(
            slots=(SlotDecision(), SlotDecision()),
            team_preview=True,
            team_size=roster_size,
        )

    own_side = state.sides[perspective]
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
    # A revive request offers the fainted members to the Revival Blessing user's slot.
    revive_slot = (
        _reviving_slot(state, perspective)
        if any(
            _is_revive(event)
            and (target := _reference(event, 0)) is not None
            and target.pokemon_ref.side.side_index == perspective
            for event in window_events
        )
        else None
    )
    fainted = (
        ()
        if revive_slot is None
        else tuple(
            member.member_id.roster_index
            for member in state.members
            if member.member_id.side.side_index == perspective and member.fainted
        )
    )
    replacement_slots = frozenset(
        slot
        for event in (window_events if replacement else ())
        if event.event.tag == "switch"
        and (reference := _reference(event, 0)) is not None
        and reference.pokemon_ref.side.side_index == perspective
        and (slot := reference.pokemon_ref.active_slot) is not None
    )
    # Struggle is hidden from the request while an opposing Imprison disables moves.
    struggle_is_forced = not _opponent_imprison_active(state, perspective)
    forced_slots = frozenset(
        _request_slot(reference, line_slot, state)
        for event in window_events
        if event.event.tag == "move"
        and (
            _is_locked_move(event)
            or (
                struggle_is_forced and normalize_showdown_id(event.event.arguments[1]) == "struggle"
            )
        )
        and (reference := _reference(event, 0)) is not None
        and reference.member_id is not None
        and reference.member_id.side.side_index == perspective
        and (line_slot := reference.pokemon_ref.active_slot) is not None
    )
    slots: list[SlotDecision] = []
    for slot, member_id in enumerate(active):
        member = None if member_id is None else state.member(member_id)
        moves = (
            ()
            if member is None
            else tuple(
                _target_codes(_effective_move_target(member, move), slot) for move in member.moves
            )
        )
        can_mega = member is not None and _can_mega(member, own_side.used_mega)
        force_switch = member is None and bool(switches)
        if revive_slot is not None:
            force_switch = slot == revive_slot
        elif replacement:
            # Mid-turn replacements leave other fainted slots until upkeep finishes.
            force_switch = slot in replacement_slots or (between_turns and force_switch)
        slots.append(
            SlotDecision(
                switch_slots=fainted if slot == revive_slot else switches,
                move_targets=moves,
                active=member is not None,
                force_switch=force_switch,
                can_mega=can_mega,
                forced_move=slot in forced_slots,
                legality_known=False,
            )
        )
    return DecisionView(
        slots=(slots[0], slots[1]),
        team_size=roster_size,
    )


def _decision_for_window(
    window: DecisionWindow,
    events: tuple[ResolvedProtocolEvent, ...],
    snapshots: Mapping[int, ReplayBattleState],
    document: ReplayDocument,
    perspective: int,
    max_candidates: int,
    animation_targets: Mapping[tuple[ReplayMemberId, str], ResolvedPokemonRefArgument],
    decision_index: int,
) -> tuple[DecisionRecord, DecisionView] | None:
    window_events = events[window.start_line_index : window.end_line_index]
    replacement = window.decision_type in {
        DecisionType.FORCED_SWITCH,
        DecisionType.PIVOT_SWITCH,
    }
    if replacement:
        # Queued moves resume after a replacement; they answer the earlier turn request.
        window_events = tuple(
            event for event in window_events if event.event.tag == "switch" or _is_revive(event)
        )
        if not _has_player_policy_action(window_events, perspective):
            return None
    if window.decision_type is DecisionType.TEAM_PREVIEW:
        final_state = snapshots[len(events) - 1]
        observed = _preview_actions(
            window_events, document.ots[perspective], final_state, perspective
        )
        view = build_decision_view(
            final_state,
            document.ots[perspective],
            perspective,
            window_events,
            preview=True,
        )
    else:
        pre_state = snapshots[window.start_line_index - 1]
        observed = _observed_actions(window_events, pre_state, perspective, animation_targets)
        view = build_decision_view(
            pre_state,
            document.ots[perspective],
            perspective,
            window_events,
            preview=False,
            replacement=replacement,
            between_turns=window.between_turns,
        )
        if window.decision_type is DecisionType.FORCED_SWITCH and any(
            action is not None for action in observed[:2]
        ):
            implicit_pass = ObservedAction(PASS_ACTION, tag="implicit_pass")
            observed = (
                observed[0] or implicit_pass,
                observed[1] or implicit_pass,
                tuple(dict.fromkeys((*observed[2], "implicit_pass"))),
            )

    first, second, tags = observed
    unknown = first is None and second is None
    evidence = extract_action_evidence(
        EvidenceRequest(
            view=view,
            slots=(first, second),
            tags=tags,
            max_candidates=max_candidates,
            unknown=unknown,
        )
    )
    record = DecisionRecord(
        decision_index=decision_index,
        player=perspective,
        decision_type=window.decision_type,
        pre_line_index=window.start_line_index,
        post_line_index=window.end_line_index,
        evidence=evidence,
    )
    return record, view


def reconstruct_decisions_from_trace(
    document: ReplayDocument,
    events: Iterable[ResolvedProtocolEvent],
    state: ReconstructedReplayState,
    *,
    perspective: int,
    max_candidates: int = 256,
    windows: tuple[DecisionWindow, ...],
) -> DecisionReconstruction:
    """
    Build decisions from an already resolved and reduced replay trace.

    Arguments:
        document: The normalized replay document containing OTS and protocol lines.
        events: The complete resolved event stream for the document.
        state: Accepted immutable state snapshots for the same event stream.
        perspective: The player index to reconstruct, either 0 or 1.
        max_candidates: Maximum joint-action candidates to retain per decision.
        windows: Boundary classification from infer_decision_windows, shared by both perspectives.

    Returns:
        A decision reconstruction containing shared windows and player-owned records.
    """
    event_tuple = tuple(events)
    snapshots_by_line = {snapshot.line_index: snapshot for snapshot in state.snapshots}
    animation_targets = _animation_targets(event_tuple)
    records: list[DecisionRecord] = []
    views: list[DecisionView] = []
    for window in windows:
        if not window.is_policy_request:
            continue
        decision = _decision_for_window(
            window,
            event_tuple,
            snapshots_by_line,
            document,
            perspective,
            max_candidates,
            animation_targets,
            len(records),
        )
        if decision is not None:
            records.append(decision[0])
            views.append(decision[1])

    return DecisionReconstruction(tuple(records), tuple(views))


__all__ = [
    "DecisionReconstruction",
    "DecisionWindow",
    "build_decision_view",
    "infer_decision_windows",
    "reconstruct_decisions_from_trace",
]
