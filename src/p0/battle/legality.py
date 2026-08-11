"""Pure scalar legality and joint-action constraints.

A SlotDecision carries both the legality itself and whether that legality is proven.
Live play and self-play own the authoritative |request| and leave legality_known
set; a public replay can only prove part of it, and marks the rest unknown. An unknown
slot must yield a superset mask - every structurally possible action stays selectable -
so the training denominator never excludes the action the demonstrator actually took.
The observation encoder gates the same flag so an unproven mask is never read as fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from p0.battle.actions import (
    ACT_SIZE,
    FORCED_ACTION,
    MEGA_FORCED_ACTION,
    MEGA_MOVE_END,
    MEGA_MOVE_START,
    MOVE_END,
    MOVE_START,
    PASS_ACTION,
    SWITCH_END,
    SWITCH_START,
    decode_team_pair,
)


@dataclass(frozen=True, slots=True)
class SlotDecision:
    switch_slots: tuple[int, ...] = ()
    move_targets: tuple[tuple[int, ...], ...] = ()
    active: bool = True
    trapped: bool = False
    force_switch: bool = False
    can_mega: bool = False
    forced_move: bool = False
    legality_known: bool = True


@dataclass(frozen=True, slots=True)
class DecisionView:
    slots: tuple[SlotDecision, SlotDecision]
    wait: bool = False
    team_preview: bool = False
    team_size: int = 6


_TEAM_PREVIEW_CACHE: dict[int, tuple[int, ...]] = {}


def _get_team_preview(team_size: int) -> tuple[int, ...]:
    """Ordered distinct team-preview pair action IDs, in ascending roster order."""
    if team_size not in _TEAM_PREVIEW_CACHE:
        _TEAM_PREVIEW_CACHE[team_size] = tuple(
            first * team_size + second
            for first in range(team_size)
            for second in range(team_size)
            if first != second
        )

    return _TEAM_PREVIEW_CACHE[team_size]


def legal_actions(view: DecisionView, position: int) -> tuple[int, ...]:
    """Compute the legal action IDs for a single slot position given decision view."""
    if view.team_preview:
        return _get_team_preview(view.team_size)

    slot = view.slots[position]
    any_force = view.slots[0].force_switch or view.slots[1].force_switch

    if view.wait or (any_force and not slot.force_switch):
        return (PASS_ACTION,)

    switches = () if slot.trapped else tuple(SWITCH_START + index for index in slot.switch_slots)

    if view.slots[0].force_switch and view.slots[1].force_switch and len(switches) == 1:
        return (*switches, PASS_ACTION)

    if not slot.active:
        return switches or (PASS_ACTION,)

    if slot.forced_move:
        moves = (FORCED_ACTION,)
        mega_moves = (MEGA_FORCED_ACTION,) if slot.can_mega else ()
    else:
        moves = tuple(
            MOVE_START + move_slot * 5 + target + 2
            for move_slot, targets in enumerate(slot.move_targets)
            for target in targets
        )
        mega_moves = (
            tuple(action + (MOVE_END - MOVE_START) for action in moves) if slot.can_mega else ()
        )

        # An unproven slot may be locked into a move we cannot see (Outrage, Encore,
        # a recharge turn), which the runtime encodes as the single forced action.
        if not slot.legality_known:
            moves = (*moves, FORCED_ACTION)
            if slot.can_mega:
                mega_moves = (*mega_moves, MEGA_FORCED_ACTION)

    actions = (*switches, *moves, *mega_moves)
    if slot.legality_known:
        return actions or (PASS_ACTION,)

    # Passing cannot be ruled out either: a mid-turn replacement request asks one slot
    # and passes the other, and neither is visible without the request itself.
    return (*actions, PASS_ACTION)


def action_mask(view: DecisionView) -> npt.NDArray[np.bool_]:
    """Build a (2, ACT_SIZE) boolean action mask for both slots."""
    mask = np.zeros((2, ACT_SIZE), dtype=np.bool_)
    for position in (0, 1):
        mask[position, legal_actions(view, position)] = True
    return mask


def slot1_base_mask(view: DecisionView) -> npt.NDArray[np.bool_]:
    """Mask of slot-1 unconstrained legal action IDs."""
    mask = np.zeros(ACT_SIZE, dtype=np.bool_)
    mask[list(legal_actions(view, 1))] = True
    return mask


def apply_joint_constraints(mask: npt.NDArray[np.bool_], view: DecisionView, first: int) -> None:
    """Apply the sequential slot-1 joint-action constraints in place to mask."""
    if view.team_preview:
        try:
            first_pair = decode_team_pair(first, view.team_size)
        except ValueError:
            mask.fill(False)
            return

        actions = np.arange(view.team_size**2)
        second_first, second_second = np.divmod(actions, view.team_size)

        valid = (
            (second_first < view.team_size)
            & (second_second < view.team_size)
            & (second_first != second_second)
        )

        conflict = (
            (second_first == first_pair[0])
            | (second_first == first_pair[1])
            | (second_second == first_pair[0])
            | (second_second == first_pair[1])
        )

        mask[: view.team_size**2] &= valid & ~conflict
    else:
        if SWITCH_START <= first < SWITCH_END:
            mask[first] = False

        if MEGA_MOVE_START <= first < MEGA_MOVE_END or first == MEGA_FORCED_ACTION:
            mask[27:48] = False

        if first == PASS_ACTION:
            mask[PASS_ACTION] = False

    if not mask.any():
        mask[PASS_ACTION] = True


def second_action_mask(view: DecisionView, first: int) -> npt.NDArray[np.bool_]:
    """Compute the legal action mask for slot 1 conditioned on slot 0's chosen action."""
    mask = slot1_base_mask(view)
    apply_joint_constraints(mask, view, first)
    return mask


def validate_joint_action(view: DecisionView, first: int, second: int) -> bool:
    """Validate whether the pair (first, second) is a legal joint action."""
    if first not in legal_actions(view, 0):
        return False
    return bool(second_action_mask(view, first)[second])
