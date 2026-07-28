"""Pure scalar legality and joint-action constraints."""

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


@dataclass(frozen=True, slots=True)
class DecisionView:
    slots: tuple[SlotDecision, SlotDecision]
    wait: bool = False
    team_preview: bool = False
    team_size: int = 6


def legal_actions(view: DecisionView, position: int) -> tuple[int, ...]:
    """Compute the legal action IDs for a single slot position given decision view."""
    if view.team_preview:
        return tuple(
            first * 6 + second
            for first in range(view.team_size)
            for second in range(first + 1, view.team_size)
        )

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

    return (*switches, *moves, *mega_moves) or (PASS_ACTION,)


def action_mask(view: DecisionView) -> npt.NDArray[np.bool_]:
    """Build a (2, ACT_SIZE) boolean action mask for both slots."""
    mask = np.zeros((2, ACT_SIZE), dtype=np.bool_)
    for position in (0, 1):
        mask[position, legal_actions(view, position)] = True
    return mask


def validate_joint_action(view: DecisionView, first: int, second: int) -> bool:
    """Validate whether the pair (first, second) is a legal joint action."""
    if first not in legal_actions(view, 0):
        return False
    return bool(second_action_mask(view, first)[second])


def second_action_mask(view: DecisionView, first: int) -> npt.NDArray[np.bool_]:
    """Compute the legal action mask for slot 1 conditioned on slot 0's chosen action."""
    mask = np.zeros(ACT_SIZE, dtype=np.bool_)
    mask[list(legal_actions(view, 1))] = True

    if view.team_preview:
        try:
            first_pair = decode_team_pair(first, view.team_size)
        except ValueError:
            mask.fill(False)
            return mask

        for action in np.flatnonzero(mask[:36]):
            try:
                second_pair = decode_team_pair(int(action), view.team_size)
            except ValueError:
                mask[action] = False
                continue

            if (
                first_pair[0] == second_pair[0]
                or first_pair[0] == second_pair[1]
                or first_pair[1] == second_pair[0]
                or first_pair[1] == second_pair[1]
            ):
                mask[action] = False
    else:
        if SWITCH_START <= first < SWITCH_END:
            mask[first] = False

        if MEGA_MOVE_START <= first < MEGA_MOVE_END or first == MEGA_FORCED_ACTION:
            mask[27:48] = False

        if first == PASS_ACTION:
            mask[PASS_ACTION] = False

    if not mask.any():
        mask[PASS_ACTION] = True

    return mask
