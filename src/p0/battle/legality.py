"""Pure scalar legality and joint-action constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from p0.battle.actions import (
    ACT_SIZE,
    FORCED_ACTION,
    MEGA_FORCED_ACTION,
    MOVE_END,
    MOVE_START,
    PASS_ACTION,
    SWITCH_START,
    ActionKind,
    decode_action,
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
    mask = np.zeros((2, ACT_SIZE), dtype=np.bool_)
    for position in (0, 1):
        mask[position, legal_actions(view, position)] = True
    return mask


def validate_joint_action(view: DecisionView, first: int, second: int) -> bool:
    if first not in legal_actions(view, 0):
        return False
    return bool(second_action_mask(view, first)[second])


def second_action_mask(view: DecisionView, first: int) -> npt.NDArray[np.bool_]:
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
            if set(first_pair) & set(second_pair):
                mask[action] = False
    else:
        semantic = decode_action(first)
        if semantic.kind is ActionKind.SWITCH:
            mask[SWITCH_START + semantic.switch_slot] = False
        if semantic.mega:
            mask[27:48] = False
        if semantic.kind is ActionKind.PASS:
            mask[PASS_ACTION] = False
    if not mask.any():
        mask[PASS_ACTION] = True
    return mask
