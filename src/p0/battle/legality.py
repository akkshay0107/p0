"""Action legality and joint-action constraints.

Computes legal moves and switches for both active battlefield slots. In replay
reconstruction where choices cannot be fully observed, unconfirmed slots allow
all possible actions so training never rules out the demonstrator's action.
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
    TARGET_COUNT,
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
_TEAM_PREVIEW_JOINT_MASKS: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}


def _get_team_preview(team_size: int) -> tuple[int, ...]:
    """Ordered distinct team-preview pair action IDs, in ascending roster order."""
    cached = _TEAM_PREVIEW_CACHE.get(team_size)
    if cached is None:
        cached = tuple(
            first * team_size + second
            for first in range(team_size)
            for second in range(team_size)
            if first != second
        )
        _TEAM_PREVIEW_CACHE[team_size] = cached
    return cached


def _get_team_preview_joint_mask(
    team_size: int, first_pair: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """Return precomputed boolean mask of valid second team-preview actions."""
    key = (team_size, first_pair[0] * team_size + first_pair[1])
    cached = _TEAM_PREVIEW_JOINT_MASKS.get(key)
    if cached is None:
        total = team_size * team_size
        cached = np.zeros(total, dtype=np.bool_)
        p0, p1 = first_pair
        for second in range(total):
            sf, ss = divmod(second, team_size)
            if sf != ss and sf != p0 and sf != p1 and ss != p0 and ss != p1:
                cached[second] = True
        _TEAM_PREVIEW_JOINT_MASKS[key] = cached
    return cached


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
            MOVE_START + move_slot * TARGET_COUNT + target + 2
            for move_slot, targets in enumerate(slot.move_targets)
            for target in targets
        )
        mega_moves = (
            tuple(action + (MOVE_END - MOVE_START) for action in moves) if slot.can_mega else ()
        )

        # When legality is not fully known, allow forced moves (e.g. Outrage,
        # recharge turns) which map to the forced action slot.
        if not slot.legality_known:
            moves = (*moves, FORCED_ACTION)
            if slot.can_mega:
                mega_moves = (*mega_moves, MEGA_FORCED_ACTION)

    actions = (*switches, *moves, *mega_moves)
    if slot.legality_known:
        return actions or (PASS_ACTION,)

    # When legality is not fully known, allow passing to account for partial
    # mid-turn replacement requests.
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

        total_pairs = view.team_size**2
        mask[:total_pairs] &= _get_team_preview_joint_mask(view.team_size, first_pair)
    else:
        if SWITCH_START <= first < SWITCH_END:
            mask[first] = False

        if MEGA_MOVE_START <= first < MEGA_MOVE_END or first == MEGA_FORCED_ACTION:
            mask[MEGA_MOVE_START : MEGA_FORCED_ACTION + 1] = False

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
