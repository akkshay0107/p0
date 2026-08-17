"""
Spatial battlefield interaction records and turn recorder.

This module defines the 4-slot spatial battlefield representation (P1A, P1B, P2A, P2B),
tracking inter-turn action types, move identifiers, spatial target coordinates, execution
orders, HP deltas, etc.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class SpatialActionType(IntEnum):
    NONE = 0
    MOVE = 1
    SWITCH = 2
    PASS = 3
    FAINT = 4
    CANT = 5


class SpatialTargetSlot(IntEnum):
    SELF = 0
    ALLY_LEFT = 1
    ALLY_RIGHT = 2
    OPP_LEFT = 3
    OPP_RIGHT = 4
    ALL = 5
    NONE = 6


SPATIAL_SLOT_COUNT = 4
SPATIAL_CATEGORICAL_WIDTH = 3  # [action_type, move_id, target_slot]
SPATIAL_NUMERICAL_WIDTH = 8  # [order_rank, hp_delta, damage_dealt, net_boost_delta, landed_crit, took_crit, move_failed, item_consumed]
NUM_ACTION_TYPES = len(SpatialActionType)
NUM_TARGET_SLOTS = len(SpatialTargetSlot)

FLAG_LANDED_CRIT = 1
FLAG_TOOK_CRIT = 2
FLAG_MOVE_FAILED = 4
FLAG_ITEM_CONSUMED = 8


@dataclass(frozen=True, slots=True)
class SpatialSlotRecord:
    """Fixed-layout inter-turn interaction record for one active battlefield slot."""

    action_type: int = int(SpatialActionType.NONE)
    move_id: int = 0
    target_slot: int = int(SpatialTargetSlot.NONE)
    order_rank: float = 0.0
    hp_delta: float = 0.0
    damage_dealt: float = 0.0
    net_boost_delta: float = 0.0
    landed_crit: float = 0.0
    took_crit: float = 0.0
    move_failed: float = 0.0
    item_consumed: float = 0.0


def get_hp_fraction(hp_status: str) -> float:
    """Extract float HP fraction from a Showdown HP status string."""
    hp_part = hp_status.split(" ", 1)[0]
    if "/" not in hp_part:
        return 0.0
    try:
        num, den_str = hp_part.split("/")
        den_clean = den_str.strip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ% ")
        return float(num) / float(den_clean)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_slot_index(endpoint: str, perspective_role: str) -> int | None:
    """Map a Showdown entity string (e.g. 'p1a: Pikachu') to perspective slot 0..3."""
    if len(endpoint) < 3 or endpoint[0] != "p":
        return None
    player_num = endpoint[1]
    slot_letter = endpoint[2].lower()
    if player_num not in ("1", "2") or slot_letter not in ("a", "b"):
        return None

    is_ally = f"p{player_num}" == perspective_role
    slot_offset = 0 if slot_letter == "a" else 1
    return slot_offset if is_ally else 2 + slot_offset


def _parse_target_slot(endpoint: str, perspective_role: str) -> int:
    """Map a target endpoint to SpatialTargetSlot enum value."""
    slot_idx = _parse_slot_index(endpoint, perspective_role)
    if slot_idx is None:
        return int(SpatialTargetSlot.NONE)
    if slot_idx == 0:
        return int(SpatialTargetSlot.ALLY_LEFT)
    if slot_idx == 1:
        return int(SpatialTargetSlot.ALLY_RIGHT)
    if slot_idx == 2:
        return int(SpatialTargetSlot.OPP_LEFT)
    if slot_idx == 3:
        return int(SpatialTargetSlot.OPP_RIGHT)

    return int(SpatialTargetSlot.NONE)


class SpatialTurnRecorder:
    """Deterministic, pure-protocol recorder for the 4 battlefield active slots."""

    __slots__ = ("player_role", "slots", "_action_order", "_last_attacker")

    def __init__(self, player_role: str = "p1") -> None:
        self.player_role = player_role
        self.slots: list[SpatialSlotRecord] = [
            SpatialSlotRecord() for _ in range(SPATIAL_SLOT_COUNT)
        ]
        self._action_order = 0
        self._last_attacker: int | None = None

    def reset_turn(self) -> None:
        self.slots = [SpatialSlotRecord() for _ in range(SPATIAL_SLOT_COUNT)]
        self._action_order = 0
        self._last_attacker = None

    def apply_line(
        self,
        parts: Sequence[str],
        resolver: Any | None = None,
        hp_for: Any | None = None,
    ) -> None:
        if len(parts) < 2:
            return

        tag = parts[1]

        if tag in ("turn", "upkeep"):
            self._last_attacker = None
            return

        if tag == "move" and len(parts) >= 4:
            actor = _parse_slot_index(parts[2], self.player_role)
            if actor is not None:
                self._action_order += 1
                self._last_attacker = actor
                move_id = 0
                if resolver is not None:
                    try:
                        resolved, _ = resolver.resolve("moves", parts[3])
                        move_id = resolved
                    except Exception:
                        move_id = 0
                target_slot = (
                    _parse_target_slot(parts[4], self.player_role)
                    if len(parts) >= 5
                    else int(SpatialTargetSlot.NONE)
                )
                prev = self.slots[actor]
                self.slots[actor] = SpatialSlotRecord(
                    action_type=int(SpatialActionType.MOVE),
                    move_id=move_id,
                    target_slot=target_slot,
                    order_rank=min(1.0, self._action_order / 4.0),
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )

        elif tag in ("switch", "drag") and len(parts) >= 3:
            slot = _parse_slot_index(parts[2], self.player_role)
            if slot is not None:
                prev = self.slots[slot]
                self.slots[slot] = SpatialSlotRecord(
                    action_type=int(SpatialActionType.SWITCH),
                    move_id=0,
                    target_slot=int(SpatialTargetSlot.NONE),
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )

        elif tag == "faint" and len(parts) >= 3:
            slot = _parse_slot_index(parts[2], self.player_role)
            if slot is not None:
                prev = self.slots[slot]
                self.slots[slot] = SpatialSlotRecord(
                    action_type=int(SpatialActionType.FAINT),
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )

        elif tag == "cant" and len(parts) >= 3:
            slot = _parse_slot_index(parts[2], self.player_role)
            if slot is not None:
                prev = self.slots[slot]
                self.slots[slot] = SpatialSlotRecord(
                    action_type=int(SpatialActionType.CANT),
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )

        elif tag in ("-damage", "-heal") and len(parts) >= 4:
            target = _parse_slot_index(parts[2], self.player_role)
            if target is not None:
                new_hp = get_hp_fraction(parts[3])
                pre_hp = hp_for(parts[2]) if hp_for is not None else None
                delta = (new_hp - pre_hp) if pre_hp is not None else 0.0
                prev = self.slots[target]
                self.slots[target] = SpatialSlotRecord(
                    action_type=prev.action_type,
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta + delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )
                if tag == "-damage" and self._last_attacker is not None and delta < 0:
                    att_prev = self.slots[self._last_attacker]
                    self.slots[self._last_attacker] = SpatialSlotRecord(
                        action_type=att_prev.action_type,
                        move_id=att_prev.move_id,
                        target_slot=att_prev.target_slot,
                        order_rank=att_prev.order_rank,
                        hp_delta=att_prev.hp_delta,
                        damage_dealt=att_prev.damage_dealt + abs(delta),
                        net_boost_delta=att_prev.net_boost_delta,
                        landed_crit=att_prev.landed_crit,
                        took_crit=att_prev.took_crit,
                        move_failed=att_prev.move_failed,
                        item_consumed=att_prev.item_consumed,
                    )

        elif tag == "-crit" and len(parts) >= 3:
            target = _parse_slot_index(parts[2], self.player_role)
            if target is not None:
                prev = self.slots[target]
                self.slots[target] = SpatialSlotRecord(
                    action_type=prev.action_type,
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=1.0,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )
            if self._last_attacker is not None:
                att_prev = self.slots[self._last_attacker]
                self.slots[self._last_attacker] = SpatialSlotRecord(
                    action_type=att_prev.action_type,
                    move_id=att_prev.move_id,
                    target_slot=att_prev.target_slot,
                    order_rank=att_prev.order_rank,
                    hp_delta=att_prev.hp_delta,
                    damage_dealt=att_prev.damage_dealt,
                    net_boost_delta=att_prev.net_boost_delta,
                    landed_crit=1.0,
                    took_crit=att_prev.took_crit,
                    move_failed=att_prev.move_failed,
                    item_consumed=att_prev.item_consumed,
                )

        elif tag in ("-boost", "-unboost") and len(parts) >= 5:
            slot = _parse_slot_index(parts[2], self.player_role)
            if slot is not None:
                amount = int(parts[4]) / 6.0
                delta = amount if tag == "-boost" else -amount
                prev = self.slots[slot]
                self.slots[slot] = SpatialSlotRecord(
                    action_type=prev.action_type,
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta + delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=prev.item_consumed,
                )

        elif tag in ("-fail", "-miss", "-immune"):
            if self._last_attacker is not None:
                prev = self.slots[self._last_attacker]
                self.slots[self._last_attacker] = SpatialSlotRecord(
                    action_type=prev.action_type,
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=1.0,
                    item_consumed=prev.item_consumed,
                )

        elif tag in ("-enditem", "-item") and len(parts) >= 3:
            slot = _parse_slot_index(parts[2], self.player_role)
            if slot is not None:
                prev = self.slots[slot]
                self.slots[slot] = SpatialSlotRecord(
                    action_type=prev.action_type,
                    move_id=prev.move_id,
                    target_slot=prev.target_slot,
                    order_rank=prev.order_rank,
                    hp_delta=prev.hp_delta,
                    damage_dealt=prev.damage_dealt,
                    net_boost_delta=prev.net_boost_delta,
                    landed_crit=prev.landed_crit,
                    took_crit=prev.took_crit,
                    move_failed=prev.move_failed,
                    item_consumed=1.0,
                )

    def to_records(self) -> tuple[SpatialSlotRecord, ...]:
        return tuple(self.slots)


__all__ = [
    "FLAG_ITEM_CONSUMED",
    "FLAG_LANDED_CRIT",
    "FLAG_MOVE_FAILED",
    "FLAG_TOOK_CRIT",
    "NUM_ACTION_TYPES",
    "NUM_TARGET_SLOTS",
    "SPATIAL_CATEGORICAL_WIDTH",
    "SPATIAL_NUMERICAL_WIDTH",
    "SPATIAL_SLOT_COUNT",
    "SpatialActionType",
    "SpatialSlotRecord",
    "SpatialTargetSlot",
    "SpatialTurnRecorder",
    "get_hp_fraction",
]
