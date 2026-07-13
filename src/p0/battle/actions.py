"""Pure, allocation-light mapping for the 49-action doubles layout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from p0.format_config import FORMAT

ACT_SIZE = FORMAT.action_size
PASS_ACTION = 0
SWITCH_START = 1
SWITCH_END = 7
MOVE_START = 7
MOVE_END = 27
MEGA_MOVE_START = 27
MEGA_MOVE_END = 47
MEGA_FORCED_ACTION = 47
FORCED_ACTION = 48
MOVE_SLOT_COUNT = 4
TARGET_COUNT = 5
TEAM_SIZE = 6

if ACT_SIZE != 49:
    raise RuntimeError(f"The action layout requires 49 actions, got {ACT_SIZE}")


class ActionKind(IntEnum):
    PASS = 0
    SWITCH = 1
    MOVE = 2
    FORCED_MOVE = 3


@dataclass(frozen=True, slots=True)
class SlotAction:
    """Semantic action used only at runtime/replay boundaries."""

    kind: ActionKind
    switch_slot: int = -1
    move_slot: int = -1
    target: int = 0
    mega: bool = False


class ActionCodec:
    """Integer/semantic conversion with no poke-env dependency."""

    @staticmethod
    def decode(action: int) -> SlotAction:
        action = int(action)
        if action == PASS_ACTION:
            return SlotAction(ActionKind.PASS)
        if SWITCH_START <= action < SWITCH_END:
            return SlotAction(ActionKind.SWITCH, switch_slot=action - SWITCH_START)
        if MOVE_START <= action < MEGA_MOVE_END:
            offset = action - MOVE_START
            mega = offset >= MOVE_END - MOVE_START
            if mega:
                offset -= MOVE_END - MOVE_START
            return SlotAction(
                ActionKind.MOVE,
                move_slot=offset // TARGET_COUNT,
                target=offset % TARGET_COUNT - 2,
                mega=mega,
            )
        if action in (MEGA_FORCED_ACTION, FORCED_ACTION):
            return SlotAction(ActionKind.FORCED_MOVE, mega=action == MEGA_FORCED_ACTION)
        raise ValueError(f"Action must be in [0, {ACT_SIZE}), got {action}")

    @staticmethod
    def encode(action: SlotAction) -> int:
        if action.kind is ActionKind.PASS:
            return PASS_ACTION
        if action.kind is ActionKind.SWITCH:
            if not 0 <= action.switch_slot < TEAM_SIZE:
                raise ValueError(f"Invalid switch slot {action.switch_slot}")
            return SWITCH_START + action.switch_slot
        if action.kind is ActionKind.FORCED_MOVE:
            return MEGA_FORCED_ACTION if action.mega else FORCED_ACTION
        if action.kind is ActionKind.MOVE:
            if not 0 <= action.move_slot < MOVE_SLOT_COUNT:
                raise ValueError(f"Invalid move slot {action.move_slot}")
            if not -2 <= action.target <= 2:
                raise ValueError(f"Invalid move target {action.target}")
            return (
                MOVE_START
                + action.move_slot * TARGET_COUNT
                + action.target
                + 2
                + (MOVE_END - MOVE_START if action.mega else 0)
            )
        raise ValueError(f"Unsupported action kind {action.kind!r}")

    @staticmethod
    def encode_team_pair(first: int, second: int, team_size: int = TEAM_SIZE) -> int:
        if not 0 <= first < team_size or not 0 <= second < team_size:
            raise ValueError("Team-preview indices are outside the roster")
        if first >= second:
            raise ValueError("Team-preview pairs must be strictly increasing")
        return first * TEAM_SIZE + second

    @staticmethod
    def decode_team_pair(action: int, team_size: int = TEAM_SIZE) -> tuple[int, int]:
        action = int(action)
        first, second = divmod(action, TEAM_SIZE)
        if action < 0 or first >= team_size or second >= team_size or first >= second:
            raise ValueError(f"Invalid canonical team-preview action {action}")
        return first, second

    @classmethod
    def team_selection(
        cls, lead_action: int, back_action: int, team_size: int = TEAM_SIZE
    ) -> tuple[int, ...]:
        selected: list[int] = []
        for index in (
            *cls.decode_team_pair(lead_action, team_size),
            *cls.decode_team_pair(back_action, team_size),
        ):
            if index not in selected:
                selected.append(index)
        selected.extend(index for index in range(team_size) if index not in selected)
        return tuple(selected)

    @classmethod
    def canonical_team_actions(
        cls, selection: tuple[int, ...], team_size: int = TEAM_SIZE
    ) -> tuple[int, int]:
        defaults = tuple(range(team_size))
        values = tuple(index for index in selection if 0 <= index < team_size)
        values += tuple(index for index in defaults if index not in values)
        lead = tuple(sorted(values[:2]))
        back = tuple(sorted(values[2:4]))
        if lead[0] == lead[1] or back[0] == back[1]:
            raise ValueError("Team preview must select four distinct members")
        return (
            cls.encode_team_pair(lead[0], lead[1], team_size),
            cls.encode_team_pair(back[0], back[1], team_size),
        )
