"""Action encoding and decoding for the 49-action doubles layout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from p0.format_config import ACTION_CONTRACT

_RANGES = {entry["meaning"]: entry for entry in ACTION_CONTRACT["ranges"]}

ACT_SIZE = ACTION_CONTRACT["action_count"]
PASS_ACTION = _RANGES["pass"]["start"]
SWITCH_START = _RANGES["switch"]["start"]
SWITCH_END = _RANGES["switch"]["end"]
MOVE_START = _RANGES["move"]["start"]
MOVE_END = _RANGES["move"]["end"]
MEGA_MOVE_START = _RANGES["mega_move"]["start"]
MEGA_MOVE_END = _RANGES["mega_move"]["end"]
MEGA_FORCED_ACTION = _RANGES["mega_forced_move"]["start"]
FORCED_ACTION = _RANGES["forced_move"]["start"]
MOVE_SLOT_COUNT = _RANGES["move"]["move_slots"]
TARGET_COUNT = len(_RANGES["move"]["targets"])
TARGET_OFFSET = 2
TEAM_SIZE = ACTION_CONTRACT["team_preview"]["roster_size"]


class ActionKind(IntEnum):
    PASS = 0
    SWITCH = 1
    MOVE = 2
    FORCED_MOVE = 3


@dataclass(frozen=True, slots=True)
class SlotAction:
    """Action representation used at runtime and replay boundaries."""

    kind: ActionKind
    switch_slot: int = -1
    move_slot: int = -1
    target: int = 0
    mega: bool = False


def decode_action(action: int) -> SlotAction:
    """Decode an action ID into a SlotAction."""
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
            target=offset % TARGET_COUNT - TARGET_OFFSET,
            mega=mega,
        )

    if action in (MEGA_FORCED_ACTION, FORCED_ACTION):
        return SlotAction(ActionKind.FORCED_MOVE, mega=action == MEGA_FORCED_ACTION)

    raise ValueError(f"Action must be in [0, {ACT_SIZE}), got {action}")


def encode_action(action: SlotAction) -> int:
    """Encode a SlotAction into an action ID."""
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

        if not -TARGET_OFFSET <= action.target <= TARGET_OFFSET:
            raise ValueError(f"Invalid move target {action.target}")

        return (
            MOVE_START
            + action.move_slot * TARGET_COUNT
            + action.target
            + TARGET_OFFSET
            + (MOVE_END - MOVE_START if action.mega else 0)
        )

    raise ValueError(f"Unsupported action kind {action.kind!r}")


def encode_team_pair(first: int, second: int, team_size: int = TEAM_SIZE) -> int:
    """Encode an ordered pair of distinct team preview indices into an action ID."""
    if not 0 <= first < team_size or not 0 <= second < team_size:
        raise ValueError("Team-preview indices are outside the roster")
    if first == second:
        raise ValueError("Team-preview pairs must be distinct")
    return first * team_size + second


def decode_team_pair(action: int, team_size: int = TEAM_SIZE) -> tuple[int, int]:
    """Decode an action ID into an ordered pair of team preview indices."""
    action = int(action)
    first, second = divmod(action, team_size)
    if action < 0 or first >= team_size or second >= team_size or first == second:
        raise ValueError(f"Invalid canonical team-preview action {action}")
    return first, second


def team_selection(
    lead_action: int, back_action: int, team_size: int = TEAM_SIZE
) -> tuple[int, ...]:
    """Construct full team ordering from lead and back pair preview actions."""
    leads = decode_team_pair(lead_action, team_size)
    backs = decode_team_pair(back_action, team_size)

    selected: list[int] = list(dict.fromkeys((*leads, *backs)))
    selected.extend(index for index in range(team_size) if index not in selected)
    return tuple(selected)


def canonical_team_actions(
    selection: tuple[int, ...], team_size: int = TEAM_SIZE
) -> tuple[int, int]:
    """Derive ordered lead and back pair preview actions from a team order."""
    defaults = tuple(range(team_size))
    values = tuple(index for index in selection if 0 <= index < team_size)
    values += tuple(index for index in defaults if index not in values)

    lead = values[:2]
    back = values[2:4]

    if len(set((*lead, *back))) != 4:
        raise ValueError("Team preview must select four distinct members")

    return (
        encode_team_pair(lead[0], lead[1], team_size),
        encode_team_pair(back[0], back[1], team_size),
    )
