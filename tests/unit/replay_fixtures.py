"""Small deterministic replay and event fixtures used by unit tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from p0.battle.events import EventTypeId, RawBattleEvent
from p0.format_config import FORMAT


def golden_series_id(parent: str) -> str:
    """Return the independently calculated series identity for this fixture family."""
    value = "\n".join((FORMAT.bo3_format, parent, "alice", "bob"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def golden_replay_payload(
    replay_id: str,
    *,
    series_id: str = "series-1",
    game_number: int = 1,
    winner: str = "Alice",
    players: tuple[str, str] = ("Alice", "Bob"),
    first_move_target: str | None = "p1a: Pikachu",
) -> dict[str, Any]:
    """Return a replay payload with the pinned protocol shape."""
    p1_team = [
        {"species": "Pikachu", "moves": ["Protect", "Tackle"]},
        {"species": "Eevee", "moves": ["Tackle", "Helping Hand"]},
    ]
    p2_team = [
        {"species": "Bulbasaur", "moves": ["Protect", "Tackle"]},
        {"species": "Charmander", "moves": ["Tackle", "Helping Hand"]},
    ]
    lines = [
        "|start",
        "|teampreview",
        f"|showteam|p1|{json.dumps(p1_team, separators=(',', ':'))}",
        f"|showteam|p2|{json.dumps(p2_team, separators=(',', ':'))}",
        "|",
        "|switch|p1a: Pikachu|Pikachu, L50|100/100",
        "|switch|p1b: Eevee|Eevee, L50|100/100",
        "|switch|p2a: Bulbasaur|Bulbasaur, L50|100/100",
        "|switch|p2b: Charmander|Charmander, L50|100/100",
        "|turn|1",
        "|",
        "|move|p1a: Pikachu|Protect"
        + (f"|{first_move_target}" if first_move_target is not None else ""),
        "|move|p1b: Eevee|Tackle|p2b: Charmander",
        "|move|p2a: Bulbasaur|Protect|p2a: Bulbasaur",
        "|move|p2b: Charmander|Tackle|p1b: Eevee",
        "|",
        f"|win|{winner}",
    ]
    return {
        "id": replay_id,
        "formatid": FORMAT.bo3_format,
        "p1": players[0],
        "p2": players[1],
        "uploadtime": 1_750_000_000,
        "roomid": replay_id,
        "parent": series_id,
        "game_number": game_number,
        "log": "\n".join(lines),
    }


def golden_raw_events() -> tuple[RawBattleEvent, ...]:
    """Return Showdown event lines with a documented type order."""
    return (
        RawBattleEvent(("", "move", "p1a: Pikachu", "Thunderbolt", "p2a: Charizard")),
        RawBattleEvent(("", "move", "p1a: Pikachu", "definitely-not-a-move", "p2a: Charizard")),
        RawBattleEvent(("", "switch", "p1a: Pikachu", "Pikachu, L50", "100/100")),
        RawBattleEvent(("", "drag", "p2a: Charizard", "Charizard, L50", "100/100")),
        RawBattleEvent(("", "swap", "p1a: Pikachu", "p1b: Eevee")),
        RawBattleEvent(("", "faint", "p2a: Charizard")),
        RawBattleEvent(("", "-damage", "p2a: Charizard", "75/100"), pre_hp=1.0),
        RawBattleEvent(("", "-heal", "p2a: Charizard", "90/100"), pre_hp=0.75),
        RawBattleEvent(("", "-boost", "p1a: Pikachu", "atk", "2")),
        RawBattleEvent(("", "-unboost", "p1a: Pikachu", "atk", "1")),
        RawBattleEvent(("", "-status", "p2a: Charizard", "par")),
        RawBattleEvent(("", "-curestatus", "p2a: Charizard", "par")),
        RawBattleEvent(("", "-enditem", "p1a: Pikachu", "Leftovers")),
        RawBattleEvent(("", "-item", "p1a: Pikachu", "Leftovers")),
        RawBattleEvent(("", "-item", "p1a: Pikachu", "Leftovers", "[from] move: Trick")),
        RawBattleEvent(("", "-ability", "p1a: Pikachu", "Static")),
        RawBattleEvent(("", "-weather", "RainDance")),
        RawBattleEvent(("", "-weather", "none")),
        RawBattleEvent(("", "-fieldstart", "move: Trick Room")),
        RawBattleEvent(("", "-fieldend", "move: Trick Room")),
        RawBattleEvent(("", "-sidestart", "p1", "move: Stealth Rock")),
        RawBattleEvent(("", "-sideend", "p1", "move: Stealth Rock")),
        RawBattleEvent(("", "-start", "p1a: Pikachu", "move: Protect")),
        RawBattleEvent(("", "-end", "p1a: Pikachu", "move: Protect")),
        RawBattleEvent(("", "-formechange", "p1a: Pikachu", "Pikachu")),
        RawBattleEvent(("", "-fail", "p1a: Pikachu")),
        RawBattleEvent(("", "-immune", "p2a: Charizard")),
        RawBattleEvent(("", "-miss", "p1a: Pikachu", "p2a: Charizard")),
        RawBattleEvent(("", "-crit", "p1a: Pikachu")),
        RawBattleEvent(("", "-mega", "p1a: Pikachu")),
        RawBattleEvent(("", "cant", "p1a: Pikachu", "par", "Thunderbolt")),
        RawBattleEvent(("", "-prepare", "p1a: Pikachu", "Solar Beam", "p2a: Charizard")),
        RawBattleEvent(("", "-singlemove", "p1a: Pikachu", "Protect")),
        RawBattleEvent(("", "-setboost", "p1a: Pikachu", "atk", "3")),
        RawBattleEvent(("", "-clearboost", "p1a: Pikachu")),
        RawBattleEvent(("", "-swapboost", "p1a: Pikachu", "p2a: Charizard")),
        RawBattleEvent(("", "-invertboost", "p1a: Pikachu")),
        RawBattleEvent(("", "-copyboost", "p1a: Pikachu", "p2a: Charizard")),
        RawBattleEvent(("", "-transform", "p1a: Pikachu", "p2a: Charizard")),
        RawBattleEvent(("", "-endability", "p1a: Pikachu", "Static")),
        RawBattleEvent(("", "-fieldactivate", "move: Trick Room")),
        RawBattleEvent(("", "-notarget", "p1a: Pikachu")),
        RawBattleEvent(("", "chat", "this is ignored")),
    )


GOLDEN_EVENT_TYPES = (
    EventTypeId.MOVE,
    EventTypeId.MOVE,
    EventTypeId.SWITCH_IN,
    EventTypeId.DRAG,
    EventTypeId.SWAP,
    EventTypeId.FAINT,
    EventTypeId.DAMAGE,
    EventTypeId.HEAL,
    EventTypeId.BOOST,
    EventTypeId.UNBOOST,
    EventTypeId.STATUS_SET,
    EventTypeId.STATUS_CURE,
    EventTypeId.ITEM_END,
    EventTypeId.ITEM_REVEAL,
    EventTypeId.ITEM_TRANSFER,
    EventTypeId.ABILITY,
    EventTypeId.WEATHER_START,
    EventTypeId.WEATHER_END,
    EventTypeId.FIELD_START,
    EventTypeId.FIELD_END,
    EventTypeId.SIDE_START,
    EventTypeId.SIDE_END,
    EventTypeId.EFFECT_START,
    EventTypeId.EFFECT_END,
    EventTypeId.FORME_CHANGE,
    EventTypeId.FAILED,
    EventTypeId.BLOCKED,
    EventTypeId.BLOCKED,
    EventTypeId.CRIT,
    EventTypeId.MEGA,
    EventTypeId.CANT,
    EventTypeId.PREPARE,
    EventTypeId.SINGLEMOVE,
    EventTypeId.BOOST_SET,
    EventTypeId.BOOST_CLEAR,
    EventTypeId.BOOST_SWAP,
    EventTypeId.BOOST_INVERT,
    EventTypeId.BOOST_COPY,
    EventTypeId.TRANSFORM,
    EventTypeId.ABILITY_END,
    EventTypeId.FIELD_ACTIVATE,
    EventTypeId.NO_TARGET,
)
