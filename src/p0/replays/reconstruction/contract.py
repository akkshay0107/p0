"""Effect names and volatile conditions the engine accepts, read from the generated data files."""

from __future__ import annotations

import json
import re
from typing import Any

from p0.paths import DEFAULT_PATHS
from p0.replays.identity import normalize_showdown_id


def _load(filename: str) -> dict[str, Any]:
    with (DEFAULT_PATHS.data_root / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


_DEX = _load("champions_dex.json")
_INVENTORY = _load("showdown_raw_emission_inventory.json")
_LITERAL_ARGUMENT = re.compile(r"['\"]([A-Za-z][A-Za-z0-9 .'_-]*)['\"]")

LEGAL_EFFECT_IDS = {
    kind: frozenset(_DEX["legality"][kind]) for kind in ("moves", "items", "abilities")
}
_VOLATILE_CONDITIONS = _INVENTORY["volatile_conditions"]
ALL_LEGAL_EFFECT_NAMES = frozenset().union(
    *LEGAL_EFFECT_IDS.values(),
    _DEX["legalProtocolEffects"]["effect"],
    _DEX["legalProtocolEffects"]["status"],
    _DEX["protocolEffects"],
    # Causes the simulator builds at run time, which no data table lists.
    ("recoil", "drain", "stealeat", "confusion"),
    (row["id"] for row in _VOLATILE_CONDITIONS),
)
# The inventory also records source expressions such as '+effect.name. They are
# not protocol values and must not become allowed names.
KNOWN_ACTIVATION_EFFECTS = frozenset(
    normalize_showdown_id(value.split(":", 1)[1].strip() if ":" in value else value)
    for value in _INVENTORY["activation_effects"]
    if not value.startswith("'+")
)
COPYABLE_VOLATILES = frozenset(
    row["id"] for row in _VOLATILE_CONDITIONS if row["exists"] and not row["noCopy"]
)
# Move groups read from dex move fields, so a regulation change needs no edit here.
_LEGAL_MOVES = tuple(move for move in _DEX["moves"] if move["id"] in LEGAL_EFFECT_IDS["moves"])
DELAYED_MOVES = frozenset(
    move["id"] for move in _LEGAL_MOVES if move.get("flags", {}).get("futuremove")
)
BREAKING_PROTECT_MOVES = frozenset(move["id"] for move in _LEGAL_MOVES if move.get("breaksProtect"))
STALLING_MOVES = frozenset(move["id"] for move in _LEGAL_MOVES if move.get("stallingMove"))
TERRAINS = frozenset(
    normalize_showdown_id(move["terrain"]) for move in _LEGAL_MOVES if move.get("terrain")
)
# Revival Blessing sets both fields: it opens a revive request and brings no
# replacement in for its user, so it belongs to neither group.
PIVOT_MOVES = frozenset(
    move["id"] for move in _LEGAL_MOVES if move.get("selfSwitch") and not move.get("slotCondition")
)
SLOT_CONDITION_MOVES = frozenset(
    move["id"] for move in _LEGAL_MOVES if move.get("slotCondition") and not move.get("selfSwitch")
)
REVIVAL_BLESSING = "revivalblessing"
ALLOWED_CAUSE_NAMESPACES = frozenset(
    {"move", "ability", "item", "condition", "status", "pokemon", "format", "gem"}
)


def _witnessed_effects(tag: str) -> set[str]:
    """Return the effect names that the simulator source passes to one protocol tag."""
    effects: set[str] = set()
    for witness in _load("replay_protocol_contract.json")["raw_witnesses"]:
        if tag != (witness["tag"] or (witness["resolved_tags"] or [None])[0]):
            continue
        effects.update(normalize_showdown_id(effect) for effect in witness["normalized_effects"])
        effects.update(
            normalize_showdown_id(literal.group(1))
            for argument in witness["arguments"]
            if (literal := _LITERAL_ARGUMENT.fullmatch(argument))
        )
    return effects


ALLOWED_EFFECTS_BY_TAG = {
    "-activate": frozenset(_witnessed_effects("-activate") | KNOWN_ACTIVATION_EFFECTS),
    "-singlemove": frozenset(_witnessed_effects("-singlemove")),
}
