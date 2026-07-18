"""Offline replay acquisition, reconstruction, and compilation.

Layering: this package may import p0.battle, p0.teams, and p0.format_config.
It must never import p0.runtime, and p0.replays.schema must stay torch-free
so the IR survives observation-schema changes.
"""

from p0.replays.schema import (
    REPLAY_IR_SCHEMA_VERSION,
    ActionEvidence,
    DecisionRecord,
    DecisionType,
    FetchIndexEntry,
    GameEndReason,
    GameRecord,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    ReplayDiagnostics,
    SeriesRecord,
)

__all__ = [
    "REPLAY_IR_SCHEMA_VERSION",
    "ActionEvidence",
    "DecisionRecord",
    "DecisionType",
    "FetchIndexEntry",
    "GameEndReason",
    "GameRecord",
    "GroupingMethod",
    "LabelKind",
    "MaskProvenance",
    "ReplayDiagnostics",
    "SeriesRecord",
]
