"""Offline replay acquisition, reconstruction, and compilation.

Layering: this package may import p0.battle, p0.teams, and p0.format_config.
It must never import p0.runtime, and p0.replays.schema must stay torch-free
so the IR survives observation-schema changes.
"""

from p0.replays.protocol import ReplayDocument, ReplayParseError, parse_replay_payload
from p0.replays.schema import (
    REPLAY_IR_SCHEMA_VERSION,
    ActionEvidence,
    DecisionRecord,
    DecisionType,
    FetchIndexEntry,
    FetchMetadata,
    GameEndReason,
    GameRecord,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    OTSData,
    ProtocolLine,
    ReplayDiagnostics,
    ReplayMetadata,
    ReplayOutcome,
    SeriesMembership,
    SeriesRecord,
)

__all__ = [
    "REPLAY_IR_SCHEMA_VERSION",
    "ActionEvidence",
    "DecisionRecord",
    "DecisionType",
    "FetchMetadata",
    "FetchIndexEntry",
    "GameEndReason",
    "GameRecord",
    "GroupingMethod",
    "LabelKind",
    "MaskProvenance",
    "OTSData",
    "ProtocolLine",
    "ReplayMetadata",
    "ReplayDiagnostics",
    "ReplayOutcome",
    "SeriesMembership",
    "SeriesRecord",
    "ReplayDocument",
    "ReplayParseError",
    "parse_replay_payload",
]
