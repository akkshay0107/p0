"""Tensor shard format, schema definitions, and manifest validation for replay datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from p0.battle.actions import ACT_SIZE, MEGA_FORCED_ACTION, MEGA_MOVE_START
from p0.contracts import require_dataclass_fields
from p0.format_config import validate_artifact_runtime_contract
from p0.model.structured_observation import StructuredObservation
from p0.replays.schema import (
    DecisionType,
    LabelKind,
    MaskProvenance,
    _require_iso_timestamp,
)

SHARD_ARTIFACT_SCHEMA = "p0.replay_shard.v2"

# Non-observation tensors stored per shard. -1 marks a variable dimension:
# T is the shard's decision count and C its total candidate count. Candidates
# use a ragged values-plus-offsets encoding: candidate_offsets has length
# T + 1 and decision t owns candidate_values[offsets[t]:offsets[t + 1]] rows
# of joint action pairs. exact_action rows are meaningful only where
# label_kind is EXACT; loss_mask is zero on UNKNOWN decisions so they keep
# chronological context without contributing policy loss.
# game_offsets and series_offsets delimit whole chronological games and
# series within the shard. outcome is the optional value target.
SHARD_TENSOR_SPECS: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = (
    ("action_mask", (-1, 2, ACT_SIZE), torch.bool),
    ("mask_provenance", (-1,), torch.long),
    ("label_kind", (-1,), torch.long),
    ("label_confidence", (-1,), torch.float32),
    ("loss_mask", (-1,), torch.float32),
    ("decision_type", (-1,), torch.long),
    ("exact_action", (-1, 2), torch.long),
    ("candidate_values", (-1, 2), torch.long),
    ("candidate_offsets", (-1,), torch.long),
    ("game_offsets", (-1,), torch.long),
    ("series_offsets", (-1,), torch.long),
    ("outcome", (-1,), torch.float32),
)

# Per-game identity records ride along as JSON, not tensors: the series and
# canonical-player each game belongs to, plus its outcome provenance. Series
# context itself is continuous and rebuilt in process, never stored here.
SHARD_SUMMARY_KEY = "series_summaries"


def observation_field_specs() -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    """
    Observation tensors stacked along a leading decision axis.

    Derived from StructuredObservation._FIELD_SPECS so an observation-schema
    change cannot silently diverge from the shard layout.
    """
    return tuple(
        (name, (-1, *shape), dtype) for name, shape, dtype in StructuredObservation._FIELD_SPECS
    )


# Each game perspective's board after its last line, stacked along a leading
# game axis under these prefixed names. It feeds series memory only and has no
# label, mask or outcome.
FINAL_OBSERVATION_PREFIX = "final_"


def final_observation_field_specs() -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    """Final-observation tensors stacked along a leading game axis."""
    return tuple(
        (f"{FINAL_OBSERVATION_PREFIX}{name}", shape, dtype)
        for name, shape, dtype in observation_field_specs()
    )


@dataclass(frozen=True, slots=True)
class ShardIndexEntry:
    """One compiled series, identified by its compilation UUID."""

    shard_id: str
    filename: str
    series_id: str
    replay_ids: tuple[str, ...]
    decisions: int
    games: int

    def __post_init__(self) -> None:
        if not self.shard_id or not self.filename or not self.series_id:
            raise ValueError("Shard entries require an ID, filename, and series ID")
        if self.decisions <= 0 or self.games != 2 * len(self.replay_ids):
            raise ValueError("Shard counts must contain two perspectives per replay")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShardIndexEntry:
        require_dataclass_fields(value, cls)
        return cls(**{**value, "replay_ids": tuple(value["replay_ids"])})


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """Small dataset index referencing complete compiled series."""

    runtime_major: str
    shards: tuple[ShardIndexEntry, ...]
    diagnostics: Mapping[str, int]
    created_at: str
    dataset_id: str
    source_format_id: str
    build_config: Mapping[str, Any]
    source_series: Mapping[str, tuple[str, ...]]
    source_games: int
    accepted_games: int
    rejected_games: int
    artifact_schema: str = SHARD_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.artifact_schema != SHARD_ARTIFACT_SCHEMA:
            raise ValueError(f"Unsupported shard artifact schema {self.artifact_schema!r}")
        if not self.dataset_id or not self.runtime_major or not self.source_format_id:
            raise ValueError("Dataset requires an ID, runtime reference, and source format")
        replay_ids = [replay for members in self.source_series.values() for replay in members]
        if len(set(replay_ids)) != len(replay_ids) or len(replay_ids) != self.source_games:
            raise ValueError("source_series must contain every source replay exactly once")
        if self.accepted_games + self.rejected_games != self.source_games:
            raise ValueError("Accepted and rejected games must account for every source game")
        if self.games != self.accepted_games * 2:
            raise ValueError("Shard game counts must contain two perspectives per accepted game")
        seen = set()
        for entry in self.shards:
            if entry.series_id in seen:
                raise ValueError(f"Series {entry.series_id!r} spans multiple shards")
            seen.add(entry.series_id)
            if self.source_series.get(entry.series_id) != entry.replay_ids:
                raise ValueError("Shard entry does not match source series")
        _require_iso_timestamp(self.created_at, "ShardManifest.created_at")

    @property
    def decisions(self) -> int:
        return sum(entry.decisions for entry in self.shards)

    @property
    def games(self) -> int:
        return sum(entry.games for entry in self.shards)

    @property
    def series(self) -> int:
        return len(self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "shards": [entry.to_dict() for entry in self.shards],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShardManifest:
        require_dataclass_fields(value, cls)
        return cls(
            **{
                **value,
                "shards": tuple(ShardIndexEntry.from_dict(entry) for entry in value["shards"]),
                "source_series": {key: tuple(ids) for key, ids in value["source_series"].items()},
            }
        )


def load_shard_manifest(value: Mapping[str, Any]) -> ShardManifest:
    validate_artifact_runtime_contract(value)
    return ShardManifest.from_dict(value)


def validate_shard_tensors(tensors: Mapping[str, Any]) -> None:
    """
    Check a shard tensor payload against the frozen layout above.

    Shared by compilation and loading so a writer cannot emit a payload the
    reader would reject.
    """
    expected = {
        name: (shape, dtype)
        for name, shape, dtype in (
            *observation_field_specs(),
            *final_observation_field_specs(),
            *SHARD_TENSOR_SPECS,
        )
    }

    if set(tensors) != set(expected):
        raise ValueError(
            f"Shard tensor fields mismatch; missing={sorted(set(expected) - set(tensors))}, "
            f"unknown={sorted(set(tensors) - set(expected))}"
        )

    for name, (shape, dtype) in expected.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise ValueError(f"Shard tensor {name} has an invalid type or dtype")
        if len(tensor.shape) != len(shape) or any(
            declared != -1 and actual != declared
            for actual, declared in zip(tensor.shape, shape, strict=True)
        ):
            raise ValueError(
                f"Shard tensor {name} has shape {tuple(tensor.shape)}, expected {shape}"
            )

    decisions = tensors["loss_mask"].shape[0]
    candidate_offsets = tensors["candidate_offsets"]

    if decisions == 0:
        raise ValueError("Published replay shards must contain at least one decision")

    decision_fields = {name for name, _, _ in observation_field_specs()} | {
        "action_mask",
        "mask_provenance",
        "label_kind",
        "label_confidence",
        "loss_mask",
        "decision_type",
        "exact_action",
        "outcome",
    }
    if any(tensors[name].shape[0] != decisions for name in decision_fields):
        raise ValueError("Shard decision tensors must have the same leading dimension")

    if (
        candidate_offsets.shape != (decisions + 1,)
        or candidate_offsets[0].item() != 0
        or candidate_offsets[-1].item() != tensors["candidate_values"].shape[0]
        or torch.any(candidate_offsets[1:] < candidate_offsets[:-1])
    ):
        raise ValueError("Shard candidate_offsets must bound every candidate row")

    for name in ("game_offsets", "series_offsets"):
        offsets = tensors[name]
        if (
            offsets.numel() < 2
            or offsets.shape[0] > decisions + 1
            or offsets[0].item() != 0
            or offsets[-1].item() != decisions
            or torch.any(offsets[1:] <= offsets[:-1])
        ):
            raise ValueError(f"Shard {name} must increase from zero to the decision count")

    games = tensors["game_offsets"].numel() - 1
    if any(tensors[name].shape[0] != games for name, _, _ in final_observation_field_specs()):
        raise ValueError("Shard final observations must hold exactly one row per game")

    game_boundaries = set(tensors["game_offsets"].tolist())
    if any(offset not in game_boundaries for offset in tensors["series_offsets"].tolist()):
        raise ValueError("Shard series_offsets must fall on game boundaries")

    if not torch.isfinite(tensors["label_confidence"]).all():
        raise ValueError("Shard label_confidence contains non-finite values")
    if torch.any((tensors["label_confidence"] < 0) | (tensors["label_confidence"] > 1)):
        raise ValueError("Shard label_confidence must be in [0, 1]")

    if (
        not torch.isfinite(tensors["loss_mask"]).all()
        or not torch.isfinite(tensors["outcome"]).all()
    ):
        raise ValueError("Shard scalar targets contain non-finite values")

    for name, _, _ in observation_field_specs():
        tensor = tensors[name]
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Shard observation field {name} contains non-finite values")

    action_mask = tensors["action_mask"]
    if torch.any(~action_mask.any(dim=-1)):
        raise ValueError("Every action slot must contain at least one legal action")

    if torch.any(tensors["mask_provenance"] != int(MaskProvenance.CONSERVATIVE_RECONSTRUCTED)):
        raise ValueError("Shard mask_provenance contains an unsupported value")

    label_kind = tensors["label_kind"]
    counts = candidate_offsets[1:] - candidate_offsets[:-1]
    exact = label_kind == int(LabelKind.EXACT)
    partial = label_kind == int(LabelKind.PARTIAL)
    unknown = label_kind == int(LabelKind.UNKNOWN)

    if torch.any(~(exact | partial | unknown)):
        raise ValueError("Shard label_kind contains an unsupported value")
    if torch.any(exact & (counts != 1)):
        raise ValueError("EXACT labels must have exactly one candidate")
    if torch.any(partial & (counts < 2)):
        raise ValueError("PARTIAL labels must have at least two candidates")
    if torch.any(unknown & (counts != 0)):
        raise ValueError("UNKNOWN labels must not have candidates")
    if torch.any(unknown & (tensors["loss_mask"] != 0)):
        raise ValueError("UNKNOWN labels must have zero loss")
    if torch.any((exact | partial) & (tensors["loss_mask"] <= 0)):
        raise ValueError("Labeled decisions must have positive loss")

    candidates = tensors["candidate_values"]
    if torch.any((candidates < 0) | (candidates >= ACT_SIZE)):
        raise ValueError("Shard candidate action ids are outside the action contract")

    if candidates.numel():
        if torch.any(tensors["exact_action"][exact] != candidates[candidate_offsets[:-1][exact]]):
            raise ValueError("Shard exact actions must match their only candidate")
        owners = torch.repeat_interleave(torch.arange(decisions), counts)
        legal = action_mask[owners, 0, candidates[:, 0]] & action_mask[owners, 1, candidates[:, 1]]

        same_switch = (
            (candidates[:, 0] >= 1)
            & (candidates[:, 0] <= 6)
            & (candidates[:, 0] == candidates[:, 1])
        )
        # Only one mega per turn, but the same id range encodes team-preview pairs,
        # where both actions legitimately fall inside it.
        is_turn = tensors["decision_type"][owners] != int(DecisionType.TEAM_PREVIEW)
        mega_first = (
            is_turn
            & (candidates[:, 0] >= MEGA_MOVE_START)
            & (candidates[:, 0] <= MEGA_FORCED_ACTION)
        )
        mega_second = (
            is_turn
            & (candidates[:, 1] >= MEGA_MOVE_START)
            & (candidates[:, 1] <= MEGA_FORCED_ACTION)
        )

        if torch.any(~legal | same_switch | (mega_first & mega_second)):
            raise ValueError("Shard contains an illegal labeled candidate")

    decision_type = tensors["decision_type"]
    if torch.any(
        ~torch.isin(
            decision_type,
            torch.tensor(
                tuple(int(value) for value in DecisionType if value), device=decision_type.device
            ),
        )
    ):
        raise ValueError("Shard decision_type contains an unsupported value")
    if torch.any((tensors["outcome"] < -1) | (tensors["outcome"] > 1)):
        raise ValueError("Shard outcomes must be in [-1, 1]")
