"""Tests for replay shard schema, manifest validation, and tensor invariants."""

from __future__ import annotations

import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.format_config import load_active_runtime_manifest
from p0.replays.schema import MaskProvenance
from p0.replays.shards import (
    ShardIndexEntry,
    ShardManifest,
    observation_field_specs,
    validate_shard_tensors,
)


def _valid_shard_entry() -> ShardIndexEntry:
    return ShardIndexEntry(
        filename="shard-00000.pt",
        sha256="a" * 64,
        decisions=10,
        games=2,
        series=1,
        byte_size=2048,
    )


def _valid_shard_manifest() -> ShardManifest:
    global_sha = load_active_runtime_manifest().global_sha256
    return ShardManifest(
        global_contract_sha256=global_sha,
        shards=(_valid_shard_entry(),),
        diagnostics={"replays": 2, "accepted_games": 2, "rejected_games": 0},
        created_at="2026-01-01T00:00:00Z",
        dataset_hash="b" * 64,
        source_format_id="gen9championsvgc2026regmbbo3",
        build_config={"max_candidates": 256},
        raw_replays={"game-1": "c" * 64, "game-2": "d" * 64},
        source_series={"series-1": ("game-1", "game-2")},
        source_games=2,
        accepted_games=2,
        rejected_games=0,
        artifact_hashes={"shard-00000.pt": "a" * 64},
    )


def _valid_tensors(decisions: int = 2) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, shape, dtype in observation_field_specs():
        concrete_shape = (decisions, *(s if s != -1 else 1 for s in shape[1:]))
        tensors[name] = torch.zeros(concrete_shape, dtype=dtype)

    action_mask = torch.zeros((decisions, 2, ACT_SIZE), dtype=torch.bool)
    action_mask[:, :, 0] = True  # Move 0 legal
    action_mask[:, :, 7] = True  # Move 7 legal

    tensors["action_mask"] = action_mask
    tensors["mask_provenance"] = torch.full(
        (decisions,), int(MaskProvenance.CONSERVATIVE_RECONSTRUCTED), dtype=torch.long
    )
    tensors["label_kind"] = torch.tensor([1, 2], dtype=torch.long)  # EXACT, PARTIAL
    tensors["label_confidence"] = torch.tensor([1.0, 0.8], dtype=torch.float32)
    tensors["loss_mask"] = torch.tensor([1.0, 1.0], dtype=torch.float32)
    tensors["decision_type"] = torch.tensor([1, 1], dtype=torch.long)  # TURN
    tensors["exact_action"] = torch.tensor([[7, 7], [-1, -1]], dtype=torch.long)
    tensors["candidate_values"] = torch.tensor([[7, 7], [7, 7], [0, 0]], dtype=torch.long)
    tensors["candidate_offsets"] = torch.tensor([0, 1, 3], dtype=torch.long)
    tensors["game_offsets"] = torch.tensor([0, decisions], dtype=torch.long)
    tensors["series_offsets"] = torch.tensor([0, decisions], dtype=torch.long)
    tensors["outcome"] = torch.tensor([1.0, -1.0], dtype=torch.float32)
    return tensors


class TestShardIndexEntry:
    def test_round_trip_dict(self) -> None:
        entry = _valid_shard_entry()
        assert ShardIndexEntry.from_dict(entry.to_dict()) == entry

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"filename": ""}, "must be non-empty"),
            ({"sha256": "not-sha256"}, "must be a lowercase SHA-256"),
            ({"decisions": -1}, "must be a nonnegative integer"),
            ({"games": -1}, "must be a nonnegative integer"),
            ({"series": -1}, "must be a nonnegative integer"),
            ({"byte_size": -1}, "must be a nonnegative integer"),
        ],
    )
    def test_validation_errors(self, kwargs: dict[str, object], match: str) -> None:
        base = {
            "filename": "shard-00000.pt",
            "sha256": "a" * 64,
            "decisions": 10,
            "games": 2,
            "series": 1,
            "byte_size": 2048,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ShardIndexEntry(**base)  # type: ignore[arg-type]


class TestShardManifest:
    def test_round_trip_dict(self) -> None:
        manifest = _valid_shard_manifest()
        assert ShardManifest.from_dict(manifest.to_dict()) == manifest
        assert manifest.decisions == 10
        assert manifest.games == 2
        assert manifest.series == 1

    def test_rejects_unsupported_schema(self) -> None:
        with pytest.raises(ValueError, match="Unsupported shard artifact schema"):
            manifest = _valid_shard_manifest()
            data = manifest.to_dict()
            data["artifact_schema"] = "unknown.v99"
            ShardManifest.from_dict(data)

    def test_rejects_partition_mismatch(self) -> None:
        manifest = _valid_shard_manifest()
        data = manifest.to_dict()
        data["raw_replays"]["game-extra"] = "e" * 64
        data["source_games"] = 3
        with pytest.raises(ValueError, match="partition all raw replays"):
            ShardManifest.from_dict(data)


class TestValidateShardTensors:
    def test_valid_tensors_pass(self) -> None:
        tensors = _valid_tensors()
        validate_shard_tensors(tensors)

    def test_missing_field_raises(self) -> None:
        tensors = _valid_tensors()
        del tensors["action_mask"]
        with pytest.raises(ValueError, match="Shard tensor fields mismatch"):
            validate_shard_tensors(tensors)

    def test_invalid_dtype_raises(self) -> None:
        tensors = _valid_tensors()
        tensors["loss_mask"] = tensors["loss_mask"].to(torch.float64)
        with pytest.raises(ValueError, match="invalid type or dtype"):
            validate_shard_tensors(tensors)

    def test_non_finite_values_raise(self) -> None:
        tensors = _valid_tensors()
        tensors["loss_mask"][0] = float("nan")
        with pytest.raises(ValueError, match="non-finite values"):
            validate_shard_tensors(tensors)

    def test_exact_label_candidate_count_mismatch_raises(self) -> None:
        tensors = _valid_tensors()
        # Decision 0 is EXACT, give it 2 candidates instead of 1
        tensors["candidate_offsets"] = torch.tensor([0, 2, 3], dtype=torch.long)
        with pytest.raises(ValueError, match="EXACT labels must have exactly one candidate"):
            validate_shard_tensors(tensors)

    def test_unknown_label_with_positive_loss_raises(self) -> None:
        tensors = _valid_tensors()
        tensors["label_kind"][0] = 3  # UNKNOWN
        tensors["candidate_offsets"] = torch.tensor([0, 0, 2], dtype=torch.long)
        tensors["candidate_values"] = torch.tensor([[7, 7], [0, 0]], dtype=torch.long)
        tensors["loss_mask"][0] = 1.0  # Should be 0.0
        with pytest.raises(ValueError, match="UNKNOWN labels must have zero loss"):
            validate_shard_tensors(tensors)
