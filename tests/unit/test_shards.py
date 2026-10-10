"""Tests for replay shard schema, manifest validation, and tensor invariants."""

from __future__ import annotations

import pytest
import torch

from p0.battle.actions import ACT_SIZE
from p0.format_config import active_runtime_contract
from p0.replays.schema import MaskProvenance
from p0.replays.shards import (
    ShardIndexEntry,
    ShardManifest,
    final_observation_field_specs,
    load_shard_manifest,
    observation_field_specs,
    validate_shard_tensors,
)


def _valid_shard_entry() -> ShardIndexEntry:
    return ShardIndexEntry(
        shard_id="shard-1",
        filename="shard-1.pt",
        series_id="series-1",
        replay_ids=("game-1", "game-2"),
        decisions=10,
        games=4,
    )


def _valid_shard_manifest() -> ShardManifest:
    global_sha = active_runtime_contract().major_sha256
    return ShardManifest(
        runtime_major=global_sha,
        shards=(_valid_shard_entry(),),
        diagnostics={"replays": 2, "accepted_games": 2, "rejected_games": 0},
        created_at="2026-01-01T00:00:00Z",
        dataset_id="b" * 64,
        source_format_id="gen9championsvgc2026regmcbo3",
        build_config={"max_candidates": 256},
        source_series={"series-1": ("game-1", "game-2")},
        source_games=2,
        accepted_games=2,
        rejected_games=0,
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
    for name, shape, dtype in final_observation_field_specs():
        tensors[name] = torch.zeros((1, *shape[1:]), dtype=dtype)
    return tensors


class TestShardIndexEntry:
    def test_round_trip_dict(self) -> None:
        entry = _valid_shard_entry()
        assert ShardIndexEntry.from_dict(entry.to_dict()) == entry

    def test_invalid_counts(self) -> None:
        value = _valid_shard_entry().to_dict()
        with pytest.raises(ValueError, match="two perspectives"):
            ShardIndexEntry.from_dict({**value, "games": 3})


class TestShardManifest:
    def test_round_trip_dict(self) -> None:
        manifest = _valid_shard_manifest()
        assert ShardManifest.from_dict(manifest.to_dict()) == manifest
        assert manifest.decisions == 10
        assert manifest.games == 4
        assert manifest.series == 1

    def test_loader_rejects_runtime_contract_and_obsolete_field(self) -> None:
        manifest = _valid_shard_manifest()
        value = manifest.to_dict()
        assert load_shard_manifest(value) == manifest

        with pytest.raises(ValueError, match="incompatible"):
            load_shard_manifest({**value, "runtime_major": "0" * 64})
        with pytest.raises(ValueError, match="unknown"):
            load_shard_manifest({**value, "runtime_manifest_sha256": "0" * 64})

    def test_rejects_partition_mismatch(self) -> None:
        data = _valid_shard_manifest().to_dict()
        data["source_series"] = {"series-1": ["game-1"]}
        with pytest.raises(ValueError, match="source_series"):
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

    def test_mismatched_decision_dimensions_raise(self) -> None:
        tensors = _valid_tensors()
        tensors["outcome"] = tensors["outcome"][:1]
        with pytest.raises(ValueError, match="same leading dimension"):
            validate_shard_tensors(tensors)

    def test_exact_action_must_match_candidate(self) -> None:
        tensors = _valid_tensors()
        tensors["exact_action"][0] = torch.tensor([0, 0])
        with pytest.raises(ValueError, match="exact actions must match"):
            validate_shard_tensors(tensors)

    def test_series_offsets_must_follow_game_boundaries(self) -> None:
        tensors = _valid_tensors()
        tensors["series_offsets"] = torch.tensor([0, 1, 2], dtype=torch.long)
        with pytest.raises(ValueError, match="fall on game boundaries"):
            validate_shard_tensors(tensors)
