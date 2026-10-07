"""Tests for replay schema and artifact validation."""

from __future__ import annotations

import pytest
import torch

from p0.battle.events import SPATIAL_SLOT_COUNT
from p0.model.architecture_contract import (
    CURRENT_REDUCER_TOKEN_COUNT,
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    POOLED_EVENT_COUNT,
    REDUCER_MAX_LENGTH,
    SERIES_SLOTS,
)
from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_NUMERICAL_WIDTH,
    MAX_EFFECTS,
    MAX_EVENT_RECORDS,
    NUM_IDX_EFFECT_COUNT,
    NUM_IDX_EFFECT_OVERFLOW,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
    TokenType,
)
from p0.replays.schema import (
    ActionEvidence,
    FetchIndexEntry,
    GroupingMethod,
    LabelKind,
    MaskProvenance,
    SeriesRecord,
)
from p0.replays.shards import (
    SHARD_TENSOR_SPECS,
    observation_field_specs,
)


def _evidence(kind: LabelKind) -> ActionEvidence:
    candidates = {
        LabelKind.EXACT: ((7, 1),),
        LabelKind.PARTIAL: ((7, 1), (8, 1)),
        LabelKind.UNKNOWN: (),
    }[kind]
    return ActionEvidence(
        label_kind=kind,
        candidates=candidates,
        confidence=0.5 if kind is not LabelKind.UNKNOWN else 0.0,
        mask_provenance=MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        tags=("fixture",),
    )


def _series_record() -> SeriesRecord:
    return SeriesRecord(
        series_id="s1",
        format_id="gen9championsvgc2026regmbbo3",
        players=("alice", "bob"),
        game_replay_ids=("r1", "r2"),
        game_player_roles=((0, 1), (1, 0)),
        team_hashes=("a" * 64, "b" * 64),
        is_complete=True,
        score=(2, 0),
        grouping_method=GroupingMethod.PARENT_ROOM,
        grouping_confidence=1.0,
    )


class TestReplaySchemas:
    def test_evidence_shapes(self) -> None:
        """Verify ActionEvidence enforces candidate counts and ranges according to EXACT/PARTIAL/UNKNOWN taxonomy."""
        assert _evidence(LabelKind.EXACT).exact_action == (7, 1)
        with pytest.raises(ValueError, match="only defined for EXACT"):
            _evidence(LabelKind.PARTIAL).exact_action
        with pytest.raises(ValueError, match="exactly one candidate"):
            ActionEvidence(LabelKind.EXACT, (), 1.0, MaskProvenance.CONSERVATIVE_RECONSTRUCTED)
        with pytest.raises(ValueError, match="two or more"):
            ActionEvidence(
                LabelKind.PARTIAL,
                ((7, 1),),
                0.5,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )
        with pytest.raises(ValueError, match="no candidates"):
            ActionEvidence(
                LabelKind.UNKNOWN,
                ((7, 1),),
                0.0,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )
        with pytest.raises(ValueError, match="outside"):
            ActionEvidence(
                LabelKind.EXACT,
                ((49, 0),),
                1.0,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )
        with pytest.raises(ValueError, match="Duplicate"):
            ActionEvidence(
                LabelKind.PARTIAL,
                ((7, 1), (7, 1)),
                0.5,
                MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            )

    def test_ir_round_trips(self) -> None:
        """Verify SeriesRecord and FetchIndexEntry serialize and deserialize cleanly."""
        series = _series_record()
        assert SeriesRecord.from_dict(series.to_dict()) == series
        fetch = FetchIndexEntry(
            replay_id="r1",
            format_id="gen9championsvgc2026regmbbo3",
            source_url="https://replay.pokemonshowdown.com/r1",
            fetched_at="2026-07-17T00:00:00Z",
            http_status=200,
            content_sha256="d" * 64,
            byte_size=100,
        )
        assert FetchIndexEntry.from_dict(fetch.to_dict()) == fetch

    def test_ir_rejects_bad_serializations(self) -> None:
        """Verify IR deserialization raises ValueError on schema version mismatch or missing/unknown fields."""
        payload = _series_record().to_dict()
        del payload["score"]
        payload["bogus"] = 1
        with pytest.raises(ValueError, match=r"missing=\['score'\], unknown=\['bogus'\]"):
            SeriesRecord.from_dict(payload)

    def test_ir_validates_construction(self) -> None:
        """Verify SeriesRecord validates structural invariants on deserialization."""
        with pytest.raises(ValueError, match="two wins"):
            SeriesRecord.from_dict({**_series_record().to_dict(), "score": [1, 0]})
        with pytest.raises(ValueError, match="include a timezone"):
            FetchIndexEntry(
                replay_id="r1",
                format_id="format",
                source_url="https://example.invalid/r1",
                fetched_at="2026-07-17T00:00:00",
                http_status=200,
                content_sha256="d" * 64,
                byte_size=1,
            )

    def test_observation_specs_are_derived(self) -> None:
        """Verify public observation and shard specs expose the runtime tensor contract."""
        specs = observation_field_specs()
        assert [spec[0] for spec in specs] == [
            "token_type_ids",
            "side_ids",
            "slot_ids",
            "categorical",
            "numerical",
            "spatial_cat",
            "spatial_num",
        ]
        observation = StructuredObservation.empty_batch(1)
        assert all(shape[0] == -1 for _, shape, _ in specs)
        assert all(
            dtype is tensor.dtype
            for (_, _, dtype), tensor in zip(specs, observation.tensors(), strict=True)
        )
        assert [spec[0] for spec in SHARD_TENSOR_SPECS] == [
            "action_mask",
            "mask_provenance",
            "label_kind",
            "label_confidence",
            "loss_mask",
            "decision_type",
            "exact_action",
            "candidate_values",
            "candidate_offsets",
            "game_offsets",
            "series_offsets",
            "outcome",
        ]

    def test_fixed_memory_and_observation_contract(self) -> None:
        """Verify the persisted architecture dimensions and the serialized observation layout."""
        assert SEQUENCE_LENGTH == 15
        assert (CATEGORICAL_WIDTH, NUMERICAL_WIDTH) == (60, 116)
        assert (SPATIAL_SLOT_COUNT, MAX_EVENT_RECORDS) == (4, 32)
        assert set(TokenType) == {TokenType.POKEMON, TokenType.FIELD, TokenType.EVENT}
        assert (CURRENT_TOKEN_COUNT, CURRENT_REDUCER_TOKEN_COUNT, REDUCER_MAX_LENGTH) == (
            20,
            21,
            77,
        )
        assert (HISTORY_WINDOW, SERIES_SLOTS, POOLED_EVENT_COUNT) == (48, 8, 4)

        observation = StructuredObservation.empty_batch(2)
        assert observation.token_type_ids.shape == (2, SEQUENCE_LENGTH)
        assert observation.spatial_cat.shape == (2, MAX_EVENT_RECORDS, EVENT_CATEGORICAL_WIDTH)
        assert observation.spatial_num.shape == (2, MAX_EVENT_RECORDS, EVENT_NUMERICAL_WIDTH)
        for (_, shape, _), tensor in zip(
            observation_field_specs(), observation.tensors(), strict=True
        ):
            assert tuple(tensor.shape[1:]) == tuple(shape[1:])

    def test_observation_overflow_contract_holds_at_capacity_boundaries(self) -> None:
        """Verify validate_overflow_contract verifies effect overflow totals and rejects mismatches."""
        observation = StructuredObservation.empty_batch(1)[0]
        observation.numerical[:, NUM_IDX_EFFECT_COUNT] = torch.tensor(
            (0,) * 12 + (MAX_EFFECTS, MAX_EFFECTS + 2, 0),
            dtype=torch.float32,
        )
        observation.numerical[:, NUM_IDX_EFFECT_OVERFLOW] = torch.tensor(
            (0.0,) * 12 + (0.0, 2.0, 0.0),
            dtype=torch.float32,
        )
        observation.validate_overflow_contract()
        assert observation.overflow_totals() == (2, 0)

        observation.numerical[13, NUM_IDX_EFFECT_OVERFLOW] = 1.0
        with pytest.raises(
            ValueError, match="Effect overflow does not match the number of dropped effects"
        ):
            observation.validate_overflow_contract()

    def test_schema_modules_stay_pure(self) -> None:
        """Verify intermediate representation modules stay pure without importing torch or runtime."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import p0.replays.schema, p0.battle.series\n"
            "assert 'torch' not in sys.modules, 'IR layer must stay torch-free'\n"
            "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
            "import p0.replays.shards\n"
            "assert not any(m.startswith('p0.runtime') for m in sys.modules)\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
