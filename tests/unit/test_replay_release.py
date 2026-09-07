from __future__ import annotations

import pytest

from p0.replays.release import (
    ReleaseGateStatus,
    evaluate_release_gates,
    validate_rejection_categories,
)
from p0.replays.schema import LabelKind


class TestReplayReleaseGates:
    def test_unknown_has_zero_weight_and_partial_is_marginalized(self) -> None:
        report = evaluate_release_gates(
            rejection_categories=("UNSUPPORTED_EVENT",),
            label_kinds=(LabelKind.EXACT, LabelKind.PARTIAL, LabelKind.UNKNOWN),
            loss_masks=(1.0, 1.0, 0.0),
            sensitivity_data_configured=True,
            sensitivity_evaluation_unchanged=True,
        )
        assert report.status is ReleaseGateStatus.PASSED
        assert report.releasable

    def test_sensitivity_gate_remains_unmet_without_high_risk_data(self) -> None:
        report = evaluate_release_gates(
            rejection_categories=(),
            label_kinds=(LabelKind.EXACT,),
            loss_masks=(1.0,),
        )
        assert report.status is ReleaseGateStatus.UNMET
        assert report.checks["training_sensitivity"] is False

    def test_generic_rejection_bucket_is_a_build_failure(self) -> None:
        with pytest.raises(ValueError, match="Unexpected replay rejection"):
            validate_rejection_categories(("KeyError",))
