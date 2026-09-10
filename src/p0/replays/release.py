"""Quality gates for publishing compiled replay datasets."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from p0.replays.schema import LabelKind

SHOWDOWN_COMMIT = "8282e63102fa824fd2f7472778ec09793ceb7cac"


class ReleaseGateStatus(StrEnum):
    PASSED = "passed"
    UNMET = "unmet"


@dataclass(frozen=True, slots=True)
class ReleaseGateReport:
    """Release gate status and check results."""

    status: ReleaseGateStatus
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]

    @property
    def releasable(self) -> bool:
        return self.status is ReleaseGateStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "releasable": self.releasable,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


def validate_rejection_categories(categories: Iterable[str]) -> tuple[str, ...]:
    """Validate that all rejection reasons are recognized."""
    allowed = {"UNSUPPORTED_EVENT", "AMBIGUOUS_IDENTITY", "INVALID_INPUT_CONTRACT"}
    invalid = tuple(sorted(set(categories) - allowed))
    if invalid:
        raise ValueError(f"Unexpected replay rejection categories: {invalid!r}")
    return tuple(sorted(set(categories)))


def run_pinned_showdown_oracle(
    *,
    repository_root: str | Path,
    timeout_seconds: float = 10.0,
) -> tuple[str, ...]:
    """Run the local Showdown BattleStream reference check."""
    root = Path(repository_root)
    showdown_root = root / "pokemon-showdown"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=showdown_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        ).stdout.strip()
        if revision != SHOWDOWN_COMMIT:
            raise ValueError(f"Pinned Showdown commit mismatch: {revision!r}")
        result = subprocess.run(
            ["node", str(root / "scripts" / "showdown_battlestream_oracle.js")],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError("Pinned Showdown BattleStream oracle failed") from exc
    if payload.get("commit") != SHOWDOWN_COMMIT or not isinstance(payload.get("protocol"), list):
        raise ValueError("BattleStream oracle returned an invalid pinned payload")
    return tuple(str(line) for line in payload["protocol"])


def evaluate_release_gates(
    *,
    rejection_categories: Iterable[str],
    label_kinds: Iterable[LabelKind],
    loss_masks: Iterable[float],
    sensitivity_data_configured: bool = False,
    sensitivity_evaluation_unchanged: bool = False,
) -> ReleaseGateReport:
    """Evaluate dataset quality gates against rejection reasons and label masks."""
    categories = validate_rejection_categories(rejection_categories)
    kinds = tuple(label_kinds)
    masks = tuple(float(mask) for mask in loss_masks)
    if len(kinds) != len(masks):
        raise ValueError("label_kinds and loss_masks must have equal lengths")
    label_contract = all(
        (kind is LabelKind.UNKNOWN and mask == 0.0)
        or (kind in {LabelKind.EXACT, LabelKind.PARTIAL} and mask > 0.0)
        for kind, mask in zip(kinds, masks, strict=True)
    )
    sensitivity_ready = sensitivity_data_configured and sensitivity_evaluation_unchanged
    checks = {
        "rejection_categories": not categories
        or all(
            category in {"UNSUPPORTED_EVENT", "AMBIGUOUS_IDENTITY", "INVALID_INPUT_CONTRACT"}
            for category in categories
        ),
        "label_loss_contract": label_contract,
        "training_sensitivity": sensitivity_ready,
    }
    reasons = (
        ("training sensitivity data is not configured or evaluation is not unchanged",)
        if not sensitivity_ready
        else ()
    )
    return ReleaseGateReport(
        ReleaseGateStatus.PASSED if all(checks.values()) else ReleaseGateStatus.UNMET,
        checks,
        reasons,
    )


__all__ = [
    "ReleaseGateReport",
    "ReleaseGateStatus",
    "SHOWDOWN_COMMIT",
    "evaluate_release_gates",
    "run_pinned_showdown_oracle",
    "validate_rejection_categories",
]
