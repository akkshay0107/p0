"""Contract tests for the pinned replay protocol surface."""

from __future__ import annotations

import subprocess
from copy import deepcopy

import pytest

from p0.paths import DEFAULT_PATHS
from p0.replays.reconstruction.classification import (
    CLASSIFICATION_REGISTRY,
    EventClassification,
)
from p0.replays.reconstruction.contract import (
    PROTOCOL_CONTRACT,
    RAW_EMISSION_INVENTORY,
    SHOWDOWN_COMMIT,
    validate_protocol_contract,
    validate_raw_emission_inventory,
)
from p0.replays.reconstruction.events import parse_protocol_event
from p0.replays.schema import ProtocolLine


def _line(raw: str) -> ProtocolLine:
    return ProtocolLine(0, raw, tuple(raw.split("|")), None)


class TestReplayProtocolContract:
    def test_inventory_is_total_for_classifiers_and_pinned(self) -> None:
        checked_out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=DEFAULT_PATHS.showdown_root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

        assert (
            PROTOCOL_CONTRACT["showdown_commit"]
            == RAW_EMISSION_INVENTORY["showdown_commit"]
            == SHOWDOWN_COMMIT
            == checked_out
        )
        assert {entry["tag"] for entry in PROTOCOL_CONTRACT["entries"]} == set(
            CLASSIFICATION_REGISTRY
        )

    def test_state_neutral_source_shapes(self) -> None:
        assert (
            parse_protocol_event("r", _line("|-ohko")).classification
            is EventClassification.NO_STATE_CHANGE
        )
        assert (
            parse_protocol_event("r", _line("|bigerror|warning")).classification
            is EventClassification.NO_STATE_CHANGE
        )
        assert (
            parse_protocol_event("r", _line("|-sethp|p1a: A|50/100|[silent]")).classification
            is EventClassification.PUBLIC_STATE
        )

    def test_malformed_shapes_fail_closed(self) -> None:
        assert (
            parse_protocol_event("r", _line("|-ohko|p1a: A")).classification
            is EventClassification.MALFORMED
        )
        assert (
            parse_protocol_event("r", _line("|-sethp|p1a: A|50/100|extra")).classification
            is EventClassification.MALFORMED
        )

    def test_missing_source_file_and_unresolved_site_fail_coverage(self) -> None:
        missing_file = deepcopy(RAW_EMISSION_INVENTORY)
        missing_file["files"] = [
            item for item in missing_file["files"] if item["path"] != "data/rulesets.ts"
        ]
        with pytest.raises(ValueError, match="source files"):
            validate_raw_emission_inventory(missing_file)

        unresolved = deepcopy(RAW_EMISSION_INVENTORY)
        counts = unresolved["reachability_counts"]
        counts[unresolved["entries"][0]["reachability"]] -= 1
        counts["unresolved"] = 1
        unresolved["entries"][0]["reachability"] = "unresolved"
        with pytest.raises(ValueError, match="unresolved"):
            validate_raw_emission_inventory(unresolved)

        missing_witness = deepcopy(PROTOCOL_CONTRACT)
        removed_id = next(
            entry["id"]
            for entry in missing_witness["raw_witnesses"]
            if entry["path"] == "sim/battle.ts" and entry["resolved_tags"] == ["teampreview"]
        )
        missing_witness["raw_witnesses"] = [
            row for row in missing_witness["raw_witnesses"] if row["id"] != removed_id
        ]
        with pytest.raises(ValueError, match="source witnesses"):
            validate_protocol_contract(missing_witness)

    def test_compiled_volatile_metadata_checks_existence_and_copy_hooks(self) -> None:
        validate_raw_emission_inventory()
        rows = {row["id"]: row for row in RAW_EMISSION_INVENTORY["volatile_conditions"]}
        assert rows["substitute"]["exists"] is True
        assert rows["substitute"]["noCopy"] is False
        assert rows["stockpile"]["noCopy"] is True
        assert rows["trapped"]["noCopy"] is True
        assert rows["dragoncheer"]["noCopy"] is False
        assert rows["gastroacid"]["onCopy"] is True
        assert rows["typechange"]["exists"] is False
        assert rows["typeadd"]["exists"] is False
        for condition_id in ("aquaring", "ingrain", "leechseed", "magnetrise"):
            assert rows[condition_id]["exists"] is True
            assert rows[condition_id]["reachable"] is True
        assert rows["substitute"]["source"]["path"] == "data/moves.ts"
