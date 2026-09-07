"""Contract tests for the pinned replay protocol surface."""

from __future__ import annotations

from copy import deepcopy

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
        assert PROTOCOL_CONTRACT["showdown_commit"] == SHOWDOWN_COMMIT
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

    def test_stored_stat_predicate_is_unsupported(self) -> None:
        event = parse_protocol_event("r", _line("|-activate|p1a: A|move: Power Split"))
        assert event.classification is EventClassification.UNSUPPORTED_STATE

    def test_raw_inventory_contains_source_shaped_witnesses(self) -> None:
        validate_raw_emission_inventory()
        ohko = [entry for entry in RAW_EMISSION_INVENTORY["entries"] if entry["tag"] == "-ohko"]
        assert any(entry["arguments"] == ["'-ohko'"] for entry in ohko)
        pressure = [
            entry
            for entry in RAW_EMISSION_INVENTORY["entries"]
            if "Pressure" in entry["expression"]
        ]
        assert any(entry["path"] == "data/abilities.ts" for entry in pressure)
        spite = [
            entry for entry in RAW_EMISSION_INVENTORY["entries"] if "Spite" in entry["expression"]
        ]
        assert any(
            entry["data_owner"] == "spite" and entry["reachability"] != "excluded"
            for entry in spite
        )

    def test_compiled_volatile_metadata_checks_existence_and_copy_hooks(self) -> None:
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
        assert rows["substitute"]["source"]["line"] is not None

    def test_stateful_witness_key_mutation_fails_closed(self) -> None:
        mutated = deepcopy(PROTOCOL_CONTRACT)
        mutated["stateful_witnesses"].pop("-activate")
        try:
            validate_protocol_contract(mutated)
        except ValueError as exc:
            assert "stateful witness keys" in str(exc)
        else:
            raise AssertionError("removing a reachable stateful witness must fail validation")
