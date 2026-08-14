from __future__ import annotations

import pytest
import torch

from tests.stress._helpers import (
    stress_batch_sizes,
    stress_dex_catalog,
    stress_int,
    stress_random_raw_events,
    stress_random_replay_payloads,
    stress_random_team_record,
    stress_rng,
)
from tests.stress.conftest import _stress_devices


@pytest.mark.stress
def test_stress_batch_sizes_deduplicate_without_reordering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P0_STRESS_BATCHES", "8, 1, 8, 32")

    assert stress_batch_sizes() == (8, 1, 32)


@pytest.mark.stress
@pytest.mark.parametrize("value", ["", "0,2", "1,-1"])
def test_stress_batch_sizes_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("P0_STRESS_BATCHES", value)

    with pytest.raises(ValueError, match="P0_STRESS_BATCHES"):
        stress_batch_sizes()


@pytest.mark.stress
def test_stress_int_enforces_the_configured_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P0_STRESS_CONTROL", "2")

    with pytest.raises(ValueError, match="at least 3"):
        stress_int("P0_STRESS_CONTROL", 1, minimum=3)


@pytest.mark.stress
def test_random_team_records_stay_inside_the_active_dex() -> None:
    catalog = stress_dex_catalog()
    rng = stress_rng()

    records = tuple(stress_random_team_record(rng, label=f"helper-{index}") for index in range(32))
    assert len({record.team.team_hash for record in records}) > 1
    for record in records:
        assert len(record.team.members) == 6
        assert {member.species for member in record.team.members} <= set(catalog.species)
        assert {member.item for member in record.team.members} <= set(catalog.items)
        assert {member.ability for member in record.team.members} <= set(catalog.abilities)
        assert {member.nature for member in record.team.members} <= set(catalog.natures)
        assert all(set(member.moves) <= set(catalog.moves) for member in record.team.members)
        assert all(sum(spread.as_dict().values()) <= 66 for spread in record.spreads)


@pytest.mark.stress
def test_random_event_corpus_covers_parser_shapes_without_golden_reuse() -> None:
    events = stress_random_raw_events(stress_rng())
    tags = {event.message[1] for event in events}

    assert {"move", "switch", "drag", "-damage", "-heal", "-activate", "cant"} <= tags
    assert any(
        event.message[3] == "not-a-real-move" for event in events if event.message[1] == "move"
    )
    assert any(event.pre_hp is None for event in events if event.message[1] == "-damage")


@pytest.mark.stress
def test_random_replay_payload_builder_rejects_empty_workloads() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        stress_random_replay_payloads(
            stress_rng(),
            0,
            replay_prefix="empty",
            series_prefix="empty-series",
        )


@pytest.mark.stress
def test_stress_device_selection_deduplicates_and_rejects_unknown_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P0_STRESS_DEVICES", "cpu,cpu")
    assert _stress_devices() == (torch.device("cpu"),)

    monkeypatch.setenv("P0_STRESS_DEVICES", "cpu,tpu")
    with pytest.raises(ValueError, match="Unsupported stress-test device"):
        _stress_devices()
