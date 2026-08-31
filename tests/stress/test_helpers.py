from __future__ import annotations

import pytest
import torch

from tests.stress._helpers import (
    stress_batch_sizes,
    stress_dex_catalog,
    stress_int,
    stress_random_replay_payloads,
    stress_random_team_record,
    stress_rng,
)
from tests.stress.conftest import _stress_devices


@pytest.mark.stress
def test_stress_batch_sizes_deduplicate_without_reordering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that batch size string parser removes duplicates while preserving original order."""
    monkeypatch.setenv("P0_STRESS_BATCHES", "8, 1, 8, 32")

    assert stress_batch_sizes() == (8, 1, 32)


@pytest.mark.stress
@pytest.mark.parametrize("value", ["", "0,2", "1,-1"])
def test_stress_batch_sizes_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Verify that empty, zero, or negative batch sizes raise ValueError."""
    monkeypatch.setenv("P0_STRESS_BATCHES", value)

    with pytest.raises(ValueError, match="P0_STRESS_BATCHES"):
        stress_batch_sizes()


@pytest.mark.stress
def test_stress_int_enforces_the_configured_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that stress_int raises ValueError when parsed env integer violates the minimum threshold."""
    monkeypatch.setenv("P0_STRESS_CONTROL", "2")

    with pytest.raises(ValueError, match="at least 3"):
        stress_int("P0_STRESS_CONTROL", 1, minimum=3)


@pytest.mark.stress
def test_random_team_records_stay_inside_the_active_dex() -> None:
    """
    Verify that randomly generated synthetic team records satisfy all dex legality constraints.

    Checks that species, items, abilities, natures, and moves belong to the active format catalog,
    and that stat spreads satisfy the 66-point EV budget rule.
    """
    catalog = stress_dex_catalog()
    rng = stress_rng()

    records = tuple(stress_random_team_record(rng, label=f"helper-{index}") for index in range(32))
    # Ensure random sampling produced diverse distinct teams rather than degenerate repeats
    assert len({record.team.team_hash for record in records}) > 1
    pools = {pool.species: pool for pool in catalog.species_pools}
    for record in records:
        assert len(record.team.members) == 6
        assert len({pools[member.species].base_species for member in record.team.members}) == 6
        assert len({member.item for member in record.team.members}) == 6
        assert {member.species for member in record.team.members} <= set(catalog.species)
        for member in record.team.members:
            pool = pools[member.species]
            assert member.item in pool.items
            assert member.ability in pool.abilities
            assert set(member.moves) <= set(pool.moves)
            assert 1 <= len(member.moves) <= 4
            if len(pool.moves) >= 2:
                assert 2 <= len(member.moves) <= min(4, len(pool.moves))
        assert {member.item for member in record.team.members} <= set(catalog.items)
        assert {member.ability for member in record.team.members} <= set(catalog.abilities)
        assert {member.nature for member in record.team.members} <= set(catalog.natures)
        assert all(set(member.moves) <= set(catalog.moves) for member in record.team.members)
        # Sum of 6 stat points (HP, Atk, Def, SpA, SpD, Spe) must not exceed total EV point allowance
        assert all(sum(spread.as_dict().values()) <= 66 for spread in record.spreads)


@pytest.mark.stress
def test_random_replay_payload_builder_rejects_empty_workloads() -> None:
    """Verify that requesting 0 replay payloads raises ValueError."""
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
    """Verify that P0_STRESS_DEVICES parses, deduplicates, and rejects unsupported hardware targets."""
    monkeypatch.setenv("P0_STRESS_DEVICES", "cpu,cpu")
    assert _stress_devices() == (torch.device("cpu"),)

    monkeypatch.setenv("P0_STRESS_DEVICES", "cpu,tpu")
    with pytest.raises(ValueError, match="Unsupported stress-test device"):
        _stress_devices()
