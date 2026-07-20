"""Tests for batched and persistent Showdown validation."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from p0.teams.stat_points import StatPoints
from p0.teams.team import CanonicalTeam, TeamMember, TeamMetadata, TeamVariant
from p0.teams.validation import (
    PersistentShowdownValidator,
    validate_many,
    validate_many_batched,
    validate_variant,
)


def _variant(species: str = "Pikachu", item: str = "Light Ball") -> TeamVariant:
    members = (
        TeamMember(
            species=species,
            item=item,
            ability="Static",
            moves=("Fake Out", "Protect", "Thunderbolt", "Electroweb"),
            nature="Jolly",
        ),
        TeamMember(
            species="Charizard",
            item="Charizardite Y",
            ability="Blaze",
            moves=("Heat Wave", "Solar Beam", "Protect", "Weather Ball"),
            nature="Modest",
        ),
        TeamMember(
            species="Whimsicott",
            item="Focus Sash",
            ability="Prankster",
            moves=("Moonblast", "Tailwind", "Encore", "Protect"),
            nature="Timid",
        ),
        TeamMember(
            species="Garchomp",
            item="Sitrus Berry",
            ability="Rough Skin",
            moves=("Earthquake", "Dragon Claw", "Rock Slide", "Protect"),
            nature="Jolly",
        ),
        TeamMember(
            species="Kingambit",
            item="Black Glasses",
            ability="Defiant",
            moves=("Kowtow Cleave", "Sucker Punch", "Protect", "Low Kick"),
            nature="Adamant",
        ),
        TeamMember(
            species="Glimmora",
            item="Shuca Berry",
            ability="Toxic Debris",
            moves=("Power Gem", "Sludge Bomb", "Earth Power", "Protect"),
            nature="Modest",
        ),
    )
    return TeamVariant(
        team=CanonicalTeam(members),
        spreads=tuple(StatPoints(hp=2, spa=32, spe=32) for _ in members),
        metadata=TeamMetadata(
            source_series=("series-1",),
            source_replays=("series-1-game-1",),
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-02T00:00:00Z",
            usage_count=1,
            archetype_tags=("balance",),
        ),
    )


def test_validate_many_empty_returns_empty_tuple() -> None:
    calls: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs.get("input"))
        return subprocess.CompletedProcess(args[0], 0, stdout="[]", stderr="")

    assert validate_many((), runner=runner) == ()
    assert validate_many_batched((), runner=runner) == ()
    assert len(calls) == 0


def test_validate_many_batched_splits_chunks_and_preserves_order() -> None:
    calls: list[str] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        payload = str(kwargs.get("input", ""))
        calls.append(payload)
        items = json.loads(payload)
        response = [
            {"valid": True, "packedTeam": f"packed_{index}", "problems": []}
            for index, _ in enumerate(items)
        ]
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(response), stderr="")

    variants = (_variant("Pikachu"), _variant("Raichu"), _variant("Zapdos"))
    results = validate_many_batched(variants, batch_size=2, runner=runner)
    assert len(results) == 3
    assert [result.team_hash for result in results] == [
        variant.team.team_hash for variant in variants
    ]
    assert results[0].packed_team == "packed_0"
    assert len(calls) == 2
    assert len(json.loads(calls[0])) == 2
    assert len(json.loads(calls[1])) == 1


def test_validate_many_delegates_to_batched_runner() -> None:
    calls: list[str] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        payload = str(kwargs.get("input", ""))
        calls.append(payload)
        items = json.loads(payload)
        response = [{"valid": True, "packedTeam": "packed_team", "problems": []} for _ in items]
        return subprocess.CompletedProcess(args[0], 0, stdout=json.dumps(response), stderr="")

    variants = (_variant("Pikachu"), _variant("Raichu"))
    results = validate_many(variants, runner=runner)
    assert len(results) == 2
    assert len(calls) == 1
    assert len(json.loads(calls[0])) == 2


def test_validate_many_batched_handles_process_failure() -> None:
    def failing_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="Node error")

    with pytest.raises(RuntimeError, match="failed: Node error"):
        validate_many_batched((_variant(),), runner=failing_runner)


def test_validate_many_batched_handles_timeout() -> None:
    def timeout_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args[0], 30.0)

    with pytest.raises(RuntimeError, match="timed out"):
        validate_many_batched((_variant(),), runner=timeout_runner)


def test_validate_many_batched_handles_malformed_json() -> None:
    def malformed_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, stdout="not json", stderr="")

    with pytest.raises(RuntimeError, match="malformed response"):
        validate_many_batched((_variant(),), runner=malformed_runner)


def test_persistent_showdown_validator_lifecycle_and_validation() -> None:
    calls: list[str] = []
    closed = [False]

    class MockProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.stdin = self
            self.stdout = self
            self.stderr = self
            self.returncode = None

        def write(self, data: str) -> None:
            trimmed = data.strip()
            calls.append(trimmed)
            try:
                payload = json.loads(trimmed)
                if payload.get("command") == "stop":
                    closed[0] = True
            except Exception:
                pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def readline(self) -> str:
            if not calls:
                return ""
            payload = json.loads(calls[-1])
            if payload.get("command") == "stop":
                return ""
            items = payload.get("batch", [])
            response = [
                {"valid": True, "packedTeam": f"persistent_{index}", "problems": []}
                for index, _ in enumerate(items)
            ]
            return json.dumps({"status": "ok", "results": response}) + "\n"

        def terminate(self) -> None:
            closed[0] = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def mock_popen(*args: Any, **kwargs: Any) -> Any:
        return MockProcess()

    variants = (_variant("Pikachu"), _variant("Raichu"))
    with PersistentShowdownValidator(popen_factory=mock_popen) as validator:
        results = validator.validate_many(variants)
        assert len(results) == 2
        assert results[0].packed_team == "persistent_0"
        assert results[1].packed_team == "persistent_1"

    assert closed[0] is True
    assert len(calls) >= 1
    first_request = json.loads(calls[0])
    assert len(first_request["batch"]) == 2


@pytest.mark.integration
def test_batched_validator_integration_matches_single_validator() -> None:
    variants = (_variant("Pikachu"), _variant("Raichu"))
    single_results = tuple(validate_variant(variant) for variant in variants)
    batched_results = validate_many_batched(variants)
    assert len(single_results) == len(batched_results)
    for single, batched in zip(single_results, batched_results, strict=True):
        assert single.team_hash == batched.team_hash
        assert single.valid == batched.valid
        assert single.packed_team == batched.packed_team
        assert single.problems == batched.problems
