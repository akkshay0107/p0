from __future__ import annotations

import json
import math
import subprocess
from typing import Any

import pytest

from p0.teams.validation import (
    PersistentShowdownValidator,
    showdown_payload,
    validate_many_batched,
)
from tests.stress._helpers import stress_count, stress_rng
from tests.unit.test_teams import _variant_team_validation_batch


@pytest.mark.stress
def test_batched_team_validation_preserves_identity_and_payload_contract() -> None:
    count = stress_count("P0_STRESS_TEAM_VARIANTS", 1024)
    species = ("Pikachu", "Raichu", "Zapdos", "Miraidon", "Gholdengo")
    rng = stress_rng()
    variants = tuple(
        _variant_team_validation_batch(rng.choice(species), item=f"Stress Item {index}")
        for index in range(count)
    )
    calls: list[list[dict[str, Any]]] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args
        payload = json.loads(kwargs["input"])
        offset = sum(len(batch) for batch in calls)
        calls.append(payload)
        response = [
            {
                "valid": (offset + index) % 2 == 0,
                "packedTeam": f"packed-{offset + index}",
                "problems": [],
            }
            for index, _ in enumerate(payload)
        ]
        return subprocess.CompletedProcess("node", 0, stdout=json.dumps(response), stderr="")

    batch_size = stress_count("P0_STRESS_TEAM_BATCH_SIZE", 64)
    results = validate_many_batched(variants, batch_size=batch_size, runner=runner)
    assert [result.team_hash for result in results] == [
        variant.team.team_hash for variant in variants
    ]
    assert [result.valid for result in results] == [index % 2 == 0 for index in range(count)]
    assert len(calls) == math.ceil(count / batch_size)
    assert sum(map(len, calls)) == count
    payload = json.loads(showdown_payload(variants[0]))
    assert payload["team"]
    assert len(payload["team"]) == 6
    assert {"species", "moves", "evs", "ivs"} <= payload["team"][0].keys()


@pytest.mark.stress
def test_persistent_validator_handles_repeated_batches_and_closes_worker() -> None:
    count = stress_count("P0_STRESS_PERSISTENT_TEAM_VARIANTS", 512)
    species = ("Pikachu", "Raichu", "Zapdos", "Miraidon", "Gholdengo")
    rng = stress_rng()
    variants = tuple(
        _variant_team_validation_batch(rng.choice(species), item=f"Persistent Stress Item {index}")
        for index in range(count)
    )
    writes: list[dict[str, Any]] = []
    closed = False

    class Stream:
        def write(self, value: str) -> None:
            nonlocal closed
            payload = json.loads(value)
            writes.append(payload)
            if payload.get("command") == "stop":
                closed = True

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

        def readline(self) -> str:
            batch = writes[-1]["batch"]
            return (
                json.dumps(
                    {
                        "status": "ok",
                        "results": [
                            {"valid": True, "packedTeam": f"packed-{i}", "problems": []}
                            for i, _ in enumerate(batch)
                        ],
                    }
                )
                + "\n"
            )

    class Process:
        def __init__(self) -> None:
            self.stdin = Stream()
            self.stdout = self.stdin
            self.stderr = Stream()

        def wait(self, timeout: float) -> int:
            del timeout
            return 0

        def terminate(self) -> None:
            pass

    with PersistentShowdownValidator(popen_factory=lambda *args, **kwargs: Process()) as validator:
        batch_size = stress_count("P0_STRESS_PERSISTENT_TEAM_BATCH_SIZE", 32)
        results = validator.validate_many(variants, batch_size=batch_size)
        assert len(results) == count
        assert [result.team_hash for result in results] == [
            variant.team.team_hash for variant in variants
        ]
        assert [len(request["batch"]) for request in writes] == [
            *([batch_size] * (count // batch_size)),
            *(([count % batch_size]) if count % batch_size else []),
        ]
    assert closed
