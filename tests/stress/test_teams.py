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
from tests.stress._helpers import stress_count, stress_random_team_record, stress_rng


class _BatchedValidationRunner:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args
        payload = json.loads(kwargs["input"])
        offset = sum(len(batch) for batch in self.calls)
        self.calls.append(payload)
        self.payloads.extend(payload)
        response = [
            {
                "valid": (offset + index) % 2 == 0,
                "packedTeam": f"packed-{offset + index}",
                "problems": [],
            }
            for index, _ in enumerate(payload)
        ]
        return subprocess.CompletedProcess("node", 0, stdout=json.dumps(response), stderr="")


class _PersistentValidatorStream:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    def write(self, value: str) -> None:
        payload = json.loads(value)
        if payload.get("command") == "stop":
            self.closed = True
        else:
            self.requests.append(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def readline(self) -> str:
        batch = self.requests[-1]["batch"]
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


class _PersistentValidatorProcess:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.stdin = _PersistentValidatorStream()
        self.stdout = self.stdin
        self.stderr = _PersistentValidatorStream()

    def wait(self, timeout: float) -> int:
        del timeout
        return 0

    def terminate(self) -> None:
        pass


class _PersistentValidatorProcessFactory:
    def __init__(self) -> None:
        self.process: _PersistentValidatorProcess | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> _PersistentValidatorProcess:
        self.process = _PersistentValidatorProcess(*args, **kwargs)
        return self.process


@pytest.mark.stress
def test_batched_team_validation_preserves_identity_and_payload_contract() -> None:
    count = stress_count("P0_STRESS_TEAM_VARIANTS", 1024)
    rng = stress_rng()
    variants = tuple(
        stress_random_team_record(rng, label=f"batched-stress-{index}") for index in range(count)
    )
    batch_size = stress_count("P0_STRESS_TEAM_BATCH_SIZE", 64)
    runner = _BatchedValidationRunner()
    results = validate_many_batched(variants, batch_size=batch_size, runner=runner)
    assert [result.team_hash for result in results] == [
        variant.team.team_hash for variant in variants
    ]
    assert [result.valid for result in results] == [index % 2 == 0 for index in range(count)]
    assert len(runner.calls) == math.ceil(count / batch_size)
    assert all(runner.calls)
    assert sum(map(len, runner.calls)) == count
    assert len(runner.payloads) == count
    assert all(payload["format"] for payload in runner.payloads)
    assert {member["species"] for payload in runner.payloads for member in payload["team"]} == {
        member.canonical().species for variant in variants for member in variant.team.members
    }
    payload = json.loads(showdown_payload(variants[0]))
    assert payload["team"]
    assert len(payload["team"]) == 6
    assert {"species", "moves", "evs", "ivs"} <= payload["team"][0].keys()


@pytest.mark.stress
def test_persistent_validator_handles_repeated_batches_and_closes_worker() -> None:
    batch_size = stress_count("P0_STRESS_PERSISTENT_TEAM_BATCH_SIZE", 32)
    count = stress_count("P0_STRESS_PERSISTENT_TEAM_VARIANTS", 512)
    if count % batch_size == 0:
        count += 1
    rng = stress_rng()
    variants = tuple(
        stress_random_team_record(rng, label=f"persistent-stress-{index}") for index in range(count)
    )
    factory = _PersistentValidatorProcessFactory()
    with PersistentShowdownValidator(popen_factory=factory) as validator:
        results = validator.validate_many(variants, batch_size=batch_size)
        assert len(results) == count
        assert [result.team_hash for result in results] == [
            variant.team.team_hash for variant in variants
        ]
        assert factory.process is not None
        assert [len(request["batch"]) for request in factory.process.stdin.requests] == [
            *([batch_size] * (count // batch_size)),
            *(([count % batch_size]) if count % batch_size else []),
        ]
    assert factory.process.stdin.closed
