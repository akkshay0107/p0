from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from p0.teams.validation import (
    PersistentShowdownValidator,
    showdown_payload,
    validate_many_batched,
    validate_variant,
)
from tests.unit.test_teams import _variant_team_validation_batch


@pytest.mark.stress
def test_batched_team_validation_preserves_identity_and_payload_contract() -> None:
    variants = tuple(
        _variant_team_validation_batch(species)
        for species in ("Pikachu", "Raichu", "Zapdos", "Miraidon", "Gholdengo")
    )
    calls: list[list[dict[str, Any]]] = []

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args
        payload = json.loads(kwargs["input"])
        calls.append(payload)
        response = [
            {"valid": index % 2 == 0, "packedTeam": f"packed-{index}", "problems": []}
            for index, _ in enumerate(payload)
        ]
        return subprocess.CompletedProcess("node", 0, stdout=json.dumps(response), stderr="")

    results = validate_many_batched(variants, batch_size=2, runner=runner)
    assert [result.team_hash for result in results] == [
        variant.team.team_hash for variant in variants
    ]
    assert [result.valid for result in results] == [True, False, True, False, True]
    assert [len(batch) for batch in calls] == [2, 2, 1]
    payload = json.loads(showdown_payload(variants[0]))
    assert payload["team"]
    assert len(payload["team"]) == 6
    assert {"species", "moves", "evs", "ivs"} <= payload["team"][0].keys()


@pytest.mark.stress
def test_single_team_validation_reports_pinned_runner_failures() -> None:
    variant = _variant_team_validation_batch()

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess("node", 1, stdout="", stderr="invalid team")

    with pytest.raises(RuntimeError, match="validator failed: invalid team"):
        validate_variant(variant, runner=runner)


@pytest.mark.stress
def test_persistent_validator_handles_repeated_batches_and_closes_worker() -> None:
    variants = tuple(
        _variant_team_validation_batch(species) for species in ("Pikachu", "Raichu", "Zapdos")
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
        result = validator.validate_many(variants, batch_size=2)
        assert len(result) == 3
        assert [len(request["batch"]) for request in writes] == [2, 1]
    assert closed
