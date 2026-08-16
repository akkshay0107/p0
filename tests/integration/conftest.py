from __future__ import annotations

import os

import pytest
import torch
from poke_env import LocalhostServerConfiguration, ServerConfiguration

from p0.evaluation.harness import DEFAULT_TEST_TEAM
from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.paths import DEFAULT_PATHS
from p0.runtime.showdown import allocate_loopback_ports, start_showdown_servers
from p0.teams.source import ValidatedTeam


@pytest.fixture(scope="session")
def battle_format() -> str:
    """Canonical battle format identifier (e.g. gen9ou) under test."""
    return FORMAT.battle_format


@pytest.fixture(scope="session")
def sample_team() -> str:
    """Return a packed team string converted from standard Showdown export text.

    Packed format is required for direct wire transmission when initializing
    Showdown player sessions.
    """
    return ValidatedTeam.from_showdown(DEFAULT_TEST_TEAM).packed


@pytest.fixture(scope="function")
def showdown_server():
    """Start a local ephemeral Showdown server process for live battle integration tests.

    Dynamically binds an unused loopback port to prevent port collisions between
    concurrent test workers, launching an isolated Showdown server instance for the test lifecycle.
    """
    if not DEFAULT_PATHS.showdown_root.exists():
        pytest.skip("pokemon-showdown directory not found. Skipping live server tests.")

    port = allocate_loopback_ports(1)[0]
    server_configuration = ServerConfiguration(
        websocket_url=f"ws://localhost:{port}/showdown/websocket",
        authentication_url=LocalhostServerConfiguration.authentication_url,
    )
    with start_showdown_servers(1, ports=(port,)):
        yield server_configuration


def _integration_devices() -> tuple[torch.device, ...]:
    """Parse requested test devices from P0_INTEGRATION_DEVICES environment variable.

    Defaults to checking both CPU and CUDA (if CUDA is actually available on the host).
    Deduplicates device entries while preserving order, falling back to CPU if empty.
    """
    requested = tuple(
        value.strip().lower()
        for value in os.getenv("P0_INTEGRATION_DEVICES", "cpu,cuda").split(",")
        if value.strip()
    )
    devices: list[torch.device] = []
    for name in requested:
        if name == "cpu":
            devices.append(torch.device("cpu"))
        elif name == "cuda" and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        elif name != "cuda":
            raise ValueError(f"Unsupported integration-test device {name!r}")
    return tuple(dict.fromkeys(devices)) or (torch.device("cpu"),)


@pytest.fixture(params=_integration_devices(), ids=lambda device: device.type)
def model_device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrize tests across available execution devices (e.g., CPU, CUDA)."""
    return request.param


@pytest.fixture
def model_policy(model_device: torch.device):
    """Instantiate a minimal Transformer policy in eval mode for integration testing.

    Uses small dimensions (d_model=32, 1 reducer layer) to keep forward pass execution
    fast while exercising the complete forward pipeline and action masking on the target device.
    """
    policy = build_policy(
        ModelConfig(d_model=32, nhead=2, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    return policy.to(model_device).eval()
