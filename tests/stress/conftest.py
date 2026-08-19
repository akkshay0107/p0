from __future__ import annotations

import os

import pytest
import torch
from poke_env import LocalhostServerConfiguration, ServerConfiguration

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.runtime.showdown import allocate_loopback_ports, start_showdown_servers


@pytest.fixture(scope="function")
def showdown_server(showdown_assets):
    """Start live stress tests only when the checked-out server is runnable.

    Dynamically binds an unused loopback port to prevent port collisions between
    concurrent test workers, launching an isolated Showdown server instance for the test lifecycle.
    """
    del showdown_assets

    # Allocate a fresh ephemeral port per-test so parallel stress jobs cannot cross-connect
    port = allocate_loopback_ports(1)[0]
    server_configuration = ServerConfiguration(
        websocket_url=f"ws://localhost:{port}/showdown/websocket",
        authentication_url=LocalhostServerConfiguration.authentication_url,
    )
    with start_showdown_servers(1, ports=(port,), build_assets=False):
        yield server_configuration


def _stress_devices() -> tuple[torch.device, ...]:
    """Parse requested stress test compute devices from P0_STRESS_DEVICES.

    Filters out CUDA when host hardware lacks GPU support, deduplicates entries,
    and falls back to CPU.
    """
    requested = tuple(
        value.strip().lower()
        for value in os.getenv("P0_STRESS_DEVICES", "cpu,cuda").split(",")
        if value.strip()
    )
    devices: list[torch.device] = []
    for name in requested:
        if name == "cpu":
            devices.append(torch.device("cpu"))
        elif name == "cuda" and torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        elif name != "cuda":
            raise ValueError(f"Unsupported stress-test device {name!r}")
    if not devices:
        return (torch.device("cpu"),)
    return tuple(dict.fromkeys(devices))


@pytest.fixture(params=_stress_devices(), ids=lambda device: device.type)
def stress_device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrize device-sensitive stress tests across CPU and available CUDA accelerators."""
    return request.param


@pytest.fixture
def stress_policy(stress_device: torch.device):
    """Build a lightweight Transformer policy for fast multi-step stress workloads on CPU or GPU."""
    policy = build_policy(
        ModelConfig(d_model=32, nhead=2, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    return policy.to(stress_device).eval()
