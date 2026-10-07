from __future__ import annotations

import pytest
import torch
from hypothesis.internal.conjecture import engine as conjecture_engine
from poke_env import LocalhostServerConfiguration, ServerConfiguration

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.runtime.showdown import allocate_loopback_ports, start_showdown_servers
from tests.stress._helpers import stress_devices

# Stress strategies intentionally generate large tensor batches. Hypothesis' default
# choice buffer rejects the configured 128-by-64 case before the test body runs.
HYPOTHESIS_STRESS_BUFFER_SIZE = 256 * 1024
conjecture_engine.BUFFER_SIZE = HYPOTHESIS_STRESS_BUFFER_SIZE


@pytest.fixture(scope="function")
def showdown_server(showdown_assets):
    """
    Start live stress tests only when the checked-out server is runnable.

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


@pytest.fixture(params=stress_devices(), ids=lambda device: device.type)
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
