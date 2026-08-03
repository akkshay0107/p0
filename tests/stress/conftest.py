from __future__ import annotations

import os

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources


def _stress_devices() -> tuple[torch.device, ...]:
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
    """Run device-sensitive tests on CPU and on CUDA when it is available."""
    return request.param


@pytest.fixture
def stress_policy(stress_device: torch.device):
    """Build a small policy so scaling tests remain practical on CPU-only hosts."""
    policy = build_policy(
        ModelConfig(d_model=32, nhead=2, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    return policy.to(stress_device).eval()
