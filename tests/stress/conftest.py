from __future__ import annotations

import pytest
import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from tests.stress._helpers import stress_devices


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
