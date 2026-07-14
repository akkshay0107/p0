"""Single construction path for new policies."""

from __future__ import annotations

from p0.model.config import ModelConfig
from p0.model.policy import PolicyNet
from p0.model.resources import RuntimeResources


def build_policy(config: ModelConfig, resources: RuntimeResources) -> PolicyNet:
    """Construct a policy from validated architecture and runtime resources."""
    return PolicyNet(config=config, resources=resources)
