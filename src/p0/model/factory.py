"""Single construction path for new policies."""

from __future__ import annotations

from p0.model.config import ModelConfig
from p0.model.policy import PolicyNet
from p0.model.resources import RuntimeResources, default_runtime_resources


class PolicyFactory:
    def __init__(self, resources: RuntimeResources | None = None):
        self.resources = resources or default_runtime_resources()

    def create(self, config: ModelConfig | None = None) -> PolicyNet:
        return PolicyNet(config=config or ModelConfig.baseline(), resources=self.resources)
