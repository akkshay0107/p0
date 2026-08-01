"""Single construction path for new policies."""

from __future__ import annotations

from typing import cast

import torch

from p0.model.config import ModelConfig
from p0.model.fused_token_encoder import FusedTokenEncoder
from p0.model.policy import ActorPolicy, PolicyNet, ValueHead
from p0.model.resources import RuntimeResources
from p0.model.series_context import DynamicSeriesResampler


def build_policy(config: ModelConfig, resources: RuntimeResources) -> PolicyNet:
    """Construct a policy from validated architecture and runtime resources."""
    return PolicyNet(config=config, resources=resources)


def compile_policy(
    policy: PolicyNet,
    *,
    enable: bool = True,
    dynamic: bool = True,
) -> PolicyNet:
    """Apply sub-module torch.compile to high-frequency execution paths."""
    if not enable or policy.device.type != "cuda":
        return policy

    policy.encoder = cast(FusedTokenEncoder, torch.compile(policy.encoder, dynamic=dynamic))
    policy.actor = cast(ActorPolicy, torch.compile(policy.actor, dynamic=dynamic))
    policy.critic = cast(ValueHead, torch.compile(policy.critic, dynamic=dynamic))
    policy.series = cast(DynamicSeriesResampler, torch.compile(policy.series, dynamic=dynamic))
    return policy
