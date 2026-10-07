"""Single construction path for new policies."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
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
    actor = policy.actor
    # PolicyNet.evaluate and magnet refresh use the logits-only entrypoint;
    # compile it explicitly before wrapping ActorPolicy's forward path.
    setattr(actor, "logits", torch.compile(actor.logits, dynamic=dynamic))
    policy.actor = cast(ActorPolicy, torch.compile(actor, dynamic=dynamic))
    policy.critic = cast(ValueHead, torch.compile(policy.critic, dynamic=dynamic))
    policy.series = cast(DynamicSeriesResampler, torch.compile(policy.series, dynamic=dynamic))
    return policy


def _strip_orig_mod(name: str) -> str:
    return name.replace("._orig_mod.", ".").removeprefix("_orig_mod.")


def canonical_policy_state_dict(policy: PolicyNet) -> OrderedDict[str, torch.Tensor]:
    """Return policy state dict with torch.compile prefixes removed."""
    return OrderedDict((_strip_orig_mod(k), v) for k, v in policy.state_dict().items())


def load_canonical_policy_state_dict(
    policy: PolicyNet,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load weights with canonical parameter names into a policy module."""
    target_names = {name: _strip_orig_mod(name) for name in policy.state_dict()}
    canonical_targets = set(target_names.values())
    source_names = set(state_dict)
    if source_names != canonical_targets:
        missing = sorted(canonical_targets - source_names)
        unexpected = sorted(source_names - canonical_targets)
        raise RuntimeError(
            "Canonical policy state keys do not match: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    policy.load_state_dict(
        {name: state_dict[canonical] for name, canonical in target_names.items()}, strict=True
    )
