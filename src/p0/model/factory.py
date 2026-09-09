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


def canonical_policy_state_dict(policy: PolicyNet) -> OrderedDict[str, torch.Tensor]:
    """Return policy weights without torch.compile wrapper namespaces."""
    canonical: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in policy.state_dict().items():
        canonical[name.replace("._orig_mod.", ".").removeprefix("_orig_mod.")] = value
    return canonical


def load_canonical_policy_state_dict(
    policy: PolicyNet,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load canonical weights into either compiled or uncompiled policy modules."""
    target_names = policy.state_dict()
    canonical_targets = {
        target_name.replace("._orig_mod.", ".").removeprefix("_orig_mod.")
        for target_name in target_names
    }
    source_names = set(state_dict)
    if source_names != canonical_targets:
        missing = sorted(canonical_targets - source_names)
        unexpected = sorted(source_names - canonical_targets)
        raise RuntimeError(
            "Canonical policy state keys do not match: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    wrapped = {
        target_name: state_dict[canonical_name]
        for target_name in target_names
        if (canonical_name := target_name.replace("._orig_mod.", ".").removeprefix("_orig_mod."))
        in state_dict
    }
    if len(wrapped) != len(target_names):
        missing = sorted(set(target_names) - set(wrapped))
        raise RuntimeError(f"Canonical policy state is missing keys: {missing[:3]}")
    policy.load_state_dict(wrapped, strict=True)
