"""Tests for Magnet frozen anchor policy and lifecycle."""

from __future__ import annotations

import copy

import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import MemoryInputs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.magnet import Magnet
from p0.training.ppo import magnet_kl_per_step


def _tiny_policy() -> PolicyNet:
    return build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())


def _batch(
    policy: PolicyNet, batch_size: int = 3
) -> tuple[StructuredObservation, torch.Tensor, torch.Tensor]:
    obs = StructuredObservation.empty_batch(batch_size)
    masks = torch.ones((batch_size, 2, policy.act_size), dtype=torch.bool)
    actions = torch.zeros((batch_size, 2), dtype=torch.long)
    return obs, masks, actions


def _live_and_magnet_logits(
    policy: PolicyNet,
    magnet: Magnet,
    obs: StructuredObservation,
    masks: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    live_encoded = policy.encode(obs, masks)
    live_memory = MemoryInputs.empty(
        obs.numerical.size(0),
        policy.d_model,
        policy.device,
        next(policy.parameters()).dtype,
    )
    live = policy.evaluate(policy.prepare(live_encoded, live_memory), masks, actions).logits
    magnet_encoded = magnet.policy.encode(obs, masks)
    magnet_memory = MemoryInputs.empty(
        obs.numerical.size(0),
        magnet.policy.d_model,
        magnet.policy.device,
        next(magnet.policy.parameters()).dtype,
    )
    mag = magnet.policy.evaluate(
        magnet.policy.prepare(magnet_encoded, magnet_memory), masks, actions
    ).logits
    return live, mag


class TestMagnetLifecycle:
    def test_magnet_anchor_lifecycle_immutability_and_refresh(self) -> None:
        """Verify Magnet initializes frozen at zero-KL, remains immutable under optimizer steps, tracks drift, and resets on refresh."""
        torch.manual_seed(0)
        policy = _tiny_policy()
        magnet = Magnet(policy)

        # 1. Anchor parameters must have requires_grad=False
        assert all(not p.requires_grad for p in magnet.policy.parameters())

        # 2. Initial KL between policy and magnet must be zero
        obs, masks, actions = _batch(policy)
        with torch.no_grad():
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl = magnet_kl_per_step(live, mag)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

        # 3. Live optimizer step does not perturb anchor weights
        snapshot = copy.deepcopy(magnet.policy.state_dict())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)

        encoded = policy.encode(obs, masks)
        empty_mem = MemoryInputs.empty(
            obs.numerical.size(0), policy.d_model, policy.device, next(policy.parameters()).dtype
        )
        loss = policy.evaluate(policy.prepare(encoded, empty_mem), masks, actions).value.sum()
        loss.backward()
        optimizer.step()

        for key, value in magnet.policy.state_dict().items():
            assert torch.equal(value, snapshot[key])

        # 4. Drifting policy weights increases KL divergence
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn_like(p) * 0.05)
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl_drifted = magnet_kl_per_step(live, mag)
        assert (kl_drifted > 1e-4).any()

        # 5. Refreshing magnet resets KL to 0.0 and preserves live optimizer moments
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        magnet.refresh(policy)
        after_optimizer = optimizer.state_dict()

        for pid, moments in before_optimizer["state"].items():
            for key, value in moments.items():
                if torch.is_tensor(value):
                    assert torch.equal(value, after_optimizer["state"][pid][key])

        with torch.no_grad():
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl_after = magnet_kl_per_step(live, mag)
        assert torch.allclose(kl_after, torch.zeros_like(kl_after), atol=1e-5)
