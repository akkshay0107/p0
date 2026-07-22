"""Tests for the MMD magnet: freezing, refresh semantics, and reverse-KL."""

import copy
from typing import Any

import torch

from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import StructuredObservation
from p0.training.magnet import Magnet
from p0.training.ppo import magnet_kl_per_step


def _tiny_policy():
    return build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())


def _batch(policy, batch_size=3):
    obs = StructuredObservation.empty_batch(batch_size)
    masks = torch.ones((batch_size, 2, policy.act_size), dtype=torch.bool)
    actions = torch.zeros((batch_size, 2), dtype=torch.long)
    return obs, masks, actions


def _live_and_magnet_logits(policy, magnet, obs, masks, actions):
    live = policy.evaluate_obs(obs, masks, actions, *policy.empty_memory(obs.numerical.size(0))).logits
    mag = magnet.policy.evaluate_obs(
        obs, masks, actions, *magnet.policy.empty_memory(obs.numerical.size(0))
    )
    return live, mag.logits


def test_magnet_params_are_frozen():
    policy = _tiny_policy()
    magnet = Magnet(policy)
    assert all(not p.requires_grad for p in magnet.policy.parameters())


def test_magnet_kl_is_zero_at_refresh():
    policy = _tiny_policy()
    magnet = Magnet(policy)
    obs, masks, actions = _batch(policy)
    with torch.no_grad():
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl = magnet_kl_per_step(live, mag)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_magnet_kl_grows_then_resets_after_refresh():
    torch.manual_seed(0)
    policy = _tiny_policy()
    magnet = Magnet(policy)
    obs, masks, actions = _batch(policy)

    # perturb the live policy so it diverges from the frozen magnet
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        policy.actor.pointer_temp.add_(0.5)
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl_drifted = magnet_kl_per_step(live, mag)
    assert (kl_drifted > 1e-4).any()

    magnet.refresh(policy)
    with torch.no_grad():
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl_after = magnet_kl_per_step(live, mag)
    assert torch.allclose(kl_after, torch.zeros_like(kl_after), atol=1e-5)


def test_refresh_does_not_perturb_optimizer_state():
    policy = _tiny_policy()
    magnet = Magnet(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    obs, masks, actions = _batch(policy)
    loss = policy.evaluate_obs(obs, masks, actions, *policy.empty_memory(obs.numerical.size(0))).value.sum()
    loss.backward()
    optimizer.step()

    before = copy.deepcopy(optimizer.state_dict())
    magnet.refresh(policy)
    after = optimizer.state_dict()

    for pid, moments in before["state"].items():
        for key, value in moments.items():
            if torch.is_tensor(value):
                assert torch.equal(value, after["state"][pid][key])


def test_magnet_frozen_under_live_optimizer_step():
    policy = _tiny_policy()
    magnet = Magnet(policy)
    snapshot = copy.deepcopy(magnet.policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)

    obs, masks, actions = _batch(policy)
    loss = policy.evaluate_obs(obs, masks, actions, *policy.empty_memory(obs.numerical.size(0))).value.sum()
    loss.backward()
    optimizer.step()

    for key, value in magnet.policy.state_dict().items():
        assert torch.equal(value, snapshot[key])


def test_magnet_kl_is_finite_for_degenerate_masks():
    policy = _tiny_policy()
    magnet = Magnet(policy)
    batch_size = 2
    obs = StructuredObservation.empty_batch(batch_size)
    # single legal action per slot: the KL must stay finite (zero contribution
    # from masked entries), never NaN
    masks = torch.zeros((batch_size, 2, policy.act_size), dtype=torch.bool)
    masks[:, :, 0] = True
    actions = torch.zeros((batch_size, 2), dtype=torch.long)
    with torch.no_grad():
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl = magnet_kl_per_step(live, mag)
    assert torch.isfinite(kl).all()


def test_magnet_kl_loss_sign_increases_with_divergence():
    # increasing KL(pi, rho) must increase the objective (opposite of the old
    # entropy bonus, which decreased it)
    from p0.training.config import TrainingConfig
    from p0.training.ppo import compute_ppo_objective

    config = TrainingConfig()
    common: dict[str, Any] = dict(
        current_log_probs=torch.zeros(2),
        current_values=torch.zeros(2),
        normalized_entropy=torch.zeros(2),
        old_log_probs=torch.zeros(2),
        advantages=torch.ones(2),
        returns=torch.zeros(2),
        team_preview=torch.tensor([False, False]),
        config=config,
    )
    low, *_ = compute_ppo_objective(
        magnet_kl=torch.zeros(2), alpha=0.5, critic_only=False, **common
    )
    high, *_ = compute_ppo_objective(
        magnet_kl=torch.ones(2), alpha=0.5, critic_only=False, **common
    )
    assert (high > low).all()
