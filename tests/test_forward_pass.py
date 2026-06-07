import pytest
import torch

from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)


@pytest.fixture
def policy_net():
    return PolicyNet(
        obs_dim=(SEQUENCE_LENGTH, NUMERICAL_WIDTH),
        act_size=ACT_SIZE,
        d_model=128,
        nhead=4,
        nlayer=2,
    )


def test_policy_net_forward_pass(policy_net):
    B = 16
    # dummy observation
    obs = StructuredObservation(
        categorical=torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        numerical=torch.zeros((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH), dtype=torch.float32),
        token_type_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        side_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        slot_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
    )

    # Populate valid orig_idxs to prevent random switch actions from crashing
    ally_indices = [1, 3, 5, 7, 9, 11]
    for i, idx in enumerate(ally_indices):
        obs.numerical[:, idx + 1, 26] = (i + 1) / 6.0

    with torch.no_grad():
        logits, log_probs, sampled_actions, value, next_state = policy_net(obs)

    # output shapes
    assert logits.shape == (B, 2, ACT_SIZE)
    assert log_probs.shape == (B,)
    assert sampled_actions.shape == (B, 2)
    assert value.shape == (B,)

    # recurrent state check
    hg = next_state
    assert hg.shape == (B, 4, 128)  # n_hg is 4


def test_policy_net_forward_tokens(policy_net):
    B = 16
    tokens = torch.randn((B, SEQUENCE_LENGTH, 128))
    aux = torch.randn((B, 4, 128))
    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))

    # Populate valid orig_idxs to prevent random switch actions from crashing
    ally_indices = [1, 3, 5, 7, 9, 11]
    for i, idx in enumerate(ally_indices):
        numerical[:, idx + 1, 26] = (i + 1) / 6.0

    with torch.no_grad():
        logits, log_probs, sampled_actions, value, next_state = policy_net.forward_tokens(
            tokens, aux, numerical
        )

    assert logits.shape == (B, 2, ACT_SIZE)
    assert value.shape == (B,)


def test_sequential_mask_fallback(policy_net):
    logits = torch.randn((1, 2, ACT_SIZE))
    action_mask = torch.zeros((1, 2, ACT_SIZE), dtype=torch.bool)
    action_mask[:, 0, 0] = True
    action1 = torch.tensor([0])
    is_tp = torch.zeros(1, dtype=torch.bool)

    masked_logits = policy_net.actor._apply_sequential_masks(logits, action1, action_mask, is_tp)

    assert torch.isfinite(masked_logits[0, 1, 0])
    assert torch.isneginf(masked_logits[0, 1, 1:]).all()
