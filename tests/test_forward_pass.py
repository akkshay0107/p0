import pytest
import torch

from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.structured_observation import CATEGORICAL_WIDTH, NUMERICAL_WIDTH, SEQUENCE_LENGTH


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
    obs = {
        "categorical": torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        "numerical": torch.zeros((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH), dtype=torch.float32),
        "token_type_ids": torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        "side_ids": torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        "slot_ids": torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
    }

    with torch.no_grad():
        logits, log_probs, sampled_actions, value, next_state = policy_net(obs)

    # output shapes
    assert logits.shape == (B, 2, ACT_SIZE)
    assert log_probs.shape == (B,)
    assert sampled_actions.shape == (B, 2)
    assert value.shape == (B,)

    # recurrent state check
    cls, hg = next_state
    # cls is enc[:, 0] which is (B, d_model)
    assert cls.shape == (B, 128)
    assert hg.shape == (B, 4, 128)  # n_hg is 4


def test_policy_net_forward_tokens(policy_net):
    B = 16
    tokens = torch.randn((B, SEQUENCE_LENGTH, 128))
    is_tp = torch.zeros(B, dtype=torch.bool)

    with torch.no_grad():
        logits, log_probs, sampled_actions, value, next_state = policy_net.forward_tokens(
            tokens, is_tp
        )

    assert logits.shape == (B, 2, ACT_SIZE)
    assert value.shape == (B,)
