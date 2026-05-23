import torch

from src.lookups import ACT_SIZE
from src.model.policy import PolicyNet
from src.model.structured_observation import CATEGORICAL_WIDTH, NUMERICAL_WIDTH, SEQUENCE_LENGTH


def test_policy_net_forward_pass():
    print("create PolicyNet")
    # smaller dim than actual
    net = PolicyNet(
        obs_dim=(SEQUENCE_LENGTH, NUMERICAL_WIDTH),
        act_size=ACT_SIZE,
        d_model=128,
        nhead=4,
        nlayer=2,
    )
    print("created PolicyNet successfully")

    B = 16
    # dummy observation
    obs = {
        "categorical": torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        "numerical": torch.zeros((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH), dtype=torch.float32),
        "token_type_ids": torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        "side_ids": torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        "slot_ids": torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
    }

    print("forward pass start")
    with torch.no_grad():
        logits, log_probs, sampled_actions, value, next_state = net(obs)

    # Assert output shapes are correct
    assert logits.shape == (B, 2, ACT_SIZE), (
        f"Expected logits shape ({B}, 2, {ACT_SIZE}), got {logits.shape}"
    )
    assert log_probs.shape == (B,), f"Expected log_probs shape ({B},), got {log_probs.shape}"
    assert sampled_actions.shape == (B, 2), (
        f"Expected sampled_actions shape ({B}, 2), got {sampled_actions.shape}"
    )
    assert value.shape == (B,), f"Expected value shape ({B},), got {value.shape}"

    print("forward pass complete")
