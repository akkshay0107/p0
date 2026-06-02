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


def test_policy_net_padding_mask_real(policy_net):
    import sys
    from pathlib import Path

    from poke_env.battle.status import Status

    sys.path.append(str(Path(__file__).parent))
    from test_observation import make_real_battle, make_real_pokemon

    from src.model.observation_builder import from_battle
    from src.model.tokenizer import tokenizer

    p1 = make_real_pokemon(species="charizard", status=None)
    p2_fainted = make_real_pokemon(species="camerupt", status=Status.FNT)
    p_bench = make_real_pokemon(species="pikachu")

    battle = make_real_battle(
        active_pokemon=[p1, p2_fainted],
        opponent_active_pokemon=[None, None],
        team=[p1, p2_fainted, p_bench],
        opponent_team=[],
        teampreview=False,
    )

    obs = from_battle(battle, tokenizer, as_dict=True)

    numerical = obs["numerical"]
    if numerical.dim() == 2:
        numerical = numerical.unsqueeze(0)

    padding_mask = policy_net._get_padding_mask(numerical)

    assert padding_mask.shape == (1, SEQUENCE_LENGTH)

    # 0: CLS
    # 1: P1 Super, 2: P1 Numeric
    # 3: P2 Super, 4: P2 Numeric
    # 5: Bench Super, 6: Bench Numeric
    # Since P2 is fainted, indices 3 and 4 should be True (ignored)
    assert padding_mask[0, 1].item() is False  # p1 super
    assert padding_mask[0, 2].item() is False  # p1 numeric

    assert padding_mask[0, 3].item() is True  # p2_fainted super
    assert padding_mask[0, 4].item() is True  # p2_fainted numeric

    assert padding_mask[0, 5].item() is False  # p_bench super
    assert padding_mask[0, 6].item() is False  # p_bench numeric

    assert padding_mask[0, 25].item() is False
    assert padding_mask[0, 26].item() is False
    assert padding_mask[0, 27].item() is False

    with torch.no_grad():
        batched_obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        logits, log_probs, sampled_actions, value, next_state = policy_net(batched_obs)
        assert logits.shape == (1, 2, ACT_SIZE)
