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

    obs = from_battle(battle, tokenizer)

    numerical = obs.numerical
    if numerical.dim() == 2:
        numerical = numerical.unsqueeze(0)

    with torch.no_grad():
        batched_obs = obs.unsqueeze(0)
        logits, log_probs, sampled_actions, value, next_state = policy_net(batched_obs)
        assert logits.shape == (1, 2, ACT_SIZE)
