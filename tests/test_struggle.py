from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from poke_env.battle import DoubleBattle

from src.env import MegaEnv
from src.format_config import FORMAT
from src.model.policy import PolicyNet
from src.model.structured_observation import (
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)

ACT_SIZE = FORMAT.action_size


def test_struggle_env_roundtrip():
    # Mock move
    mock_struggle = SimpleNamespace(id="struggle")
    mock_tackle = SimpleNamespace(id="tackle")
    mock_active = SimpleNamespace(moves={"tackle": mock_tackle}, fainted=False)

    # Mock battle
    battle = cast(
        DoubleBattle,
        SimpleNamespace(
            player_username="player",
            battle_tag="battle",
            teampreview=False,
            _wait=False,
            force_switch=[False, False],
            trapped=[False, False],
            maybe_trapped=[False, False],
            active_pokemon=[mock_active, None],
            available_moves=[[mock_struggle], []],
            available_switches=[[], []],
            team={},
            can_mega_evolve=[False, False],
            valid_orders=[[], []],
            get_possible_showdown_targets=lambda move, mon: [0],
        ),
    )

    mask = MegaEnv.single_action_mask(battle, 0)
    assert mask == [48]

    order = MegaEnv._action_to_order_individual(np.int64(48), battle, fake=True, pos=0)
    assert cast(Any, order.order).id == "struggle"
    assert not order.mega

    action = MegaEnv._order_to_action_individual(order, battle, fake=True, pos=0)
    assert action == 48


def test_mega_struggle_env_roundtrip():
    mock_struggle = SimpleNamespace(id="recharge")
    mock_tackle = SimpleNamespace(id="tackle")
    mock_active = SimpleNamespace(moves={"tackle": mock_tackle}, fainted=False)

    battle = cast(
        DoubleBattle,
        SimpleNamespace(
            player_username="player",
            battle_tag="battle",
            teampreview=False,
            _wait=False,
            force_switch=[False, False],
            trapped=[False, False],
            maybe_trapped=[False, False],
            active_pokemon=[mock_active, None],
            available_moves=[[mock_struggle], []],
            available_switches=[[], []],
            team={},
            can_mega_evolve=[True, False],
            valid_orders=[[], []],
            get_possible_showdown_targets=lambda move, mon: [0],
        ),
    )

    mask = MegaEnv.single_action_mask(battle, 0)
    assert 48 in mask
    assert 47 in mask

    order = MegaEnv._action_to_order_individual(np.int64(47), battle, fake=True, pos=0)
    assert cast(Any, order.order).id == "recharge"
    assert order.mega

    action = MegaEnv._order_to_action_individual(order, battle, fake=True, pos=0)
    assert action == 47


def test_struggle_policy_logits():
    B = 2
    policy = PolicyNet(
        obs_dim=(SEQUENCE_LENGTH, NUMERICAL_WIDTH),
        act_size=ACT_SIZE,
        d_model=64,
        nhead=2,
        nlayer=1,
    )

    obs = StructuredObservation.empty_batch(B)
    obs.numerical[:, :, -1] = 0.5  # fake ratios so orig_ids aren't all 0

    action_mask = torch.zeros((B, 2, ACT_SIZE), dtype=torch.bool)
    action_mask[:, 0, 48] = True
    action_mask[:, 1, 47] = True

    state = policy.initial_state(B)
    enc = policy.encode(obs, action_mask)

    z, next_state, tokens_ctx = policy.actor.reducer(enc.tokens, state, None)
    k_entity_extended = policy.actor._compute_keys(tokens_ctx)
    logits, keys = policy.actor._compute_pointer_logits(
        z, k_entity_extended, enc.aux[:, 0], enc.numerical, head_idx=0
    )

    torch.testing.assert_close(keys[:, 48], policy.actor.struggle_key.unsqueeze(0).expand(B, -1))

    torch.testing.assert_close(
        keys[:, 47], (policy.actor.struggle_key + policy.actor.mega_emb).unsqueeze(0).expand(B, -1)
    )

    actions = torch.tensor([[48, 47], [48, 47]])
    out = policy.evaluate(enc, action_mask, actions, state)

    assert torch.isfinite(out.logits[:, 0, 48]).all()

    logits2 = policy.actor._apply_sequential_masks(
        out.logits, torch.tensor([47, 47]), action_mask, torch.tensor([False, False])
    )
    assert (logits2[:, 1, 47] == float("-inf")).all()
