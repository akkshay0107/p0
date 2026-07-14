from types import SimpleNamespace
from typing import Any, cast

import torch
from poke_env.battle import DoubleBattle

from p0.battle.legality import legal_actions
from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    StructuredObservation,
)
from p0.runtime.poke_env_action_adapter import action_to_single_order, single_order_to_action
from p0.runtime.poke_env_battle_adapter import decision_view

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

    mask = list(legal_actions(decision_view(battle), 0))
    assert mask == [48]

    order = action_to_single_order(48, battle, fake=True, position=0)
    assert cast(Any, order.order).id == "struggle"
    assert not order.mega

    action = single_order_to_action(order, battle, fake=True, position=0)
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

    mask = list(legal_actions(decision_view(battle), 0))
    assert 48 in mask
    assert 47 in mask

    order = action_to_single_order(47, battle, fake=True, position=0)
    assert cast(Any, order.order).id == "recharge"
    assert order.mega

    action = single_order_to_action(order, battle, fake=True, position=0)
    assert action == 47


def test_struggle_policy_logits():
    B = 2
    policy = build_policy(ModelConfig(64, 2, 1, 8, 256), default_runtime_resources())

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
