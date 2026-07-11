import pytest
import torch

from src.format_config import FORMAT
from src.model.policy import PolicyNet
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_NUMERICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    SideId,
    StructuredObservation,
    TokenType,
)
from src.train.config import TrainingConfig
from src.train.train_loop import _run_batched_ppo

ACT_SIZE = FORMAT.action_size


@pytest.fixture
def dummy_obs():
    B = 2

    token_type_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    token_type_ids[:, 0] = TokenType.CLS
    token_type_ids[:, 1:13] = TokenType.POKEMON_SUPER
    token_type_ids[:, 13:25] = TokenType.POKEMON_NUMERIC
    token_type_ids[:, 25] = TokenType.FIELD_SUPER
    token_type_ids[:, 26] = TokenType.FIELD_NUMERIC
    token_type_ids[:, 27] = TokenType.FIELD_SUPER
    token_type_ids[:, 28] = TokenType.FIELD_NUMERIC
    token_type_ids[:, 29] = TokenType.FIELD_SUPER
    token_type_ids[:, 30] = TokenType.FIELD_NUMERIC

    side_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    side_ids[:, 1:13] = SideId.ALLY
    side_ids[:, 13:25] = SideId.OPPONENT
    side_ids[:, 25:27] = SideId.NONE
    side_ids[:, 27:29] = SideId.ALLY
    side_ids[:, 29:31] = SideId.OPPONENT

    slot_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    for i in range(6):
        # 1, 2 for first pokemon, 3, 4 for second pokemon, etc.
        slot_ids[:, 1 + 2 * i : 1 + 2 * i + 2] = i + 1
        slot_ids[:, 13 + 2 * i : 13 + 2 * i + 2] = i + 1

    # Populate categorical with random IDs respecting vocab limits
    categorical = torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long)

    # Pokemon tokens (1-24)
    # species (0): 1-34
    categorical[:, 1:25, 0] = torch.randint(1, 35, (B, 24))
    # ability (1): 1-23
    categorical[:, 1:25, 1] = torch.randint(1, 24, (B, 24))
    # item (2): 1-18
    categorical[:, 1:25, 2] = torch.randint(1, 19, (B, 24))
    # types (3,4): 1-18
    categorical[:, 1:25, 3:5] = torch.randint(1, 19, (B, 24, 2))
    # moves (5-8): 1-69
    categorical[:, 1:25, 5:9] = torch.randint(1, 70, (B, 24, 4))
    # move_types (9-12): 1-18
    categorical[:, 1:25, 9:13] = torch.randint(1, 19, (B, 24, 4))
    # move_categories (13-16): 1-3
    categorical[:, 1:25, 13:17] = torch.randint(1, 4, (B, 24, 4))
    # status (17): 1-6
    categorical[:, 1:25, 17] = torch.randint(1, 7, (B, 24))
    # volatiles (18-23): 1-5
    categorical[:, 1:25, 18:24] = torch.randint(1, 6, (B, 24, 6))

    # weather_emb has size 5 (0-4), trickroom_emb has size 2 (0-1)
    categorical[:, 25, 0] = torch.randint(1, 5, (B,))
    categorical[:, 25, 1] = torch.randint(1, 2, (B,))

    # side_condition_emb has size 5 (0-4)
    categorical[:, (27, 29), :4] = torch.randint(1, 5, (B, 2, 4))

    # Numerical features
    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))

    # Populate valid orig_idxs to prevent random switch actions from crashing
    ally_indices = [1, 3, 5, 7, 9, 11]
    for i, idx in enumerate(ally_indices):
        numerical[:, idx + 1, 26] = (i + 1) / 6.0

    numerical[:, 26, 2] = 1.0

    events_cat = torch.zeros((B, EVENT_COUNT, EVENT_CATEGORICAL_WIDTH), dtype=torch.long)
    events_cat[..., 0] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 1] = torch.randint(1, 70, (B, EVENT_COUNT))
    events_cat[..., 2] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 3] = torch.randint(1, 7, (B, EVENT_COUNT))
    events_cat[..., 4] = torch.randint(1, 25, (B, EVENT_COUNT))

    events_num = torch.randn((B, EVENT_COUNT, EVENT_NUMERICAL_WIDTH))
    events_side_ids = torch.randint(0, 3, (B, EVENT_COUNT), dtype=torch.long)
    events_slot_ids = torch.randint(0, 7, (B, EVENT_COUNT), dtype=torch.long)

    return StructuredObservation(
        token_type_ids=token_type_ids,
        side_ids=side_ids,
        slot_ids=slot_ids,
        categorical=categorical,
        numerical=numerical,
        events_cat=events_cat,
        events_num=events_num,
        events_side_ids=events_side_ids,
        events_slot_ids=events_slot_ids,
    )


def test_gradient_flow(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # smaller model for faster testing
    policy = PolicyNet(d_model=64, nhead=2, nlayer=1).to(device)
    policy.train()

    obs = dummy_obs.to(device)

    # allow all actions for now
    action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.uint8).to(device)

    out = policy.act_obs(obs, action_mask, policy.initial_state(2))
    # random loss fn involving both value and policy paths
    loss = out.value.mean() - out.log_probs.mean()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    missing_grads = []
    zero_grads = []

    # not all used every turn
    conditional_params = [
        "actor.pass_emb",
        "actor.switch_meta_emb",
        "actor.move_meta_emb",
        "actor.mega_meta_emb",
        "actor.target_ally_emb",
        "actor.target_opp_emb",
        "actor.target_self_multi_emb",
        "actor.move_proj",
        "actor.tp_meta_emb",
        "actor.reducer.hg_gate",  # since test only on a single step
        "actor.q_switch_proj1",
        "actor.q_pass_proj1",
    ]

    for name, param in policy.named_parameters():
        if any(cond in name for cond in conditional_params):
            continue
        if param.requires_grad:
            if param.grad is None:
                missing_grads.append(name)
            elif torch.all(param.grad == 0):
                zero_grads.append(name)

    assert not missing_grads, f"Parameters missing gradients: {missing_grads}"

    if zero_grads:
        print(f"Warning: Parameters with zero gradients: {zero_grads}")

    # verify gradient coverage across logical components
    components = {
        "shared_encoder": "encoder",
        "actor_reducer": "actor.reducer",
        "actor_w_k": "actor.w_k",
        "actor_q_proj": "actor.q_",
        "critic_head": "critic.net",
    }

    failed_components = []
    for comp_name, prefix in components.items():
        comp_params = [p for n, p in policy.named_parameters() if n.startswith(prefix)]
        if not comp_params:
            failed_components.append(f"{comp_name} (no parameters found with prefix {prefix})")
            continue

        has_grad = any(p.grad is not None and torch.abs(p.grad).sum() > 0 for p in comp_params)
        if not has_grad:
            failed_components.append(comp_name)

    assert not failed_components, (
        f"The following components are not receiving gradients: {failed_components}"
    )

    # verify gradient coverage for embedding matrix
    embeddings = [
        "species_emb",
        "ability_emb",
        "item_emb",
        "move_emb",
        "type_emb",
        "category_emb",
        "status_emb",
        "volatile_emb",
        "weather_emb",
        "trickroom_emb",
        "side_condition_emb",
        "token_type_emb",
        "side_emb",
        "slot_emb",
        "event_type_emb",
        "order_pos_emb",
    ]

    missing_emb_grads = []
    for emb in embeddings:
        found = False
        for n, p in policy.encoder.named_parameters():
            if emb in n:
                found = True
                if p.grad is None or torch.abs(p.grad).sum() == 0:
                    missing_emb_grads.append(emb)
        if not found:
            missing_emb_grads.append(f"{emb} (not found)")

    assert not missing_emb_grads, (
        f"The following embeddings are not receiving gradients: {missing_emb_grads}"
    )


def test_ppo_warmup(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet(d_model=64, nhead=2, nlayer=1).to(device)
    policy.train()

    episode = {
        "obs": dummy_obs[0].unsqueeze(0),
        "actions": torch.tensor([[1, 2]], dtype=torch.long),
        "log_probs": torch.zeros(1),
        "advantages": torch.ones(1),
        "returns": torch.ones(1),
        "values": torch.zeros(1),
        "action_masks": torch.ones((1, 2, ACT_SIZE), dtype=torch.bool),
        "length": 1,
    }
    config = TrainingConfig(warmup_episodes=10)

    loss, _, steps = _run_batched_ppo([episode], policy, config, device, episode=0)
    assert steps == 1

    policy.zero_grad(set_to_none=True)
    loss.backward()

    assert all(p.grad is None for p in policy.encoder.parameters())
    assert all(p.grad is None for p in policy.actor.parameters())
    assert any(
        p.grad is not None and not torch.all(p.grad == 0) for p in policy.critic.parameters()
    )


if __name__ == "__main__":
    pytest.main([__file__])
