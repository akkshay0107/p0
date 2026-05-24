import pytest
import torch

from src.model.policy import PolicyNet
from src.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    SideId,
    TokenType,
)


@pytest.fixture
def dummy_obs():
    B = 2

    token_type_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    token_type_ids[:, 0] = TokenType.CLS
    token_type_ids[:, 1:13] = TokenType.POKEMON_SUPER
    token_type_ids[:, 13:25] = TokenType.POKEMON_NUMERIC
    token_type_ids[:, 25] = TokenType.GLOBAL_FIELD
    token_type_ids[:, 26] = TokenType.ALLY_SIDE
    token_type_ids[:, 27] = TokenType.OPPONENT_SIDE

    side_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    side_ids[:, 1:7] = SideId.ALLY
    side_ids[:, 7:13] = SideId.OPPONENT

    slot_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    for i in range(6):
        slot_ids[:, 1 + i] = i + 1
        slot_ids[:, 7 + i] = i + 1

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
    # volatiles (18-23): 1-6
    categorical[:, 1:25, 18:24] = torch.randint(1, 7, (B, 24, 6))

    # global_condition_emb has size 10 (0-9)
    categorical[:, 25, :6] = torch.randint(1, 10, (B, 6))

    # side_condition_emb has size 6 (0-5)
    categorical[:, 26:28, :6] = torch.randint(1, 6, (B, 2, 6))

    # Numerical features
    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))
    # Set the is_tp flag (numerical[:, 25, 6]) randomly
    numerical[:, 25, 6] = 1.0

    return {
        "token_type_ids": token_type_ids,
        "side_ids": side_ids,
        "slot_ids": slot_ids,
        "categorical": categorical,
        "numerical": numerical,
    }


def test_gradient_flow(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # smaller model for faster testing
    policy = PolicyNet(d_model=64, nhead=2, nlayer=1).to(device)
    policy.train()

    obs = {k: v.to(device) for k, v in dummy_obs.items()}

    # allow all actions for now
    action_mask = torch.ones((2, 2, 47), dtype=torch.uint8).to(device)

    logits, log_probs, sampled_actions, value, next_state = policy(
        obs, action_mask=action_mask, sample_actions=True
    )

    # random loss fn involving both value and policy paths
    loss = value.mean() - log_probs.mean()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    missing_grads = []
    zero_grads = []

    # there is technically a chance that one of these parameters
    # could genuinely have a zero gradient, but that is really
    # unprobably to the point if it does its mostly due to a bug
    for name, param in policy.named_parameters():
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
        "actor_head1": "actor.head1",
        "actor_head2": "actor.head2",
        "critic_reducer": "critic.reducer",
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
        "global_condition_emb",
        "side_condition_emb",
        "token_type_emb",
        "side_emb",
        "slot_emb",
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


def test_value_head_scaling(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs = {k: v.to(device) for k, v in dummy_obs.items()}

    # Create two identical policies except for the scale
    torch.manual_seed(1)
    p1 = PolicyNet(d_model=64, nhead=2, nlayer=1).to(device)
    p2 = PolicyNet(d_model=64, nhead=2, nlayer=1).to(device)
    p2.load_state_dict(p1.state_dict())

    p1.critic.scale = 1.0
    p2.critic.scale = 0.1

    # backprop using value loss only
    p1.zero_grad()
    _, _, _, value1, _ = p1(obs)
    value1.mean().backward()
    grads1 = {n: p.grad.clone() for n, p in p1.encoder.named_parameters() if p.grad is not None}

    p2.zero_grad()
    _, _, _, value2, _ = p2(obs)
    value2.mean().backward()
    grads2 = {n: p.grad.clone() for n, p in p2.encoder.named_parameters() if p.grad is not None}

    for name in grads1:
        g1 = grads1[name]
        g2 = grads2[name]

        mask = torch.abs(g1) > 1e-7
        if mask.any():
            # ratio should match scale above
            ratio = g2[mask] / g1[mask]
            torch.testing.assert_close(ratio, torch.full_like(ratio, 0.1), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
