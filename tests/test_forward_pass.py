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


def test_nature_embedding_correctness(policy_net):
    encoder = policy_net.encoder
    assert encoder.nature_emb.num_embeddings == 25
    assert encoder.nature_emb.embedding_dim == 128
    assert encoder.nature_proj.in_features == 128
    assert encoder.nature_proj.out_features == encoder.d_model

    # Create dummy categorical tensors with different natures
    cat1 = torch.zeros((1, CATEGORICAL_WIDTH), dtype=torch.long)
    cat1[0, 24] = 5  # arbitrary nature ID
    cat2 = torch.zeros((1, CATEGORICAL_WIDTH), dtype=torch.long)
    cat2[0, 24] = 12  # different nature ID

    out1 = encoder._embed_pokemon_super(cat1)
    out2 = encoder._embed_pokemon_super(cat2)
    assert not torch.allclose(out1, out2), (
        "Changing nature did not change the Pokemon super embedding"
    )


def test_fainted_pokemon_masking(policy_net):
    B = 1
    obs = StructuredObservation(
        categorical=torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        numerical=torch.zeros((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH), dtype=torch.float32),
        token_type_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        side_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        slot_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
    )

    obs.token_type_ids[0, 0] = 0  # CLS
    for i in range(1, 13):
        obs.token_type_ids[0, i] = 1 if i % 2 == 1 else 2  # Super or Numeric
        obs.side_ids[0, i] = 1  # ALLY
        obs.slot_ids[0, i] = (i - 1) // 2 + 1
    for i in range(13, 25):
        obs.token_type_ids[0, i] = 1 if i % 2 == 1 else 2  # Super or Numeric
        obs.side_ids[0, i] = 2  # OPPONENT
        obs.slot_ids[0, i] = (i - 13) // 2 + 1

    obs.token_type_ids[0, 25] = 3  # FIELD_SUPER
    obs.token_type_ids[0, 26] = 4  # FIELD_NUMERIC
    obs.token_type_ids[0, 27] = 3  # FIELD_SUPER
    obs.token_type_ids[0, 28] = 4  # FIELD_NUMERIC
    obs.side_ids[0, 27] = 1
    obs.side_ids[0, 28] = 1
    obs.token_type_ids[0, 29] = 3  # FIELD_SUPER
    obs.token_type_ids[0, 30] = 4  # FIELD_NUMERIC
    obs.side_ids[0, 29] = 2
    obs.side_ids[0, 30] = 2

    # Populate valid orig_idxs to prevent random switch actions from crashing
    ally_indices = [1, 3, 5, 7, 9, 11]
    for i, idx in enumerate(ally_indices):
        obs.numerical[:, idx + 1, 26] = (i + 1) / 6.0

    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    actions = torch.zeros((B, 2), dtype=torch.long)

    with torch.no_grad():
        logits_active, _, _, value_active, _ = policy_net(
            obs, action_mask=action_mask, actions=actions
        )

    # Mark Ally Pokemon 2 (numeric at index 4) as fainted (fainted flag at 27)
    obs.numerical[:, 4, 27] = 1.0

    with torch.no_grad():
        logits_fainted, _, _, value_fainted, _ = policy_net(
            obs, action_mask=action_mask, actions=actions
        )

    assert not torch.allclose(logits_active, logits_fainted, atol=1e-5)

    # Modify the features of the fainted pokemon:
    obs.categorical[:, 3, 0] = 41  # species
    obs.categorical[:, 3, 14] = 2  # move category
    obs.numerical[:, 4, 0] = 0.99  # numeric stat

    with torch.no_grad():
        logits_modified, _, _, value_modified, _ = policy_net(
            obs, action_mask=action_mask, actions=actions
        )

    assert torch.allclose(logits_fainted, logits_modified, atol=1e-5)
    assert torch.allclose(value_fainted, value_modified, atol=1e-5)
