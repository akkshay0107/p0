import pytest
import torch

from src.lookups import ACT_SIZE
from src.model.policy import EncodedObs, PolicyNet
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

    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    state = policy_net.initial_state(B)

    with torch.no_grad():
        out = policy_net.act_obs(obs, action_mask, state)

    assert out.log_probs.shape == (B,)
    assert out.actions.shape == (B, 2)
    assert out.value.shape == (B,)

    assert out.state.shape == (B, 4, 128)  # n_hg is 4


def test_policy_net_encoded_evaluate(policy_net):
    B = 16
    tokens = torch.randn((B, SEQUENCE_LENGTH + 1, 128))
    aux = torch.randn((B, 4, 128))
    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))

    # Populate valid orig_idxs to prevent random switch actions from crashing
    ally_indices = [1, 3, 5, 7, 9, 11]
    for i, idx in enumerate(ally_indices):
        numerical[:, idx + 1, 26] = (i + 1) / 6.0

    enc = EncodedObs(tokens, aux, numerical)
    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    actions = torch.full((B, 2), 7, dtype=torch.long)
    state = policy_net.initial_state(B)

    with torch.no_grad():
        out = policy_net.evaluate(enc, action_mask, actions, state)

    assert out.logits.shape == (B, 2, ACT_SIZE)
    assert out.log_probs.shape == (B,)
    assert out.entropy.shape == (B,)
    assert out.norm_entropy.shape == (B,)
    assert out.value.shape == (B,)
    assert out.state.shape == (B, 4, 128)


def test_encode_requires_batched_observation_and_mask(policy_net):
    obs = StructuredObservation(
        categorical=torch.zeros((SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        numerical=torch.zeros((SEQUENCE_LENGTH, NUMERICAL_WIDTH), dtype=torch.float32),
        token_type_ids=torch.zeros(SEQUENCE_LENGTH, dtype=torch.long),
        side_ids=torch.zeros(SEQUENCE_LENGTH, dtype=torch.long),
        slot_ids=torch.zeros(SEQUENCE_LENGTH, dtype=torch.long),
    )
    action_mask = torch.ones((2, ACT_SIZE), dtype=torch.bool)

    with pytest.raises(ValueError, match="batched"):
        policy_net.encode(obs, action_mask)

    with pytest.raises(TypeError):
        policy_net.encode(obs.unsqueeze(0))  # type: ignore[call-arg]


def test_top_p_validation(policy_net):
    B = 1
    obs = StructuredObservation(
        categorical=torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long),
        numerical=torch.zeros((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH), dtype=torch.float32),
        token_type_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        side_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
        slot_ids=torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long),
    )
    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)

    with pytest.raises(ValueError, match="top_p"):
        policy_net.act_obs(obs, action_mask, policy_net.initial_state(B), top_p=0.0)


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


def test_fainted_pokemon_visible(policy_net):
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
    actions = torch.full((B, 2), 7, dtype=torch.long)
    state = policy_net.initial_state(B)

    with torch.no_grad():
        out_active = policy_net.evaluate_obs(obs, action_mask, actions, state)

    # Mark Ally Pokemon 2 (numeric at index 4) as fainted (fainted flag at 27)
    obs.numerical[:, 4, 27] = 1.0

    with torch.no_grad():
        out_fainted = policy_net.evaluate_obs(obs, action_mask, actions, state)

    assert not torch.allclose(out_active.logits, out_fainted.logits, atol=1e-5)

    # Modify the features of the fainted pokemon:
    obs.categorical[:, 3, 0] = 41  # species
    obs.categorical[:, 3, 14] = 2  # move category
    obs.numerical[:, 4, 0] = 0.99  # numeric stat

    with torch.no_grad():
        out_modified = policy_net.evaluate_obs(obs, action_mask, actions, state)

    assert not torch.allclose(out_fainted.logits, out_modified.logits, atol=1e-5)
    assert not torch.allclose(out_fainted.value, out_modified.value, atol=1e-5)
