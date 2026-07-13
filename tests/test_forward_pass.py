import pytest
import torch

from p0.format_config import FORMAT
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    EVENT_COUNT,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)

ACT_SIZE = FORMAT.action_size


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
    obs = StructuredObservation.empty_batch(B)

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

    assert out.state.shape == (B, 8, 128)  # n_hg is 8


def test_encoder_batches_all_pokemon_in_one_fusion_call(policy_net):
    B = 2
    obs = StructuredObservation.empty_batch(B)
    obs.numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))
    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    calls: list[tuple[int, ...]] = []

    def record_shape(module, args, output):
        del module, output
        calls.append(tuple(args[0].shape))

    handle = policy_net.encoder.mon_fusion.register_forward_hook(record_shape)
    try:
        with torch.no_grad():
            batched = policy_net.encode(obs, action_mask)
    finally:
        handle.remove()

    assert calls == [(B * 12, 11, 128)]

    with torch.no_grad():
        separate = [policy_net.encode(obs[i : i + 1], action_mask[i : i + 1]) for i in range(B)]

    torch.testing.assert_close(
        batched.tokens,
        torch.cat([enc.tokens for enc in separate]),
    )
    torch.testing.assert_close(
        batched.aux,
        torch.cat([enc.aux for enc in separate]),
    )


def test_encoded_obs_step_is_contiguous_time_major():
    enc = EncodedObs(
        tokens=torch.randn((3, 4, 5, 6)),
        aux=torch.randn((3, 4, 2, 6)),
        numerical=torch.randn((3, 4, 5, 7)),
    )

    step = enc.step(3, 1)

    assert step.tokens.shape == (3, 5, 6)
    assert step.aux.shape == (3, 2, 6)
    assert step.numerical.shape == (3, 5, 7)
    assert step.tokens.is_contiguous()
    assert step.aux.is_contiguous()
    assert step.numerical.is_contiguous()


def test_policy_net_encoded_evaluate(policy_net):
    B = 16
    tokens = torch.randn((B, SEQUENCE_LENGTH + 1 + EVENT_COUNT, 128))
    aux = torch.randn((B, 2, 4, 128))
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
    assert out.state.shape == (B, 8, 128)


def test_encode_requires_batched_observation_and_mask(policy_net):
    obs = StructuredObservation.empty_batch(1)[0]
    action_mask = torch.ones((2, ACT_SIZE), dtype=torch.bool)

    with pytest.raises(ValueError, match="batched"):
        policy_net.encode(obs, action_mask)

    with pytest.raises(TypeError):
        policy_net.encode(obs.unsqueeze(0))  # type: ignore[call-arg]


def test_top_p_validation(policy_net):
    B = 1
    obs = StructuredObservation.empty_batch(B)
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
    obs = StructuredObservation.empty_batch(B)

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


def test_cls_reducer_pokemon_tokens_alignment():
    """pokemon_tokens must be exactly the 24 pokemon tokens (original indices 1-24)."""
    import torch.nn as nn

    from p0.model.cls_reducer import CLSReducer

    reducer = CLSReducer(seq_len=SEQUENCE_LENGTH + 1, d_model=32, nhead=4, nlayer=1)

    class _Passthrough(nn.Module):
        def forward(self, seq, src_key_padding_mask=None):
            return seq

    reducer.encoder = _Passthrough()  # type: ignore
    tokens = torch.randn(2, SEQUENCE_LENGTH + 1, 32)

    _, _, pokemon_tokens = reducer(tokens, None, None)

    torch.testing.assert_close(pokemon_tokens, tokens[:, 1:25])
