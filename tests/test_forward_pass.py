import pytest
import torch

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import EncodedObs
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CATEGORICAL_WIDTH,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
)

ACT_SIZE = FORMAT.action_size


@pytest.fixture
def policy_net():
    return build_policy(
        ModelConfig(128, 4, 2, 512),
        default_runtime_resources(),
    )


def test_policy_net_act_and_encoded_evaluate_shapes(policy_net):
    B = 16
    obs = StructuredObservation.empty_batch(B)

    # Populate valid orig_idxs to prevent random switch actions from crashing
    for i, idx in enumerate(range(1, 7)):
        obs.numerical[:, idx, 26] = (i + 1) / 6.0

    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    memory = policy_net.empty_memory(B)

    with torch.no_grad():
        out = policy_net.act_obs(obs, action_mask, *memory)

    assert out.log_probs.shape == (B,)
    assert out.actions.shape == (B, 2)
    assert out.value.shape == (B,)

    assert out.history_token.shape == (B, 128)

    encoded = policy_net.encode(obs, action_mask)
    actions = torch.full((B, 2), 7, dtype=torch.long)
    with torch.no_grad():
        evaluated = policy_net.evaluate(encoded, action_mask, actions, *memory)
    assert evaluated.logits.shape == (B, 2, ACT_SIZE)
    assert evaluated.log_probs.shape == (B,)
    assert evaluated.entropy.shape == (B,)
    assert evaluated.norm_entropy.shape == (B,)
    assert evaluated.value.shape == (B,)
    assert evaluated.history_token.shape == (B, 128)


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

    assert calls == [(B * 12, 15, 128)]

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


def test_policy_inputs_reject_unbatched_missing_mask_and_invalid_top_p(policy_net):
    obs = StructuredObservation.empty_batch(1)[0]
    action_mask = torch.ones((2, ACT_SIZE), dtype=torch.bool)

    with pytest.raises(ValueError, match="batched"):
        policy_net.encode(obs, action_mask)

    with pytest.raises(TypeError):
        policy_net.encode(obs.unsqueeze(0))  # type: ignore[call-arg]

    B = 1
    obs = StructuredObservation.empty_batch(B)
    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)

    with pytest.raises(ValueError, match="top_p"):
        policy_net.act_obs(obs, action_mask, *policy_net.empty_memory(B), top_p=0.0)


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
    num = torch.zeros((1, NUMERICAL_WIDTH))

    out1 = encoder._embed_pokemon_super(cat1, num)
    out2 = encoder._embed_pokemon_super(cat2, num)
    assert not torch.allclose(out1, out2), (
        "Changing nature did not change the Pokemon super embedding"
    )


def test_fainted_pokemon_visible(policy_net):
    B = 1
    obs = StructuredObservation.empty_batch(B)

    for i in range(0, 6):
        obs.token_type_ids[0, i] = 1  # POKEMON
        obs.side_ids[0, i] = 1  # ALLY
        obs.slot_ids[0, i] = i + 1
    for i in range(6, 12):
        obs.token_type_ids[0, i] = 1  # POKEMON
        obs.side_ids[0, i] = 2  # OPPONENT
        obs.slot_ids[0, i] = i - 5

    obs.token_type_ids[0, 12:15] = 2  # FIELD owners
    obs.side_ids[0, 13] = 1
    obs.side_ids[0, 14] = 2

    # Populate valid orig_idxs to prevent random switch actions from crashing
    for i, idx in enumerate(range(0, 6)):
        obs.numerical[:, idx, 26] = (i + 1) / 6.0

    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    actions = torch.full((B, 2), 7, dtype=torch.long)
    memory = policy_net.empty_memory(B)

    with torch.no_grad():
        out_active = policy_net.evaluate_obs(obs, action_mask, actions, *memory)

    # Mark Ally Pokemon 2 (token index 2) as fainted (fainted flag at 27)
    obs.numerical[:, 2, 27] = 1.0

    with torch.no_grad():
        out_fainted = policy_net.evaluate_obs(obs, action_mask, actions, *memory)

    assert not torch.allclose(out_active.logits, out_fainted.logits, atol=1e-5)

    # Modify the features of the fainted pokemon:
    obs.categorical[:, 2, 0] = 41  # species
    obs.categorical[:, 2, 14] = 2  # move category
    obs.numerical[:, 2, 0] = 0.99  # numeric stat

    with torch.no_grad():
        out_modified = policy_net.evaluate_obs(obs, action_mask, actions, *memory)

    assert not torch.allclose(out_fainted.logits, out_modified.logits, atol=1e-5)
    assert not torch.allclose(out_fainted.value, out_modified.value, atol=1e-5)


def test_memory_reducer_pokemon_tokens_alignment():
    from p0.model.cls_reducer import MemoryReducer

    reducer = MemoryReducer(32, 4, 1, 128)
    current = torch.randn(2, 24, 32)
    series = torch.zeros(2, 2, 32)
    series_mask = torch.zeros(2, 2, dtype=torch.bool)
    history = torch.zeros(2, 48, 32)
    history_mask = torch.zeros(2, 48, dtype=torch.bool)
    ages = torch.zeros(2, 48, dtype=torch.long)
    reduced = reducer(current, series, series_mask, history, history_mask, ages)
    assert reduced.pokemon.shape == (2, 12, 32)


def test_event_targets_do_not_alias(policy_net):
    """Crossed actor/target slots must produce distinct event tokens (audit §1.5)."""
    from p0.battle.events import EventTypeId
    from p0.model.structured_observation import SideId

    def encode_event(actor_slot: int, target_slot: int) -> torch.Tensor:
        obs = StructuredObservation.empty_batch(1)
        obs.events_cat[0, 0, 0] = EventTypeId.MOVE
        obs.events_side_ids[0, 0] = SideId.ALLY
        obs.events_slot_ids[0, 0] = actor_slot
        obs.events_cat[0, 0, 8] = SideId.OPPONENT
        obs.events_cat[0, 0, 9] = target_slot
        action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
        with torch.no_grad():
            tokens, _ = policy_net.encoder(obs, action_mask)
        return tokens[0, -8]  # first pooled event token

    crossed_a = encode_event(actor_slot=1, target_slot=2)
    crossed_b = encode_event(actor_slot=2, target_slot=1)

    assert not torch.allclose(crossed_a, crossed_b, atol=1e-6)


def test_event_effect_namespaces(policy_net):
    """Event effect ids are tagged with the vocab table they index (audit §1.1)."""
    from p0.battle.events import EventTypeId
    from p0.model.structured_observation import EffectNamespace

    namespaces = policy_net.encoder._event_effect_namespace
    assert namespaces[EventTypeId.WEATHER_START] == EffectNamespace.WEATHER
    assert namespaces[EventTypeId.WEATHER_END] == EffectNamespace.WEATHER
    assert namespaces[EventTypeId.FIELD_START] == EffectNamespace.FIELD
    assert namespaces[EventTypeId.FIELD_END] == EffectNamespace.FIELD
    assert namespaces[EventTypeId.SIDE_START] == EffectNamespace.SIDE
    assert namespaces[EventTypeId.SIDE_END] == EffectNamespace.SIDE
    assert namespaces[EventTypeId.EFFECT_START] == EffectNamespace.POKEMON
    assert namespaces[EventTypeId.EFFECT_END] == EffectNamespace.POKEMON
    assert namespaces[EventTypeId.CANT] == EffectNamespace.POKEMON
    assert namespaces[EventTypeId.MOVE] == EffectNamespace.NONE

    def encode_first_event(event_type: EventTypeId) -> torch.Tensor:
        obs = StructuredObservation.empty_batch(1)
        obs.events_cat[0, 0, 0] = event_type
        obs.events_cat[0, 0, 5] = 1  # same effect id in both namespaces
        action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
        with torch.no_grad():
            tokens, _ = policy_net.encoder(obs, action_mask)
        return tokens[0, -8]

    weather = encode_first_event(EventTypeId.WEATHER_START)
    volatile = encode_first_event(EventTypeId.EFFECT_START)
    assert not torch.allclose(weather, volatile, atol=1e-6)
