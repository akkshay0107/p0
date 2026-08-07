import copy
from typing import Any, cast

import pytest
import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import (
    CURRENT_REDUCER_TOKEN_COUNT,
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    POOLED_EVENT_COUNT,
    REDUCER_MAX_LENGTH,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
from p0.model.cls_reducer import MemoryReducer, pack_history_tokens
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.series_context import DynamicSeriesResampler
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_STATUS_COUNTER_KIND,
    CAT_KNOWNNESS_START,
    CAT_KNOWNNESS_WIDTH,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EFFECT_NUMERICAL_WIDTH,
    EVENT_CATEGORICAL_WIDTH,
    EVENT_COUNT,
    EVENT_METADATA_WIDTH,
    EVENT_NUMERICAL_WIDTH,
    NUM_EFFECT_START,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.model.swiglu_encoder import SwiGLUEncoderLayer
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import _run_batched_ppo, magnet_kl_per_step
from p0.training.trajectory import TrajectoryBatch

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
    from p0.model.architecture_contract import SERIES_SLOTS
    from p0.model.cls_reducer import MemoryReducer

    reducer = MemoryReducer(32, 4, 1, 128)
    current = torch.randn(2, 24, 32)
    series = torch.zeros(2, SERIES_SLOTS, 32)
    series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
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


ACT_SIZE = FORMAT.action_size


@pytest.fixture
def dummy_obs():
    B = 2

    token_type_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    token_type_ids[:, 0:12] = TokenType.POKEMON
    token_type_ids[:, 12:15] = TokenType.FIELD

    side_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    side_ids[:, 0:6] = SideId.ALLY
    side_ids[:, 6:12] = SideId.OPPONENT
    side_ids[:, 12] = SideId.NONE
    side_ids[:, 13] = SideId.ALLY
    side_ids[:, 14] = SideId.OPPONENT

    slot_ids = torch.zeros((B, SEQUENCE_LENGTH), dtype=torch.long)
    for i in range(6):
        slot_ids[:, i] = i + 1
        slot_ids[:, 6 + i] = i + 1

    # Populate categorical with random IDs respecting vocab limits
    categorical = torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long)

    # Pokemon tokens (0-11)
    # species (0): 1-34
    categorical[:, 0:12, 0] = torch.randint(1, 35, (B, 12))
    # ability (1): 1-23
    categorical[:, 0:12, 1] = torch.randint(1, 24, (B, 12))
    # item (2): 1-18
    categorical[:, 0:12, 2] = torch.randint(1, 19, (B, 12))
    # types (3,4): 1-18
    categorical[:, 0:12, 3:5] = torch.randint(1, 19, (B, 12, 2))
    # moves (5-8): 1-69
    categorical[:, 0:12, 5:9] = torch.randint(1, 70, (B, 12, 4))
    # move_types (9-12): 1-18
    categorical[:, 0:12, 9:13] = torch.randint(1, 19, (B, 12, 4))
    # move_categories (13-16): 1-3
    categorical[:, 0:12, 13:17] = torch.randint(1, 4, (B, 12, 4))
    # status (17): 1-6
    categorical[:, 0:12, 17] = torch.randint(1, 7, (B, 12))
    # status counter kind: 0-4
    categorical[:, 0:12, CAT_IDX_STATUS_COUNTER_KIND] = torch.randint(0, 5, (B, 12))
    categorical[:, 0:12, CAT_KNOWNNESS_START : CAT_KNOWNNESS_START + CAT_KNOWNNESS_WIDTH] = (
        torch.randint(1, 5, (B, 12, CAT_KNOWNNESS_WIDTH))
    )

    # Numerical features
    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))

    for token_idx in range(15):
        categorical[
            :, token_idx, CAT_EFFECT_START : CAT_EFFECT_START + EFFECT_CATEGORICAL_WIDTH
        ] = torch.tensor((1, 1, 1))
        numerical[:, token_idx, NUM_EFFECT_START : NUM_EFFECT_START + EFFECT_NUMERICAL_WIDTH] = 1.0

    # Populate valid orig_idxs to prevent random switch actions from crashing
    for i, idx in enumerate(range(0, 6)):
        numerical[:, idx, 26] = (i + 1) / 6.0

    numerical[:, 12, 2] = 1.0

    events_cat = torch.zeros((B, EVENT_COUNT, EVENT_CATEGORICAL_WIDTH), dtype=torch.long)
    events_cat[..., 0] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 1] = torch.randint(1, 70, (B, EVENT_COUNT))
    events_cat[..., 2] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 3] = torch.randint(1, 7, (B, EVENT_COUNT))
    events_cat[..., 4] = torch.randint(1, 25, (B, EVENT_COUNT))
    events_cat[..., 5] = torch.randint(1, 6, (B, EVENT_COUNT))
    events_cat[..., 6] = torch.randint(1, 19, (B, EVENT_COUNT))
    events_cat[..., 7] = torch.randint(0, 8, (B, EVENT_COUNT))
    events_cat[..., 8] = torch.randint(0, 3, (B, EVENT_COUNT))
    events_cat[..., 9] = torch.randint(0, 7, (B, EVENT_COUNT))

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
        events_metadata=torch.zeros((B, 2)),
    )


def test_gradient_flow(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # smaller model for faster testing
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources()).to(device)
    policy.train()

    obs = dummy_obs.to(device)

    # allow all actions for now
    action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.uint8).to(device)

    out = policy.act_obs(obs, action_mask, *policy.empty_memory(2))
    # random loss fn involving both value and policy paths
    loss = out.value.mean() - out.log_probs.mean()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    missing_grads = []
    zero_grads = []

    # not all used every turn
    conditional_params = [
        "series.",  # this single-step test supplies masked Game 1 memory slots
        "actor.pass_emb",
        "actor.switch_meta_emb",
        "actor.move_meta_emb",
        "actor.mega_meta_emb",
        "actor.target_ally_emb",
        "actor.target_opp_emb",
        "actor.target_self_multi_emb",
        "actor.move_proj",
        "actor.tp_meta_emb",
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
        "effect_emb",
        "counter_kind_emb",
        "effect_namespace_emb",
        "knownness_emb",
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


def test_ppo_updates_all_policy_paths(dummy_obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources()).to(device)
    policy.train()

    episode = TrajectoryBatch(
        observations=dummy_obs[0].unsqueeze(0),
        actions=torch.tensor([[1, 2]], dtype=torch.long),
        log_probs=torch.zeros(1),
        advantages=torch.ones(1),
        returns=torch.ones(1),
        values=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1),
        action_masks=torch.ones((1, 2, ACT_SIZE), dtype=torch.bool),
        length=1,
    )
    config = TrainingConfig()
    magnet = Magnet(policy)

    loss, _, steps = _run_batched_ppo(
        [episode], policy, magnet, config, device, episode=0, alpha=config.magnet_alpha
    )
    assert steps == 1

    policy.zero_grad(set_to_none=True)
    loss.backward()

    assert any(
        p.grad is not None and torch.abs(p.grad).sum() > 0 for p in policy.encoder.parameters()
    )
    assert any(
        p.grad is not None and torch.abs(p.grad).sum() > 0 for p in policy.actor.parameters()
    )
    assert any(
        p.grad is not None and not torch.all(p.grad == 0) for p in policy.critic.parameters()
    )


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.fixture
def policy():
    result = build_policy(
        ModelConfig(64, 4, 1, 128),
        default_runtime_resources(),
    )
    result.eval()
    return result


def _inputs(policy, batch_size: int = 2):
    observations = StructuredObservation.empty_batch(batch_size)
    action_mask = torch.ones((batch_size, 2, FORMAT.action_size), dtype=torch.bool)
    encoded = policy.encode(observations, action_mask)
    return encoded, action_mask, policy.empty_memory(batch_size)


def test_singleton_candidate_scores_match_standard_joint_scoring(policy) -> None:
    encoded, action_mask, memory = _inputs(policy)
    candidates = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    offsets = torch.tensor([0, 1, 2], dtype=torch.long)

    with torch.no_grad():
        candidate_scores = policy.actor.score_joint_candidates(
            encoded, action_mask, *memory, candidates, offsets
        )
        expected = torch.stack(
            [
                policy.actor.score(
                    EncodedObs(
                        encoded.tokens[index : index + 1],
                        encoded.aux[index : index + 1],
                        encoded.numerical[index : index + 1],
                    ),
                    action_mask[index : index + 1],
                    candidates[index : index + 1],
                    *[item[index : index + 1] for item in memory],
                )[1][0]
                for index in range(2)
            ]
        )
    torch.testing.assert_close(candidate_scores, expected)


def test_candidate_scoring_runs_reducer_once_per_observation_batch(policy) -> None:
    encoded, action_mask, memory = _inputs(policy)
    candidates = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)
    offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    calls = []

    def record_call(*_args):
        calls.append(True)

    handle = policy.actor.reducer.register_forward_hook(record_call)
    try:
        policy.actor.score_joint_candidates(encoded, action_mask, *memory, candidates, offsets)
    finally:
        handle.remove()
    assert calls == [True]


def test_candidate_order_does_not_change_scores(policy) -> None:
    encoded, action_mask, memory = _inputs(policy, batch_size=1)
    candidates = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)
    offsets = torch.tensor([0, 3], dtype=torch.long)
    with torch.no_grad():
        first = policy.actor.score_joint_candidates(
            encoded, action_mask, *memory, candidates, offsets
        )
        permutation = torch.tensor([2, 0, 1])
        second = policy.actor.score_joint_candidates(
            encoded,
            action_mask,
            *memory,
            candidates[permutation],
            offsets,
        )
    torch.testing.assert_close(first, second[torch.argsort(permutation)])


def test_candidate_scoring_applies_second_action_mask(policy) -> None:
    encoded, _, memory = _inputs(policy, batch_size=1)
    action_mask = torch.zeros((1, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 1, 8] = True
    candidates = torch.tensor([[7, 8], [7, 7]], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    scores = policy.actor.score_joint_candidates(encoded, action_mask, *memory, candidates, offsets)
    assert torch.isfinite(scores[0])
    assert torch.isneginf(scores[1])


def test_greedy_inference_selects_actions_autoregressively(policy) -> None:
    encoded, _, memory = _inputs(policy, batch_size=2)
    action_mask = torch.zeros((2, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 0, 9] = True
    action_mask[:, 1, 8] = True
    action_mask[:, 1, 10] = True

    with torch.inference_mode():
        first = policy.actor.greedy(encoded, action_mask, *memory)
        second = policy.actor.greedy(encoded, action_mask, *memory)

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert first[0].shape == (2, 2)
    assert torch.all((first[0][:, 0] == 7) | (first[0][:, 0] == 9))
    assert torch.all((first[0][:, 1] == 8) | (first[0][:, 1] == 10))


def test_candidate_scoring_rejects_malformed_ragged_inputs(policy) -> None:
    encoded, action_mask, memory = _inputs(policy, batch_size=1)
    candidates = torch.tensor([[7, 8]], dtype=torch.long)
    with pytest.raises(ValueError, match="one boundary"):
        policy.actor.score_joint_candidates(
            encoded, action_mask, *memory, candidates, torch.tensor([0, 1, 1])
        )
    with pytest.raises(ValueError, match="action ids"):
        policy.actor.score_joint_candidates(
            encoded,
            action_mask,
            *memory,
            candidates.to(torch.float32),
            torch.tensor([0, 1]),
        )


def _tiny_policy():
    return build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())


def _batch(policy, batch_size=3):
    obs = StructuredObservation.empty_batch(batch_size)
    masks = torch.ones((batch_size, 2, policy.act_size), dtype=torch.bool)
    actions = torch.zeros((batch_size, 2), dtype=torch.long)
    return obs, masks, actions


def _live_and_magnet_logits(policy, magnet, obs, masks, actions):
    live = policy.evaluate_obs(
        obs, masks, actions, *policy.empty_memory(obs.numerical.size(0))
    ).logits
    mag = magnet.policy.evaluate_obs(
        obs, masks, actions, *magnet.policy.empty_memory(obs.numerical.size(0))
    )
    return live, mag.logits


def test_magnet_freezes_and_preserves_optimizer_state():
    def _test_magnet_params_are_frozen():
        policy = _tiny_policy()
        magnet = Magnet(policy)
        assert all(not p.requires_grad for p in magnet.policy.parameters())

    _test_magnet_params_are_frozen()

    def _test_refresh_does_not_perturb_optimizer_state():
        policy = _tiny_policy()
        magnet = Magnet(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        obs, masks, actions = _batch(policy)
        loss = policy.evaluate_obs(
            obs, masks, actions, *policy.empty_memory(obs.numerical.size(0))
        ).value.sum()
        loss.backward()
        optimizer.step()

        before = copy.deepcopy(optimizer.state_dict())
        magnet.refresh(policy)
        after = optimizer.state_dict()

        for pid, moments in before["state"].items():
            for key, value in moments.items():
                if torch.is_tensor(value):
                    assert torch.equal(value, after["state"][pid][key])

    _test_refresh_does_not_perturb_optimizer_state()

    def _test_magnet_frozen_under_live_optimizer_step():
        policy = _tiny_policy()
        magnet = Magnet(policy)
        snapshot = copy.deepcopy(magnet.policy.state_dict())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)

        obs, masks, actions = _batch(policy)
        loss = policy.evaluate_obs(
            obs, masks, actions, *policy.empty_memory(obs.numerical.size(0))
        ).value.sum()
        loss.backward()
        optimizer.step()

        for key, value in magnet.policy.state_dict().items():
            assert torch.equal(value, snapshot[key])

    _test_magnet_frozen_under_live_optimizer_step()


def test_magnet_kl_divergence_computation():
    def _test_magnet_kl_is_zero_at_refresh():
        policy = _tiny_policy()
        magnet = Magnet(policy)
        obs, masks, actions = _batch(policy)
        with torch.no_grad():
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl = magnet_kl_per_step(live, mag)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    _test_magnet_kl_is_zero_at_refresh()

    def _test_magnet_kl_grows_then_resets_after_refresh():
        torch.manual_seed(0)
        policy = _tiny_policy()
        magnet = Magnet(policy)
        obs, masks, actions = _batch(policy)

        # perturb the live policy so it diverges from the frozen magnet
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(torch.randn_like(p) * 0.05)
            policy.actor.pointer_temp.add_(0.5)
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl_drifted = magnet_kl_per_step(live, mag)
        assert (kl_drifted > 1e-4).any()

        magnet.refresh(policy)
        with torch.no_grad():
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl_after = magnet_kl_per_step(live, mag)
        assert torch.allclose(kl_after, torch.zeros_like(kl_after), atol=1e-5)

    _test_magnet_kl_grows_then_resets_after_refresh()

    def _test_magnet_kl_is_finite_for_degenerate_masks():
        policy = _tiny_policy()
        magnet = Magnet(policy)
        batch_size = 2
        obs = StructuredObservation.empty_batch(batch_size)
        # single legal action per slot: the KL must stay finite (zero contribution
        # from masked entries), never NaN
        masks = torch.zeros((batch_size, 2, policy.act_size), dtype=torch.bool)
        masks[:, :, 0] = True
        actions = torch.zeros((batch_size, 2), dtype=torch.long)
        with torch.no_grad():
            live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
            kl = magnet_kl_per_step(live, mag)
        assert torch.isfinite(kl).all()

    _test_magnet_kl_is_finite_for_degenerate_masks()

    def _test_magnet_kl_loss_sign_increases_with_divergence():
        from p0.training.config import TrainingConfig
        from p0.training.ppo import compute_ppo_objective

        config = TrainingConfig()
        common: dict[str, Any] = dict(
            current_log_probs=torch.zeros(2),
            current_values=torch.zeros(2),
            normalized_entropy=torch.zeros(2),
            old_log_probs=torch.zeros(2),
            advantages=torch.ones(2),
            returns=torch.zeros(2),
            team_preview=torch.tensor([False, False]),
            config=config,
        )
        low, *_ = compute_ppo_objective(magnet_kl=torch.zeros(2), alpha=0.5, **common)
        high, *_ = compute_ppo_objective(magnet_kl=torch.ones(2), alpha=0.5, **common)
        assert (high > low).all()

    _test_magnet_kl_loss_sign_increases_with_divergence()


def _policy():
    return build_policy(ModelConfig(64, 4, 1, 128), default_runtime_resources())


def test_fixed_memory_and_observation_contract() -> None:
    observation = StructuredObservation.empty_batch(2)
    assert SEQUENCE_LENGTH == 15
    assert EVENT_COUNT == 64
    assert observation.token_type_ids.shape == (2, 15)
    assert observation.events_num.shape == (2, 64, EVENT_NUMERICAL_WIDTH)
    assert observation.events_metadata.shape == (2, EVENT_METADATA_WIDTH)
    assert set(TokenType) == {TokenType.POKEMON, TokenType.FIELD, TokenType.EVENT}
    assert (CURRENT_TOKEN_COUNT, CURRENT_REDUCER_TOKEN_COUNT, REDUCER_MAX_LENGTH) == (24, 25, 81)
    assert (HISTORY_WINDOW, SERIES_SLOTS, POOLED_EVENT_COUNT) == (48, 8, 8)


def test_empty_events_are_finite_deterministic_and_pooled() -> None:
    policy = _policy()
    obs = StructuredObservation.empty_batch(2)
    first = policy.encoder._encode_events(obs, policy.device)
    second = policy.encoder._encode_events(obs, policy.device)
    assert first.shape == (2, POOLED_EVENT_COUNT, policy.d_model)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_padded_events_do_not_change_valid_pooling_and_metadata_is_aggregate() -> None:
    policy = _policy()
    valid = StructuredObservation.empty_batch(1)
    valid.events_cat[0, 0, :10] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    valid.events_num[0, 0, 0] = 0.5
    padded = valid.clone()
    padded.events_cat[0, 10, 1] = 2
    padded.events_num[0, 10, 0] = 99.0
    without_padding = policy.encoder._encode_events(valid, policy.device)
    with_padding = policy.encoder._encode_events(padded, policy.device)
    torch.testing.assert_close(without_padding, with_padding)
    assert valid.events_num.shape[-1] == 2
    assert valid.events_metadata.shape[-1] == 2


def test_event_roles_namespace_order_and_overflow_remain_observable() -> None:
    policy = _policy()
    base = StructuredObservation.empty_batch(1)
    base.events_cat[0, 0, :10] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    base.events_num[0, 0, 0] = 1.0
    target_swap = base.clone()
    target_swap.events_cat[0, 0, 8:10] = torch.tensor([2, 1])
    namespace_swap = base.clone()
    namespace_swap.events_cat[0, 0, 0] = 2
    order_swap = base.clone()
    order_swap.events_cat[0, 0, 4] = 2
    overflow = base.clone()
    overflow.events_metadata[0] = torch.tensor([64.0, 32.0])

    outputs = [
        policy.encoder._encode_events(item, policy.device)
        for item in (base, target_swap, namespace_swap, order_swap, overflow)
    ]
    assert all(not torch.allclose(outputs[0], other) for other in outputs[1:])


def test_event_compression_has_gradient_paths() -> None:
    policy = _policy()
    obs = StructuredObservation.empty_batch(2)
    obs.events_cat[:, 0, :10] = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 1, 1])
    output = policy.encoder._encode_events(obs, policy.device)
    output.square().mean().backward()
    assert policy.encoder.event_type_emb.weight.grad is not None
    event_layer = cast(SwiGLUEncoderLayer, policy.encoder.event_encoder.layers[0])
    assert event_layer.qkv_proj.weight.grad is not None
    assert policy.encoder.event_pool_queries.grad is not None
    assert policy.encoder.event_value_proj.weight.grad is not None


def test_reducer_uses_fixed_padding_only_memory_attention() -> None:
    torch.manual_seed(0)
    reducer = MemoryReducer(32, 4, 1, 64)
    current = torch.randn(2, CURRENT_TOKEN_COUNT, 32)
    series = torch.randn(2, SERIES_SLOTS, 32)
    series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
    series_mask[1, :4] = True
    history = torch.randn(2, HISTORY_WINDOW, 32)
    history_mask = torch.zeros(2, HISTORY_WINDOW, dtype=torch.bool)
    ages = torch.zeros(2, HISTORY_WINDOW, dtype=torch.long)
    first = reducer(current, series, series_mask, history, history_mask, ages)

    changed_padding = series.clone()
    changed_padding[0] = 1000.0
    second = reducer(current, changed_padding, series_mask, history, history_mask, ages)
    torch.testing.assert_close(first.cls, second.cls)

    changed_valid_history = history.clone()
    changed_valid_history[1, -1] = 1000.0
    changed_mask = history_mask.clone()
    changed_mask[1, -1] = True
    third = reducer(current, series, series_mask, changed_valid_history, changed_mask, ages)
    assert not torch.allclose(first.cls[1], third.cls[1])
    assert first.pokemon.shape == (2, 12, 32)
    assert first.local_history_token.shape == (2, 32)


def test_local_history_tokens_are_independent_of_prior_memory_and_window_is_sliding() -> None:
    torch.manual_seed(1)
    reducer = MemoryReducer(32, 4, 1, 64)
    current = torch.randn(1, CURRENT_TOKEN_COUNT, 32)
    summary_a = reducer.local_summary(current)
    summary_b = reducer.local_summary(current)
    torch.testing.assert_close(summary_a, summary_b)

    all_history = torch.arange((HISTORY_WINDOW + 3) * 32, dtype=torch.float32).reshape(
        1, HISTORY_WINDOW + 3, 32
    )
    packed, mask, ages = pack_history_tokens(all_history[:, -HISTORY_WINDOW:])
    assert packed.shape == (1, HISTORY_WINDOW, 32)
    assert mask.all()
    assert ages[0, 0] == HISTORY_WINDOW - 1 and ages[0, -1] == 0


def test_policy_exposes_24_current_tokens_and_immutable_history_token() -> None:
    policy = _policy()
    obs = StructuredObservation.empty_batch(2)
    mask = torch.ones((2, 2, FORMAT.action_size), dtype=torch.bool)
    encoded = policy.encode(obs, mask)
    memory = policy.empty_memory(2)
    with torch.no_grad():
        output = policy.act(encoded, mask, *memory)
    assert encoded.tokens.shape == (2, CURRENT_TOKEN_COUNT, policy.d_model)
    assert output.history_token.shape == (2, policy.d_model)
    assert not hasattr(policy, "initial_state")
    assert not hasattr(output, "state")


def test_ppo_keeps_series_encoder_out_of_the_bo1_graph() -> None:
    policy = _policy()
    observation = StructuredObservation.empty_batch(1)
    action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)
    episode = TrajectoryBatch(
        observations=observation,
        action_masks=action_mask,
        actions=torch.zeros((1, 2), dtype=torch.long),
        log_probs=torch.zeros(1),
        values=torch.zeros(1),
        rewards=torch.zeros(1),
        dones=torch.ones(1),
        length=1,
        returns=torch.zeros(1),
        advantages=torch.ones(1),
    )
    loss, _, _ = _run_batched_ppo(
        [episode],
        policy,
        Magnet(policy),
        TrainingConfig(enable_optim=False),
        policy.device,
        episode=1,
        alpha=0.0,
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.series.parameters())


D_MODEL = 32


def _resampler() -> DynamicSeriesResampler:
    torch.manual_seed(0)
    return DynamicSeriesResampler(
        d_model=D_MODEL,
        nhead=4,
        dim_feedforward=64,
        num_summary_tokens=SERIES_TOKENS_PER_GAME,
        num_layers=2,
    )


def test_resample_single_game_shape() -> None:
    resampler = _resampler()
    batch_size = 2
    turns = 15
    history = torch.randn(batch_size, turns, D_MODEL)
    output = resampler.resample_single_game(history)
    assert output.shape == (batch_size, SERIES_TOKENS_PER_GAME, D_MODEL)
    assert torch.isfinite(output).all()


def test_padded_resampling_matches_independent_game_histories() -> None:
    resampler = _resampler()
    short = torch.randn(5, D_MODEL)
    long = torch.randn(9, D_MODEL)
    padded = torch.zeros((2, 9, D_MODEL))
    padded[0, :5] = short
    padded[1] = long
    mask = torch.arange(9).unsqueeze(0) < torch.tensor((5, 9)).unsqueeze(1)

    actual = resampler.resample_single_game(padded, mask)
    expected = torch.cat(
        (
            resampler.resample_single_game(short.unsqueeze(0)),
            resampler.resample_single_game(long.unsqueeze(0)),
        )
    )

    torch.testing.assert_close(actual, expected)


def test_resample_empty_game() -> None:
    resampler = _resampler()
    batch_size = 2
    empty_history = torch.zeros(batch_size, 0, D_MODEL)
    output = resampler.resample_single_game(empty_history)
    assert output.shape == (batch_size, SERIES_TOKENS_PER_GAME, D_MODEL)
    assert torch.equal(output[0], resampler.empty_game_context[0])


def test_series_context_encoding_shapes() -> None:
    resampler = _resampler()
    batch_size = 2
    game1 = torch.randn(batch_size, 10, D_MODEL)
    game2 = torch.randn(batch_size, 20, D_MODEL)

    series_tokens, series_mask = resampler([game1, game2])
    assert series_tokens.shape == (batch_size, SERIES_SLOTS, D_MODEL)
    assert series_mask.shape == (batch_size, SERIES_SLOTS)
    assert series_mask.all()

    series_tokens_1, series_mask_1 = resampler([game1])
    assert series_tokens_1.shape == (batch_size, SERIES_SLOTS, D_MODEL)
    assert series_mask_1[:, :4].all()
    assert not series_mask_1[:, 4:].any()


def _series_policy() -> PolicyNet:
    torch.manual_seed(0)
    config = ModelConfig(
        d_model=64,
        nhead=4,
        reducer_layers=1,
        dim_feedforward=128,
    )
    return build_policy(config, default_runtime_resources())


def test_policy_series_resampler() -> None:
    policy = _series_policy()
    game1 = torch.randn(1, 12, policy.d_model)
    tokens, mask = policy.encode_series([game1])
    assert tokens.shape == (1, SERIES_SLOTS, policy.d_model)
    assert mask.shape == (1, SERIES_SLOTS)


def test_compile_policy_device_guard() -> None:
    config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
    resources = default_runtime_resources()
    policy = build_policy(config, resources).to("cpu")

    # On CPU, compile_policy should return policy uncompiled (bypass CUDA guard)
    compiled_cpu = compile_policy(policy, enable=True)
    assert compiled_cpu is policy
    assert not hasattr(policy.encoder, "_orig_mod")


def test_compile_policy_state_dict_integrity() -> None:
    config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
    resources = default_runtime_resources()
    policy = build_policy(config, resources)
    keys_before = set(policy.state_dict().keys())

    compiled = compile_policy(policy, enable=False)
    keys_after = set(compiled.state_dict().keys())

    assert keys_before == keys_after


def test_unknown_legality_gate_replaces_the_mask_it_cannot_prove(policy_net):
    """The gate must be a distinct state, and it must actually suppress the mask."""
    observations = StructuredObservation.empty_batch(2)
    gates = slice(NUM_IDX_SLOT_LEGALITY_UNKNOWN, NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2)
    observations.numerical[1, TOKEN_IDX_ALLY_SIDE, gates] = 1.0

    mask = torch.zeros((2, 2, ACT_SIZE), dtype=torch.bool)
    mask[:, :, :8] = True

    with torch.no_grad():
        tokens, _ = policy_net.encoder(observations, mask)
        other_mask = mask.clone()
        other_mask[:, :, 8:16] = True
        other_tokens, _ = policy_net.encoder(observations, other_mask)

    mask_token = -POOLED_EVENT_COUNT - 1
    proven, unknown = tokens[0, mask_token], tokens[1, mask_token]
    assert torch.isfinite(tokens).all()
    assert not torch.allclose(proven, unknown)

    # a proven row tracks its mask; an unproven row ignores mask values entirely
    assert not torch.allclose(other_tokens[0, mask_token], proven)
    torch.testing.assert_close(other_tokens[1, mask_token], unknown)
