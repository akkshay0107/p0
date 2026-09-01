from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from p0.battle.actions import encode_team_pair
from p0.battle.events import (
    SPATIAL_CATEGORICAL_WIDTH,
    SPATIAL_NUMERICAL_WIDTH,
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialTargetSlot,
)
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
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import MemoryInputs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.series_context import DynamicSeriesResampler
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_IDENTITY_KNOWNNESS,
    CAT_IDX_MECHANIC_STATE,
    CAT_IDX_NATURE,
    CAT_IDX_PRESENCE_STATUS,
    CAT_IDX_STAT_PROVENANCE,
    CAT_IDX_STATUS_COUNTER_KIND,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EFFECT_NUMERICAL_WIDTH,
    NUM_EFFECT_START,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUM_IDX_TEAM_PREVIEW,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
    TOKEN_IDX_GLOBAL_FIELD,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import compute_ppo_objective, magnet_kl_per_step

ACT_SIZE = FORMAT.action_size


@pytest.fixture
def policy_net() -> PolicyNet:
    torch.manual_seed(0)
    return build_policy(
        ModelConfig(128, 4, 2, 512),
        default_runtime_resources(),
    )


@pytest.fixture
def policy() -> PolicyNet:
    torch.manual_seed(0)
    res = build_policy(
        ModelConfig(64, 4, 1, 128),
        default_runtime_resources(),
    )
    res.eval()
    return res


def _empty_memory(policy: PolicyNet, batch_size: int) -> MemoryInputs:
    return MemoryInputs.empty(
        batch_size,
        policy.d_model,
        policy.device,
        next(policy.parameters()).dtype,
    )


def _inputs(policy: PolicyNet, batch_size: int = 2) -> tuple[Any, torch.Tensor, MemoryInputs]:
    observations = StructuredObservation.empty_batch(batch_size)
    action_mask = torch.ones((batch_size, 2, ACT_SIZE), dtype=torch.bool)
    encoded = policy.encode(observations, action_mask)
    return encoded, action_mask, _empty_memory(policy, batch_size)


def _tiny_policy() -> PolicyNet:
    return build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())


def test_policy_uses_rmsnorm_and_only_explicit_position_tables(policy: PolicyNet) -> None:
    """Verify the simplified baseline has one documented table per fixed semantic layout."""
    assert any(isinstance(module, torch.nn.RMSNorm) for module in policy.modules())
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in policy.modules())
    assert policy.encoder.move_pos_emb.num_embeddings == 4
    assert policy.encoder.entity_position_emb.num_embeddings == SEQUENCE_LENGTH
    assert policy.encoder.event_slot_emb.num_embeddings == SPATIAL_SLOT_COUNT
    assert policy.actor.reducer.memory_position_emb.num_embeddings == REDUCER_MAX_LENGTH
    assert not hasattr(policy.actor.reducer, "history_age_emb")
    assert not hasattr(policy.actor.reducer, "segment_emb")


def _batch(
    policy: PolicyNet, batch_size: int = 3
) -> tuple[StructuredObservation, torch.Tensor, torch.Tensor]:
    obs = StructuredObservation.empty_batch(batch_size)
    masks = torch.ones((batch_size, 2, policy.act_size), dtype=torch.bool)
    actions = torch.zeros((batch_size, 2), dtype=torch.long)
    return obs, masks, actions


def _live_and_magnet_logits(
    policy: PolicyNet,
    magnet: Magnet,
    obs: StructuredObservation,
    masks: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    live_encoded = policy.encode(obs, masks)
    live_memory = _empty_memory(policy, obs.numerical.size(0))
    live = policy.evaluate(policy.prepare(live_encoded, live_memory), masks, actions).logits
    magnet_encoded = magnet.policy.encode(obs, masks)
    magnet_memory = _empty_memory(magnet.policy, obs.numerical.size(0))
    mag = magnet.policy.evaluate(
        magnet.policy.prepare(magnet_encoded, magnet_memory), masks, actions
    )
    return live, mag.logits


@pytest.fixture
def dummy_obs() -> StructuredObservation:
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

    categorical = torch.zeros((B, SEQUENCE_LENGTH, CATEGORICAL_WIDTH), dtype=torch.long)
    categorical[:, 0:12, 0] = torch.randint(1, 35, (B, 12))
    categorical[:, 0:12, 1] = torch.randint(1, 24, (B, 12))
    categorical[:, 0:12, 2] = torch.randint(1, 19, (B, 12))
    categorical[:, 0:12, 3:5] = torch.randint(1, 19, (B, 12, 2))
    categorical[:, 0:12, 5:9] = torch.randint(1, 70, (B, 12, 4))
    categorical[:, 0:12, 9:13] = torch.randint(1, 19, (B, 12, 4))
    categorical[:, 0:12, 13:17] = torch.randint(1, 4, (B, 12, 4))
    categorical[:, 0:12, 17] = torch.randint(1, 7, (B, 12))
    categorical[:, 0:12, CAT_IDX_STATUS_COUNTER_KIND] = torch.randint(0, 5, (B, 12))
    categorical[:, 0:12, CAT_IDX_IDENTITY_KNOWNNESS] = torch.randint(1, 4, (B, 12))
    categorical[:, 0:12, CAT_IDX_STAT_PROVENANCE] = torch.randint(1, 4, (B, 12))
    categorical[:, 0:12, CAT_IDX_PRESENCE_STATUS] = torch.randint(1, 5, (B, 12))
    categorical[:, 0:12, CAT_IDX_MECHANIC_STATE] = torch.randint(0, 3, (B, 12))

    numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))
    for token_idx in range(15):
        categorical[
            :, token_idx, CAT_EFFECT_START : CAT_EFFECT_START + EFFECT_CATEGORICAL_WIDTH
        ] = torch.tensor((1, 1, 1))
        numerical[:, token_idx, NUM_EFFECT_START : NUM_EFFECT_START + EFFECT_NUMERICAL_WIDTH] = 1.0

    for i, idx in enumerate(range(0, 6)):
        numerical[:, idx, 26] = (i + 1) / 6.0

    numerical[:, 12, 2] = 1.0

    spatial_cat = torch.zeros((B, SPATIAL_SLOT_COUNT, SPATIAL_CATEGORICAL_WIDTH), dtype=torch.long)
    spatial_cat[..., 0] = torch.randint(0, len(SpatialActionType), (B, SPATIAL_SLOT_COUNT))
    spatial_cat[..., 1] = torch.randint(0, 100, (B, SPATIAL_SLOT_COUNT))
    spatial_cat[..., 2] = torch.randint(0, len(SpatialTargetSlot), (B, SPATIAL_SLOT_COUNT))

    spatial_num = torch.randn((B, SPATIAL_SLOT_COUNT, SPATIAL_NUMERICAL_WIDTH))

    return StructuredObservation(
        token_type_ids=token_type_ids,
        side_ids=side_ids,
        slot_ids=slot_ids,
        categorical=categorical,
        numerical=numerical,
        spatial_cat=spatial_cat,
        spatial_num=spatial_num,
    )


def test_policy_net_act_and_encoded_evaluate_shapes(policy_net: PolicyNet) -> None:
    """Verify PolicyNet act() and evaluate() return expected tensor dimensions across policy, value, and history channels."""
    B = 16
    obs = StructuredObservation.empty_batch(B)

    for i, idx in enumerate(range(1, 7)):
        obs.numerical[:, idx, 26] = (i + 1) / 6.0

    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    memory = _empty_memory(policy_net, B)

    with torch.no_grad():
        encoded = policy_net.encode(obs, action_mask)
        out = policy_net.act(policy_net.prepare(encoded, memory), action_mask)

    assert out.log_probs.shape == (B,)
    assert out.actions.shape == (B, 2)
    assert out.value.shape == (B,)
    assert out.history_token.shape == (B, 128)

    actions = torch.full((B, 2), 7, dtype=torch.long)
    with torch.no_grad():
        evaluated = policy_net.evaluate(policy_net.prepare(encoded, memory), action_mask, actions)
    assert evaluated.logits.shape == (B, 2, ACT_SIZE)
    assert evaluated.log_probs.shape == (B,)
    assert evaluated.entropy.shape == (B,)
    assert evaluated.norm_entropy.shape == (B,)
    assert evaluated.value.shape == (B,)
    assert evaluated.history_token.shape == (B, 128)


def test_batched_encoding_matches_individual_observations(policy_net: PolicyNet) -> None:
    """Verify the public encoder gives the same result for batched and individual observations."""
    B = 2
    obs = StructuredObservation.empty_batch(B)
    obs.numerical = torch.randn((B, SEQUENCE_LENGTH, NUMERICAL_WIDTH))
    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    with torch.no_grad():
        batched = policy_net.encode(obs, action_mask)

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


def test_encoded_phase_is_an_immutable_snapshot(policy_net: PolicyNet) -> None:
    """Verify EncodedObservation is a frozen immutable container whose tensor buffers cannot be modified in-place."""
    observation = StructuredObservation.empty_batch(1)
    action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
    encoded = policy_net.encode(observation, action_mask)
    phase = encoded.phase.clone()

    observation.numerical[:, TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] = 1.0
    torch.testing.assert_close(encoded.phase, phase)

    with pytest.raises(FrozenInstanceError):
        setattr(encoded, "phase", torch.ones_like(encoded.phase))


def test_reducer_keeps_current_and_history_summaries_trainable() -> None:
    """Verify differentiable training reconstruction reaches current and history tokens."""
    reducer = MemoryReducer(32, 4, 1, 64)
    current = torch.randn(2, CURRENT_TOKEN_COUNT, 32, requires_grad=True)
    local_summary = reducer.local_summary(current)
    history = torch.randn(2, HISTORY_WINDOW, 32, requires_grad=True)
    series = torch.zeros(2, SERIES_SLOTS, 32)
    series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
    history_mask = torch.ones(2, HISTORY_WINDOW, dtype=torch.bool)

    output = reducer.reduce(
        local_summary,
        current,
        series,
        series_mask,
        history,
        history_mask,
    )
    output.cls.square().mean().backward()

    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert torch.count_nonzero(history.grad) > 0


def test_policy_inputs_reject_unbatched_missing_mask_and_invalid_top_p(
    policy_net: PolicyNet,
) -> None:
    """Verify PolicyNet validates input ranks and raises errors on unbatched tensors or top_p <= 0.0."""
    obs = StructuredObservation.empty_batch(1)[0]
    action_mask = torch.ones((2, ACT_SIZE), dtype=torch.bool)

    with pytest.raises(ValueError, match="batched"):
        policy_net.encode(obs, action_mask)

    with pytest.raises(TypeError):
        policy_net.encode(obs.unsqueeze(0))  # type: ignore[call-arg]

    B = 1
    obs = StructuredObservation.empty_batch(B)
    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    encoded = policy_net.encode(obs, action_mask)

    with pytest.raises(ValueError, match="top_p"):
        policy_net.act(
            policy_net.prepare(encoded, _empty_memory(policy_net, B)),
            action_mask,
            top_p=0.0,
        )


def test_sequential_mask_fallback(policy_net: PolicyNet) -> None:
    """Verify the public evaluator masks every invalid second-slot action."""
    observation = StructuredObservation.empty_batch(1)
    action_mask = torch.zeros((1, 2, ACT_SIZE), dtype=torch.bool)
    action_mask[:, 0, 0] = True
    action_mask[:, 1, 0] = True
    memory = _empty_memory(policy_net, 1)

    with torch.inference_mode():
        encoded = policy_net.encode(observation, action_mask)
        output = policy_net.evaluate(
            policy_net.prepare(encoded, memory),
            action_mask,
            torch.zeros((1, 2), dtype=torch.long),
        )

    assert torch.isfinite(output.logits[0, 1, 0])
    assert torch.isneginf(output.logits[0, 1, 1:]).all()


def test_nature_embedding_correctness(policy_net: PolicyNet) -> None:
    """Verify changing a public nature token changes the encoded Pokémon representation."""
    first = StructuredObservation.empty_batch(1)
    second = first.clone()
    first.categorical[0, 0, CAT_IDX_NATURE] = 5
    second.categorical[0, 0, CAT_IDX_NATURE] = 12
    action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)

    with torch.inference_mode():
        first_tokens = policy_net.encode(first, action_mask).tokens
        second_tokens = policy_net.encode(second, action_mask).tokens

    assert not torch.allclose(first_tokens[:, 0], second_tokens[:, 0])


def test_fainted_pokemon_visible(policy_net: PolicyNet) -> None:
    """Verify fainted bench Pokémon tokens remain visible to the transformer encoder and affect policy output distributions."""
    B = 1
    obs = StructuredObservation.empty_batch(B)

    for i in range(0, 6):
        obs.token_type_ids[0, i] = 1
        obs.side_ids[0, i] = 1
        obs.slot_ids[0, i] = i + 1
    for i in range(6, 12):
        obs.token_type_ids[0, i] = 1
        obs.side_ids[0, i] = 2
        obs.slot_ids[0, i] = i - 5

    obs.token_type_ids[0, 12:15] = 2
    obs.side_ids[0, 13] = 1
    obs.side_ids[0, 14] = 2

    for i, idx in enumerate(range(0, 6)):
        obs.numerical[:, idx, 26] = (i + 1) / 6.0

    action_mask = torch.ones((B, 2, ACT_SIZE), dtype=torch.bool)
    actions = torch.full((B, 2), 7, dtype=torch.long)
    memory = _empty_memory(policy_net, B)

    with torch.no_grad():
        encoded = policy_net.encode(obs, action_mask)
        out_active = policy_net.evaluate(policy_net.prepare(encoded, memory), action_mask, actions)

    # Mark slot 2 as fainted
    obs.numerical[:, 2, 27] = 1.0

    with torch.no_grad():
        encoded = policy_net.encode(obs, action_mask)
        out_fainted = policy_net.evaluate(policy_net.prepare(encoded, memory), action_mask, actions)

    assert not torch.allclose(out_active.logits, out_fainted.logits, atol=1e-5)

    obs.categorical[:, 2, 0] = 41
    obs.categorical[:, 2, 14] = 2
    obs.numerical[:, 2, 0] = 0.99

    with torch.no_grad():
        encoded = policy_net.encode(obs, action_mask)
        out_modified = policy_net.evaluate(
            policy_net.prepare(encoded, memory), action_mask, actions
        )

    assert not torch.allclose(out_fainted.logits, out_modified.logits, atol=1e-5)
    assert not torch.allclose(out_fainted.value, out_modified.value, atol=1e-5)


def test_memory_reducer_pokemon_tokens_alignment() -> None:
    """Verify MemoryReducer outputs 12 aligned Pokémon tokens matching the ally and opponent roster slots."""
    reducer = MemoryReducer(32, 4, 1, 128)
    current = torch.randn(2, CURRENT_TOKEN_COUNT, 32)
    series = torch.zeros(2, SERIES_SLOTS, 32)
    series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
    history = torch.zeros(2, 48, 32)
    history_mask = torch.zeros(2, 48, dtype=torch.bool)
    reduced = reducer.reduce(
        reducer.local_summary(current),
        current,
        series,
        series_mask,
        history,
        history_mask,
    )
    assert reduced.pokemon.shape == (2, 12, 32)


def test_spatial_event_slot_encoding_distinguishes_slots(policy_net: PolicyNet) -> None:
    """Verify spatial event encoder distinguishes between different spatial slots and action targets."""
    from p0.battle.actions import ACT_SIZE

    def encode_spatial(slot_idx: int, target_slot: int) -> torch.Tensor:
        obs = StructuredObservation.empty_batch(1)
        obs.spatial_cat[0, slot_idx, 0] = int(SpatialActionType.MOVE)
        obs.spatial_cat[0, slot_idx, 1] = 10
        obs.spatial_cat[0, slot_idx, 2] = target_slot
        action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
        with torch.no_grad():
            tokens, _ = policy_net.encoder(obs, action_mask)
        # Spatial event tokens are at positions 16..19
        return tokens[0, 16 + slot_idx]

    slot0_target_opp_left = encode_spatial(slot_idx=0, target_slot=int(SpatialTargetSlot.OPP_LEFT))
    slot0_target_opp_right = encode_spatial(
        slot_idx=0, target_slot=int(SpatialTargetSlot.OPP_RIGHT)
    )
    slot1_target_opp_left = encode_spatial(slot_idx=1, target_slot=int(SpatialTargetSlot.OPP_LEFT))

    assert not torch.allclose(slot0_target_opp_left, slot0_target_opp_right, atol=1e-6)
    assert not torch.allclose(slot0_target_opp_left, slot1_target_opp_left, atol=1e-6)


def test_gradient_flow(dummy_obs: StructuredObservation) -> None:
    """Verify loss backpropagation computes non-zero gradients across encoder, actor reducer, query projections, and critic head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources()).to(device)
    policy.train()

    obs = dummy_obs.to(device)
    action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.uint8).to(device)

    encoded = policy.encode(obs, action_mask)
    out = policy.act(policy.prepare(encoded, _empty_memory(policy, 2)), action_mask)
    loss = out.value.mean() - out.log_probs.mean()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    missing_grads: list[str] = []
    zero_grads: list[str] = []

    conditional_params = [
        "series.",
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

    components = {
        "shared_encoder": "encoder",
        "actor_reducer": "actor.reducer",
        "actor_w_k": "actor.w_k",
        "actor_q_proj": "actor.q_",
        "critic_head": "critic.net",
    }

    failed_components: list[str] = []
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


def test_forced_move_keys_use_the_move_pointer_and_mega_constraint(policy: PolicyNet) -> None:
    """Verify public policy evaluation accepts forced moves and rejects other actions."""
    batch_size = 2
    obs = StructuredObservation.empty_batch(batch_size)
    action_mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool)
    action_mask[:, 0, 48] = True
    action_mask[:, 1, 47] = True

    memory = _empty_memory(policy, batch_size)
    with torch.inference_mode():
        encoded = policy.encode(obs, action_mask)
        output = policy.evaluate(
            policy.prepare(encoded, memory),
            action_mask,
            torch.tensor([[48, 47], [48, 47]]),
        )

    assert torch.isfinite(output.logits[:, 0, 48]).all()
    assert torch.isfinite(output.logits[:, 1, 47]).all()
    assert torch.isneginf(output.logits[:, 0, :48]).all()
    assert torch.isneginf(output.logits[:, 1, :47]).all()


def test_team_preview_pointer_is_symmetric_and_uses_phase_roles(policy: PolicyNet) -> None:
    """Verify public team-preview logits are invariant to member order within a pair."""
    obs = StructuredObservation.empty_batch(1)
    obs.numerical[:, TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] = 1.0
    action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)

    memory = _empty_memory(policy, 1)
    left_right = encode_team_pair(0, 1)
    right_left = encode_team_pair(1, 0)
    with torch.inference_mode():
        encoded = policy.encode(obs, action_mask)
        output = policy.evaluate(
            policy.prepare(encoded, memory),
            action_mask,
            torch.tensor([[left_right, right_left]]),
        )

    torch.testing.assert_close(output.logits[:, 0, left_right], output.logits[:, 0, right_left])
    torch.testing.assert_close(output.logits[:, 1, left_right], output.logits[:, 1, right_left])


def test_singleton_candidate_scores_match_standard_joint_scoring(policy: PolicyNet) -> None:
    """Verify score_candidates with candidate count 1 returns exact joint log probabilities matching policy.evaluate()."""
    encoded, action_mask, memory = _inputs(policy)
    candidates = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    offsets = torch.tensor([0, 1, 2], dtype=torch.long)
    prepared = policy.prepare(encoded, memory)

    with torch.no_grad():
        candidate_scores = policy.score_candidates(prepared, action_mask, candidates, offsets)
        expected = torch.stack(
            [
                policy.evaluate(
                    policy.prepare(
                        encoded[index : index + 1],
                        MemoryInputs(
                            memory.series_tokens[index : index + 1],
                            memory.series_mask[index : index + 1],
                            memory.history_tokens[index : index + 1],
                            memory.history_mask[index : index + 1],
                        ),
                    ),
                    action_mask[index : index + 1],
                    candidates[index : index + 1],
                ).log_probs[0]
                for index in range(2)
            ]
        )
    torch.testing.assert_close(candidate_scores, expected)


def test_reducer_rejects_a_local_summary_that_does_not_match_the_batch(policy: PolicyNet) -> None:
    """Verify MemoryReducer validates batch size and dtype alignment on local_summary arguments."""
    encoded, _, memory = _inputs(policy, batch_size=2)
    summary = policy.actor.reducer.local_summary(encoded.tokens)

    with pytest.raises(ValueError, match="local summary"):
        policy.actor.reducer.reduce(
            summary[:1],
            encoded.tokens,
            memory.series_tokens,
            memory.series_mask,
            memory.history_tokens,
            memory.history_mask,
        )
    with pytest.raises(ValueError, match="local summary"):
        policy.actor.reducer.reduce(
            summary.to(torch.float64),
            encoded.tokens,
            memory.series_tokens,
            memory.series_mask,
            memory.history_tokens,
            memory.history_mask,
        )


def test_candidate_order_does_not_change_scores(policy: PolicyNet) -> None:
    """Verify score_candidates scores are invariant to candidate action permutation."""
    encoded, action_mask, memory = _inputs(policy, batch_size=1)
    candidates = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)
    offsets = torch.tensor([0, 3], dtype=torch.long)
    prepared = policy.prepare(encoded, memory)
    with torch.no_grad():
        first = policy.score_candidates(prepared, action_mask, candidates, offsets)
        permutation = torch.tensor([2, 0, 1])
        second = policy.score_candidates(prepared, action_mask, candidates[permutation], offsets)
    torch.testing.assert_close(first, second[torch.argsort(permutation)])


def test_candidate_scoring_applies_second_action_mask(policy: PolicyNet) -> None:
    """Verify score_candidates applies slot 2 sequential masking, setting illegal 2nd actions to -inf score."""
    encoded, _, memory = _inputs(policy, batch_size=1)
    action_mask = torch.zeros((1, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 1, 8] = True
    candidates = torch.tensor([[7, 8], [7, 7]], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    scores = policy.score_candidates(
        policy.prepare(encoded, memory), action_mask, candidates, offsets
    )
    assert torch.isfinite(scores[0])
    assert torch.isneginf(scores[1])


def test_greedy_inference_selects_actions_autoregressively(policy: PolicyNet) -> None:
    """Verify deterministic greedy action sampling (deterministic=True) chooses valid actions autoregressively across slots."""
    encoded, _, memory = _inputs(policy, batch_size=2)
    action_mask = torch.zeros((2, 2, FORMAT.action_size), dtype=torch.bool)
    action_mask[:, 0, 7] = True
    action_mask[:, 0, 9] = True
    action_mask[:, 1, 8] = True
    action_mask[:, 1, 10] = True

    prepared = policy.prepare(encoded, memory)
    with torch.inference_mode():
        first = policy.act(prepared, action_mask, deterministic=True)
        second = policy.act(prepared, action_mask, deterministic=True)

    torch.testing.assert_close(first.actions, second.actions)
    torch.testing.assert_close(first.log_probs, second.log_probs)
    assert first.actions.shape == (2, 2)
    assert torch.all((first.actions[:, 0] == 7) | (first.actions[:, 0] == 9))
    assert torch.all((first.actions[:, 1] == 8) | (first.actions[:, 1] == 10))


def test_candidate_scoring_rejects_malformed_ragged_inputs(policy: PolicyNet) -> None:
    """Verify score_candidates validates ragged CSR offset dimensions and action ID data types."""
    encoded, action_mask, memory = _inputs(policy, batch_size=1)
    candidates = torch.tensor([[7, 8]], dtype=torch.long)
    with pytest.raises(ValueError, match="one boundary"):
        policy.score_candidates(
            policy.prepare(encoded, memory),
            action_mask,
            candidates,
            torch.tensor([0, 1, 1]),
        )
    with pytest.raises(ValueError, match="action ids"):
        policy.score_candidates(
            policy.prepare(encoded, memory),
            action_mask,
            candidates.to(torch.float32),
            torch.tensor([0, 1]),
        )


def test_policy_scores_all_empty_candidate_rows(policy: PolicyNet) -> None:
    """Verify score_candidates returns an empty tensor when every batch row has no candidates."""
    encoded, action_mask, memory = _inputs(policy, batch_size=4)
    candidate_values = torch.empty((0, 2), dtype=torch.long)
    candidate_offsets = torch.zeros(5, dtype=torch.long)

    with torch.inference_mode():
        log_probs = policy.score_candidates(
            policy.prepare(encoded, memory),
            action_mask,
            candidate_values,
            candidate_offsets,
        )

    assert log_probs.shape == (0,)


def test_magnet_params_are_frozen() -> None:
    """Verify Magnet anchor network parameters are set to requires_grad=False."""
    policy = _tiny_policy()
    magnet = Magnet(policy)
    assert all(not p.requires_grad for p in magnet.policy.parameters())


def test_magnet_refresh_does_not_perturb_optimizer_state() -> None:
    """Verify magnet.refresh(policy) updates anchor weights without corrupting or resetting live optimizer state."""
    policy = _tiny_policy()
    magnet = Magnet(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    obs, masks, actions = _batch(policy)
    encoded = policy.encode(obs, masks)
    loss = policy.evaluate(
        policy.prepare(encoded, _empty_memory(policy, obs.numerical.size(0))), masks, actions
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


def test_magnet_frozen_under_live_optimizer_step() -> None:
    """Verify Magnet anchor weights remain unchanged when the active policy undergoes optimizer step."""
    policy = _tiny_policy()
    magnet = Magnet(policy)
    snapshot = copy.deepcopy(magnet.policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)

    obs, masks, actions = _batch(policy)
    encoded = policy.encode(obs, masks)
    loss = policy.evaluate(
        policy.prepare(encoded, _empty_memory(policy, obs.numerical.size(0))), masks, actions
    ).value.sum()
    loss.backward()
    optimizer.step()

    for key, value in magnet.policy.state_dict().items():
        assert torch.equal(value, snapshot[key])


def test_magnet_kl_is_zero_at_refresh() -> None:
    """Verify magnet_kl_per_step is 0.0 immediately upon initializing or refreshing Magnet."""
    policy = _tiny_policy()
    magnet = Magnet(policy)
    obs, masks, actions = _batch(policy)
    with torch.no_grad():
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl = magnet_kl_per_step(live, mag)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_magnet_kl_grows_then_resets_after_refresh() -> None:
    """Verify Magnet KL divergence increases when weights drift and resets to 0.0 following refresh."""
    torch.manual_seed(0)
    policy = _tiny_policy()
    magnet = Magnet(policy)
    obs, masks, actions = _batch(policy)

    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl_drifted = magnet_kl_per_step(live, mag)
    assert (kl_drifted > 1e-4).any()

    magnet.refresh(policy)
    with torch.no_grad():
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl_after = magnet_kl_per_step(live, mag)
    assert torch.allclose(kl_after, torch.zeros_like(kl_after), atol=1e-5)


def test_magnet_kl_is_finite_for_degenerate_masks() -> None:
    """Verify magnet_kl_per_step produces finite numbers even under single-action degenerate masks."""
    policy = _tiny_policy()
    magnet = Magnet(policy)
    batch_size = 2
    obs = StructuredObservation.empty_batch(batch_size)
    masks = torch.zeros((batch_size, 2, policy.act_size), dtype=torch.bool)
    masks[:, :, 0] = True
    actions = torch.zeros((batch_size, 2), dtype=torch.long)
    with torch.no_grad():
        live, mag = _live_and_magnet_logits(policy, magnet, obs, masks, actions)
        kl = magnet_kl_per_step(live, mag)
    assert torch.isfinite(kl).all()


def test_magnet_kl_loss_sign_increases_with_divergence() -> None:
    """Verify compute_ppo_objective penalizes larger Magnet KL divergences proportionally."""
    config = TrainingConfig()
    common: dict[str, Any] = dict(
        current_log_probs=torch.zeros(2),
        current_values=torch.zeros(2),
        normalized_entropy=torch.zeros(2),
        old_log_probs=torch.zeros(2),
        advantages=torch.ones(2),
        returns=torch.zeros(2),
        config=config,
    )
    low, *_ = compute_ppo_objective(magnet_kl=torch.zeros(2), alpha=0.5, **common)
    high, *_ = compute_ppo_objective(magnet_kl=torch.ones(2), alpha=0.5, **common)
    assert (high > low).all()


def test_fixed_memory_and_observation_contract() -> None:
    """Verify architecture contract constants (SEQUENCE_LENGTH=15, SPATIAL_SLOT_COUNT=4, HISTORY_WINDOW=48, SERIES_SLOTS=8)."""
    observation = StructuredObservation.empty_batch(2)
    assert SEQUENCE_LENGTH == 15
    assert CATEGORICAL_WIDTH == 60
    assert NUMERICAL_WIDTH == 116
    assert SPATIAL_SLOT_COUNT == 4
    assert observation.token_type_ids.shape == (2, 15)
    assert observation.spatial_cat.shape == (2, 4, SPATIAL_CATEGORICAL_WIDTH)
    assert observation.spatial_num.shape == (2, 4, SPATIAL_NUMERICAL_WIDTH)
    assert set(TokenType) == {TokenType.POKEMON, TokenType.FIELD, TokenType.EVENT}
    assert (CURRENT_TOKEN_COUNT, CURRENT_REDUCER_TOKEN_COUNT, REDUCER_MAX_LENGTH) == (20, 21, 77)
    assert (HISTORY_WINDOW, SERIES_SLOTS, POOLED_EVENT_COUNT) == (48, 8, 4)


def test_empty_events_are_finite_deterministic_and_pooled(policy: PolicyNet) -> None:
    """Verify public policy encoding returns finite, deterministic empty-event tokens."""
    obs = StructuredObservation.empty_batch(2)
    mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.bool)
    with torch.inference_mode():
        first = policy.encode(obs, mask).tokens[:, 16:20]
        second = policy.encode(obs, mask).tokens[:, 16:20]
    assert first.shape == (2, 4, policy.d_model)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_spatial_event_channels_remain_observable(policy: PolicyNet) -> None:
    """Verify public policy encoding responds to each spatial event channel."""
    base = StructuredObservation.empty_batch(1)
    base.spatial_cat[0, 0, 0] = int(SpatialActionType.MOVE)
    base.spatial_cat[0, 0, 1] = 10
    base.spatial_cat[0, 0, 2] = int(SpatialTargetSlot.OPP_LEFT)
    base.spatial_num[0, 0, 0] = 0.25

    target_swap = base.clone()
    target_swap.spatial_cat[0, 0, 2] = int(SpatialTargetSlot.OPP_RIGHT)

    action_swap = base.clone()
    action_swap.spatial_cat[0, 0, 0] = int(SpatialActionType.SWITCH)

    damage_change = base.clone()
    damage_change.spatial_num[0, 0, 2] = 0.75

    crit_change = base.clone()
    crit_change.spatial_num[0, 0, 4] = 1.0

    mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
    outputs = [
        policy.encode(item, mask).tokens[:, 16:20]
        for item in (base, target_swap, action_swap, damage_change, crit_change)
    ]
    assert all(not torch.allclose(outputs[0], other) for other in outputs[1:])


def test_event_compression_has_gradient_paths(policy: PolicyNet) -> None:
    """Verify spatial event tokens participate in the public policy computation graph."""
    obs = StructuredObservation.empty_batch(2)
    obs.spatial_cat[:, 0, 0] = int(SpatialActionType.MOVE)
    obs.spatial_cat[:, 0, 1] = 15
    obs.spatial_cat[:, 0, 2] = int(SpatialTargetSlot.OPP_LEFT)
    action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.bool)
    output = policy.encode(obs, action_mask).tokens[:, 16:20]
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_reducer_uses_fixed_padding_only_memory_attention() -> None:
    """Verify MemoryReducer ignores masked padding tokens in series and history memory banks."""
    torch.manual_seed(0)
    reducer = MemoryReducer(32, 4, 1, 64)
    current = torch.randn(2, CURRENT_TOKEN_COUNT, 32)
    series = torch.randn(2, SERIES_SLOTS, 32)
    series_mask = torch.zeros(2, SERIES_SLOTS, dtype=torch.bool)
    series_mask[1, :4] = True
    history = torch.randn(2, HISTORY_WINDOW, 32)
    history_mask = torch.zeros(2, HISTORY_WINDOW, dtype=torch.bool)
    first = reducer.reduce(
        reducer.local_summary(current),
        current,
        series,
        series_mask,
        history,
        history_mask,
    )

    # Mutating unmasked padding tokens must not change reduced CLS token
    changed_padding = series.clone()
    changed_padding[0] = 1000.0
    second = reducer.reduce(
        reducer.local_summary(current),
        current,
        changed_padding,
        series_mask,
        history,
        history_mask,
    )
    torch.testing.assert_close(first.cls, second.cls)

    # Mutating valid unmasked history tokens alters reduced representation
    changed_valid_history = history.clone()
    changed_valid_history[1, -1] = 1000.0
    changed_mask = history_mask.clone()
    changed_mask[1, -1] = True
    third = reducer.reduce(
        reducer.local_summary(current),
        current,
        series,
        series_mask,
        changed_valid_history,
        changed_mask,
    )
    assert not torch.allclose(first.cls[1], third.cls[1])
    assert first.pokemon.shape == (2, 12, 32)
    assert first.local_history_token.shape == (2, 32)


def test_local_history_summary_is_independent_of_prior_memory_and_window_is_sliding() -> None:
    """Verify local summaries use only the current turn and history packing is right-aligned."""
    torch.manual_seed(1)
    reducer = MemoryReducer(32, 4, 1, 64)
    current = torch.randn(1, CURRENT_TOKEN_COUNT, 32)
    summary_a = reducer.local_summary(current)
    summary_b = reducer.local_summary(current)
    torch.testing.assert_close(summary_a, summary_b)

    all_history = torch.arange((HISTORY_WINDOW + 3) * 32, dtype=torch.float32).reshape(
        1, HISTORY_WINDOW + 3, 32
    )
    packed, mask = pack_history_tokens(all_history[:, -HISTORY_WINDOW:])
    assert packed.shape == (1, HISTORY_WINDOW, 32)
    assert mask.all()


def test_policy_exposes_fixed_current_tokens_and_immutable_history_token(policy: PolicyNet) -> None:
    """Verify encoder and local-history outputs follow the fixed token contracts."""
    obs = StructuredObservation.empty_batch(2)
    mask = torch.ones((2, 2, FORMAT.action_size), dtype=torch.bool)
    encoded = policy.encode(obs, mask)
    memory = _empty_memory(policy, 2)
    with torch.no_grad():
        output = policy.act(policy.prepare(encoded, memory), mask)
    assert encoded.tokens.shape == (2, CURRENT_TOKEN_COUNT, policy.d_model)
    assert output.history_token.shape == (2, policy.d_model)


@pytest.mark.parametrize("history_length", (1, HISTORY_WINDOW))
def test_policy_training_path_backpropagates_into_earlier_same_game_summary(
    history_length: int,
) -> None:
    """Verify a later public policy decision trains the encoder for a prior history row."""
    torch.manual_seed(3)
    policy = build_policy(
        ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64),
        default_runtime_resources(),
    )
    observations = StructuredObservation.empty_batch(2)
    action_mask = torch.ones((2, 2, FORMAT.action_size), dtype=torch.bool)
    encoded = policy.encode(observations, action_mask)
    earlier_summary = encoded.local_history_token[0:1]
    earlier_summary.retain_grad()
    history_padding = torch.zeros((1, HISTORY_WINDOW - history_length, policy.d_model))
    history_suffix = torch.zeros((1, history_length - 1, policy.d_model))
    memory = MemoryInputs(
        series_tokens=torch.zeros((1, SERIES_SLOTS, policy.d_model)),
        series_mask=torch.zeros((1, SERIES_SLOTS), dtype=torch.bool),
        history_tokens=torch.cat(
            (
                history_padding,
                earlier_summary.unsqueeze(1),
                history_suffix,
            ),
            dim=1,
        ),
        history_mask=torch.cat(
            (
                torch.zeros((1, HISTORY_WINDOW - history_length), dtype=torch.bool),
                torch.ones((1, history_length), dtype=torch.bool),
            ),
            dim=1,
        ),
    )

    output = policy.evaluate(
        policy.prepare(encoded[1:2], memory),
        action_mask[1:2],
        torch.zeros((1, 2), dtype=torch.long),
    )
    output.log_probs.sum().backward()

    assert earlier_summary.grad is not None
    assert torch.isfinite(earlier_summary.grad).all()
    assert torch.count_nonzero(earlier_summary.grad) > 0


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


def test_series_resampler_shape() -> None:
    """Verify the masked series kernel compresses histories into summary tokens."""
    resampler = _resampler()
    batch_size = 2
    turns = 15
    history = torch.randn(batch_size, turns, D_MODEL)
    history_mask = torch.ones(batch_size, turns, dtype=torch.bool)
    output = resampler(history, history_mask)
    assert output.shape == (batch_size, SERIES_TOKENS_PER_GAME, D_MODEL)
    assert torch.isfinite(output).all()


def test_padded_resampling_matches_independent_game_histories() -> None:
    """Verify vectorized masked resampling matches independent game resampling."""
    resampler = _resampler()
    short = torch.randn(5, D_MODEL)
    long = torch.randn(9, D_MODEL)
    padded = torch.zeros((2, 9, D_MODEL))
    padded[0, :5] = short
    padded[1] = long
    mask = torch.arange(9).unsqueeze(0) < torch.tensor((5, 9)).unsqueeze(1)

    actual = resampler(padded, mask)
    expected = torch.cat(
        (
            resampler(short.unsqueeze(0), torch.ones((1, 5), dtype=torch.bool)),
            resampler(long.unsqueeze(0), torch.ones((1, 9), dtype=torch.bool)),
        )
    )

    torch.testing.assert_close(actual, expected)


def test_series_resampler_rejects_empty_game() -> None:
    """Verify empty games are handled outside the tensor-only neural kernel."""
    resampler = _resampler()
    empty_history = torch.zeros(2, 1, D_MODEL)
    empty_mask = torch.zeros(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one"):
        resampler(empty_history, empty_mask)


def test_series_resampler_handles_mixed_lengths() -> None:
    """Verify the vectorized kernel handles mixed game lengths and padding masks."""
    resampler = _resampler()
    history = torch.zeros(2, 20, D_MODEL)
    history[0, :10] = torch.randn(10, D_MODEL)
    history[1] = torch.randn(20, D_MODEL)
    history_mask = torch.arange(20).unsqueeze(0) < torch.tensor((10, 20)).unsqueeze(1)

    output = resampler(history, history_mask)

    assert output.shape == (2, SERIES_TOKENS_PER_GAME, D_MODEL)
    assert torch.isfinite(output).all()


def test_policy_series_resampler() -> None:
    """Verify PolicyNet's series module emits the per-game token contract."""
    torch.manual_seed(0)
    config = ModelConfig(
        d_model=64,
        nhead=4,
        reducer_layers=1,
        dim_feedforward=128,
    )
    policy = build_policy(config, default_runtime_resources())
    history = torch.randn(1, 12, policy.d_model)
    history_mask = torch.ones(1, 12, dtype=torch.bool)
    tokens = policy.series(history, history_mask)
    assert tokens.shape == (1, SERIES_TOKENS_PER_GAME, policy.d_model)


@pytest.mark.parametrize("prior_games", (1, 2))
def test_policy_series_path_trains_every_used_resampler_group(prior_games: int) -> None:
    """Verify one and two prior games reach every live series-resampler parameter group."""
    torch.manual_seed(9)
    policy = build_policy(
        ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64),
        default_runtime_resources(),
    )
    observations = StructuredObservation.empty_batch(2)
    action_mask = torch.ones((2, 2, FORMAT.action_size), dtype=torch.bool)
    encoded = policy.encode(observations, action_mask)
    histories = torch.randn(prior_games, 7, policy.d_model)
    history_mask = torch.ones((prior_games, 7), dtype=torch.bool)
    series_tokens = policy.series(histories, history_mask).flatten(0, 1)
    series_tokens = torch.cat(
        (
            series_tokens,
            torch.zeros(
                (SERIES_SLOTS - series_tokens.size(0), policy.d_model),
                dtype=series_tokens.dtype,
            ),
        ),
        dim=0,
    ).unsqueeze(0)
    series_tokens = series_tokens.expand(2, -1, -1)
    series_mask = torch.zeros((2, SERIES_SLOTS), dtype=torch.bool)
    series_mask[:, : prior_games * SERIES_TOKENS_PER_GAME] = True
    memory = MemoryInputs(
        series_tokens=series_tokens,
        series_mask=series_mask,
        history_tokens=torch.zeros((2, HISTORY_WINDOW, policy.d_model)),
        history_mask=torch.zeros((2, HISTORY_WINDOW), dtype=torch.bool),
    )

    output = policy.evaluate(
        policy.prepare(encoded, memory),
        action_mask,
        torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
    )
    output.log_probs.sum().backward()

    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.series.parameters()
    )


def test_compile_policy_device_guard() -> None:
    """Verify compile_policy leaves CPU policy nets unmodified to prevent torch.compile overhead during CPU tests."""
    config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
    resources = default_runtime_resources()
    policy = build_policy(config, resources).to("cpu")

    compiled_cpu = compile_policy(policy, enable=True)
    assert compiled_cpu is policy
    assert not hasattr(policy.encoder, "_orig_mod")


def test_compile_policy_state_dict_integrity() -> None:
    """Verify compile_policy preserves state dictionary keys."""
    config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
    resources = default_runtime_resources()
    policy = build_policy(config, resources)
    keys_before = set(policy.state_dict().keys())

    compiled = compile_policy(policy, enable=False)
    keys_after = set(compiled.state_dict().keys())

    assert keys_before == keys_after


def test_unknown_legality_gate_replaces_the_mask_it_cannot_prove(policy_net: PolicyNet) -> None:
    """Verify that when legality is unknown, the encoder substitutes learned unknown-gate embeddings in place of action masks."""
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
    assert not torch.allclose(other_tokens[0, mask_token], proven)
    # When legality gate is active, changing the input mask has no effect on encoded output
    torch.testing.assert_close(other_tokens[1, mask_token], unknown)


def test_factory_shares_resources_and_preserves_state_dict_layout() -> None:
    """Verify build_policy shares underlying RuntimeResources singletons and matches direct PolicyNet instantiation."""
    resources = default_runtime_resources()
    config = ModelConfig(32, 2, 1, 128)
    direct = PolicyNet(config, resources)
    policy = build_policy(config, resources)
    builder = ObservationBuilder(resources=resources)
    assert policy.resources is policy.encoder.resources is builder.resources is resources
    assert policy.config == config == ModelConfig.from_dict(config.to_dict())
    assert direct.state_dict().keys() == policy.state_dict().keys()


def test_policy_act_deterministic_and_top_p(policy: PolicyNet) -> None:
    """Verify act() produces deterministic actions when deterministic=True and finite probabilities under top_p nucleus sampling."""
    encoded, action_mask, memory = _inputs(policy, batch_size=2)
    prepared = policy.prepare(encoded, memory)

    with torch.no_grad():
        det_out = policy.act(prepared, action_mask, deterministic=True)
        det_out_2 = policy.act(prepared, action_mask, deterministic=True)
        torch.testing.assert_close(det_out.actions, det_out_2.actions)

        top_p_out = policy.act(prepared, action_mask, top_p=0.5)
        assert top_p_out.actions.shape == (2, 2)
        assert torch.isfinite(top_p_out.log_probs).all()


def test_critic_head_shape_and_gradients(policy: PolicyNet) -> None:
    """Verify critic head predicts scalar state values (B,) and propagates non-zero gradients to critic parameters."""
    encoded, _, memory = _inputs(policy, batch_size=2)
    prepared = policy.prepare(encoded, memory)
    value = policy.critic(prepared.reduced.cls)
    assert value.shape == (2,)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert any(
        p.grad is not None and torch.abs(p.grad).sum() > 0 for p in policy.critic.parameters()
    )
