"""Tests for policy architecture, encoding, and action scoring."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from p0.battle.actions import encode_team_pair
from p0.battle.events import (
    EVENT_NUMERICAL_WIDTH,
    MAX_EVENT_RECORDS,
    NUM_EVENT_DETAILS,
    NUM_EVENT_KINDS,
    NUM_EVENT_POSITIONS,
    EventKind,
    EventPosition,
)
from p0.format_config import FORMAT
from p0.model.architecture_contract import (
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    REDUCER_MAX_LENGTH,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.policy import ActorPolicy, MemoryInputs, PolicyNet
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_EFFECT_START,
    CAT_IDX_IDENTITY_KNOWNNESS,
    CAT_IDX_MECHANIC_STATE,
    CAT_IDX_PRESENCE_STATUS,
    CAT_IDX_STAT_PROVENANCE,
    CAT_IDX_STATUS_COUNTER_KIND,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EFFECT_NUMERICAL_WIDTH,
    NUM_EFFECT_START,
    NUM_IDX_ORIG_IDX_RATIO,
    NUM_IDX_TEAM_PREVIEW,
    NUMERICAL_WIDTH,
    POKEMON_TOKENS,
    SEQUENCE_LENGTH,
    TOKEN_IDX_GLOBAL_FIELD,
    SideId,
    StructuredObservation,
    TokenType,
)
from p0.training.magnet import Magnet

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

    records = (B, MAX_EVENT_RECORDS)
    spatial_cat = torch.stack(
        (
            torch.randint(0, NUM_EVENT_KINDS, records),
            torch.randint(0, NUM_EVENT_POSITIONS, records),
            torch.randint(0, NUM_EVENT_POSITIONS, records),
            torch.randint(0, 100, records),
            torch.randint(0, NUM_EVENT_DETAILS, records),
            *[torch.zeros(records, dtype=torch.long) for _ in range(4)],
        ),
        dim=-1,
    )
    spatial_num = torch.randn((B, MAX_EVENT_RECORDS, EVENT_NUMERICAL_WIDTH))

    return StructuredObservation(
        token_type_ids=token_type_ids,
        side_ids=side_ids,
        slot_ids=slot_ids,
        categorical=categorical,
        numerical=numerical,
        spatial_cat=spatial_cat,
        spatial_num=spatial_num,
    )


class TestPolicyArchitecture:
    def test_policy_uses_rmsnorm_and_only_explicit_position_tables(self, policy: PolicyNet) -> None:
        """Verify the simplified baseline has one documented table per fixed semantic layout."""
        assert any(isinstance(module, torch.nn.RMSNorm) for module in policy.modules())
        assert not any(isinstance(module, torch.nn.LayerNorm) for module in policy.modules())
        assert policy.encoder.move_pos_emb.num_embeddings == 4
        assert policy.encoder.entity_position_emb.num_embeddings == SEQUENCE_LENGTH
        assert policy.encoder.event_order_emb.num_embeddings == MAX_EVENT_RECORDS
        assert policy.actor.reducer.memory_position_emb.num_embeddings == REDUCER_MAX_LENGTH


class TestPolicy:
    def test_switch_context_preserves_roster_identity_when_bench_rows_move(self) -> None:
        torch.manual_seed(0)
        actor = ActorPolicy(32, 4, 1, ACT_SIZE, 64)
        readout = torch.randn(1, 32)
        pokemon = torch.randn(1, 12, 32)
        moves = torch.randn(1, 2, 4, 32)
        numerical = torch.zeros(1, SEQUENCE_LENGTH, NUMERICAL_WIDTH)
        numerical[:, :6, NUM_IDX_ORIG_IDX_RATIO] = torch.arange(1, 7) / 6
        phase = torch.zeros(1, dtype=torch.bool)
        mask = torch.ones(1, 2, ACT_SIZE, dtype=torch.bool)
        actions = torch.tensor([[3, 7]])
        permutation = torch.tensor([0, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11])
        reordered_numerical = numerical.clone()
        reordered_numerical[:, :12] = numerical[:, permutation]

        logits, log_probs, _ = actor(readout, pokemon, moves, numerical, phase, mask, actions)
        reordered_logits, reordered_log_probs, _ = actor(
            readout, pokemon[:, permutation], moves, reordered_numerical, phase, mask, actions
        )

        torch.testing.assert_close(logits, reordered_logits)
        torch.testing.assert_close(log_probs, reordered_log_probs)

    def test_policy_net_act_and_encoded_evaluate_shapes(self, policy_net: PolicyNet) -> None:
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

        assert encoded.tokens.shape == (B, CURRENT_TOKEN_COUNT, policy_net.d_model)
        assert out.log_probs.shape == (B,)
        assert out.actions.shape == (B, 2)
        assert out.value.shape == (B,)
        assert out.history_token.shape == (B, 128)

        actions = torch.full((B, 2), 7, dtype=torch.long)
        with torch.no_grad():
            evaluated = policy_net.evaluate(
                policy_net.prepare(encoded, memory), action_mask, actions
            )
        assert evaluated.logits.shape == (B, 2, ACT_SIZE)
        assert evaluated.log_probs.shape == (B,)
        assert evaluated.entropy.shape == (B,)
        assert evaluated.norm_entropy.shape == (B,)
        assert evaluated.value.shape == (B,)
        assert evaluated.history_token.shape == (B, 128)

    def test_batched_encoding_matches_individual_observations(self, policy_net: PolicyNet) -> None:
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

    def test_encoding_captures_phase_and_freezes_container(self, policy_net: PolicyNet) -> None:
        """Verify the phase snapshot survives input mutation and its field cannot be replaced."""
        observation = StructuredObservation.empty_batch(1)
        action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
        encoded = policy_net.encode(observation, action_mask)
        phase = encoded.phase.clone()

        observation.numerical[:, TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] = 1.0
        torch.testing.assert_close(encoded.phase, phase)

        with pytest.raises(FrozenInstanceError):
            setattr(encoded, "phase", torch.ones_like(encoded.phase))

    def test_policy_inputs_reject_unbatched_missing_mask_and_invalid_top_p(
        self,
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

    def test_sequential_mask_fallback(self, policy_net: PolicyNet) -> None:
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

    def test_fainted_pokemon_visible(self, policy_net: PolicyNet) -> None:
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
            out_active = policy_net.evaluate(
                policy_net.prepare(encoded, memory), action_mask, actions
            )

        # Mark slot 2 as fainted
        obs.numerical[:, 2, 27] = 1.0

        with torch.no_grad():
            encoded = policy_net.encode(obs, action_mask)
            out_fainted = policy_net.evaluate(
                policy_net.prepare(encoded, memory), action_mask, actions
            )

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

    def test_gradient_flow(self, dummy_obs: StructuredObservation) -> None:
        """Verify loss backpropagation computes non-zero gradients across encoder, actor reducer, query projections, and critic head."""
        policy = build_policy(ModelConfig(64, 2, 1, 256), default_runtime_resources())
        policy.train()

        action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.uint8)

        encoded = policy.encode(dummy_obs, action_mask)
        out = policy.act(policy.prepare(encoded, _empty_memory(policy, 2)), action_mask)
        loss = out.value.mean() - out.log_probs.mean()

        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()

        missing_grads: list[str] = []

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

    def test_forced_moves_stay_legal_but_a_second_mega_is_rejected(self, policy: PolicyNet) -> None:
        """Verify public evaluation scores forced moves and masks a second Mega after a first."""
        # 7 is a normal move, 27 a Mega move, 47 the Mega forced move, 48 the forced move.
        action_mask = torch.zeros((4, 2, ACT_SIZE), dtype=torch.bool)
        action_mask[:, 0, [7, 27, 47, 48]] = True
        action_mask[:, 1, [8, 28, 47, 48]] = True
        actions = torch.tensor([[7, 47], [48, 48], [47, 47], [27, 28]])

        memory = _empty_memory(policy, 4)
        with torch.inference_mode():
            encoded = policy.encode(StructuredObservation.empty_batch(4), action_mask)
            logits = policy.evaluate(policy.prepare(encoded, memory), action_mask, actions).logits

        assert torch.isfinite(logits[0, 1, 47])
        assert torch.isfinite(logits[1, 1, 48])
        assert torch.isneginf(logits[2, 1, 47])
        assert torch.isneginf(logits[3, 1, 28])
        assert torch.isfinite(logits[3, 1, 8])

    def test_team_preview_pointer_is_symmetric_within_lead_and_back_pairs(
        self, policy: PolicyNet
    ) -> None:
        """Verify public team-preview logits are invariant to member order within each pair."""
        obs = StructuredObservation.empty_batch(1)
        obs.numerical[:, TOKEN_IDX_GLOBAL_FIELD, NUM_IDX_TEAM_PREVIEW] = 1.0
        action_mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)

        memory = _empty_memory(policy, 1)
        lead = encode_team_pair(0, 1)
        back = encode_team_pair(2, 3)
        swapped_back = encode_team_pair(3, 2)
        with torch.inference_mode():
            encoded = policy.encode(obs, action_mask)
            logits = policy.evaluate(
                policy.prepare(encoded, memory),
                action_mask,
                torch.tensor([[lead, back]]),
            ).logits

        assert torch.isfinite(logits[0, 0, lead])
        assert torch.isfinite(logits[0, 1, back])
        torch.testing.assert_close(logits[:, 0, lead], logits[:, 0, encode_team_pair(1, 0)])
        torch.testing.assert_close(logits[:, 1, back], logits[:, 1, swapped_back])

    def test_singleton_candidate_scores_match_standard_joint_scoring(
        self, policy: PolicyNet
    ) -> None:
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

    def test_candidate_order_does_not_change_scores(self, policy: PolicyNet) -> None:
        """Verify score_candidates scores are invariant to candidate action permutation."""
        encoded, action_mask, memory = _inputs(policy, batch_size=1)
        candidates = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)
        offsets = torch.tensor([0, 3], dtype=torch.long)
        prepared = policy.prepare(encoded, memory)
        with torch.no_grad():
            first = policy.score_candidates(prepared, action_mask, candidates, offsets)
            permutation = torch.tensor([2, 0, 1])
            second = policy.score_candidates(
                prepared, action_mask, candidates[permutation], offsets
            )
        torch.testing.assert_close(first, second[torch.argsort(permutation)])

    def test_candidate_scoring_applies_sequential_second_action_masks(
        self, policy: PolicyNet
    ) -> None:
        """Verify score_candidates masks a repeated switch and a second Mega that the raw mask allows."""
        encoded, _, memory = _inputs(policy, batch_size=1)
        # 1-2 are switches, 7-8 normal moves, 27-28 Mega moves.
        action_mask = torch.zeros((1, 2, FORMAT.action_size), dtype=torch.bool)
        action_mask[:, 0, [1, 7, 27]] = True
        action_mask[:, 1, [1, 2, 7, 8, 28]] = True
        candidates = torch.tensor(
            [[7, 8], [7, 9], [1, 2], [1, 1], [27, 8], [27, 28]], dtype=torch.long
        )
        offsets = torch.tensor([0, 6], dtype=torch.long)

        scores = policy.score_candidates(
            policy.prepare(encoded, memory), action_mask, candidates, offsets
        )

        assert torch.isfinite(scores[[0, 2, 4]]).all()
        assert torch.isneginf(scores[[1, 3, 5]]).all()

    def test_greedy_inference_selects_actions_autoregressively(self, policy: PolicyNet) -> None:
        """Verify greedy action selection takes each slot's best legal action given the first slot."""
        encoded, _, memory = _inputs(policy, batch_size=2)
        action_mask = torch.zeros((2, 2, FORMAT.action_size), dtype=torch.bool)
        action_mask[:, 0, 7] = True
        action_mask[:, 0, 9] = True
        action_mask[:, 1, 8] = True
        action_mask[:, 1, 10] = True

        prepared = policy.prepare(encoded, memory)
        with torch.inference_mode():
            chosen = policy.act(prepared, action_mask, deterministic=True).actions
            logits = policy.evaluate(prepared, action_mask, chosen).logits

        assert chosen.shape == (2, 2)
        torch.testing.assert_close(chosen, logits.argmax(dim=-1))

    def test_top_p_sampling_stays_inside_the_nucleus(self, policy: PolicyNet) -> None:
        """Verify top_p sampling only picks first-slot actions inside the smallest mass-p set."""
        encoded, _, memory = _inputs(policy, batch_size=1)
        action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)
        prepared = policy.prepare(encoded, memory)
        top_p = 0.5

        with torch.inference_mode():
            greedy = policy.act(prepared, action_mask, deterministic=True).actions[0, 0]
            logits = policy.evaluate(prepared, action_mask, torch.zeros((1, 2), dtype=torch.long))
            probabilities, order = torch.softmax(logits.logits[0, 0], dim=-1).sort(descending=True)
            nucleus = order[probabilities.cumsum(dim=0) - probabilities < top_p]
            torch.manual_seed(0)
            sampled = torch.stack(
                [policy.act(prepared, action_mask, top_p=top_p).actions[0, 0] for _ in range(100)]
            )
            narrowest = policy.act(prepared, action_mask, top_p=1e-6).actions[0, 0]

        assert nucleus.numel() < FORMAT.action_size
        assert torch.isin(sampled, nucleus).all()
        assert sampled.unique().numel() > 1
        assert narrowest == greedy

    def test_candidate_scoring_rejects_malformed_ragged_inputs(self, policy: PolicyNet) -> None:
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

    def test_policy_scores_all_empty_candidate_rows(self, policy: PolicyNet) -> None:
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

    def test_empty_events_are_finite_deterministic_and_pooled(self, policy: PolicyNet) -> None:
        """Verify public policy encoding returns finite, deterministic empty-event tokens."""
        obs = StructuredObservation.empty_batch(2)
        mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.bool)
        with torch.inference_mode():
            first = policy.encode(obs, mask).tokens[:, 16:20]
            second = policy.encode(obs, mask).tokens[:, 16:20]
        assert first.shape == (2, 4, policy.d_model)
        assert torch.isfinite(first).all()
        torch.testing.assert_close(first, second)

    def test_event_compression_has_gradient_paths(self, policy: PolicyNet) -> None:
        """Verify spatial event tokens participate in the public policy computation graph."""
        obs = StructuredObservation.empty_batch(2)
        obs.spatial_cat[:, 0] = torch.tensor(
            [EventKind.MOVE, EventPosition.OWN_LEFT, EventPosition.OPPONENT_LEFT, 15, 0, 0, 0, 0, 0]
        )
        action_mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.bool)
        obs.spatial_num.requires_grad_()
        output = policy.encode(obs, action_mask).tokens[:, 16:20]
        output[..., 0].sum().backward()

        assert obs.spatial_num.grad is not None
        assert torch.isfinite(obs.spatial_num.grad).all()
        assert torch.count_nonzero(obs.spatial_num.grad[:, 0]) > 0
        assert torch.count_nonzero(obs.spatial_num.grad[:, 1:]) == 0

    def test_event_only_change_reaches_board_tokens_before_the_local_summary(
        self, policy: PolicyNet
    ) -> None:
        """Verify the local mixer lets board tokens read events, which the fused encoder leaves out."""
        base = StructuredObservation.empty_batch(1)
        with_event = base.clone()
        with_event.spatial_cat[0, 0] = torch.tensor(
            [EventKind.MOVE, EventPosition.OWN_LEFT, EventPosition.OPPONENT_LEFT, 10, 0, 0, 0, 0, 0]
        )
        mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)

        with torch.no_grad():
            mixed_base = policy.encode(base, mask).tokens
            mixed_event = policy.encode(with_event, mask).tokens

        pokemon = slice(0, len(POKEMON_TOKENS))
        assert not torch.allclose(mixed_base[:, pokemon], mixed_event[:, pokemon])

    def test_local_summary_is_unchanged_by_history_and_series_inputs(
        self, policy: PolicyNet
    ) -> None:
        """The archived summary reads current tokens only, while the readout reads memory."""
        torch.manual_seed(5)
        obs = StructuredObservation.empty_batch(1)
        mask = torch.ones((1, 2, ACT_SIZE), dtype=torch.bool)
        empty = _empty_memory(policy, 1)
        filled = MemoryInputs(
            series_tokens=torch.randn(1, SERIES_SLOTS, policy.d_model),
            series_mask=torch.ones((1, SERIES_SLOTS), dtype=torch.bool),
            history_tokens=torch.randn(1, HISTORY_WINDOW, policy.d_model),
            history_mask=torch.ones((1, HISTORY_WINDOW), dtype=torch.bool),
        )

        with torch.no_grad():
            encoded = policy.encode(obs, mask)
            without_memory = policy.act(policy.prepare(encoded, empty), mask, deterministic=True)
            with_memory = policy.act(policy.prepare(encoded, filled), mask, deterministic=True)

        torch.testing.assert_close(without_memory.history_token, with_memory.history_token)
        torch.testing.assert_close(with_memory.history_token, encoded.local_history_token)
        assert not torch.allclose(without_memory.value, with_memory.value)

    def test_local_mixer_receives_gradients_from_the_local_summary(self, policy: PolicyNet) -> None:
        obs = StructuredObservation.empty_batch(2)
        obs.spatial_cat[:, 0] = torch.tensor(
            [EventKind.MOVE, EventPosition.OWN_LEFT, EventPosition.OPPONENT_LEFT, 15, 0, 0, 0, 0, 0]
        )
        mask = torch.ones((2, 2, ACT_SIZE), dtype=torch.bool)

        policy.encode(obs, mask).local_history_token.square().sum().backward()

        gradients = [parameter.grad for parameter in policy.local_mixer.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)

    def test_policy_exposes_history_token_from_act(self, policy: PolicyNet) -> None:
        """Verify the public action output carries a model-width history token."""
        obs = StructuredObservation.empty_batch(2)
        mask = torch.ones((2, 2, FORMAT.action_size), dtype=torch.bool)
        encoded = policy.encode(obs, mask)
        memory = _empty_memory(policy, 2)
        with torch.no_grad():
            output = policy.act(policy.prepare(encoded, memory), mask)
        assert output.history_token.shape == (2, policy.d_model)

    @pytest.mark.parametrize("history_length", (1, HISTORY_WINDOW))
    def test_policy_training_path_backpropagates_into_earlier_same_game_summary(
        self,
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

    @pytest.mark.parametrize("prior_games", (1, 2))
    def test_policy_series_path_trains_every_used_resampler_group(self, prior_games: int) -> None:
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
