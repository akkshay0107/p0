"""Tests for policy factory and series-context construction."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from p0.battle.events import (
    SPATIAL_CATEGORICAL_WIDTH,
    SPATIAL_NUMERICAL_WIDTH,
    SPATIAL_SLOT_COUNT,
    SpatialActionType,
    SpatialTargetSlot,
)
from p0.format_config import FORMAT
from p0.model.architecture_contract import (
    HISTORY_WINDOW,
    POOLED_EVENT_COUNT,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
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
    CAT_IDX_PRESENCE_STATUS,
    CAT_IDX_STAT_PROVENANCE,
    CAT_IDX_STATUS_COUNTER_KIND,
    CATEGORICAL_WIDTH,
    EFFECT_CATEGORICAL_WIDTH,
    EFFECT_NUMERICAL_WIDTH,
    NUM_EFFECT_START,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TOKEN_IDX_ALLY_SIDE,
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


class TestFactoryAndSeries:
    def test_series_resampler_shape(self) -> None:
        """Verify the masked series kernel compresses histories into summary tokens."""
        resampler = _resampler()
        batch_size = 2
        turns = 15
        history = torch.randn(batch_size, turns, D_MODEL)
        history_mask = torch.ones(batch_size, turns, dtype=torch.bool)
        output = resampler(history, history_mask)
        assert output.shape == (batch_size, SERIES_TOKENS_PER_GAME, D_MODEL)
        assert torch.isfinite(output).all()

    def test_padded_resampling_matches_independent_game_histories(self) -> None:
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

    def test_series_resampler_rejects_empty_game(self) -> None:
        """Verify empty games are handled outside the tensor-only neural kernel."""
        resampler = _resampler()
        empty_history = torch.zeros(2, 1, D_MODEL)
        empty_mask = torch.zeros(2, 1, dtype=torch.bool)
        with pytest.raises(ValueError, match="at least one"):
            resampler(empty_history, empty_mask)

    def test_series_resampler_handles_mixed_lengths(self) -> None:
        """Verify the vectorized kernel handles mixed game lengths and padding masks."""
        resampler = _resampler()
        history = torch.zeros(2, 20, D_MODEL)
        history[0, :10] = torch.randn(10, D_MODEL)
        history[1] = torch.randn(20, D_MODEL)
        history_mask = torch.arange(20).unsqueeze(0) < torch.tensor((10, 20)).unsqueeze(1)

        output = resampler(history, history_mask)

        assert output.shape == (2, SERIES_TOKENS_PER_GAME, D_MODEL)
        assert torch.isfinite(output).all()

    def test_policy_series_resampler(self) -> None:
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

    def test_compile_policy_device_guard(self) -> None:
        """Verify compile_policy leaves CPU policy nets unmodified to prevent torch.compile overhead during CPU tests."""
        config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
        resources = default_runtime_resources()
        policy = build_policy(config, resources).to("cpu")

        compiled_cpu = compile_policy(policy, enable=True)
        assert compiled_cpu is policy

    def test_compile_policy_state_dict_integrity(self) -> None:
        """Verify compile_policy preserves state dictionary keys."""
        config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
        resources = default_runtime_resources()
        policy = build_policy(config, resources)
        keys_before = set(policy.state_dict().keys())

        compiled = compile_policy(policy, enable=False)
        keys_after = set(compiled.state_dict().keys())

        assert keys_before == keys_after

    def test_unknown_legality_gate_replaces_the_mask_it_cannot_prove(
        self, policy_net: PolicyNet
    ) -> None:
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

    def test_factory_shares_resources_and_preserves_state_dict_layout(self) -> None:
        """Verify build_policy shares underlying RuntimeResources singletons and matches direct PolicyNet instantiation."""
        resources = default_runtime_resources()
        config = ModelConfig(32, 2, 1, 128)
        direct = PolicyNet(config, resources)
        policy = build_policy(config, resources)
        builder = ObservationBuilder(resources=resources)
        assert policy.resources is policy.encoder.resources is builder.resources is resources
        assert policy.config == config == ModelConfig.from_dict(config.to_dict())
        assert direct.state_dict().keys() == policy.state_dict().keys()

    def test_policy_act_deterministic_and_top_p(self, policy: PolicyNet) -> None:
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

    def test_critic_head_shape_and_gradients(self, policy: PolicyNet) -> None:
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
