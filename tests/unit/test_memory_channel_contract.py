from typing import cast

import torch

from p0.format_config import FORMAT
from p0.model.architecture_contract import (
    CURRENT_REDUCER_TOKEN_COUNT,
    CURRENT_TOKEN_COUNT,
    HISTORY_WINDOW,
    POOLED_EVENT_COUNT,
    REDUCER_MAX_LENGTH,
    SERIES_SLOTS,
)
from p0.model.cls_reducer import MemoryReducer, pack_history_tokens
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    EVENT_COUNT,
    EVENT_METADATA_WIDTH,
    EVENT_NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    StructuredObservation,
    TokenType,
)
from p0.model.swiglu_encoder import SwiGLUEncoderLayer
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.ppo import _run_batched_ppo
from p0.training.trajectory import TrajectoryBatch


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
        TrainingConfig(enable_optim=False, warmup_episodes=0),
        policy.device,
        episode=1,
        alpha=0.0,
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in policy.series.parameters())
