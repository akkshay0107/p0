"""Tests for policy factory and series-context construction."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from p0.battle.events import (
    EVENT_NUMERICAL_WIDTH,
    MAX_EVENT_RECORDS,
    NUM_EVENT_DETAILS,
    NUM_EVENT_KINDS,
    NUM_EVENT_POSITIONS,
)
from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import MemoryInputs, PolicyNet
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
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    SideId,
    StructuredObservation,
    TokenType,
)

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


class TestFactoryAndSeries:
    def test_compile_policy_device_guard(self) -> None:
        """Verify compile_policy leaves CPU policy nets unmodified to prevent torch.compile overhead during CPU tests."""
        config = ModelConfig(d_model=32, nhead=4, reducer_layers=1, dim_feedforward=64)
        resources = default_runtime_resources()
        policy = build_policy(config, resources).to("cpu")

        compiled_cpu = compile_policy(policy, enable=True)
        assert compiled_cpu is policy

    def test_factory_preserves_resources_and_model_config(self) -> None:
        """Verify the policy and observation builder keep the supplied resources."""
        resources = default_runtime_resources()
        config = ModelConfig(32, 2, 1, 128)
        policy = build_policy(config, resources)
        builder = ObservationBuilder(resources=resources)
        assert policy.resources is builder.resources is resources
        assert policy.config == config == ModelConfig.from_dict(config.to_dict())
