from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from p0.battle.actions import FORCED_ACTION, MOVE_START
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.policy import MemoryInputs
from p0.model.structured_observation import (
    CAT_IDX_IDENTITY_KNOWNNESS,
    CAT_IDX_NATURE,
    CAT_IDX_STATUS,
    CAT_IDX_STATUS_COUNTER_KIND,
    CounterKind,
    IdentityKnownness,
    StructuredObservation,
)
from tests.stress._helpers import (
    stress_batch_sizes,
)

ACT_SIZE = FORMAT.action_size


def _empty_memory(policy, batch_size: int) -> MemoryInputs:
    return MemoryInputs.empty(
        batch_size,
        policy.d_model,
        policy.device,
        next(policy.parameters()).dtype,
    )


def _minimal_model_action_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=device)
    mask[:, :, 0] = True
    mask[:, :, MOVE_START] = True
    mask[:, :, FORCED_ACTION] = True
    return mask


def _random_ids(
    target: torch.Tensor,
    upper_bound: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        0,
        upper_bound,
        target.shape,
        generator=generator,
        dtype=target.dtype,
    ).to(device)


def _resource_vocab_size(policy, name: str) -> int:
    return len(policy.resources.vocab[name]) + 1


@pytest.mark.stress
@pytest.mark.parametrize("batch_size", stress_batch_sizes())
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_policy_handles_empty_and_maximum_memory_inputs(
    stress_policy,
    stress_device: torch.device,
    batch_size: int,
    seed: int,
) -> None:
    """Stress test policy forward pass across empty memory and maximum Best-of-3 / turn history contexts.

    Verifies that:
    1. Categorical features across all embedding tables (species, moves, items, status, etc.) are handled without index errors.
    2. Zero-memory baseline inference produces valid, masked, finite action decisions.
    3. Full recurrent memory (prior Bo3 games + intra-game history window) produces valid decisions.
    4. Action mask constraints are strictly respected under both zero-memory and full-memory conditions.
    """
    action_mask = _minimal_model_action_mask(batch_size, stress_device)

    generator = torch.Generator().manual_seed(seed)
    observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
    # Random categorical values exercise unknown, known, and padding paths
    # without depending on a particular vocabulary entry.
    categorical = observation.categorical
    resources = stress_policy.resources
    categorical[..., 0] = _random_ids(
        categorical[..., 0],
        _resource_vocab_size(stress_policy, "species"),
        generator,
        stress_device,
    )
    categorical[..., 1] = _random_ids(
        categorical[..., 1],
        _resource_vocab_size(stress_policy, "abilities"),
        generator,
        stress_device,
    )
    categorical[..., 2] = _random_ids(
        categorical[..., 2], _resource_vocab_size(stress_policy, "items"), generator, stress_device
    )
    categorical[..., 3:5] = _random_ids(
        categorical[..., 3:5],
        _resource_vocab_size(stress_policy, "types"),
        generator,
        stress_device,
    )
    categorical[..., 5:9] = _random_ids(
        categorical[..., 5:9],
        _resource_vocab_size(stress_policy, "moves"),
        generator,
        stress_device,
    )
    categorical[..., 9:13] = _random_ids(
        categorical[..., 9:13],
        _resource_vocab_size(stress_policy, "types"),
        generator,
        stress_device,
    )
    categorical[..., 13:17] = _random_ids(
        categorical[..., 13:17],
        _resource_vocab_size(stress_policy, "categories"),
        generator,
        stress_device,
    )
    categorical[..., CAT_IDX_STATUS] = _random_ids(
        categorical[..., CAT_IDX_STATUS],
        _resource_vocab_size(stress_policy, "status"),
        generator,
        stress_device,
    )
    categorical[..., CAT_IDX_STATUS_COUNTER_KIND] = _random_ids(
        categorical[..., CAT_IDX_STATUS_COUNTER_KIND], len(CounterKind), generator, stress_device
    )
    categorical[..., CAT_IDX_NATURE] = _random_ids(
        categorical[..., CAT_IDX_NATURE], len(resources.dex["natures"]), generator, stress_device
    )
    categorical[..., CAT_IDX_IDENTITY_KNOWNNESS] = _random_ids(
        categorical[..., CAT_IDX_IDENTITY_KNOWNNESS],
        len(IdentityKnownness),
        generator,
        stress_device,
    )

    empty_memory = _empty_memory(stress_policy, batch_size)
    with torch.inference_mode():
        encoded = stress_policy.encode(observation, action_mask)
        empty_acted = stress_policy.act(stress_policy.prepare(encoded, empty_memory), action_mask)
    assert torch.isfinite(empty_acted.log_probs).all()
    assert torch.isfinite(empty_acted.value).all()
    # Verify gathered boolean mask is True at all chosen action positions
    assert torch.all(action_mask.gather(2, empty_acted.actions.unsqueeze(-1)).squeeze(-1))

    # Construct maximum capacity recurrent memory: intra-game turn history + inter-game Bo3 tokens
    history = torch.randn(
        (batch_size, HISTORY_WINDOW, stress_policy.d_model), generator=generator
    ).to(stress_device)
    history_mask = torch.ones((batch_size, HISTORY_WINDOW), dtype=torch.bool, device=stress_device)
    history_age_ids = torch.arange(HISTORY_WINDOW, device=stress_device).expand(batch_size, -1)
    prior_games = [
        [
            torch.randn((HISTORY_WINDOW, stress_policy.d_model), generator=generator).to(
                stress_device
            ),
            torch.randn((HISTORY_WINDOW, stress_policy.d_model), generator=generator).to(
                stress_device
            ),
        ]
        for _ in range(batch_size)
    ]
    series_tokens, series_mask = stress_policy.encode_series(prior_games)

    with torch.inference_mode():
        memory = MemoryInputs(
            series_tokens,
            series_mask,
            history,
            history_mask,
            history_age_ids,
        )
        encoded = stress_policy.encode(observation, action_mask)
        acted = stress_policy.act(stress_policy.prepare(encoded, memory), action_mask)

    assert acted.actions.shape == (batch_size, 2)
    assert torch.isfinite(acted.log_probs).all()
    assert torch.isfinite(acted.value).all()
    assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
    assert series_mask.all()
    assert history_mask.all()

    # The same full-memory path must remain stable across repeated evaluation.
    with torch.inference_mode():
        repeated = stress_policy.act(stress_policy.prepare(encoded, memory), action_mask)

    assert repeated.actions.shape == (batch_size, 2)
    assert torch.isfinite(repeated.log_probs).all()
    assert torch.isfinite(repeated.value).all()
    assert torch.all(action_mask.gather(2, repeated.actions.unsqueeze(-1)).squeeze(-1))
    assert series_mask.all()
    assert history_mask.all()


@pytest.mark.stress
@settings(
    max_examples=16,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    counts=st.lists(st.integers(min_value=0, max_value=16), min_size=64, max_size=64),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_policy_scores_ragged_candidates_with_empty_decision_rows(
    stress_policy,
    stress_device: torch.device,
    counts: list[int],
    seed: int,
) -> None:
    """Verify candidate scoring handles variable candidate counts per batch item via CSR offset indexing.

    In search or imitation workflows, each battle state may have a variable number of legal candidate
    joint action pairs (or zero candidates for non-acting rows). Candidate scoring must correctly
    map ragged flat candidate tensors to batch rows using cumulative offsets without NaNs.
    """
    batch_size = 64
    observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
    action_mask = torch.ones((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=stress_device)
    memory = _empty_memory(stress_policy, batch_size)
    encoded = stress_policy.encode(observation, action_mask)
    generator = torch.Generator().manual_seed(seed)
    # Keep at least one non-empty row so the candidate scorer exercises both empty and populated CSR segments.
    if not any(counts):
        counts[0] = 1
    candidate_count = sum(counts)
    candidate_values = torch.randint(
        0,
        ACT_SIZE,
        (candidate_count, 2),
        generator=generator,
        dtype=torch.long,
        device=stress_device,
    )
    candidate_offsets = torch.tensor(
        (0, *torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()), dtype=torch.long
    )
    nonempty_starts = candidate_offsets[:-1][candidate_offsets[1:] > candidate_offsets[:-1]].to(
        stress_device
    )
    candidate_values[nonempty_starts] = MOVE_START

    with torch.inference_mode():
        log_probs = stress_policy.score_candidates(
            stress_policy.prepare(encoded, memory),
            action_mask,
            candidate_values,
            candidate_offsets,
        )

    assert log_probs.shape == (candidate_values.shape[0],)
    assert not torch.isnan(log_probs).any()
    # Find starting indices of non-empty candidate segments
    # Guaranteed valid moves at non-empty starts must have finite log-probabilities
    assert torch.isfinite(log_probs[nonempty_starts]).all()
