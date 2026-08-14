from __future__ import annotations

import pytest
import torch

from p0.battle.actions import FORCED_ACTION, MOVE_START
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
from p0.model.policy import MemoryInputs
from p0.model.structured_observation import (
    CAT_IDX_STATUS,
    CAT_IDX_STATUS_COUNTER_KIND,
    CAT_KNOWNNESS_START,
    CAT_KNOWNNESS_WIDTH,
    StructuredObservation,
)
from tests.stress._helpers import (
    stress_batch_sizes,
    stress_repetitions,
    stress_rng,
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


@pytest.mark.stress
@pytest.mark.parametrize("batch_size", stress_batch_sizes())
def test_policy_handles_empty_and_maximum_memory_inputs(
    stress_policy,
    stress_device: torch.device,
    batch_size: int,
) -> None:
    """Stress test policy forward pass across empty memory and maximum Best-of-3 / turn history contexts.
    
    Verifies that:
    1. Categorical features across all embedding tables (species, moves, items, status, etc.) are handled without index errors.
    2. Zero-memory baseline inference produces valid, masked, finite action decisions.
    3. Full recurrent memory (prior Bo3 games + intra-game history window) produces valid decisions.
    4. Action mask constraints are strictly respected under both zero-memory and full-memory conditions.
    """
    action_mask = _minimal_model_action_mask(batch_size, stress_device)

    generator = torch.Generator().manual_seed(20260805 + batch_size)
    for _ in range(stress_repetitions(default=32)):
        observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
        # Random categorical values exercise unknown, known, and padding paths
        # without depending on a particular vocabulary entry.
        categorical = observation.categorical

        categorical[..., 0] = _random_ids(
            categorical[..., 0],
            stress_policy.encoder.species_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 1] = _random_ids(
            categorical[..., 1],
            stress_policy.encoder.ability_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 2] = _random_ids(
            categorical[..., 2],
            stress_policy.encoder.item_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 3:5] = _random_ids(
            categorical[..., 3:5],
            stress_policy.encoder.type_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 5:9] = _random_ids(
            categorical[..., 5:9],
            stress_policy.encoder.move_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 9:13] = _random_ids(
            categorical[..., 9:13],
            stress_policy.encoder.type_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 13:17] = _random_ids(
            categorical[..., 13:17],
            stress_policy.encoder.category_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., CAT_IDX_STATUS] = _random_ids(
            categorical[..., CAT_IDX_STATUS],
            stress_policy.encoder.status_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., CAT_IDX_STATUS_COUNTER_KIND] = _random_ids(
            categorical[..., CAT_IDX_STATUS_COUNTER_KIND],
            stress_policy.encoder.counter_kind_emb.num_embeddings,
            generator,
            stress_device,
        )
        categorical[..., 24] = _random_ids(
            categorical[..., 24],
            stress_policy.encoder.nature_emb.num_embeddings,
            generator,
            stress_device,
        )
        knownness = observation.categorical[
            ..., CAT_KNOWNNESS_START : CAT_KNOWNNESS_START + CAT_KNOWNNESS_WIDTH
        ]
        knownness[:] = _random_ids(knownness, 5, generator, stress_device)

        empty_memory = _empty_memory(stress_policy, batch_size)
        with torch.inference_mode():
            encoded = stress_policy.encode(observation, action_mask)
            empty_acted = stress_policy.act(
                stress_policy.prepare(encoded, empty_memory), action_mask
            )
        assert torch.isfinite(empty_acted.log_probs).all()
        assert torch.isfinite(empty_acted.value).all()
        # Verify gathered boolean mask is True at all chosen action positions
        assert torch.all(action_mask.gather(2, empty_acted.actions.unsqueeze(-1)).squeeze(-1))

        # Construct maximum capacity recurrent memory: intra-game turn history + inter-game Bo3 tokens
        history = torch.randn(
            (batch_size, HISTORY_WINDOW, stress_policy.d_model),
            generator=generator,
        ).to(stress_device)
        history_mask = torch.ones(
            (batch_size, HISTORY_WINDOW), dtype=torch.bool, device=stress_device
        )
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

        # The same full-memory path must remain stable across randomized calls.
        with torch.inference_mode():
            acted = stress_policy.act(stress_policy.prepare(encoded, memory), action_mask)

        assert acted.actions.shape == (batch_size, 2)
        assert torch.isfinite(acted.log_probs).all()
        assert torch.isfinite(acted.value).all()
        assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
        assert series_mask.all()
        assert history_mask.all()


@pytest.mark.stress
def test_policy_scores_ragged_candidates_with_empty_decision_rows(
    stress_policy,
    stress_device: torch.device,
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
    rng = stress_rng()
    for _ in range(stress_repetitions(default=64)):
        # Randomly assign candidate counts per row, with some rows having 0 candidates
        counts = [0 if index % 7 == 0 else rng.randrange(1, 17) for index in range(batch_size)]
        if not any(counts):
            counts[1] = 1
        candidate_rows = []
        for count in counts:
            if count:
                candidate_rows.append((MOVE_START, MOVE_START))
                candidate_rows.extend(
                    (rng.randrange(ACT_SIZE), rng.randrange(ACT_SIZE)) for _ in range(count - 1)
                )
        candidate_values = torch.tensor(
            candidate_rows,
            dtype=torch.long,
            device=stress_device,
        )
        # Construct CSR cumulative offset array of shape (batch_size + 1,)
        candidate_offsets = torch.tensor(
            (0, *torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()),
            dtype=torch.long,
        )

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
        nonempty_starts = candidate_offsets[:-1][candidate_offsets[1:] > candidate_offsets[:-1]].to(
            stress_device
        )
        # Guaranteed valid moves at non-empty starts must have finite log-probabilities
        assert torch.isfinite(log_probs[nonempty_starts]).all()


@pytest.mark.stress
def test_policy_scores_all_empty_candidate_rows(
    stress_policy,
    stress_device: torch.device,
) -> None:
    """Verify that score_candidates returns an empty 1D tensor when total candidate count is zero."""
    batch_size = 4
    observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
    action_mask = _minimal_model_action_mask(batch_size, stress_device)
    memory = _empty_memory(stress_policy, batch_size)
    candidate_values = torch.empty((0, 2), dtype=torch.long, device=stress_device)
    candidate_offsets = torch.zeros(batch_size + 1, dtype=torch.long)

    with torch.inference_mode():
        prepared = stress_policy.prepare(stress_policy.encode(observation, action_mask), memory)
        log_probs = stress_policy.score_candidates(
            prepared,
            action_mask,
            candidate_values,
            candidate_offsets,
        )

    assert log_probs.shape == (0,)
