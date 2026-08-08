from __future__ import annotations

import pytest
import torch

from p0.battle.actions import FORCED_ACTION, MOVE_START
from p0.format_config import FORMAT
from p0.model.architecture_contract import HISTORY_WINDOW
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


def _minimal_model_action_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=device)
    mask[:, :, 0] = True
    mask[:, :, MOVE_START] = True
    mask[:, :, FORCED_ACTION] = True
    return mask


@pytest.mark.stress
@pytest.mark.parametrize("batch_size", stress_batch_sizes())
def test_policy_handles_empty_and_maximum_memory_inputs(
    stress_policy,
    stress_device: torch.device,
    batch_size: int,
) -> None:
    """Exercise the documented zero-memory and full Bo3/history contracts."""
    action_mask = _minimal_model_action_mask(batch_size, stress_device)

    generator = torch.Generator().manual_seed(20260805 + batch_size)
    for _ in range(stress_repetitions(default=32)):
        observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
        # Random categorical values exercise unknown, known, and padding paths
        # without depending on a particular vocabulary entry.
        categorical = observation.categorical

        def random_ids(target: torch.Tensor, upper_bound: int) -> torch.Tensor:
            return torch.randint(
                0,
                upper_bound,
                target.shape,
                generator=generator,
                dtype=target.dtype,
            ).to(stress_device)

        categorical[..., 0] = random_ids(
            categorical[..., 0], stress_policy.encoder.species_emb.num_embeddings
        )
        categorical[..., 1] = random_ids(
            categorical[..., 1], stress_policy.encoder.ability_emb.num_embeddings
        )
        categorical[..., 2] = random_ids(
            categorical[..., 2], stress_policy.encoder.item_emb.num_embeddings
        )
        categorical[..., 3:5] = random_ids(
            categorical[..., 3:5], stress_policy.encoder.type_emb.num_embeddings
        )
        categorical[..., 5:9] = random_ids(
            categorical[..., 5:9], stress_policy.encoder.move_emb.num_embeddings
        )
        categorical[..., 9:13] = random_ids(
            categorical[..., 9:13], stress_policy.encoder.type_emb.num_embeddings
        )
        categorical[..., 13:17] = random_ids(
            categorical[..., 13:17], stress_policy.encoder.category_emb.num_embeddings
        )
        categorical[..., CAT_IDX_STATUS] = random_ids(
            categorical[..., CAT_IDX_STATUS], stress_policy.encoder.status_emb.num_embeddings
        )
        categorical[..., CAT_IDX_STATUS_COUNTER_KIND] = random_ids(
            categorical[..., CAT_IDX_STATUS_COUNTER_KIND],
            stress_policy.encoder.counter_kind_emb.num_embeddings,
        )
        categorical[..., 24] = random_ids(
            categorical[..., 24], stress_policy.encoder.nature_emb.num_embeddings
        )
        observation.categorical[
            ..., CAT_KNOWNNESS_START : CAT_KNOWNNESS_START + CAT_KNOWNNESS_WIDTH
        ] = 4

        empty_memory = stress_policy.empty_memory(batch_size)
        with torch.inference_mode():
            empty_acted = stress_policy.act_obs(observation, action_mask, *empty_memory)
        assert torch.isfinite(empty_acted.log_probs).all()
        assert torch.isfinite(empty_acted.value).all()
        assert torch.all(action_mask.gather(2, empty_acted.actions.unsqueeze(-1)).squeeze(-1))

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
            acted = stress_policy.act_obs(
                observation,
                action_mask,
                series_tokens,
                series_mask,
                history,
                history_mask,
                history_age_ids,
            )

        assert acted.actions.shape == (batch_size, 2)
        assert torch.isfinite(acted.log_probs).all()
        assert torch.isfinite(acted.value).all()
        assert torch.all(action_mask.gather(2, acted.actions.unsqueeze(-1)).squeeze(-1))
        assert series_mask.all()
        assert history_mask.all()

        # The same full-memory path must remain stable across randomized calls.
        with torch.inference_mode():
            acted = stress_policy.act_obs(
                observation,
                action_mask,
                series_tokens,
                series_mask,
                history,
                history_mask,
                history_age_ids,
            )

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
    """Verify candidate offsets support variable counts without flattening assumptions."""
    batch_size = 64
    observation = StructuredObservation.empty_batch(batch_size).to(stress_device)
    action_mask = torch.ones((batch_size, 2, ACT_SIZE), dtype=torch.bool, device=stress_device)
    memory = stress_policy.empty_memory(batch_size)
    encoded = stress_policy.encode(observation, action_mask)
    rng = stress_rng()
    for _ in range(stress_repetitions(default=64)):
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
        candidate_offsets = torch.tensor(
            (0, *torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()),
            dtype=torch.long,
        )

        with torch.inference_mode():
            log_probs = stress_policy.actor.score_joint_candidates(
                encoded,
                action_mask,
                *memory,
                candidate_values,
                candidate_offsets,
            )

        assert log_probs.shape == (candidate_values.shape[0],)
        assert not torch.isnan(log_probs).any()
        nonempty_starts = candidate_offsets[:-1][candidate_offsets[1:] > candidate_offsets[:-1]].to(
            stress_device
        )
        assert torch.isfinite(log_probs[nonempty_starts]).all()
