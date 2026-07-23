from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_bc_trainer import _chunk
from test_replay_dataset import _payload

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy
from p0.model.resources import default_runtime_resources
from p0.replays.compile import build_shards_from_cache
from p0.replays.dataset import LazyReplayDataset, assign_series_splits
from p0.replays.schema import LabelKind
from p0.replays.scrape import HttpResponse, ReplayFetcher, ScrapeConfig, load_raw_replay
from p0.training.bc import BCTrainer, MultiGameBCCollator
from p0.training.checkpoint import CheckpointStore
from p0.training.config import BCConfig


def test_scrape_soft_limit_completes_the_final_linked_series(tmp_path: Path) -> None:
    format_id = FORMAT.bo3_format
    seeds = [f"{format_id}-{number}" for number in (100, 200, 300)]
    siblings = [f"{format_id}-{number}" for number in (101, 201, 301)]
    bodies: dict[str, bytes] = {}
    for seed, sibling in zip(seeds, siblings, strict=True):
        first = _payload(seed, parent=f"series-{seed}")
        first["log"] = f'|uhtml|next|<a href="/battle-{sibling}">Game 2</a>\n{first["log"]}'
        bodies[seed] = json.dumps(first).encode()
        bodies[sibling] = json.dumps(_payload(sibling, parent=f"series-{seed}")).encode()

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        if "search.invalid" in url:
            return HttpResponse(
                200,
                json.dumps([{"id": seed, "formatid": format_id} for seed in seeds]).encode(),
            )
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, bodies[replay_id])

    config = ScrapeConfig(
        format_id=format_id,
        cache_dir=tmp_path,
        search_url="https://search.invalid",
        replay_url_template="https://replay.invalid/{replay_id}.json",
        page_size=50,
        limit_games=3,
        rate_limit_per_second=0,
    )
    entries = ReplayFetcher(config, transport=transport).acquire()

    assert {entry.replay_id for entry in entries} == {
        seeds[0],
        siblings[0],
        seeds[1],
        siblings[1],
    }


def test_raw_cache_keeps_malformed_bytes_for_later_quality_rejection(
    tmp_path: Path,
) -> None:
    replay_id = f"{FORMAT.bo3_format}-malformed"

    def transport(url: str, timeout: float) -> HttpResponse:
        del url, timeout
        return HttpResponse(200, b"not-json")

    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=tmp_path,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    ReplayFetcher(config, transport=transport).acquire((replay_id,))

    assert (
        load_raw_replay(tmp_path / FORMAT.bo3_format / "raw" / f"{replay_id}.json.gz")
        == b"not-json"
    )


def test_cache_build_is_dataset_bound_and_enforces_empty_bo1_history(
    tmp_path: Path,
) -> None:
    good_id = f"{FORMAT.bo3_format}-good"
    bad_id = f"{FORMAT.bo3_format}-bad"
    good = _payload(good_id, parent="source-series")
    bad = _payload(bad_id, parent="source-series")
    bad["log"] = "\n".join(
        line for line in str(bad["log"]).splitlines() if "|showteam|" not in line
    )
    bodies = {good_id: json.dumps(good).encode(), bad_id: json.dumps(bad).encode()}

    def transport(url: str, timeout: float) -> HttpResponse:
        del timeout
        replay_id = url.rsplit("/", 1)[-1].removesuffix(".json")
        return HttpResponse(200, bodies[replay_id])

    cache = tmp_path / "replays"
    config = ScrapeConfig(
        format_id=FORMAT.bo3_format,
        cache_dir=cache,
        replay_url_template="https://replay.invalid/{replay_id}.json",
        rate_limit_per_second=0,
    )
    ReplayFetcher(config, transport=transport).acquire((good_id, bad_id))
    first = build_shards_from_cache(cache, tmp_path / "shards")
    second = build_shards_from_cache(cache, tmp_path / "shards")

    assert first.manifest_path == second.manifest_path
    assert first.manifest.dataset_hash == second.manifest.dataset_hash
    assert first.manifest.source_games == 2
    assert first.manifest.accepted_games == 1
    assert first.manifest.rejected_games == 1
    chunks = list(LazyReplayDataset(first.manifest_path))
    assert len(chunks) == 2
    assert all(chunk.summary_inputs == () for chunk in chunks)


def test_collator_fills_budget_across_games_and_rebases_candidates() -> None:
    first = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT), int(LabelKind.EXACT)],
        [(7, 8), (7, 8)],
        [0, 0, 1, 2],
    )
    second = _chunk(
        [int(LabelKind.EXACT), int(LabelKind.PARTIAL)],
        [(7, 8), (7, 8), (9, 10)],
        [0, 1, 3],
    )
    second = replace(second, series_id="series-2")

    batches = list(MultiGameBCCollator(4)((first, second)))

    assert [batch.decisions for batch in batches] == [4, 1]
    assert batches[0].games == 2
    assert batches[0].candidate_offsets.tolist() == [0, 0, 1, 2, 3]
    assert batches[1].candidate_offsets.tolist() == [0, 2]
    assert torch.all(batches[0].history_local_indices[3] == -1)


def test_split_assignment_populates_all_requested_splits_when_possible() -> None:
    manifest = assign_series_splits(
        ("a", "b", "c", "d", "e"),
        runtime_contract_sha256="a" * 64,
        dataset_hash="b" * 64,
    )

    assert set(manifest.assignments.values()) == {"train", "validation", "test"}


def test_validation_is_deterministic_inference_only_and_reports_all_counts() -> None:
    game = _chunk(
        [int(LabelKind.UNKNOWN), int(LabelKind.EXACT), int(LabelKind.PARTIAL)],
        [(7, 8), (7, 8), (9, 10)],
        [0, 0, 1, 3],
    )
    policy = build_policy(
        ModelConfig(d_model=64, nhead=4, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    trainer = BCTrainer(
        policy,
        (game,),
        BCConfig(batch_decisions=3, amp=False),
        device="cpu",
    )
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}

    first = trainer.evaluate()
    second = trainer.evaluate()

    assert first.to_dict() == second.to_dict()
    assert first.decisions == 3
    assert first.labeled_decisions == 2
    assert first.unknown_decisions == 1
    assert first.exact_decisions == 1
    assert first.partial_decisions == 1
    assert first.non_finite_values == 0
    assert first.illegal_predictions == 0
    assert all(
        torch.equal(before[name], parameter) for name, parameter in policy.named_parameters()
    )


def test_bc_training_state_cannot_resume_ppo_but_policy_weights_can_transfer(
    tmp_path: Path,
) -> None:
    policy = build_policy(
        ModelConfig(d_model=64, nhead=4, reducer_layers=1, dim_feedforward=128),
        default_runtime_resources(),
    )
    store = CheckpointStore()
    training_path = tmp_path / "bc-training.pt"
    policy_path = tmp_path / "bc-policy.pt"
    store.save_training_state(
        training_path,
        1,
        policy,
        optimizer=torch.optim.AdamW(policy.parameters()),
        trainer_kind="bc",
    )
    store.save_policy(policy_path, policy, metadata={"dataset_hash": "a" * 64})
    restored = store.load_policy(policy_path, "cpu")

    with pytest.raises(ValueError, match="trainer"):
        store.load_training_state(
            training_path,
            restored,
            expected_trainer_kind="ppo",
            require_training_state=True,
        )
    with pytest.raises(ValueError, match="weights-only"):
        store.load_training_state(
            policy_path,
            restored,
            expected_trainer_kind="ppo",
            require_training_state=True,
        )
