from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from p0.training.config import PoolConfig
from p0.training.league.league import INIT_WR, SCORE_EPS, OpponentPool, _normalize


def _fake_policy(value: float = 0.0) -> Any:
    return SimpleNamespace(
        state_dict=lambda: {
            "weight": torch.tensor([value], dtype=torch.float32),
            "index": torch.tensor([1], dtype=torch.long),
        }
    )


class _SnapshotStore:
    def save_policy(self, path, policy, **kwargs):
        torch.save(policy.state_dict(), path)

    def load_policy(self, path, device):
        state = torch.load(path, weights_only=True)
        return _fake_policy(float(state["weight"].item()))


@pytest.fixture
def snapshot_store():
    return _SnapshotStore()


def test_sample_many_builds_roster_from_shadow_anchors_and_regulars(tmp_path, snapshot_store):
    pool = OpponentPool(tmp_path, PoolConfig(), snapshot_store)
    pool.shadow_id = "shadow"
    pool.anchor_ids = ["seed-a", "seed-b"]
    pool.regular_ids = ["ep20", "ep40"]
    pool.win_rates = {opponent_id: INIT_WR for opponent_id in pool.active_ids()}

    sampled = pool.sample_many(10)

    assert set(sampled) == {"shadow", "seed-a", "seed-b", "ep20", "ep40"}
    assert len(sampled) == len(set(sampled))


def test_add_regular_evicts_lowest_win_rate_regular_and_keeps_anchors(tmp_path, snapshot_store):
    config = PoolConfig(pool_size=4)
    pool = OpponentPool(tmp_path, config, snapshot_store)
    pool.shadow_id = "shadow"
    pool.anchor_ids = ["seed"]
    pool.regular_ids = ["weak", "strong"]
    pool.win_rates = {
        "shadow": 0.5,
        "seed": 0.5,
        "weak": 0.1,
        "strong": 0.9,
    }

    added = pool.add(cast(Any, _fake_policy(2.0)), "latest")

    assert added
    assert pool.anchor_ids == ["seed"]
    assert pool.regular_ids == ["strong", "latest"]
    assert pool.win_rates["latest"] == INIT_WR
    assert "weak" not in pool.win_rates


def _dist(*probs: list[float]) -> torch.Tensor:
    """Build a (1, 2, ACT) signature from per-player probability rows."""
    return torch.tensor([probs], dtype=torch.float32)


def test_normalize_neutral_when_spread_too_small():
    # flat values -> uninformative -> neutral 1.0 so the factor drops out of a product
    assert _normalize({"a": 0.5, "b": 0.5}) == {"a": 1.0, "b": 1.0}

    # real spread -> min maps to eps, max maps to 1.0
    out = _normalize({"a": 0.0, "b": 1.0})
    assert out["a"] == SCORE_EPS
    assert out["b"] == 1.0


def test_update_win_rate_tracks_games_and_persists(tmp_path, snapshot_store):
    config = PoolConfig()
    pool = OpponentPool(tmp_path, config, snapshot_store)
    pool.add(cast(Any, _fake_policy()), "ep20")

    pool.update_win_rate("ep20", agent_wins=1, num_games=4)
    pool.update_win_rate("ep20", agent_wins=0, num_games=2)
    assert pool.games["ep20"] == 6

    pool.save_state()
    reloaded = OpponentPool.load_or_create(tmp_path, config, snapshot_store)
    assert reloaded.games["ep20"] == 6


def test_strict_load_reports_missing_referenced_snapshot(tmp_path, snapshot_store):
    config = PoolConfig()
    pool = OpponentPool(tmp_path, config, snapshot_store)
    pool.add(cast(Any, _fake_policy()), "ep20")
    pool.save_state()
    (tmp_path / "ep20.pt").unlink()
    with pytest.raises(ValueError, match="missing policy snapshots"):
        OpponentPool.load_or_create(tmp_path, config, snapshot_store)


def test_maybe_promote_skips_when_no_candidate_meets_floor_or_games(tmp_path, snapshot_store):
    config = PoolConfig(pool_anchor_every=1, pool_anchor_min_wr=0.4, pool_anchor_min_games=10)
    pool = OpponentPool(tmp_path, config, snapshot_store)
    pool.regular_ids = ["lowwr", "fewgames"]
    pool.win_rates = {"lowwr": 0.3, "fewgames": 0.9}
    pool.games = {"lowwr": 50, "fewgames": 2}
    pool.snapshots_since_anchor = 1

    assert pool.maybe_promote() is None
    assert pool.anchor_ids == []
    # counter is not reset so we retry on the next snapshot
    assert pool.snapshots_since_anchor == 1


def test_maybe_promote_prefers_candidate_strong_on_both_axes(tmp_path, snapshot_store):
    config = PoolConfig(pool_anchor_every=1, pool_anchor_min_wr=0.4, pool_anchor_min_games=0)
    pool = OpponentPool(tmp_path, config, snapshot_store)
    pool.anchor_ids = ["anchor"]
    pool.regular_ids = ["comp_only", "both", "div_only"]
    pool.win_rates = {
        "anchor": 0.5,
        "comp_only": 0.95,  # hardest but identical to anchor (no diversity)
        "both": 0.85,  # strong on both
        "div_only": 0.45,  # most diverse but barely a threat
    }
    pool.snapshots_since_anchor = 1

    # diversity comes from JS divergence of these signatures vs the anchor's
    pool.signatures = {
        "anchor": _dist([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        "comp_only": _dist([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),  # dist 0
        "both": _dist([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]),  # moderate dist
        "div_only": _dist([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),  # max dist
    }

    promoted = pool.maybe_promote()
    assert promoted == "both"
    assert "both" in pool.anchor_ids
    assert pool.snapshots_since_anchor == 0


def test_set_reference_batch_invalidates_stale_signatures(tmp_path, snapshot_store):
    config = PoolConfig()
    pool = OpponentPool(tmp_path, config, snapshot_store)
    pool.anchor_ids = ["ghost"]  # active but has no checkpoint on disk
    pool.win_rates = {"ghost": 0.5}
    pool.signatures = {"ghost": _dist([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])}

    # a new batch makes old signatures incomparable; recompute fails for the
    # checkpoint-less id and is swallowed, leaving signatures cleared
    pool.set_reference_batch({"action_masks": torch.zeros(1)})
    assert pool.signatures == {}
