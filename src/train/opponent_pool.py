import json
import logging
import random
from pathlib import Path
from typing import Self

import torch

from src.model.policy import PolicyNet
from src.train.config import PPOConfig

# new snapshots and seeds join the pool at parity (opponent-vs-agent win rate)
INITIAL_WIN_RATE = 0.5


# manager for past checkpoints of the model
# past checkpoints used as opponents in a league manner
# similar to alphastar (no specific exploiter agents though)
#
# the pool is split in two:
#   - anchors: never evicted (the bc seeds plus periodically promoted snapshots)
#   - rotating: ordinary snapshots that get evicted once the agent outgrows them
class OpponentPool:
    def __init__(self, pool_dir: Path, config: PPOConfig):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.opponent_ids: list[str] = []  # ordered list of opponents (first = oldest)
        self.win_rates: dict[str, float] = {}  # ema win rate for each of the policies in the pool
        self.anchor_ids: list[str] = []  # never-evicted subset of opponent_ids
        # rotating snapshots admitted since the last promotion; the best of every
        # `pool_anchor_every` of them gets promoted into the anchor set
        self.promotion_window: list[str] = []

    def __len__(self) -> int:
        return len(self.opponent_ids)

    def __repr__(self) -> str:
        return (
            f"OpponentPool(size={len(self)}/{self.config.pool_size}, "
            f"anchors={self.anchor_ids}, ids={self.opponent_ids})"
        )

    def save_state(self) -> None:
        state = {
            "opponent_ids": self.opponent_ids,
            "win_rates": self.win_rates,
            "anchor_ids": self.anchor_ids,
            "promotion_window": self.promotion_window,
        }
        with open(self.pool_dir / "pool_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        path = self.pool_dir / "pool_state.json"
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.opponent_ids = state.get("opponent_ids", [])
            self.win_rates = state.get("win_rates", {})
            self.anchor_ids = state.get("anchor_ids", [])
            self.promotion_window = state.get("promotion_window", [])

        # sync with pool directory (in case json got deleted)
        # sort by creation time to maintain order
        existing_files = sorted(self.pool_dir.glob("*.pt"), key=lambda p: p.stat().st_ctime)
        existing_ids = {pt.stem for pt in existing_files}

        for pt_file in existing_files:
            opponent_id = pt_file.stem
            if opponent_id not in self.opponent_ids:
                self.opponent_ids.append(opponent_id)
                if opponent_id not in self.win_rates:
                    self.win_rates[opponent_id] = INITIAL_WIN_RATE

        # bc seeds are always anchors (migrates pools saved before anchoring existed)
        for opponent_id in self.opponent_ids:
            if opponent_id.startswith("seed") and opponent_id not in self.anchor_ids:
                self.anchor_ids.append(opponent_id)

        # drop entries that no longer exist on disk
        self.opponent_ids = [oid for oid in self.opponent_ids if oid in existing_ids]
        self.win_rates = {oid: wr for oid, wr in self.win_rates.items() if oid in existing_ids}
        self.anchor_ids = [oid for oid in self.anchor_ids if oid in existing_ids]
        self.promotion_window = [
            oid
            for oid in self.promotion_window
            if oid in existing_ids and oid not in self.anchor_ids
        ]

    @classmethod
    def load_or_create(cls, pool_dir: Path, config: PPOConfig) -> Self:
        """Load an existing pool from disk, or create an empty one."""
        pool = cls(pool_dir, config)
        pool._load_state()
        return pool

    def _remove(self, opponent_id: str) -> None:
        evict_path = self.pool_dir / f"{opponent_id}.pt"
        if evict_path.exists():
            evict_path.unlink()
        self.opponent_ids.remove(opponent_id)
        self.win_rates.pop(opponent_id, None)
        if opponent_id in self.anchor_ids:
            self.anchor_ids.remove(opponent_id)
        if opponent_id in self.promotion_window:
            self.promotion_window.remove(opponent_id)

    def add(self, policy: PolicyNet, id: str, anchor: bool = False) -> bool:
        if id in self.opponent_ids:
            return False

        # when full, make room by evicting the weakest rotating (non-anchor) opponent.
        # win_rates are opponent-vs-agent, so the lowest one is the snapshot the agent
        # beats the most. a new snapshot joins at parity, so it only displaces an
        # opponent the agent already outgrew (anchors bypass this gate).
        if len(self.opponent_ids) >= self.config.pool_size:
            rotating = [oid for oid in self.opponent_ids if oid not in self.anchor_ids]
            if not rotating:
                return False
            weakest = min(rotating, key=lambda oid: self.win_rates[oid])
            if not anchor and self.win_rates[weakest] >= INITIAL_WIN_RATE:
                return False
            self._remove(weakest)

        save_path = self.pool_dir / f"{id}.pt"
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
            },
            save_path,
        )

        self.opponent_ids.append(id)
        self.win_rates[id] = INITIAL_WIN_RATE
        if anchor:
            self.anchor_ids.append(id)
        else:
            self.promotion_window.append(id)
        return True

    def maybe_promote(self) -> str | None:
        """Promote the strongest of every `pool_anchor_every` rotating snapshots
        into the permanent anchor set. Returns the promoted id, or None."""
        k = self.config.pool_anchor_every
        if k <= 0 or len(self.promotion_window) < k:
            return None

        candidates = [oid for oid in self.promotion_window if oid in self.win_rates]
        self.promotion_window = []
        if not candidates:
            return None

        # promote the snapshot the agent struggles most against (highest
        # opponent-vs-agent win rate) == the strongest of the window
        best = max(candidates, key=lambda oid: self.win_rates[oid])
        self.anchor_ids.append(best)
        logging.info(
            f"Promoted '{best}' to anchor pool "
            f"(win rate {self.win_rates[best]:.3f}). Anchors: {self.anchor_ids}"
        )
        return best

    def load_policy(self, opponent_id: str, device: str) -> PolicyNet:
        path = self.pool_dir / f"{opponent_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for opponent '{opponent_id}' not found at {path}")

        checkpoint = torch.load(path, weights_only=True, map_location=device)
        net = PolicyNet()
        net.load_state_dict(checkpoint["model_state_dict"])
        return net.to(device).eval()

    def update_win_rate(self, opponent_id: str, agent_wins: int, num_games: int = 1) -> None:
        if opponent_id not in self.win_rates:
            return

        # agent_wins <=> games lost by the opp id against latest policy
        observed_wr = 1.0 - agent_wins / num_games
        curr_wr = self.win_rates[opponent_id]

        alpha = self.config.pool_win_rate_smoothing
        self.win_rates[opponent_id] = (1 - alpha) * curr_wr + alpha * observed_wr

    def _pfsp_weight(self, opponent_id: str) -> float:
        # competitive weighting: starves both too good and too bad agents
        wr = self.win_rates[opponent_id]
        return max(self.config.pool_wr_floor, wr * (1.0 - wr))

    def sample(self, device: str) -> tuple[PolicyNet, str]:
        """Returns a frozen policy (loaded to device) from the pool sampled with PFSP weights."""
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample().")

        weights = [self._pfsp_weight(oid) for oid in self.opponent_ids]
        (opponent_id,) = random.choices(self.opponent_ids, weights=weights, k=1)
        return self.load_policy(opponent_id, device), opponent_id

    def sample_many(self, count: int) -> list[str]:
        """Sample opponent IDs without replacement using PFSP weights."""
        if count <= 0:
            raise ValueError("Opponent count must be greater than zero.")
        if not self.opponent_ids:
            raise RuntimeError("OpponentPool is empty. Call pool.add() before pool.sample_many().")

        count = min(count, len(self.opponent_ids))
        weights = torch.tensor(
            [self._pfsp_weight(opponent_id) for opponent_id in self.opponent_ids],
            dtype=torch.float32,
        )
        indices = torch.multinomial(weights, count, replacement=False)
        return [self.opponent_ids[index] for index in indices.tolist()]
