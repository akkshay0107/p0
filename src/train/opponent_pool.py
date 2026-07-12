import json
import logging
from pathlib import Path
from typing import Self

import torch

from src.format_config import (
    policy_model_config,
    runtime_manifest_sha256,
    validate_artifact_manifest_reference,
)
from src.model.policy import PolicyNet
from src.model.structured_observation import StructuredObservation
from src.train.config import PoolConfig

INIT_WR = 0.5
SHADOW_ID = "shadow"
alpha_shadow = 0.99  # ema decay rate of the shadow model

SCORE_EPS = 0.05  # floor for not zeroing the product
DIV_MIN_SPREAD = 0.01  # JSD spread below this is treated as uninformative


def _normalize(
    values: dict[str, float], min_spread: float = DIV_MIN_SPREAD, eps: float = SCORE_EPS
) -> dict[str, float]:
    """Min-max candidate values into [eps, 1]. When the spread is too small to
    be meaningful, return a neutral 1.0 for every candidate so the factor drops
    out of a multiplicative score instead of annihilating it."""
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < min_spread:
        return {k: 1.0 for k in values}
    return {k: eps + (1 - eps) * (v - lo) / (hi - lo) for k, v in values.items()}


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Computes the Jensen-Shannon divergence between two batched probability distributions.
    Expects p and q to be valid distributions that sum to 1 over their last dimension."""
    m = 0.5 * (p + q)
    safe_m = m.clamp_min(1e-12)

    kl_p = torch.sum(torch.where(p > 0, p * torch.log2(p / safe_m), 0.0), dim=-1)
    kl_q = torch.sum(torch.where(q > 0, q * torch.log2(q / safe_m), 0.0), dim=-1)

    return (0.5 * kl_p + 0.5 * kl_q).mean().item()


class OpponentPool:
    def __init__(self, pool_dir: Path, config: PoolConfig):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.shadow_id: str | None = None
        self.anchor_ids: list[str] = []
        self.regular_ids: list[str] = []
        self.win_rates: dict[str, float] = {}
        self.games: dict[str, int] = {}
        self.snapshots_since_anchor = 0
        self.signatures: dict[str, torch.Tensor] = {}
        self.reference_batch: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.active_ids())

    def __repr__(self) -> str:
        return (
            f"OpponentPool(size={len(self)}/{self.config.pool_size}, "
            f"shadow={self.shadow_id}, anchors={self.anchor_ids}, regular={self.regular_ids})"
        )

    @property
    def opponent_ids(self) -> list[str]:
        return self.active_ids()

    def active_ids(self) -> list[str]:
        ids: list[str] = []
        if self.shadow_id is not None:
            ids.append(self.shadow_id)
        ids.extend(self.anchor_ids)
        ids.extend(self.regular_ids)
        return ids

    def contains(self, opponent_id: str) -> bool:
        return opponent_id in self.active_ids()

    def sample_many(self, count: int) -> list[str]:
        """Sample opponent IDs across shadow, anchors, and regulars without replacement."""
        if count <= 0:
            raise ValueError("Opponent count must be greater than zero.")

        roster = self.active_ids()
        if not roster:
            raise RuntimeError("OpponentPool is empty. Add an opponent before sampling.")

        count = min(count, len(roster))
        weights = torch.tensor(
            [self._pfsp_weight(opponent_id) for opponent_id in roster],
            dtype=torch.float32,
        )
        indices = torch.multinomial(weights, count, replacement=False)
        return [roster[index] for index in indices.tolist()]

    def update_win_rate(self, opponent_id: str, agent_wins: int, num_games: int = 1) -> None:
        if opponent_id not in self.win_rates:
            return

        # agent_wins <=> games lost by the opp id against latest policy
        observed_wr = 1.0 - agent_wins / num_games
        curr_wr = self.win_rates[opponent_id]

        alpha = self.config.pool_win_rate_smoothing
        self.win_rates[opponent_id] = (1 - alpha) * curr_wr + alpha * observed_wr
        self.games[opponent_id] = self.games.get(opponent_id, 0) + num_games

    def load_policy(self, opponent_id: str, device: str) -> PolicyNet:
        path = self._checkpoint_path(opponent_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint for opponent '{opponent_id}' not found at {path}")

        checkpoint = torch.load(path, weights_only=True, map_location=device)
        validate_artifact_manifest_reference(checkpoint)
        config_value = checkpoint.get("model_config")
        if not isinstance(config_value, dict):
            raise ValueError(f"Opponent checkpoint {path} has no valid model configuration")
        config = dict(config_value)
        if not config:
            raise ValueError(f"Opponent checkpoint {path} has no serialized model configuration")
        try:
            config["obs_dim"] = tuple(config["obs_dim"])
            net = PolicyNet(**config)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid model configuration in opponent checkpoint {path}") from exc
        net.load_state_dict(checkpoint["model_state_dict"])
        return net.to(device).eval()

    def add_anchor(self, policy: PolicyNet, id: str) -> bool:
        if self.contains(id):
            return False
        self._register_opponent(id, policy, "anchor")
        return True

    def add(self, policy: PolicyNet, id: str) -> bool:
        """Add the latest snapshot to the regular pool, evicting the weakest regular."""
        if self.contains(id):
            return False

        while len(self.regular_ids) >= self._regular_capacity():
            weakest = min(self.regular_ids, key=lambda oid: self.win_rates.get(oid, INIT_WR))
            self._deregister_opponent(weakest)

        self._register_opponent(id, policy, "regular")
        self.snapshots_since_anchor += 1
        return True

    def maybe_promote(self, reference_batch: dict[str, torch.Tensor] | None = None) -> str | None:
        """Promote the regular that best balances competitiveness and diversity.

        A candidate must still be a genuine threat (win rate >= floor) and have
        been played enough to trust that win rate (games >= min). Among those, the
        score is the *product* of normalized competitiveness and diversity, so a
        promotion has to be good at both. When committing to a promotion, refresh
        all signatures against the freshly captured `reference_batch` first.
        """
        self._drop_outgrown_anchors()

        k = self.config.pool_anchor_every
        if k <= 0 or self.snapshots_since_anchor < k or not self.regular_ids:
            return None

        floor = self.config.pool_anchor_min_wr
        min_games = self.config.pool_anchor_min_games
        candidates = [
            oid
            for oid in self.regular_ids
            if self.win_rates.get(oid, INIT_WR) >= floor and self.games.get(oid, 0) >= min_games
        ]
        if not candidates:
            # nothing worth protecting yet; leave the counter so we retry next snapshot
            return None

        # committed to a promotion -> refresh signatures against the fresh batch first
        if reference_batch is not None:
            self.set_reference_batch(reference_batch)
        self.snapshots_since_anchor = 0

        comp = _normalize({oid: self.win_rates[oid] for oid in candidates})
        div = _normalize({oid: self._min_anchor_distance(oid) for oid in candidates})
        best = max(candidates, key=lambda oid: comp[oid] * div[oid])

        self.regular_ids.remove(best)
        self.anchor_ids.append(best)

        logging.info(
            f"Promoted '{best}' to anchor pool "
            f"(win rate {self.win_rates.get(best, INIT_WR):.3f}, "
            f"diversity dist {self._min_anchor_distance(best):.4f}, "
            f"games {self.games.get(best, 0)}). Anchors: {self.anchor_ids}"
        )
        return best

    def set_shadow(self, policy: PolicyNet) -> None:
        self._register_opponent(SHADOW_ID, policy, "shadow")

    def update_shadow(self, policy: PolicyNet) -> None:
        if self.shadow_id is None:
            self.set_shadow(policy)
            return

        path = self._checkpoint_path(self.shadow_id)
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        shadow_state = checkpoint["model_state_dict"]
        policy_state = policy.state_dict()
        for key, shadow_value in shadow_state.items():
            policy_value = policy_state[key].detach().cpu()
            if torch.is_floating_point(shadow_value):
                shadow_value *= alpha_shadow
                shadow_value += (1 - alpha_shadow) * policy_value
            else:
                shadow_state[key] = policy_value
        torch.save(
            {
                "model_state_dict": shadow_state,
                "runtime_manifest_sha256": runtime_manifest_sha256(),
                "model_config": policy_model_config(policy),
            },
            path,
        )
        self.signatures.pop(self.shadow_id, None)

    def _regular_capacity(self) -> int:
        reserved = len(self.anchor_ids) + int(self.shadow_id is not None)
        return max(1, self.config.pool_size - reserved)

    def _drop_outgrown_anchors(self) -> None:
        threshold = self.config.pool_anchor_drop_wr
        outgrown = [oid for oid in self.anchor_ids if self.win_rates.get(oid, INIT_WR) < threshold]
        for oid in outgrown:
            wr = self.win_rates.get(oid, INIT_WR)
            self._deregister_opponent(oid)
            logging.info(
                f"Dropped outgrown anchor '{oid}' (win rate {wr:.3f} < {threshold}). "
                f"Anchors: {self.anchor_ids}"
            )

    def _pfsp_weight(self, opponent_id: str) -> float:
        # mix of competitive weighting + preferential hard weighting
        wr = max(self.config.pool_wr_floor, self.win_rates[opponent_id])
        base = wr * (1 - wr) + 0.3 * wr
        # up-weight under-sampled opponents so their win rate (and thus anchor
        # eligibility) converges faster; decays to ~0 as games accumulate
        explore = self.config.pool_explore_coef / (1 + 0.2 * self.games.get(opponent_id, 0))
        return base + explore

    def set_reference_batch(self, batch: dict[str, torch.Tensor]) -> None:
        """Install a fresh reference batch and recompute every signature against
        it. A new batch makes old signatures incomparable so all are rebuilt."""
        self.reference_batch = {k: v.cpu() for k, v in batch.items()}
        torch.save(
            {
                "runtime_manifest_sha256": runtime_manifest_sha256(),
                "batch": self.reference_batch,
            },
            self.pool_dir / "reference_batch.pt",
        )

        self.signatures.clear()
        for oid in self.active_ids():
            if oid == self.shadow_id:
                continue
            try:
                sig = self._compute_signature(self.load_policy(oid, "cpu"))
                if sig is not None:
                    self.signatures[oid] = sig
            except Exception as e:
                logging.warning(f"Failed to compute signature for {oid}: {e}")

        self.save_state()

    def load_reference_batch(self) -> None:
        path = self.pool_dir / "reference_batch.pt"
        if path.exists():
            artifact = torch.load(path, weights_only=True, map_location="cpu")
            validate_artifact_manifest_reference(artifact)
            self.reference_batch = artifact["batch"]

    def _compute_signature(self, policy: PolicyNet) -> torch.Tensor | None:
        if self.reference_batch is None:
            return None
        device = policy.device
        obs_tensors = {
            k: v.to(device)
            for k, v in self.reference_batch.items()
            if k not in ("action_masks", "states")
        }

        # the hidden state does matter when it comes to decision
        # making, and there is the slight chance that when you do
        # end up replacing the actual hidden state, it might have
        # wildly different responses, but calculating the actual history
        # for each of these policies would be time consuming, so this is
        # a cheap approximation
        obs = StructuredObservation(**obs_tensors)
        mask = self.reference_batch["action_masks"].to(device)
        B = mask.size(0)
        state = policy.initial_state(B)
        dummy_actions = torch.zeros((B, 2), dtype=torch.long, device=device)

        with torch.inference_mode():
            out = policy.evaluate_obs(obs, mask, dummy_actions, state)

        sig = out.logits.softmax(dim=-1).cpu()
        return sig  # (B, 2, ACT_SIZE)

    def _min_anchor_distance(self, oid: str) -> float:
        if not self.anchor_ids or oid not in self.signatures:
            return 0.0

        sig_c = self.signatures[oid]
        min_d = float("inf")
        for a in self.anchor_ids:
            if a in self.signatures:
                min_d = min(min_d, _js_divergence(sig_c, self.signatures[a]))

        return min_d if min_d != float("inf") else 0.0

    def _checkpoint_path(self, opponent_id: str) -> Path:
        return self.pool_dir / f"{opponent_id}.pt"

    def _register_opponent(self, opponent_id: str, policy: PolicyNet, role: str) -> None:
        """Saves the policy to disk, computes its signature, and tracks its state."""
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "runtime_manifest_sha256": runtime_manifest_sha256(),
                "model_config": policy_model_config(policy),
            },
            self._checkpoint_path(opponent_id),
        )

        if role != "shadow":
            sig = self._compute_signature(policy)
            if sig is not None:
                self.signatures[opponent_id] = sig

        if role == "anchor":
            self.anchor_ids.append(opponent_id)
        elif role == "regular":
            self.regular_ids.append(opponent_id)
        elif role == "shadow":
            self.shadow_id = opponent_id
        else:
            raise ValueError(f"Unknown role: {role}")

        self.win_rates.setdefault(opponent_id, INIT_WR)
        self.games.setdefault(opponent_id, 0)

    def _deregister_opponent(self, opponent_id: str) -> None:
        """Removes an opponent from disk and all tracking structures."""
        self._checkpoint_path(opponent_id).unlink(missing_ok=True)
        self.win_rates.pop(opponent_id, None)
        self.games.pop(opponent_id, None)
        self.signatures.pop(opponent_id, None)

        if opponent_id in self.regular_ids:
            self.regular_ids.remove(opponent_id)
        elif opponent_id in self.anchor_ids:
            self.anchor_ids.remove(opponent_id)
        elif self.shadow_id == opponent_id:
            self.shadow_id = None

    def save_state(self) -> None:
        state = {
            "runtime_manifest_sha256": runtime_manifest_sha256(),
            "shadow_id": self.shadow_id,
            "anchor_ids": self.anchor_ids,
            "regular_ids": self.regular_ids,
            "win_rates": self.win_rates,
            "games": self.games,
            "snapshots_since_anchor": self.snapshots_since_anchor,
        }
        with open(self.pool_dir / "pool_state.json", "w") as f:
            json.dump(state, f, indent=2)
        torch.save(
            {
                "runtime_manifest_sha256": runtime_manifest_sha256(),
                "signatures": self.signatures,
            },
            self.pool_dir / "pool_signatures.pt",
        )

    def _load_state(self) -> None:
        path = self.pool_dir / "pool_state.json"
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            validate_artifact_manifest_reference(state)
            self.shadow_id = state.get("shadow_id")
            self.anchor_ids = state.get("anchor_ids", [])
            self.regular_ids = state.get("regular_ids", [])
            self.win_rates = state.get("win_rates", {})
            self.games = state.get("games", {})
            self.snapshots_since_anchor = state.get("snapshots_since_anchor", 0)
        else:
            self._load_from_checkpoints()

        self.load_reference_batch()
        path_sig = self.pool_dir / "pool_signatures.pt"
        if path_sig.exists():
            artifact = torch.load(path_sig, weights_only=True, map_location="cpu")
            validate_artifact_manifest_reference(artifact)
            self.signatures = artifact["signatures"]

        self._prune_missing_files()

    def _load_from_checkpoints(self) -> None:
        existing_files = sorted(
            (
                path
                for path in self.pool_dir.glob("*.pt")
                if path.name not in {"reference_batch.pt", "pool_signatures.pt"}
            ),
            key=lambda p: p.stat().st_ctime,
        )
        for pt_file in existing_files:
            opponent_id = pt_file.stem
            if opponent_id == SHADOW_ID:
                self.shadow_id = SHADOW_ID
            elif opponent_id.startswith("seed"):
                self.anchor_ids.append(opponent_id)
            else:
                self.regular_ids.append(opponent_id)
            self.win_rates.setdefault(opponent_id, INIT_WR)
            self.games.setdefault(opponent_id, 0)

    def _prune_missing_files(self) -> None:
        existing_ids = {pt.stem for pt in self.pool_dir.glob("*.pt")}
        if self.shadow_id not in existing_ids:
            self.shadow_id = None
        self.anchor_ids = [oid for oid in self.anchor_ids if oid in existing_ids]
        self.regular_ids = [oid for oid in self.regular_ids if oid in existing_ids]
        self.win_rates = {oid: wr for oid, wr in self.win_rates.items() if oid in self.active_ids()}
        self.games = {oid: g for oid, g in self.games.items() if oid in self.active_ids()}
        self.signatures = {
            oid: sig for oid, sig in self.signatures.items() if oid in self.active_ids()
        }
        for opponent_id in self.active_ids():
            self.win_rates.setdefault(opponent_id, INIT_WR)
            self.games.setdefault(opponent_id, 0)

    @classmethod
    def load_or_create(cls, pool_dir: Path, config: PoolConfig) -> Self:
        """Load an existing pool from disk, or create an empty one."""
        pool = cls(pool_dir, config)
        pool._load_state()
        return pool
