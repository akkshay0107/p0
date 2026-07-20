"""Harness for policy evaluation against seen, unseen, and archetype team sets."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle
from poke_env.player import RandomPlayer
from poke_env.teambuilder import Teambuilder

from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import PolicyNet
from p0.rl_player import RLPlayer
from p0.runtime import poke_env_patches
from p0.teams.corpus import CorpusSourceSpec, CorpusSplit, SamplingPolicy
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.source import FixedTeamSource, TeamSource

# Default Pikachu/Charizard test team used as fallback when no corpus is available
DEFAULT_TEST_TEAM = """
Pikachu @ Light Ball
Ability: Static
Level: 50
Jolly Nature
- Fake Out
- Protect
- Thunderbolt
- Electroweb

Charizard @ Charizardite Y
Ability: Blaze
Level: 50
Modest Nature
- Heat Wave
- Solar Beam
- Protect
- Weather Ball

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
Timid Nature
- Moonblast
- Tailwind
- Encore
- Protect

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Protect
- Low Kick

Glimmora @ Shuca Berry
Ability: Corrosion
Level: 50
Modest Nature
- Power Gem
- Sludge Bomb
- Earth Power
- Protect
"""


logger = logging.getLogger(__name__)


def wilson_score_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the Wilson score interval for a binomial proportion.

    Arguments:
      wins: Number of successes (wins)
      total: Total number of trials (games)
      confidence: Confidence level (e.g. 0.95)

    Returns:
      A tuple (lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    # For 95% confidence, z = 1.96
    z = 1.96

    denominator = 1 + (z**2) / total
    center = p + (z**2) / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)

    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    return max(0.0, lower), min(1.0, upper)


class EvalPlayer(RLPlayer):
    """An RLPlayer subclass that tracks the history and teams used during evaluation."""

    current_team_packed: str | None
    history: list[tuple[str | None, bool]]

    def __init__(
        self,
        policy: PolicyNet,
        *args: Any,
        team_rng: random.Random,
        team_source: TeamSource | None = None,
        **kwargs: Any,
    ) -> None:
        if team_source is not None:
            if "team" in kwargs:
                raise ValueError("Pass either team or team_source, not both")
            initial_team = team_source.sample(team_rng).packed
            kwargs["team"] = initial_team
            self.current_team_packed = initial_team
        else:
            self.current_team_packed = kwargs.get("team")

        self.history = []
        super().__init__(
            policy,
            *args,
            team_rng=team_rng,
            team_source=team_source,
            **kwargs,
        )

    def update_team(self, team: str | Teambuilder) -> None:
        super().update_team(team)
        if isinstance(team, str):
            self.current_team_packed = team

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        won = battle.won if battle.won is not None else False
        self.history.append((self.current_team_packed, won))
        self.state = None
        if self.team_source is not None:
            self.update_team(self.team_source.sample(self.team_rng).packed)


class EvalRandomPlayer(RandomPlayer):
    """A RandomPlayer subclass that tracks history and teams used during evaluation."""

    current_team_packed: str | None
    history: list[tuple[str | None, bool]]

    def __init__(
        self,
        *args: Any,
        team_rng: random.Random,
        team_source: TeamSource | None = None,
        **kwargs: Any,
    ) -> None:
        self.team_source = team_source
        self.team_rng = team_rng
        if team_source is not None:
            if "team" in kwargs:
                raise ValueError("Pass either team or team_source, not both")
            initial_team = team_source.sample(team_rng).packed
            kwargs["team"] = initial_team
            self.current_team_packed = initial_team
        else:
            self.current_team_packed = kwargs.get("team")

        self.history = []
        super().__init__(*args, **kwargs)
        poke_env_patches.install(self.logger)

    def update_team(self, team: str | Teambuilder) -> None:
        super().update_team(team)
        if isinstance(team, str):
            self.current_team_packed = team

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        won = battle.won if battle.won is not None else False
        self.history.append((self.current_team_packed, won))
        if self.team_source is not None:
            self.update_team(self.team_source.sample(self.team_rng).packed)


@dataclass(frozen=True, slots=True)
class MatchupResult:
    policy_a: str
    policy_b: str
    team_category: str
    total_games: int
    wins_a: int
    wins_b: int
    win_rate_a: float
    confidence_interval_a: tuple[float, float]
    per_team_results: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "team_category": self.team_category,
            "total_games": self.total_games,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "win_rate_a": self.win_rate_a,
            "confidence_interval_a": list(self.confidence_interval_a),
            "per_team_results": dict(self.per_team_results),
        }


class EvaluationHarness:
    """Harness to coordinate and run multiple matchups over different team splits."""

    def __init__(
        self,
        *,
        corpus_path: Path | None = None,
        corpus_hash: str = "",
        format_id: str = FORMAT.battle_format,
        episodes_per_matchup: int = 20,
        seed: int = 0,
        port: int = 8120,
    ) -> None:
        self.corpus_path = corpus_path
        self.corpus_hash = corpus_hash
        self.format_id = format_id
        self.episodes_per_matchup = episodes_per_matchup
        self.seed = seed
        self.port = port
        self.rng = random.Random(seed)

    def _build_team_sources(self) -> dict[str, TeamSource]:
        """Build team sources for different categories based on the corpus manifest."""
        sources: dict[str, TeamSource] = {}
        if self.corpus_path is not None and self.corpus_path.exists() and self.corpus_hash:
            logger.info("Loading team splits from corpus manifest: %s", self.corpus_path)
            splits = {
                "seen": (CorpusSplit.TRAIN, SamplingPolicy.USAGE_WEIGHTED),
                "unseen_canonical": (CorpusSplit.VALIDATION, SamplingPolicy.UNIFORM_CANONICAL),
                "unseen_archetypes": (
                    CorpusSplit.HELD_OUT_ARCHETYPE,
                    SamplingPolicy.UNIFORM_ARCHETYPE,
                ),
                "rare_species": (CorpusSplit.TRAIN, SamplingPolicy.RARE_COVERAGE),
            }
            for key, (split, policy) in splits.items():
                spec = CorpusSourceSpec(
                    corpus_path=str(self.corpus_path),
                    corpus_hash=self.corpus_hash,
                    format_id=self.format_id,
                    split=split,
                    seed=self.rng.randint(0, 1_000_000),
                    sampling_policy=policy,
                )
                try:
                    sources[key] = CorpusTeamSource(spec)
                except Exception as exc:
                    logger.warning(
                        "Could not build CorpusTeamSource for split %s: %s", split.name, exc
                    )

        if not sources:
            logger.info("Using fallback FixedTeamSource for all categories.")
            fallback = FixedTeamSource(DEFAULT_TEST_TEAM)
            sources = {
                "seen": fallback,
                "unseen_canonical": fallback,
                "unseen_archetypes": fallback,
                "rare_species": fallback,
            }
        return sources

    async def run_matchup(
        self,
        name_a: str,
        policy_a: PolicyNet | None,
        name_b: str,
        policy_b: PolicyNet | None,
        team_category: str,
        team_source: TeamSource,
        server_configuration: Any,
    ) -> MatchupResult:
        """Run a single matchup between two policies on a specific team source."""
        logger.info(
            "Starting matchup: %s vs %s on team category '%s' (%d episodes)",
            name_a,
            name_b,
            team_category,
            self.episodes_per_matchup,
        )

        # Build players
        rng_a = random.Random(self.rng.randint(0, 1_000_000))
        rng_b = random.Random(self.rng.randint(0, 1_000_000))

        # Policy A
        account_config_a = AccountConfiguration(f"evala{self.rng.randint(1000, 9999)}", None)
        if policy_a is not None:
            player_a = EvalPlayer(
                policy=policy_a,
                observation_builder=ObservationBuilder(policy_a.resources),
                team_rng=rng_a,
                team_source=team_source,
                battle_format=self.format_id,
                server_configuration=server_configuration,
                account_configuration=account_config_a,
                max_concurrent_battles=1,
            )
        else:
            player_a = EvalRandomPlayer(
                team_rng=rng_a,
                team_source=team_source,
                battle_format=self.format_id,
                server_configuration=server_configuration,
                account_configuration=account_config_a,
                max_concurrent_battles=1,
            )

        # Policy B
        account_config_b = AccountConfiguration(f"evalb{self.rng.randint(1000, 9999)}", None)
        if policy_b is not None:
            player_b = EvalPlayer(
                policy=policy_b,
                observation_builder=ObservationBuilder(policy_b.resources),
                team_rng=rng_b,
                team_source=team_source,
                battle_format=self.format_id,
                server_configuration=server_configuration,
                account_configuration=account_config_b,
                max_concurrent_battles=1,
            )
        else:
            player_b = EvalRandomPlayer(
                team_rng=rng_b,
                team_source=team_source,
                battle_format=self.format_id,
                server_configuration=server_configuration,
                account_configuration=account_config_b,
                max_concurrent_battles=1,
            )

        # Run battles
        try:
            await player_a.battle_against(player_b, n_battles=self.episodes_per_matchup)
        finally:
            await player_a.ps_client.stop_listening()
            await player_b.ps_client.stop_listening()

        # Compile results
        wins_a = sum(1 for _, won in player_a.history if won)
        wins_b = sum(1 for _, won in player_b.history if won)
        total_games = len(player_a.history)

        win_rate_a = wins_a / max(1, total_games)
        ci_a = wilson_score_interval(wins_a, total_games)

        # Per team stats (using team string hashes or raw values)
        per_team: dict[str, dict[str, int]] = {}
        for team, won in player_a.history:
            if team is None:
                continue
            # Use short summary or species listing for readability if possible
            team_key = hashlib_team(team)
            stats = per_team.setdefault(team_key, {"wins": 0, "games": 0})
            stats["games"] += 1
            if won:
                stats["wins"] += 1

        per_team_results: dict[str, dict[str, Any]] = {}
        for team_key, stats in per_team.items():
            wins = stats["wins"]
            games = stats["games"]
            per_team_results[team_key] = {
                "wins": wins,
                "games": games,
                "win_rate": wins / games,
            }

        return MatchupResult(
            policy_a=name_a,
            policy_b=name_b,
            team_category=team_category,
            total_games=total_games,
            wins_a=wins_a,
            wins_b=wins_b,
            win_rate_a=win_rate_a,
            confidence_interval_a=ci_a,
            per_team_results=per_team_results,
        )


def hashlib_team(team_packed: str) -> str:
    """Generate a stable short identifier for a team string."""
    import hashlib

    return hashlib.sha256(team_packed.encode("utf-8")).hexdigest()[:8]
