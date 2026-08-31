"""Harness for policy evaluation against seen and unseen team sets."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import RandomPlayer

from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import PolicyNet
from p0.rl_player import RLPlayer, TeamPlayerMixin
from p0.runtime import poke_env_patches
from p0.teams.corpus import CorpusSplit
from p0.teams.corpus_source import CorpusTeamSource
from p0.teams.factory import build_team_source
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


def wilson_score_interval(wins: int, total: int) -> tuple[float, float]:
    """
    Calculate the Wilson 95% score interval for a binomial proportion.

    Arguments:
      wins: Number of successes (wins)
      total: Total number of trials (games)

    Returns:
      A tuple (lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    z = 1.96

    denominator = 1 + (z**2) / total
    center = p + (z**2) / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)

    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    return max(0.0, lower), min(1.0, upper)


@dataclass(slots=True)
class _EvaluationSeriesState:
    """Track child-game results until one Bo3 series is complete."""

    games_played: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def complete(self) -> bool:
        return self.wins >= 2 or self.losses >= 2 or self.games_played >= 3


class EvalPlayerMixin:
    """Track one result per completed Bo3 series during evaluation."""

    history: list[tuple[str | None, bool]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history = []
        self._evaluation_series: dict[str, _EvaluationSeriesState] = {}
        self._evaluation_resample_team = True
        if getattr(self, "current_team_packed", None) is None:
            raise ValueError("EvalPlayer requires either team_source or a team in kwargs")

    def _should_resample_team_after_battle(self, battle: AbstractBattle) -> bool:
        del battle
        return self._evaluation_resample_team

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if not isinstance(battle, DoubleBattle):
            raise TypeError(f"Evaluation requires DoubleBattle, got {type(battle).__name__}")

        opponent = battle.opponent_username
        if not opponent or not opponent.strip():
            raise ValueError("Evaluation battle has no opponent identity")
        opponent_id = opponent.strip().casefold()
        state = self._evaluation_series.setdefault(opponent_id, _EvaluationSeriesState())
        state.games_played += 1
        if battle.won:
            state.wins += 1
        elif battle.lost:
            state.losses += 1

        self._evaluation_resample_team = state.complete
        if state.complete:
            self.history.append((getattr(self, "current_team_packed", None), state.wins >= 2))
            self._evaluation_series.pop(opponent_id, None)

        super()._battle_finished_callback(battle)  # type: ignore


class EvalPlayer(EvalPlayerMixin, RLPlayer):
    """An RLPlayer subclass that tracks history during evaluation."""


class EvalRandomPlayer(EvalPlayerMixin, TeamPlayerMixin, RandomPlayer):
    """A RandomPlayer subclass that tracks history and teams used during evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        poke_env_patches.install(self.logger)
        setattr(self.ps_client, "_p0_force_open_team_sheet", True)


@dataclass(frozen=True, slots=True)
class MatchupResult:
    policy_a: str
    policy_b: str
    team_category: str
    total_games: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    confidence_interval_a: tuple[float, float]
    per_team_results: Mapping[str, Mapping[str, Any]]
    source_description: Mapping[str, Any] = field(default_factory=dict)
    per_team_a_results: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    per_team_b_results: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "team_category": self.team_category,
            "total_games": self.total_games,
            "wins_a": self.wins_a,
            "wins_b": self.wins_b,
            "ties": self.ties,
            "win_rate_a": self.win_rate_a,
            "confidence_interval_a": list(self.confidence_interval_a),
            "per_team_results": dict(self.per_team_results),
            "source_description": dict(self.source_description),
            "per_team_a_results": dict(self.per_team_a_results),
            "per_team_b_results": dict(self.per_team_b_results),
        }


class EvaluationHarness:
    """Harness to coordinate and run multiple matchups over different team splits."""

    def __init__(
        self,
        *,
        teams_path: Path | None = None,
        format_id: str = FORMAT.bo3_format,
        episodes_per_matchup: int = 20,
        seed: int = 0,
        port: int = 8120,
        smoke_test: bool = False,
    ) -> None:
        if format_id != FORMAT.bo3_format:
            raise ValueError(
                f"EvaluationHarness only supports the Bo3 format {FORMAT.bo3_format!r}; "
                f"got {format_id!r}"
            )
        self.teams_path = teams_path
        self.format_id = format_id
        self.episodes_per_matchup = episodes_per_matchup
        self.seed = seed
        self.port = port
        self.rng = random.Random(seed)
        self.smoke_test = smoke_test
        self.category_metadata: dict[str, Mapping[str, Any]] = {}

    def build_team_sources(self) -> dict[str, TeamSource]:
        """Build team sources for different categories based on the corpus manifest."""
        sources: dict[str, TeamSource] = {}
        categories = {
            "seen": CorpusSplit.TRAIN,
            "validation_unseen_canonical": CorpusSplit.VALIDATION,
            "test_unseen_canonical": CorpusSplit.TEST,
        }
        failures: dict[str, str] = {}
        if self.teams_path is not None:
            logger.info("Loading team splits from team pool: %s", self.teams_path)
            for key, split in categories.items():
                try:
                    source = build_team_source(
                        self.teams_path,
                        split=split,
                        expected_format_id=self.format_id,
                    )
                    if not isinstance(source, CorpusTeamSource):
                        raise ValueError("team pool does not contain a corpus manifest")
                    sources[key] = source
                    self.category_metadata[key] = {
                        **dict(source.describe()),
                        "status": "ready",
                        "fallback": False,
                    }
                except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                    failures[key] = str(exc)

        else:
            failures = {key: "corpus manifest is unavailable" for key in categories}

        if failures and not self.smoke_test:
            raise ValueError(f"Evaluation corpus categories unavailable: {failures}")

        if failures and self.smoke_test:
            logger.info("Using fallback FixedTeamSource for all categories.")
            fallback = FixedTeamSource(DEFAULT_TEST_TEAM)
            for key in categories:
                sources.setdefault(key, fallback)
                self.category_metadata[key] = {
                    **dict(fallback.describe()),
                    "status": "smoke_fallback",
                    "fallback": True,
                    "error": failures.get(key, ""),
                }
        if set(sources) != set(categories):
            raise ValueError("Evaluation did not construct the complete category set")
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
        """
        Run a single matchup between two policies on a specific team source.

        Arguments:
            name_a: Display name for the first policy.
            policy_a: First policy, or None for a random player.
            name_b: Display name for the second policy.
            policy_b: Second policy, or None for a random player.
            team_category: Label used to aggregate the matchup results.
            team_source: Team sampler shared by both players.
            server_configuration: Showdown server connection settings.

        Returns:
            Aggregated win rates and pair/marginal per-team results for the matchup.
        """
        logger.info(
            "Starting matchup: %s vs %s on team category '%s' (%d episodes)",
            name_a,
            name_b,
            team_category,
            self.episodes_per_matchup,
        )

        rng_a = random.Random(self.rng.randint(0, 1_000_000))
        rng_b = random.Random(self.rng.randint(0, 1_000_000))

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

        try:
            for _ in range(self.episodes_per_matchup):
                expected_series_count = len(player_a.history) + 1
                await player_a.battle_against(player_b, n_battles=1)
                await self._wait_for_series_completion(
                    player_a,
                    player_b,
                    expected_series_count,
                )
        finally:
            await player_a.ps_client.stop_listening()
            await player_b.ps_client.stop_listening()

        if len(player_a.history) != len(player_b.history):
            raise RuntimeError("Evaluation players reported different game counts")
        wins_a = sum(1 for _, won in player_a.history if won)
        wins_b = sum(1 for _, won in player_b.history if won)
        total_games = len(player_a.history)
        for (_, won_a), (_, won_b) in zip(player_a.history, player_b.history, strict=True):
            if won_a and won_b:
                raise RuntimeError("Evaluation histories report both players winning a game")
        if wins_a + wins_b > total_games:
            raise RuntimeError("Evaluation win counts exceed the number of games")
        ties = total_games - wins_a - wins_b
        if wins_a + wins_b + ties != total_games:
            raise RuntimeError("Evaluation game outcomes do not reconcile")

        win_rate_a = wins_a / max(1, total_games)
        ci_a = wilson_score_interval(wins_a, total_games)

        per_team: dict[str, dict[str, int]] = {}
        per_team_a: dict[str, dict[str, int]] = {}
        per_team_b: dict[str, dict[str, int]] = {}
        for (team_a, won_a), (team_b, won_b) in zip(
            player_a.history, player_b.history, strict=True
        ):
            if team_a is None or team_b is None:
                continue
            team_key = f"{hashlib_team(team_a)}:{hashlib_team(team_b)}"
            stats = per_team.setdefault(team_key, {"wins": 0, "games": 0})
            stats["games"] += 1
            if won_a:
                stats["wins"] += 1
            stats_a = per_team_a.setdefault(hashlib_team(team_a), {"wins": 0, "games": 0})
            stats_a["games"] += 1
            if won_a:
                stats_a["wins"] += 1
            stats_b = per_team_b.setdefault(hashlib_team(team_b), {"wins": 0, "games": 0})
            stats_b["games"] += 1
            if won_b:
                stats_b["wins"] += 1

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
            ties=ties,
            win_rate_a=win_rate_a,
            confidence_interval_a=ci_a,
            per_team_results=per_team_results,
            source_description=self.category_metadata.get(
                team_category, dict(team_source.describe())
            ),
            per_team_a_results=_finalize_team_results(per_team_a),
            per_team_b_results=_finalize_team_results(per_team_b),
        )

    @staticmethod
    async def _wait_for_series_completion(
        player_a: EvalPlayerMixin,
        player_b: EvalPlayerMixin,
        expected_series_count: int,
        timeout: float = 120.0,
    ) -> None:
        """Wait until both evaluation players report the requested complete series."""
        deadline = asyncio.get_running_loop().time() + timeout
        while (
            len(player_a.history) < expected_series_count
            or len(player_b.history) < expected_series_count
        ):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the live Bo3 series to complete "
                    f"after {timeout:.0f} seconds"
                )
            await asyncio.sleep(0.05)


def hashlib_team(team_packed: str) -> str:
    """Generate a stable short identifier for a team string."""
    return hashlib.sha256(team_packed.encode("utf-8")).hexdigest()[:8]


def _finalize_team_results(
    values: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, Any]]:
    """Add marginal team win rates to integer game/win counters."""
    return {
        team_hash: {
            "wins": int(stats["wins"]),
            "games": int(stats["games"]),
            "win_rate": stats["wins"] / stats["games"],
        }
        for team_hash, stats in values.items()
        if stats["games"] > 0
    }
