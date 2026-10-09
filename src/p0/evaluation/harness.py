"""Evaluation harness for policy and baseline opponents in Pokemon Showdown Bo3."""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import PolicyNet
from p0.rl_player import RLPlayer, TeamPlayerMixin
from p0.runtime import poke_env_patches
from p0.teams.corpus import TeamCorpus, load_team_corpus

logger = logging.getLogger(__name__)


def wilson_score_interval(wins: int, total: int) -> tuple[float, float]:
    """Calculate the Wilson 95% score confidence interval."""
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
    """Track child game results for one Bo3 series."""

    games_played: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def complete(self) -> bool:
        threshold = (3 - self.ties) // 2 + 1
        return self.wins >= threshold or self.losses >= threshold or self.games_played >= 3


class EvalPlayerMixin:
    """Track results per completed Bo3 series during evaluation."""

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
        else:
            state.ties += 1

        self._evaluation_resample_team = state.complete
        if state.complete:
            self._evaluation_series.pop(opponent_id, None)

        super()._battle_finished_callback(battle)  # type: ignore


class EvalPlayer(EvalPlayerMixin, RLPlayer):
    """RLPlayer that tracks series history during evaluation."""


class _BaselineClientMixin:
    """Install the p0 poke-env patches on a baseline player's Bo3 client."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The Player base class later in the MRO supplies the logger.
        poke_env_patches.install(self.logger)  # type: ignore[attr-defined]
        poke_env_patches.enable_forced_open_team_sheet(self)


class EvalRandomPlayer(_BaselineClientMixin, EvalPlayerMixin, TeamPlayerMixin, RandomPlayer):
    """RandomPlayer that tracks series history and teams during evaluation."""


class EvalMaxBasePowerPlayer(
    _BaselineClientMixin, EvalPlayerMixin, TeamPlayerMixin, MaxBasePowerPlayer
):
    """MaxBasePowerPlayer that tracks series history and teams during evaluation."""


class EvalSimpleHeuristicsPlayer(
    _BaselineClientMixin, EvalPlayerMixin, TeamPlayerMixin, SimpleHeuristicsPlayer
):
    """SimpleHeuristicsPlayer that tracks series history and teams during evaluation."""


type EvalPlayerType = (
    EvalPlayer | EvalRandomPlayer | EvalMaxBasePowerPlayer | EvalSimpleHeuristicsPlayer
)


def create_eval_player(
    spec: PolicyNet | str | None,
    *,
    team_rng: random.Random,
    team_source: TeamCorpus,
    battle_format: str,
    server_configuration: ServerConfiguration,
    account_configuration: AccountConfiguration,
    max_concurrent_battles: int = 1,
    **kwargs: Any,
) -> EvalPlayerType:
    """Create an evaluation player from a policy network or opponent name."""
    if isinstance(spec, PolicyNet):
        return EvalPlayer(
            policy=spec,
            observation_builder=ObservationBuilder(spec.resources),
            team_rng=team_rng,
            team_source=team_source,
            battle_format=battle_format,
            server_configuration=server_configuration,
            account_configuration=account_configuration,
            max_concurrent_battles=max_concurrent_battles,
            **kwargs,
        )

    opponent_type = (spec or "random").lower().strip()
    if opponent_type == "random":
        player_cls = EvalRandomPlayer
    elif opponent_type in {"max_power", "max_base_power"}:
        player_cls = EvalMaxBasePowerPlayer
    elif opponent_type in {"simple_heuristics", "heuristic", "heuristics"}:
        player_cls = EvalSimpleHeuristicsPlayer
    else:
        raise ValueError(
            f"Unknown evaluation opponent type {spec!r}; "
            "supported: 'random', 'max_power', 'simple_heuristics', or a PolicyNet checkpoint."
        )

    return player_cls(
        team_rng=team_rng,
        team_source=team_source,
        battle_format=battle_format,
        server_configuration=server_configuration,
        account_configuration=account_configuration,
        max_concurrent_battles=max_concurrent_battles,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class MatchupResult:
    """Summary of games and win rates from an evaluation matchup."""

    policy_a: str
    policy_b: str
    total_games: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    confidence_interval_a: tuple[float, float]
    team_category: str = "default"
    per_team_results: Mapping[str, Any] = field(default_factory=dict)
    per_team_a_results: Mapping[str, Any] = field(default_factory=dict)
    per_team_b_results: Mapping[str, Any] = field(default_factory=dict)

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
            "per_team_a_results": dict(self.per_team_a_results),
            "per_team_b_results": dict(self.per_team_b_results),
        }


def _record(store: dict[str, dict[str, Any]], key: str, won: bool) -> None:
    """Count one series result for a team key and refresh its win rate."""
    stats = store.setdefault(key, {"wins": 0, "games": 0, "win_rate": 0.0})
    stats["games"] += 1
    if won:
        stats["wins"] += 1
    stats["win_rate"] = stats["wins"] / stats["games"]


class EvaluationHarness:
    """Coordinates and runs matchups between policies and baseline opponents."""

    def __init__(
        self,
        *,
        teams_path: Path | None = None,
        episodes_per_matchup: int = 20,
        seed: int = 0,
    ) -> None:
        self.teams_path = teams_path
        self.format_id = FORMAT.bo3_format
        self.episodes_per_matchup = episodes_per_matchup
        self.rng = random.Random(seed)

    def build_team_source(self) -> TeamCorpus:
        """Load the team corpus from the configured teams path."""
        if self.teams_path is None:
            raise ValueError("Evaluation requires teams_path to build a team source")

        return load_team_corpus(self.teams_path, self.format_id)

    async def run_matchup(
        self,
        name_a: str,
        policy_a: PolicyNet | str | None,
        name_b: str,
        policy_b: PolicyNet | str | None,
        *,
        team_source: TeamCorpus,
        server_configuration: ServerConfiguration,
        team_category: str = "default",
    ) -> MatchupResult:
        """
        Run a matchup between two players on the given team source.

        Plays episodes_per_matchup Bo3 series one after another, one battle
        at a time, and scores each series by the parent-series winner that
        both clients receive. Counts and win rates are per series, not per
        game. Team seeds come from the evaluator RNG, so a fixed evaluator
        seed repeats the team draws. Both player connections are closed on
        exit, including on failure.

        Raises ValueError when a policy name is not a supported built-in
        opponent. Raises RuntimeError when the two clients report different
        series results or counts, or the winner is neither player. Raises
        TimeoutError when a series result does not arrive within 120 seconds.

        Arguments:
            name_a: Label for player A in the result.
            policy_a: Policy network for player A, or a built-in opponent name
                ("random", "max_power", "simple_heuristics"); None means "random".
            name_b: Label for player B in the result.
            policy_b: Policy network or built-in opponent name for player B.
            team_source: Source both players sample their teams from.
            server_configuration: Showdown server both players connect to.
            team_category: Label stored in the result for this team source.

        Returns:
            MatchupResult with series counts, ties, player A's win rate and
            Wilson confidence interval, and win counts keyed by the first 8 hex
            characters of each packed team's SHA-256, for team pairs and for
            each side's teams.
        """
        logger.info(
            "Starting matchup: %s vs %s (%d episodes)",
            name_a,
            name_b,
            self.episodes_per_matchup,
        )
        rng_a = random.Random(self.rng.randint(0, 1_000_000))
        rng_b = random.Random(self.rng.randint(0, 1_000_000))

        async with AsyncExitStack() as stack:
            players: list[EvalPlayerType] = []
            for label, policy, team_rng in (("a", policy_a, rng_a), ("b", policy_b, rng_b)):
                player = create_eval_player(
                    policy,
                    team_rng=team_rng,
                    team_source=team_source,
                    battle_format=self.format_id,
                    server_configuration=server_configuration,
                    account_configuration=AccountConfiguration(
                        f"eval{label}{self.rng.randint(1000, 9999)}", None
                    ),
                    max_concurrent_battles=1,
                )
                stack.push_async_callback(player.ps_client.stop_listening)
                players.append(player)
            player_a, player_b = players

            for _ in range(self.episodes_per_matchup):
                expected = len(player_a.history) + 1
                team_a = player_a.current_team_packed
                team_b = player_b.current_team_packed
                await player_a.battle_against(player_b, n_battles=1)
                winner = await self._wait_for_series_completion(player_a, player_b, expected)
                player_a.history.append((team_a, winner == player_a.username.casefold()))
                player_b.history.append((team_b, winner == player_b.username.casefold()))

        if len(player_a.history) != len(player_b.history):
            raise RuntimeError("Evaluation players reported different series counts")

        total_games = len(player_a.history)
        wins_a = sum(1 for _, won in player_a.history if won)
        wins_b = sum(1 for _, won in player_b.history if won)
        ties = total_games - wins_a - wins_b

        win_rate_a = wins_a / max(1, total_games)
        ci_a = wilson_score_interval(wins_a, total_games)

        per_team: dict[str, dict[str, Any]] = {}
        per_team_a: dict[str, dict[str, Any]] = {}
        per_team_b: dict[str, dict[str, Any]] = {}
        for (team_a, won_a), (team_b, won_b) in zip(
            player_a.history, player_b.history, strict=True
        ):
            if team_a and team_b:
                ha = team_a
                hb = team_b
                _record(per_team, f"{ha}:{hb}", won_a)
                _record(per_team_a, ha, won_a)
                _record(per_team_b, hb, won_b)

        return MatchupResult(
            policy_a=name_a,
            policy_b=name_b,
            total_games=total_games,
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            win_rate_a=win_rate_a,
            confidence_interval_a=ci_a,
            team_category=team_category,
            per_team_results=per_team,
            per_team_a_results=per_team_a,
            per_team_b_results=per_team_b,
        )

    @staticmethod
    async def _wait_for_series_completion(
        player_a: EvalPlayerType,
        player_b: EvalPlayerType,
        expected_series_count: int,
        timeout: float = 120.0,
    ) -> str | None:
        """Wait for both clients to receive the same parent-series result."""
        results = await asyncio.wait_for(
            asyncio.gather(
                poke_env_patches.wait_for_parent_result(player_a.ps_client, expected_series_count),
                poke_env_patches.wait_for_parent_result(player_b.ps_client, expected_series_count),
            ),
            timeout=timeout,
        )
        if results[0] != results[1]:
            raise RuntimeError(f"Evaluation clients received different parent results: {results}")
        _, winner = results[0]
        if winner is not None:
            winner = winner.casefold()
            if winner not in {player_a.username.casefold(), player_b.username.casefold()}:
                raise RuntimeError(f"Unexpected parent-series winner {winner!r}")
        return winner
