"""Evaluation harness for policy and baseline opponents in Pokemon Showdown Bo3."""

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
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from p0.format_config import FORMAT
from p0.model.observation_builder import ObservationBuilder
from p0.model.policy import PolicyNet
from p0.rl_player import RLPlayer, TeamPlayerMixin
from p0.runtime import poke_env_patches
from p0.teams.corpus import CorpusSplit
from p0.teams.factory import build_team_source
from p0.teams.source import TeamSource

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

    @property
    def complete(self) -> bool:
        return self.wins >= 2 or self.losses >= 2 or self.games_played >= 3


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

        self._evaluation_resample_team = state.complete
        if state.complete:
            self.history.append((getattr(self, "current_team_packed", None), state.wins >= 2))
            self._evaluation_series.pop(opponent_id, None)

        super()._battle_finished_callback(battle)  # type: ignore


class EvalPlayer(EvalPlayerMixin, RLPlayer):
    """RLPlayer that tracks series history during evaluation."""


class EvalRandomPlayer(EvalPlayerMixin, TeamPlayerMixin, RandomPlayer):
    """RandomPlayer that tracks series history and teams during evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        poke_env_patches.install(self.logger)
        setattr(self.ps_client, "_p0_force_open_team_sheet", True)


class EvalMaxBasePowerPlayer(EvalPlayerMixin, TeamPlayerMixin, MaxBasePowerPlayer):
    """MaxBasePowerPlayer that tracks series history and teams during evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        poke_env_patches.install(self.logger)
        setattr(self.ps_client, "_p0_force_open_team_sheet", True)


class EvalSimpleHeuristicsPlayer(EvalPlayerMixin, TeamPlayerMixin, SimpleHeuristicsPlayer):
    """SimpleHeuristicsPlayer that tracks series history and teams during evaluation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        poke_env_patches.install(self.logger)
        setattr(self.ps_client, "_p0_force_open_team_sheet", True)


type EvalPlayerType = (
    EvalPlayer | EvalRandomPlayer | EvalMaxBasePowerPlayer | EvalSimpleHeuristicsPlayer
)


def _to_corpus_split(split: CorpusSplit | str | None) -> CorpusSplit:
    if split is None:
        return CorpusSplit.TRAIN
    if isinstance(split, CorpusSplit):
        return split
    normalized = split.strip().lower()
    if normalized in {"train", "seen"}:
        return CorpusSplit.TRAIN
    if normalized in {"val", "validation"}:
        return CorpusSplit.VALIDATION
    if normalized in {"test", "unseen"}:
        return CorpusSplit.TEST
    raise ValueError(f"Unknown corpus split: {split}")


def create_eval_player(
    spec: PolicyNet | str | None,
    *,
    team_rng: random.Random,
    team_source: TeamSource,
    battle_format: str,
    server_configuration: Any,
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
    source_description: Mapping[str, Any] = field(default_factory=dict)
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
            "source_description": dict(self.source_description),
            "per_team_a_results": dict(self.per_team_a_results),
            "per_team_b_results": dict(self.per_team_b_results),
        }


class EvaluationHarness:
    """Coordinates and runs matchups between policies and baseline opponents."""

    def __init__(
        self,
        *,
        teams_path: Path | None = None,
        format_id: str = FORMAT.bo3_format,
        episodes_per_matchup: int = 20,
        seed: int = 0,
        port: int = 8120,
        split: CorpusSplit | str | None = None,
    ) -> None:
        if format_id != FORMAT.bo3_format:
            raise ValueError(
                f"EvaluationHarness only supports {FORMAT.bo3_format!r}; got {format_id!r}"
            )
        self.teams_path = teams_path
        self.format_id = format_id
        self.episodes_per_matchup = episodes_per_matchup
        self.seed = seed
        self.port = port
        self.split = split
        self.rng = random.Random(seed)

    def build_team_source(self, split: CorpusSplit | str | None = None) -> TeamSource:
        """Build team source from the configured teams path."""
        if self.teams_path is None:
            raise ValueError("Evaluation requires teams_path to build a team source")
        if not self.teams_path.exists():
            raise FileNotFoundError(f"Evaluation teams path not found: {self.teams_path}")

        chosen_split = _to_corpus_split(split if split is not None else self.split)
        return build_team_source(
            self.teams_path,
            split=chosen_split,
            expected_format_id=self.format_id,
        )

    def build_team_sources(self) -> dict[str, TeamSource]:
        """Build team sources mapping for evaluation."""
        split_name = (
            self.split.name.lower()
            if isinstance(self.split, CorpusSplit)
            else str(self.split or "default")
        )
        return {split_name: self.build_team_source()}

    async def run_matchup(
        self,
        name_a: str,
        policy_a: PolicyNet | str | None,
        name_b: str,
        policy_b: PolicyNet | str | None,
        team_category: str | TeamSource = "default",
        team_source: TeamSource | Any = None,
        server_configuration: Any = None,
    ) -> MatchupResult:
        """Run a matchup between two players on the given team source."""
        source: Any
        if hasattr(team_category, "sample"):
            server_config = team_source
            source = team_category
            category = "default"
        else:
            category = str(team_category)
            source = team_source
            server_config = server_configuration

        if not hasattr(source, "sample"):
            raise TypeError(
                f"Expected TeamSource with .sample() method, got {type(source).__name__}"
            )

        logger.info(
            "Starting matchup: %s vs %s (%d episodes)",
            name_a,
            name_b,
            self.episodes_per_matchup,
        )
        rng_a = random.Random(self.rng.randint(0, 1_000_000))
        rng_b = random.Random(self.rng.randint(0, 1_000_000))

        account_config_a = AccountConfiguration(f"evala{self.rng.randint(1000, 9999)}", None)
        player_a = create_eval_player(
            policy_a,
            team_rng=rng_a,
            team_source=source,
            battle_format=self.format_id,
            server_configuration=server_config,
            account_configuration=account_config_a,
            max_concurrent_battles=1,
        )

        account_config_b = AccountConfiguration(f"evalb{self.rng.randint(1000, 9999)}", None)
        player_b = create_eval_player(
            policy_b,
            team_rng=rng_b,
            team_source=source,
            battle_format=self.format_id,
            server_configuration=server_config,
            account_configuration=account_config_b,
            max_concurrent_battles=1,
        )

        try:
            for _ in range(self.episodes_per_matchup):
                expected = len(player_a.history) + 1
                await player_a.battle_against(player_b, n_battles=1)
                await self._wait_for_series_completion(player_a, player_b, expected)
        finally:
            await player_a.ps_client.stop_listening()
            await player_b.ps_client.stop_listening()

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

        def _record(store: dict[str, dict[str, Any]], key: str, won: bool) -> None:
            stats = store.setdefault(key, {"wins": 0, "games": 0, "win_rate": 0.0})
            stats["games"] += 1
            if won:
                stats["wins"] += 1
            stats["win_rate"] = stats["wins"] / stats["games"]

        for (team_a, won_a), (team_b, won_b) in zip(
            player_a.history, player_b.history, strict=True
        ):
            if team_a and team_b:
                ha = hashlib.sha256(team_a.encode()).hexdigest()[:8]
                hb = hashlib.sha256(team_b.encode()).hexdigest()[:8]
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
            team_category=category,
            per_team_results=per_team,
            source_description=dict(source.describe()),
            per_team_a_results=per_team_a,
            per_team_b_results=per_team_b,
        )

    @staticmethod
    async def _wait_for_series_completion(
        player_a: EvalPlayerType,
        player_b: EvalPlayerType,
        expected_series_count: int,
        timeout: float = 120.0,
    ) -> None:
        """Wait until both players finish the requested series count."""
        deadline = asyncio.get_running_loop().time() + timeout
        while (
            len(player_a.history) < expected_series_count
            or len(player_b.history) < expected_series_count
        ):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"Timed out waiting for series to complete after {timeout:.0f}s")
            await asyncio.sleep(0.05)
