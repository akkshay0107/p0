import asyncio
import logging
from pathlib import Path

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from src.heuristic.heuristic import FuzzyHeuristic
from src.model.policy import PolicyNet
from src.rl_player import RLPlayer
from src.team_picker import RandomTeamFromPool
from src.train.config import PROJECT_ROOT

DEFAULT_BATTLE_FORMAT = "gen9championsvgc2026regma"

# the fixed, absolute yardstick the policy is benchmarked against
HEURISTIC_OPPONENTS: dict[str, type[Player]] = {
    "Random": RandomPlayer,
    "MaxBP": MaxBasePowerPlayer,
    "SimpleH": SimpleHeuristicsPlayer,
    "FuzzyH": FuzzyHeuristic,
}


def _server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )


def load_team(teams_dir: Path | str | None = None) -> RandomTeamFromPool:
    teams_dir = Path(teams_dir) if teams_dir is not None else PROJECT_ROOT / "teams"
    team_files = [
        path.read_text(encoding="utf-8")
        for path in teams_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    ]
    if not team_files:
        raise FileNotFoundError(f"No team files found in {teams_dir}.")
    return RandomTeamFromPool(team_files)


async def evaluate_against_heuristics(
    policy: PolicyNet,
    *,
    team: RandomTeamFromPool,
    port: int,
    n_battles: int,
    battle_format: str = DEFAULT_BATTLE_FORMAT,
    max_concurrent_battles: int = 10,
    log_level: int = logging.WARNING,
) -> dict[str, float]:
    """Play `policy` against each of the fixed heuristics for `n_battles` games
    on the showdown server at `port`.

    Returns per-opponent win rates, all in [0, 1].

    `log_level` is forwarded to the players to silence poke_env's per-message
    websocket logging (which otherwise floods the root logger at INFO).
    """
    policy.eval()
    server_config = _server_config(port)

    def make(account: str, cls: type[Player], **kwargs) -> Player:
        return cls(
            account_configuration=AccountConfiguration(account, None),
            battle_format=battle_format,
            server_configuration=server_config,
            max_concurrent_battles=max_concurrent_battles,
            team=team,
            accept_open_team_sheet=True,
            log_level=log_level,
            **kwargs,
        )

    rl_player = make("Eval_RL", RLPlayer, policy=policy)

    results: dict[str, float] = {}
    try:
        for name, cls in HEURISTIC_OPPONENTS.items():
            opponent = make(f"Eval_{name}", cls)
            await rl_player.battle_against(opponent, n_battles=n_battles)
            results[name] = rl_player.win_rate
            rl_player.reset_battles()
            opponent.reset_battles()
            await opponent.ps_client.stop_listening()
            # prevent a bunch of warnings from poke-env
            # for cancelling pending tasks
            opponent.ps_client.logger.setLevel(100)
            for task in list(opponent.ps_client._active_tasks):
                task.cancel()
    finally:
        await rl_player.ps_client.stop_listening()
        rl_player.ps_client.logger.setLevel(100)
        for task in list(rl_player.ps_client._active_tasks):
            task.cancel()

    return results


def run_evaluation(
    policy: PolicyNet,
    *,
    port: int,
    n_battles: int,
    team: RandomTeamFromPool | None = None,
    teams_dir: Path | str | None = None,
    battle_format: str = DEFAULT_BATTLE_FORMAT,
    max_concurrent_battles: int = 10,
    log_level: int = logging.WARNING,
) -> dict[str, float]:
    """Synchronous wrapper around `evaluate_against_heuristics`."""
    if team is None:
        team = load_team(teams_dir)
    return asyncio.run(
        evaluate_against_heuristics(
            policy,
            team=team,
            port=port,
            n_battles=n_battles,
            battle_format=battle_format,
            max_concurrent_battles=max_concurrent_battles,
            log_level=log_level,
        )
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    *,
    port: int,
    n_battles: int,
    teams_dir: Path | str | None = None,
    battle_format: str = DEFAULT_BATTLE_FORMAT,
) -> dict[str, float]:
    """Load a policy checkpoint onto CPU and benchmark it against the heuristics.

    Built to run in a separate process so this CPU-bound eval overlaps GPU
    training. Returns per-opponent win rates in [0, 1].
    """
    import torch

    from src.lookups import ACT_SIZE, OBS_DIM

    policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return run_evaluation(
        policy,
        port=port,
        n_battles=n_battles,
        teams_dir=teams_dir,
        battle_format=battle_format,
    )
