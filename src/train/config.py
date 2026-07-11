"""Typed, immutable application configuration loaded from YAML."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _positive_ints(config: Any, *names: str) -> None:
    for name in names:
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{config.__class__.__name__}.{name} must be a positive integer")


def _positive(config: Any, *names: str) -> None:
    for name in names:
        value = getattr(config, name)
        if value <= 0:
            raise ValueError(f"{config.__class__.__name__}.{name} must be greater than zero")


def _non_negative(config: Any, *names: str) -> None:
    for name in names:
        if getattr(config, name) < 0:
            raise ValueError(f"{config.__class__.__name__}.{name} must not be negative")


def _unit_interval(config: Any, *names: str) -> None:
    for name in names:
        value = getattr(config, name)
        if not 0 <= value <= 1:
            raise ValueError(f"{config.__class__.__name__}.{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    num_episodes: int = 2000
    n_envs: int = 8
    n_self_envs: int = 4
    n_pool_opponents: int = 4
    rollout_steps: int = 320
    batch_size: int = 128
    chunk_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_low: float = 0.2
    clip_high: float = 0.28
    lr: float = 6e-5
    value_coef: float = 0.05
    entropy_coef: float = 0.03
    max_grad_norm: float = 1.0
    target_kl: float = 0.015
    ppo_epochs: int = 6
    teampreview_loss_mult: float = 1.5
    teampreview_entropy_mult: float = 2.0
    enable_optim: bool = True
    warmup_episodes: int = 20
    ramp_up_phase: float = 0.1
    ramp_down_phase: float = 0.2

    def __post_init__(self) -> None:
        _positive_ints(
            self,
            "num_episodes",
            "n_envs",
            "n_pool_opponents",
            "rollout_steps",
            "batch_size",
            "chunk_size",
            "ppo_epochs",
        )
        if not 0 <= self.n_self_envs <= self.n_envs:
            raise ValueError("training.n_self_envs must be between 0 and training.n_envs")
        _unit_interval(self, "gamma", "gae_lambda", "ramp_up_phase", "ramp_down_phase")
        _non_negative(self, "clip_low", "clip_high", "value_coef", "entropy_coef", "target_kl")
        _positive(self, "lr", "max_grad_norm", "teampreview_loss_mult", "teampreview_entropy_mult")
        if not 0 <= self.warmup_episodes <= self.num_episodes:
            raise ValueError("training.warmup_episodes must be between 0 and training.num_episodes")
        if self.ramp_up_phase + self.ramp_down_phase > 1:
            raise ValueError("training.ramp_up_phase + training.ramp_down_phase must not exceed 1")


@dataclass(frozen=True, slots=True)
class PoolConfig:
    pool_size: int = 50
    snapshot_interval: int = 20
    pool_anchor_every: int = 10
    pool_win_rate_smoothing: float = 0.1
    pool_wr_floor: float = 0.1
    pool_anchor_drop_wr: float = 0.05
    pool_anchor_min_wr: float = 0.4
    pool_anchor_min_games: int = 20
    pool_explore_coef: float = 0.3

    def __post_init__(self) -> None:
        _positive_ints(self, "pool_size", "snapshot_interval", "pool_anchor_every")
        _non_negative(self, "pool_anchor_min_games", "pool_explore_coef")
        _unit_interval(
            self,
            "pool_win_rate_smoothing",
            "pool_wr_floor",
            "pool_anchor_drop_wr",
            "pool_anchor_min_wr",
        )
        if self.pool_anchor_drop_wr > self.pool_anchor_min_wr:
            raise ValueError("pool.pool_anchor_drop_wr must not exceed pool.pool_anchor_min_wr")


@dataclass(frozen=True, slots=True)
class PathsConfig:
    artifacts_dir: Path = ARTIFACTS_DIR
    pool_dir: Path = ARTIFACTS_DIR / "checkpoints" / "pool"
    checkpoint_path: Path = ARTIFACTS_DIR / "checkpoints" / "ppo_checkpoint.pt"
    runs_dir: Path = ARTIFACTS_DIR / "runs"
    replays_dir: Path = ARTIFACTS_DIR / "replays"
    backups_dir: Path = ARTIFACTS_DIR / "backups"
    log_path: Path = ARTIFACTS_DIR / "training.log"


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    team_pool: str = "all"
    opponent_team_pool: str = "all"


@dataclass(frozen=True, slots=True)
class BotConfig:
    username: str = "Bot"
    password: str | None = None
    battle_format: str = "gen9championsvgc2026regma"
    websocket_url: str | None = None
    authentication_url: str | None = None
    checkpoint_path: Path | None = None
    team_files: tuple[Path, ...] = ()
    top_p: float = 0.9
    max_concurrent_battles: int = 10
    challenge_limit: int = 1_000_000
    opponent: str | None = None
    accept_open_team_sheet: bool = True
    allow_random_init: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    training: TrainingConfig = TrainingConfig()
    pool: PoolConfig = PoolConfig()
    paths: PathsConfig = PathsConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    bot: BotConfig = BotConfig()

    def __post_init__(self) -> None:
        if self.training.n_pool_opponents > self.pool.pool_size:
            raise ValueError("training.n_pool_opponents must not exceed pool.pool_size")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _resolve_paths(config: GlobalConfig) -> GlobalConfig:
    paths = replace(
        config.paths,
        artifacts_dir=_resolve_path(config.paths.artifacts_dir),
        pool_dir=_resolve_path(config.paths.pool_dir),
        checkpoint_path=_resolve_path(config.paths.checkpoint_path),
        runs_dir=_resolve_path(config.paths.runs_dir),
        replays_dir=_resolve_path(config.paths.replays_dir),
        backups_dir=_resolve_path(config.paths.backups_dir),
        log_path=_resolve_path(config.paths.log_path),
    )
    bot = replace(
        config.bot,
        checkpoint_path=(
            None
            if config.bot.checkpoint_path is None
            else _resolve_path(config.bot.checkpoint_path)
        ),
        team_files=tuple(_resolve_path(path) for path in config.bot.team_files),
    )
    return replace(config, paths=paths, bot=bot)


def _build_section(cls: type, values: Any, *, bot: bool = False) -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{cls.__name__} must be a mapping")
    names = {field.name for field in fields(cls)}
    unknown = set(values) - names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {cls.__name__} field(s): {names}")
    values = dict(values)
    if bot:
        values["team_files"] = tuple(values["team_files"])
    return cls(**values)


def load_config(config_path: str | Path | None = None) -> GlobalConfig:
    """Load required ``config.yaml`` and apply its values to source defaults."""
    path = PROJECT_ROOT / "config.yaml" if config_path is None else Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.name in {".ppoconfig", ".ppoconfig.example"}:
        raise ValueError(".ppoconfig is no longer supported; migrate settings to config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        loaded = OmegaConf.load(path)
        merged = OmegaConf.merge(OmegaConf.create(asdict(GlobalConfig())), loaded)
        values = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(values, Mapping):
            raise ValueError("configuration root must be a mapping")
        sections = {field.name for field in fields(GlobalConfig)}
        unknown = set(values) - sections
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown root configuration section(s): {names}")
        config = GlobalConfig(
            training=_build_section(TrainingConfig, values["training"]),
            pool=_build_section(PoolConfig, values["pool"]),
            paths=_build_section(PathsConfig, values["paths"]),
            environment=_build_section(EnvironmentConfig, values["environment"]),
            bot=_build_section(BotConfig, values["bot"], bot=True),
        )
        return _resolve_paths(config)
    except (OSError, OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load configuration from {path}: {exc}") from exc
