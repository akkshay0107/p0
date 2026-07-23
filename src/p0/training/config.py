"""Typed, immutable application configuration loaded from YAML."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS, ProjectPaths


def _positive_ints(owner: str, *values: tuple[str, object]) -> None:
    for name, value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{owner}.{name} must be a positive integer")


def _positive(owner: str, *values: tuple[str, float]) -> None:
    for name, value in values:
        if value <= 0:
            raise ValueError(f"{owner}.{name} must be greater than zero")


def _non_negative(owner: str, *values: tuple[str, float]) -> None:
    for name, value in values:
        if value < 0:
            raise ValueError(f"{owner}.{name} must not be negative")


def _unit_interval(owner: str, *values: tuple[str, float]) -> None:
    for name, value in values:
        if not 0 <= value <= 1:
            raise ValueError(f"{owner}.{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    num_episodes: int = 2000
    n_envs: int = 8
    rollout_steps: int = 320
    batch_size: int = 128
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_low: float = 0.2
    clip_high: float = 0.28
    lr: float = 6e-5
    value_coef: float = 0.05
    magnet_alpha: float = 0.03
    magnet_refresh_interval: int = 20
    residual_entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.015
    ppo_epochs: int = 6
    teampreview_loss_mult: float = 1.5
    teampreview_alpha_mult: float = 2.0
    enable_optim: bool = True
    warmup_episodes: int = 20
    ramp_up_phase: float = 0.1

    def __post_init__(self) -> None:
        _positive_ints(
            type(self).__name__,
            ("num_episodes", self.num_episodes),
            ("n_envs", self.n_envs),
            ("rollout_steps", self.rollout_steps),
            ("batch_size", self.batch_size),
            ("minibatch_size", self.minibatch_size),
            ("ppo_epochs", self.ppo_epochs),
            ("magnet_refresh_interval", self.magnet_refresh_interval),
        )
        _unit_interval(
            type(self).__name__,
            ("gamma", self.gamma),
            ("gae_lambda", self.gae_lambda),
            ("ramp_up_phase", self.ramp_up_phase),
        )
        _non_negative(
            type(self).__name__,
            ("clip_low", self.clip_low),
            ("clip_high", self.clip_high),
            ("value_coef", self.value_coef),
            ("magnet_alpha", self.magnet_alpha),
            ("residual_entropy_coef", self.residual_entropy_coef),
            ("target_kl", self.target_kl),
        )
        _positive(
            type(self).__name__,
            ("lr", self.lr),
            ("max_grad_norm", self.max_grad_norm),
            ("teampreview_loss_mult", self.teampreview_loss_mult),
            ("teampreview_alpha_mult", self.teampreview_alpha_mult),
        )
        if not 0 <= self.warmup_episodes <= self.num_episodes:
            raise ValueError("training.warmup_episodes must be between 0 and training.num_episodes")
        if self.magnet_refresh_interval > self.num_episodes:
            raise ValueError(
                "training.magnet_refresh_interval must not exceed training.num_episodes"
            )


@dataclass(frozen=True, slots=True)
class TeamSourceConfig:
    path: Path = Path("all")

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("TeamSourceConfig.path must not be empty")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    agent_team_source: TeamSourceConfig = TeamSourceConfig()
    opponent_team_source: TeamSourceConfig = TeamSourceConfig()


@dataclass(frozen=True, slots=True)
class BotConfig:
    username: str = "Bot"
    password: str | None = None
    battle_format: str = FORMAT.battle_format
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

    def __post_init__(self) -> None:
        if self.battle_format != FORMAT.battle_format:
            raise ValueError(
                f"bot.battle_format must match configured format {FORMAT.battle_format!r}"
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("bot.top_p must be in (0, 1]")


# The bc, corpus, and evaluation sections are reserved here so their
# workstreams only ever touch their own dataclass's field list; adding a new
# root section requires editing GlobalConfig and load_config in one place.
@dataclass(frozen=True, slots=True)
class BCConfig:
    batch_decisions: int = 256
    learning_rate: float = 3e-4
    epochs: int = 1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 0
    amp: bool = True
    shards_dir: str = "artifacts/shards"
    checkpoint_path: str = "artifacts/bc_checkpoint.pt"

    def __post_init__(self) -> None:
        _positive_ints(
            type(self).__name__,
            ("batch_decisions", self.batch_decisions),
            ("epochs", self.epochs),
        )
        _positive(type(self).__name__, ("learning_rate", self.learning_rate))
        _non_negative(type(self).__name__, ("weight_decay", self.weight_decay))
        _positive(type(self).__name__, ("max_grad_norm", self.max_grad_norm))
        if type(self.seed) is not int:
            raise ValueError("bc.seed must be an integer")
        if not self.shards_dir.strip():
            raise ValueError("bc.shards_dir must not be empty")
        if not self.checkpoint_path.strip():
            raise ValueError("bc.checkpoint_path must not be empty")


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    manifest_path: str = "teams/corpus_manifest.json"
    agent_split: str = "train"
    sampling_policy: str = "usage_weighted"
    allow_mirror: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("manifest_path", self.manifest_path),
            ("agent_split", self.agent_split),
            ("sampling_policy", self.sampling_policy),
        ):
            if not value.strip():
                raise ValueError(f"corpus.{name} must not be empty")
        if self.agent_split.upper() not in {"TRAIN", "VALIDATION", "TEST"}:
            raise ValueError("corpus.agent_split must be train, validation, or test")
        if self.sampling_policy.upper() not in {
            "USAGE_WEIGHTED",
            "UNIFORM_CANONICAL",
            "UNIFORM_ARCHETYPE",
            "RARE_COVERAGE",
            "MATCHUP_BALANCED",
        }:
            raise ValueError("corpus.sampling_policy is not supported")


@dataclass(frozen=True, slots=True)
class EvalConfig:
    episodes_per_matchup: int = 20
    seed: int = 0
    report_dir: str = "artifacts/eval"

    def __post_init__(self) -> None:
        _positive_ints(type(self).__name__, ("episodes_per_matchup", self.episodes_per_matchup))
        _non_negative(type(self).__name__, ("seed", self.seed))
        if not self.report_dir.strip():
            raise ValueError("evaluation.report_dir must not be empty")


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    bo3: bool = False
    training: TrainingConfig = TrainingConfig()
    paths: ProjectPaths = DEFAULT_PATHS
    environment: EnvironmentConfig = EnvironmentConfig()
    bot: BotConfig = BotConfig()
    bc: BCConfig = BCConfig()
    corpus: CorpusConfig = CorpusConfig()
    evaluation: EvalConfig = EvalConfig()

    def __post_init__(self) -> None:
        if type(self.bo3) is not bool:
            raise ValueError("bo3 must be a boolean")


def _resolve_path(value: str | Path, root: Path = DEFAULT_PATHS.repository_root) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_paths(config: GlobalConfig) -> GlobalConfig:
    repository_root = _resolve_path(config.paths.repository_root)
    paths = replace(
        config.paths,
        repository_root=repository_root,
        data_root=_resolve_path(config.paths.data_root, repository_root),
        teams_root=_resolve_path(config.paths.teams_root, repository_root),
        artifacts_root=_resolve_path(config.paths.artifacts_root, repository_root),
        showdown_root=_resolve_path(config.paths.showdown_root, repository_root),
        gauntlet_dir=_resolve_path(config.paths.gauntlet_dir, repository_root),
        checkpoint_path=_resolve_path(config.paths.checkpoint_path, repository_root),
        runs_dir=_resolve_path(config.paths.runs_dir, repository_root),
        replays_dir=_resolve_path(config.paths.replays_dir, repository_root),
        backups_dir=_resolve_path(config.paths.backups_dir, repository_root),
        log_path=_resolve_path(config.paths.log_path, repository_root),
    )
    bot = replace(
        config.bot,
        checkpoint_path=(
            None
            if config.bot.checkpoint_path is None
            else _resolve_path(config.bot.checkpoint_path, repository_root)
        ),
        team_files=tuple(_resolve_path(path, repository_root) for path in config.bot.team_files),
    )
    environment = replace(
        config.environment,
        agent_team_source=replace(
            config.environment.agent_team_source,
            path=_resolve_path(config.environment.agent_team_source.path, paths.teams_root),
        ),
        opponent_team_source=replace(
            config.environment.opponent_team_source,
            path=_resolve_path(config.environment.opponent_team_source.path, paths.teams_root),
        ),
    )
    return replace(config, paths=paths, bot=bot, environment=environment)


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


def _build_environment(values: Any) -> EnvironmentConfig:
    if not isinstance(values, Mapping):
        raise ValueError("EnvironmentConfig must be a mapping")
    names = {field.name for field in fields(EnvironmentConfig)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown EnvironmentConfig field(s): {', '.join(sorted(unknown))}")
    return EnvironmentConfig(
        agent_team_source=_build_section(TeamSourceConfig, values["agent_team_source"]),
        opponent_team_source=_build_section(TeamSourceConfig, values["opponent_team_source"]),
    )


def load_config(config_path: str | Path | None = None) -> GlobalConfig:
    """Load required ``config.yaml`` and apply its values to source defaults."""
    path = (
        DEFAULT_PATHS.repository_root / "config.yaml" if config_path is None else Path(config_path)
    )
    if not path.is_absolute():
        path = DEFAULT_PATHS.repository_root / path
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
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unknown root configuration section(s): {names}")
        config = GlobalConfig(
            bo3=values["bo3"],
            training=_build_section(TrainingConfig, values["training"]),
            paths=_build_section(ProjectPaths, values["paths"]),
            environment=_build_environment(values["environment"]),
            bot=_build_section(BotConfig, values["bot"], bot=True),
            bc=_build_section(BCConfig, values["bc"]),
            corpus=_build_section(CorpusConfig, values["corpus"]),
            evaluation=_build_section(EvalConfig, values["evaluation"]),
        )
        return _resolve_paths(config)
    except (OSError, OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load configuration from {path}: {exc}") from exc
