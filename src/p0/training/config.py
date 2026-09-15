"""Typed, immutable application configuration loaded from YAML."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from p0.format_config import FORMAT
from p0.paths import DEFAULT_PATHS, ProjectPaths


def _positive_ints(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{type(obj).__name__}.{name} must be a positive integer")


def _positive(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if (
            not isinstance(val, (int, float))
            or isinstance(val, bool)
            or not math.isfinite(val)
            or val <= 0
        ):
            raise ValueError(f"{type(obj).__name__}.{name} must be greater than zero")


def _non_negative(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if (
            not isinstance(val, (int, float))
            or isinstance(val, bool)
            or not math.isfinite(val)
            or val < 0
        ):
            raise ValueError(f"{type(obj).__name__}.{name} must not be negative")


def _unit_interval(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not 0 <= val <= 1:
            raise ValueError(f"{type(obj).__name__}.{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    num_episodes: int = 2000
    n_envs: int = 8
    rollout_steps: int = 320
    batch_size: int = 128
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    magnet_alpha: float = 0.03
    magnet_refresh_interval: int = 20
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    ppo_epochs: int = 10
    enable_optim: bool = True
    seed: int = 0
    ramp_up_phase: float = 0.1

    def __post_init__(self) -> None:
        _positive_ints(
            self,
            "num_episodes",
            "n_envs",
            "rollout_steps",
            "batch_size",
            "minibatch_size",
            "ppo_epochs",
            "magnet_refresh_interval",
        )
        _unit_interval(self, "gamma", "gae_lambda", "ramp_up_phase")
        if self.gamma >= 1.0:
            raise ValueError("TrainingConfig.gamma must be less than 1")

        _non_negative(
            self,
            "clip_range",
            "value_coef",
            "magnet_alpha",
            "entropy_coef",
            "target_kl",
        )
        _positive(self, "lr", "max_grad_norm")

        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("training.seed must be a nonnegative integer")
        if not 0.0 < self.ramp_up_phase < 1.0:
            raise ValueError("training.ramp_up_phase must be strictly between 0 and 1")

        ramp_end = int(self.ramp_up_phase * self.num_episodes)
        if not 0 < ramp_end < self.num_episodes - 1:
            raise ValueError(
                "training.ramp_up_phase must produce an endpoint before the final episode"
            )
        if self.magnet_refresh_interval > self.num_episodes:
            raise ValueError(
                "training.magnet_refresh_interval must not exceed training.num_episodes"
            )


@dataclass(frozen=True, slots=True)
class TeamsConfig:
    all: Path = Path("all")
    reduced: Path = Path("reduced")

    def __post_init__(self) -> None:
        for name, path in (("all", self.all), ("reduced", self.reduced)):
            if not str(path).strip():
                raise ValueError(f"teams.{name} must not be empty")


@dataclass(frozen=True, slots=True)
class BotConfig:
    username: str = "Bot"
    password: str | None = None
    battle_format: str = FORMAT.bo3_format
    websocket_url: str | None = None
    authentication_url: str | None = None
    checkpoint_path: Path | None = None
    team_files: tuple[Path, ...] = ()
    top_p: float = 0.9
    max_concurrent_battles: int = 1
    challenge_limit: int = 1_000_000
    opponent: str | None = None
    allow_random_init: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not isinstance(self.team_files, tuple):
            object.__setattr__(self, "team_files", tuple(self.team_files))

        if self.battle_format != FORMAT.bo3_format:
            raise ValueError(f"bot.battle_format must be the Bo3 format {FORMAT.bo3_format!r}")

        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("bot.top_p must be in (0, 1]")

        if type(self.max_concurrent_battles) is not int or self.max_concurrent_battles != 1:
            raise ValueError("bot.max_concurrent_battles is fixed at 1 for live Bo3 play")


@dataclass(frozen=True, slots=True)
class BCConfig:
    batch_decisions: int = 256
    max_chunk_size: int = 1024
    learning_rate: float = 3e-4
    gamma: float = 0.99
    value_coef: float = 0.5
    epochs: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 0
    enable_optim: bool = True
    shard_manifest: Path = Path("artifacts/shards/manifest.json")
    split_manifest: Path = Path("artifacts/shards/splits.json")
    output_dir: Path = Path("artifacts/checkpoints/bc")
    resume_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        _positive_ints(self, "batch_decisions", "max_chunk_size", "epochs")
        if self.num_workers < 0:
            raise ValueError("bc.num_workers must be non-negative")
        if self.prefetch_factor <= 0:
            raise ValueError("bc.prefetch_factor must be positive")

        _positive(self, "learning_rate", "max_grad_norm")
        _unit_interval(self, "gamma")
        if self.gamma >= 1.0:
            raise ValueError("BCConfig.gamma must be less than 1")
        _non_negative(self, "value_coef", "weight_decay")

        if type(self.seed) is not int:
            raise ValueError("bc.seed must be an integer")

        for name, value in (
            ("shard_manifest", self.shard_manifest),
            ("split_manifest", self.split_manifest),
            ("output_dir", self.output_dir),
        ):
            if not str(value).strip():
                raise ValueError(f"bc.{name} must not be empty")


@dataclass(frozen=True, slots=True)
class EvalConfig:
    episodes_per_matchup: int = 20
    seed: int = 0
    report_dir: Path = Path("artifacts/eval")

    def __post_init__(self) -> None:
        _positive_ints(self, "episodes_per_matchup")
        _non_negative(self, "seed")

        if not str(self.report_dir).strip():
            raise ValueError("evaluation.report_dir must not be empty")


@dataclass(frozen=True, slots=True)
class GlobalConfig:
    training: TrainingConfig = TrainingConfig()
    paths: ProjectPaths = DEFAULT_PATHS
    teams: TeamsConfig = TeamsConfig()
    bot: BotConfig = BotConfig()
    bc: BCConfig = BCConfig()
    evaluation: EvalConfig = EvalConfig()

    def __post_init__(self) -> None:
        if self.bc.gamma != self.training.gamma:
            raise ValueError("bc.gamma must match training.gamma")
        if self.bc.value_coef != self.training.value_coef:
            raise ValueError("bc.value_coef must match training.value_coef")


def _resolve_path(value: str | Path, root: Path = DEFAULT_PATHS.repository_root) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_paths(config: GlobalConfig) -> GlobalConfig:
    root = _resolve_path(config.paths.repository_root)

    def _resolve(dc: Any, base: Path, path_field_names: set[str]) -> Any:
        updates: dict[str, Any] = {}
        for f in fields(dc):
            if f.name in path_field_names:
                val = getattr(dc, f.name)
                if val is not None:
                    updates[f.name] = _resolve_path(val, base)
        return replace(dc, **updates)

    path_fields = {f.name for f in fields(ProjectPaths)}
    paths = _resolve(config.paths, root, path_fields)
    bot = replace(
        config.bot,
        checkpoint_path=(
            _resolve_path(config.bot.checkpoint_path, root)
            if config.bot.checkpoint_path is not None
            else None
        ),
        team_files=tuple(_resolve_path(p, root) for p in config.bot.team_files),
    )
    teams = _resolve(config.teams, paths.teams_root, {"all", "reduced"})
    bc = _resolve(
        config.bc,
        root,
        {"shard_manifest", "split_manifest", "output_dir", "resume_checkpoint"},
    )
    evaluation = _resolve(config.evaluation, root, {"report_dir"})

    return replace(config, paths=paths, bot=bot, teams=teams, bc=bc, evaluation=evaluation)


def _build_section(cls: type, values: Any) -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{cls.__name__} must be a mapping")

    names = {field.name for field in fields(cls)}
    unknown = set(values) - names
    if unknown:
        names_str = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {cls.__name__} field(s): {names_str}")

    return cls(**values)


def load_config(config_path: str | Path | None = None) -> GlobalConfig:
    """Load configuration from config.yaml and merge with source defaults."""
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
        loaded: Any = OmegaConf.load(path)
        loaded_bc = loaded.get("bc", {})
        if loaded_bc is not None and not isinstance(loaded_bc, Mapping):
            raise ValueError("BCConfig must be a mapping")
        if isinstance(loaded_bc, Mapping):
            duplicated_objective_fields = {"gamma", "value_coef"} & set(loaded_bc)
            if duplicated_objective_fields:
                fields_text = ", ".join(sorted(duplicated_objective_fields))
                raise ValueError(
                    f"bc.{fields_text} is derived from training; configure it under training"
                )
        merged = OmegaConf.merge(OmegaConf.create(asdict(GlobalConfig())), loaded)
        values = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(values, Mapping):
            raise ValueError("configuration root must be a mapping")

        sections = {field.name for field in fields(GlobalConfig)}
        unknown = set(values) - sections
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unknown root configuration section(s): {names}")

        training = _build_section(TrainingConfig, values["training"])
        bc_values = dict(values["bc"])
        bc_values["gamma"] = training.gamma
        bc_values["value_coef"] = training.value_coef
        config = GlobalConfig(
            training=training,
            paths=_build_section(ProjectPaths, values["paths"]),
            teams=_build_section(TeamsConfig, values["teams"]),
            bot=_build_section(BotConfig, values["bot"]),
            bc=_build_section(BCConfig, bc_values),
            evaluation=_build_section(EvalConfig, values["evaluation"]),
        )
        return _resolve_paths(config)
    except (OSError, OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load configuration from {path}: {exc}") from exc
