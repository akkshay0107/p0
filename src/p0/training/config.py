"""Typed, immutable application configuration loaded from YAML."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from p0.paths import DEFAULT_PATHS


def _positive_ints(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{type(obj).__name__}.{name} must be a positive integer")


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if not _is_finite_number(val) or val <= 0:
            raise ValueError(f"{type(obj).__name__}.{name} must be greater than zero")


def _non_negative(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if not _is_finite_number(val) or val < 0:
            raise ValueError(f"{type(obj).__name__}.{name} must not be negative")


def _unit_interval(obj: object, *names: str) -> None:
    for name in names:
        val = getattr(obj, name)
        if not _is_finite_number(val) or not 0 <= val <= 1:
            raise ValueError(f"{type(obj).__name__}.{name} must be between 0 and 1")


# The value head is trained by BC and then by PPO, so both read one gamma.
DEFAULT_GAMMA = 0.99

# Settings config.yaml may contain. Every other field below keeps its default in code.
SHARED_YAML_FIELDS = frozenset({"gamma", "seed", "enable_optim"})
PPO_YAML_FIELDS = frozenset(
    {
        "output_dir",
        "resume_checkpoint",
        "initial_policy_checkpoint",
        "num_episodes",
        "n_envs",
        "rollout_steps",
        "batch_size",
        "lr",
        "entropy_coef",
        "magnet_alpha",
    }
)
BC_YAML_FIELDS = frozenset(
    {
        "dataset_dir",
        "output_dir",
        "resume_checkpoint",
        "epochs",
        "batch_decisions",
        "learning_rate",
        "num_workers",
    }
)
BOT_YAML_FIELDS = frozenset({"username", "websocket_url", "authentication_url"})


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Every value PPO training reads; config.yaml sets the PPO_YAML_FIELDS subset."""

    output_dir: Path = Path("artifacts/runs/ppo")
    resume_checkpoint: Path | None = None
    initial_policy_checkpoint: Path | None = None
    num_episodes: int = 2000
    n_envs: int = 8
    rollout_steps: int = 320
    batch_size: int = 128
    minibatch_size: int = 32
    gamma: float = DEFAULT_GAMMA
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

    @property
    def checkpoint_path(self) -> Path:
        """Return the training checkpoint written inside the output directory."""
        return self.output_dir / "ppo_checkpoint.pt"

    def __post_init__(self) -> None:
        if self.resume_checkpoint is not None and self.initial_policy_checkpoint is not None:
            raise ValueError(
                "ppo.resume_checkpoint and ppo.initial_policy_checkpoint are mutually exclusive"
            )

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
            raise ValueError("gamma must be less than 1")

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
            raise ValueError("seed must be a nonnegative integer")
        if not 0.0 < self.ramp_up_phase < 1.0:
            raise ValueError("PPOConfig.ramp_up_phase must be strictly between 0 and 1")

        ramp_end = int(self.ramp_up_phase * self.num_episodes)
        if not 0 < ramp_end < self.num_episodes - 1:
            raise ValueError(
                f"ppo.num_episodes={self.num_episodes} is too short for the learning-rate "
                f"warmup, which takes {self.ramp_up_phase:.0%} of the run"
            )


@dataclass(frozen=True, slots=True)
class BotConfig:
    username: str = "Bot"
    websocket_url: str | None = None
    authentication_url: str | None = None

    def __post_init__(self) -> None:
        if (self.websocket_url is None) != (self.authentication_url is None):
            raise ValueError(
                "bot.websocket_url and bot.authentication_url must be configured together"
            )
        for name, url in (
            ("websocket_url", self.websocket_url),
            ("authentication_url", self.authentication_url),
        ):
            if url is not None and not url.strip():
                raise ValueError(f"bot.{name} must not be empty")


@dataclass(frozen=True, slots=True)
class BCConfig:
    """Every value BC training reads; config.yaml sets the BC_YAML_FIELDS subset."""

    dataset_dir: Path = Path("artifacts/datasets")
    output_dir: Path = Path("artifacts/checkpoints/bc")
    resume_checkpoint: Path | None = None
    epochs: int = 1
    batch_decisions: int = 256
    max_chunk_size: int = 1024
    learning_rate: float = 3e-4
    gamma: float = DEFAULT_GAMMA
    value_coef: float = 0.5
    num_workers: int = 0
    prefetch_factor: int = 2
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    seed: int = 0
    enable_optim: bool = True

    def __post_init__(self) -> None:
        _positive_ints(self, "batch_decisions", "max_chunk_size", "epochs")
        if self.num_workers < 0:
            raise ValueError("bc.num_workers must be non-negative")
        if self.prefetch_factor <= 0:
            raise ValueError("BCConfig.prefetch_factor must be positive")

        _positive(self, "learning_rate", "max_grad_norm")
        _unit_interval(self, "gamma")
        if self.gamma >= 1.0:
            raise ValueError("gamma must be less than 1")
        _non_negative(self, "value_coef", "weight_decay")

        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


class GlobalConfig(NamedTuple):
    ppo: PPOConfig = PPOConfig()
    bc: BCConfig = BCConfig()
    bot: BotConfig = BotConfig()


def _resolve_fields(section: Any, path_field_names: set[str]) -> Any:
    """Return the config section with its set path fields resolved against the repository root."""
    updates = {
        name: (DEFAULT_PATHS.repository_root / Path(value).expanduser()).resolve()
        for name in path_field_names
        if (value := getattr(section, name)) is not None
    }
    return replace(section, **updates)


def _section(values: Mapping[str, Any], name: str, allowed: frozenset[str]) -> dict[str, Any]:
    """Return one YAML section after rejecting settings the file may not contain."""
    section = values.get(name)
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be a mapping")

    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown {name} setting(s): {', '.join(sorted(unknown))}")
    return dict(section)


def load_config(config_path: str | Path | None = None) -> GlobalConfig:
    """Load config.yaml; settings it omits keep their defaults."""
    path = (
        DEFAULT_PATHS.repository_root / "config.yaml" if config_path is None else Path(config_path)
    )
    if not path.is_absolute():
        path = DEFAULT_PATHS.repository_root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        values = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(values, Mapping):
            raise ValueError("configuration root must be a mapping")

        unknown = set(values) - SHARED_YAML_FIELDS - {"ppo", "bc", "bot"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unknown root setting(s): {names}")

        # gamma, seed, and enable_optim are set once and given to both trainers.
        shared = {name: values[name] for name in SHARED_YAML_FIELDS if name in values}
        ppo = PPOConfig(**_section(values, "ppo", PPO_YAML_FIELDS), **shared)
        bc = BCConfig(**_section(values, "bc", BC_YAML_FIELDS), **shared)
        bot = BotConfig(**_section(values, "bot", BOT_YAML_FIELDS))
        return GlobalConfig(
            ppo=_resolve_fields(
                ppo, {"output_dir", "resume_checkpoint", "initial_policy_checkpoint"}
            ),
            bc=_resolve_fields(bc, {"dataset_dir", "output_dir", "resume_checkpoint"}),
            bot=bot,
        )
    except (OSError, OmegaConfBaseException, TypeError, ValueError) as exc:
        raise ValueError(f"Could not load configuration from {path}: {exc}") from exc
