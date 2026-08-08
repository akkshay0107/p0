"""Application composition for PPO training."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Callable, Mapping

import torch.optim as optim
from poke_env import AccountConfiguration
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from p0.format_config import FORMAT
from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.runtime.showdown import start_showdown_servers
from p0.teams.source import FileTeamSource, TeamSource
from p0.training.checkpoint import DEFAULT_POLICY_STORE, PolicyStore
from p0.training.config import CorpusConfig, GlobalConfig, TeamSourceConfig
from p0.training.magnet import Magnet
from p0.training.ppo import PPOUpdater
from p0.training.rollout import RolloutCollector
from p0.training.trainer import PPOTrainer
from p0.training.utils import PPOScheduler, adamw_param_groups, default_device, seed_everything
from p0.training.vector_env import ThreadVecEnv


def _team_source(
    config: TeamSourceConfig,
    *,
    corpus_config: CorpusConfig | None = None,
    is_agent: bool = True,
) -> TeamSource:
    """Build a team provider from the configuration, automatically parsing corpus manifests."""
    path = config.path
    if path.is_dir() and (path / "corpus_manifest.json").exists():
        path = path / "corpus_manifest.json"

    if path.suffix == ".json" or (path.is_file() and path.name.endswith(".json")):
        from p0.teams.corpus import CorpusSourceSpec, CorpusSplit, SamplingPolicy
        from p0.teams.corpus_source import CorpusTeamSource

        raw = json.loads(path.read_text(encoding="utf-8"))
        corpus_hash = str(raw.get("corpus_hash", ""))
        format_id = str(raw.get("format_id", FORMAT.battle_format))

        split = CorpusSplit.TRAIN
        policy = SamplingPolicy.USAGE_WEIGHTED
        if corpus_config is not None:
            split_name = corpus_config.agent_split.upper() if is_agent else "TRAIN"
            split = CorpusSplit[split_name]
            policy_name = corpus_config.sampling_policy.upper()
            policy = SamplingPolicy[policy_name]

        spec = CorpusSourceSpec(
            corpus_path=str(path),
            corpus_hash=corpus_hash,
            format_id=format_id,
            split=split,
            sampling_policy=policy,
        )
        return CorpusTeamSource(spec)
    return FileTeamSource(config.path)


def _tensorboard_sink(writer: SummaryWriter):
    """Create a metric sink callback for Tensorboard logging."""

    return functools.partial(_emit_tensorboard, writer)


def _emit_tensorboard(
    writer: SummaryWriter,
    metrics: Mapping[str, float],
    step: int,
    phase: str,
) -> None:
    """Write a metric mapping under a phase-specific Tensorboard namespace."""
    for name, value in metrics.items():
        writer.add_scalar(f"{phase}/{name}", value, step)


def _close_environments(envs: list[SimEnv]) -> None:
    """Cleanly close a list of Pokemon Showdown simulation environments."""
    for env in envs:
        try:
            env.close()
            for player in (env.agent1, env.agent2):
                asyncio.run_coroutine_threadsafe(
                    player.ps_client.stop_listening(), player.ps_client.loop
                ).result(timeout=2.0)
        # Environment and client shutdown cross async/thread boundaries; cleanup
        # must continue for the remaining environments after any one failure.
        except Exception:
            logging.exception("Failed to close a simulation environment cleanly")


def _close_vector_env(vector_env: ThreadVecEnv) -> None:
    """Cleanly shutdown the vector environment pool and its underlying simulations."""
    vector_env.shutdown()
    _close_environments(vector_env.envs)


def run_training(
    config: GlobalConfig,
    *,
    policy_store: PolicyStore = DEFAULT_POLICY_STORE,
    cancel_requested: Callable[[], bool] = lambda: False,
) -> None:
    """Build the self-play stack and run training to completion.

    Arguments:
        config: Validated global runtime and training configuration.
        policy_store: Checkpoint persistence implementation.
        cancel_requested: Callback polled for cooperative cancellation.

    Returns:
        None.
    """
    training, paths = config.training, config.paths
    checkpoint_path = paths.resume_checkpoint or paths.initial_policy_checkpoint
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint_path}")
        policy_store.preflight(checkpoint_path)

    seed_everything(training.seed)
    resources = default_runtime_resources()
    device = default_device()

    if paths.resume_checkpoint is not None:
        policy = policy_store.load_policy(
            paths.resume_checkpoint,
            device,
            expected_metadata={
                "gamma": training.gamma,
                "value_target_semantics": "discounted_terminal_outcome.v1",
            },
        )
    elif paths.initial_policy_checkpoint is not None:
        policy = policy_store.load_policy(
            paths.initial_policy_checkpoint,
            device,
            expected_metadata={
                "gamma": training.gamma,
                "value_target_semantics": "discounted_terminal_outcome.v1",
            },
        )
    else:
        policy = build_policy(ModelConfig.baseline(), resources).to(device)

    optimizer = optim.AdamW(adamw_param_groups(policy, weight_decay=1e-4), lr=training.lr, eps=1e-6)
    scaler = GradScaler(
        "cuda", enabled=training.enable_optim and device.type == "cuda", init_scale=512.0
    )
    magnet = Magnet(policy)
    scheduler = PPOScheduler(training)
    resume_environment_state: object | None = None
    resume_collector_state: object | None = None

    start = (
        policy_store.load_training_state(
            paths.resume_checkpoint,
            policy,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            magnet=magnet,
            expected_trainer_kind="ppo",
            require_training_state=True,
        )
        if paths.resume_checkpoint is not None
        else 0
    )
    if paths.resume_checkpoint is not None:
        resume_metadata = policy_store.load_metadata(paths.resume_checkpoint)
        resume_environment_state = resume_metadata.get("environment_state")
        resume_collector_state = resume_metadata.get("collector_state")
    policy = compile_policy(policy, enable=training.enable_optim and device.type == "cuda")

    agent_source = _team_source(
        config.environment.agent_team_source,
        corpus_config=config.corpus,
        is_agent=True,
    )
    opponent_source = _team_source(
        config.environment.opponent_team_source,
        corpus_config=config.corpus,
        is_agent=False,
    )

    with start_showdown_servers(
        training.n_envs,
        showdown_root=paths.showdown_root,
    ) as servers:
        envs = []
        vector_env = None
        writer = None

        try:
            for index, server in enumerate(servers):
                envs.append(
                    build_sim_env(
                        account_configuration1=AccountConfiguration(f"TrainAgent_{index}", None),
                        account_configuration2=AccountConfiguration(f"BestAgent_{index}", None),
                        server_port=server.port,
                        agent_team_source=agent_source,
                        opponent_team_source=opponent_source,
                        observation_builder=ObservationBuilder(resources=resources),
                        agent_seed=index * 2,
                        opponent_seed=index * 2 + 1,
                    )
                )
            vector_env = ThreadVecEnv(envs)
            if isinstance(resume_environment_state, (tuple, list)):
                vector_env.restore_training_state(resume_environment_state)
            writer = SummaryWriter(log_dir=str(paths.runs_dir / "ppo_training"))
            collector = RolloutCollector(
                vector_env,
                policy,
                training,
            )
            if isinstance(resume_collector_state, Mapping):
                collector.restore_training_state(resume_collector_state)
            updater = PPOUpdater(
                policy,
                optimizer,
                scaler,
                training,
                magnet,
                cancel_requested=cancel_requested,
            )
            trainer = PPOTrainer(
                policy=policy,
                policy_store=policy_store,
                checkpoint_path=paths.checkpoint_path,
                collector=collector,
                updater=updater,
                magnet=magnet,
                scheduler=scheduler,
                training_config=training,
                metric_sink=_tensorboard_sink(writer),
                cancel_requested=cancel_requested,
            )
            trainer.run(start)
        finally:
            if writer is not None:
                writer.close()
            if vector_env is not None:
                _close_vector_env(vector_env)
            else:
                _close_environments(envs)
