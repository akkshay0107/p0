"""Application composition for PPO training."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Mapping

import torch.optim as optim
from poke_env import AccountConfiguration
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from p0.model.config import ModelConfig
from p0.model.factory import build_policy, compile_policy
from p0.model.observation_builder import ObservationBuilder
from p0.model.resources import default_runtime_resources
from p0.runtime.composition import build_sim_env
from p0.runtime.env import SimEnv
from p0.runtime.showdown import start_showdown_servers
from p0.teams.factory import build_team_source
from p0.training.checkpoint import DEFAULT_POLICY_STORE, PolicyStore
from p0.training.config import GlobalConfig
from p0.training.magnet import Magnet
from p0.training.ppo import PPOUpdater
from p0.training.rollout import RolloutCollector
from p0.training.trainer import PPOTrainer
from p0.training.utils import (
    PPOScheduler,
    adamw_param_groups,
    default_device,
    seed_everything,
    select_optimization_precision,
)
from p0.training.vector_env import ThreadVecEnv


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
    agent_team_source: str = "all",
) -> None:
    """
    Build the self-play stack and run training to completion.

    Arguments:
        config: Validated global runtime and training configuration.
        policy_store: Checkpoint persistence implementation.
        cancel_requested: Callback polled for cooperative cancellation.
        agent_team_source: Team pool name used for the self-play agent.

    Returns:
        None.
    """
    training, paths = config.training, config.paths
    if agent_team_source == "all":
        agent_team_path = config.teams.all
    elif agent_team_source == "reduced":
        agent_team_path = config.teams.reduced
    else:
        raise ValueError("agent_team_source must be 'all' or 'reduced'")

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
    precision = select_optimization_precision(training.enable_optim, device)
    scaler = GradScaler(device.type, enabled=precision.grad_scaler)
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

    agent_source = build_team_source(agent_team_path, expected_format_id=config.bot.battle_format)
    opponent_source = build_team_source(
        config.teams.all, expected_format_id=config.bot.battle_format
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
                metrics_path=paths.runs_dir / "ppo_training" / "metrics.json",
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
