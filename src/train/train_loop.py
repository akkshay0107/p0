import logging
import os
import random
import signal
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.env import SimEnv
from src.lookups import ACT_SIZE, OBS_DIM
from src.model.policy import PolicyNet
from src.model.structured_observation import (
    NUM_IDX_TEAM_PREVIEW,
    TOKEN_IDX_GLOBAL_FIELD_NUMERIC,
    StructuredObservation,
)
from src.train.config import PPOConfig, load_config
from src.train.opponent_pool import OpponentPool
from src.train.rollout import RolloutBuffer, collect_rollouts
from src.train.utils import (
    PPOScheduler,
    adamw_param_groups,
    initial_state,
    load_checkpoint,
    save_checkpoint,
)
from src.train.vec_env import ThreadVecEnv


def handle_sigterm(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        logging.warning("Second shutdown signal received, forcing immediate exit...")
        os._exit(1)
    logging.warning("SIGTERM received, requesting shutdown...")
    shutdown_requested = True


shutdown_requested = False
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log", mode="w"), logging.StreamHandler(sys.stdout)],
)


def _kl_exceeds_target(kl_sum: float, steps: int, target_kl: float) -> tuple[bool, float]:
    mean_kl = kl_sum / steps if steps > 0 else 0.0
    return mean_kl > target_kl, mean_kl


def _run_batched_ppo(
    episodes: list[dict],
    policy: PolicyNet,
    config: PPOConfig,
    device: torch.device,
    episode: int,
) -> tuple[torch.Tensor, dict[str, float], int]:
    """
    Run PPO BPTT over a minibatch of variable-length episodes.
    """
    if not episodes:
        return (
            torch.tensor(0.0, device=device),
            {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "kl_div": 0.0,
                "clip_frac": 0.0,
            },
            0,
        )

    batch_size = len(episodes)
    lengths = [ep["length"] for ep in episodes]
    max_steps = lengths[0]
    is_warmup = episode < config.warmup_episodes

    all_obs = StructuredObservation.cat([ep["obs"] for ep in episodes], dim=0)
    all_tokens, all_aux = policy.encoder(all_obs, aux=True)
    if is_warmup:
        all_tokens = all_tokens.detach()
        all_aux = all_aux.detach()
    split_sizes = [ep["length"] for ep in episodes]
    tokens_list = torch.split(all_tokens, split_sizes)
    aux_list = torch.split(all_aux, split_sizes)
    numerical_list = torch.split(all_obs.numerical, split_sizes)

    # pre pad all to max len
    tokens_p = torch.nn.utils.rnn.pad_sequence(list(tokens_list), batch_first=True)
    aux_p = torch.nn.utils.rnn.pad_sequence(list(aux_list), batch_first=True)
    numerical_p = torch.nn.utils.rnn.pad_sequence(list(numerical_list), batch_first=True)

    # pre-pack non-observation tensors for fast slicing [Batch, Time, ...]
    def pack(fields):
        return torch.nn.utils.rnn.pad_sequence(fields, batch_first=True).to(device)

    actions_p = pack([ep["actions"] for ep in episodes])
    old_log_probs_p = pack([ep["log_probs"] for ep in episodes])
    advantages_p = pack([ep["advantages"] for ep in episodes])
    returns_p = pack([ep["returns"] for ep in episodes])
    action_masks_p = pack([ep["action_masks"] for ep in episodes])

    state = initial_state(policy, batch_size, device)
    total_loss = torch.tensor(0.0, device=device)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "normalized_entropy": 0.0,
        "kl_div": 0.0,
        "clip_frac": 0.0,
        "entropy_coef": 0.0,
    }
    total_steps = 0

    curr_ent_coef = config.entropy_coef

    for t in range(max_steps):
        active_n = sum(1 for length in lengths if length > t)
        if active_n == 0:
            break

        tokens_t = tokens_p[:active_n, t]  # (active_n, S, D) — contiguous slice
        aux_t = aux_p[:active_n, t]  # (active_n, 4, D)
        numerical_t = numerical_p[:active_n, t]  # (active_n, S, N)
        actions_t = actions_p[:active_n, t]
        old_log_probs_t = old_log_probs_p[:active_n, t]
        advantages_t = advantages_p[:active_n, t]
        returns_t = returns_p[:active_n, t]
        action_masks_t = action_masks_p[:active_n, t]
        is_tp_t = numerical_t[:, TOKEN_IDX_GLOBAL_FIELD_NUMERIC, NUM_IDX_TEAM_PREVIEW] > 0.5

        curr_state = state[:active_n]
        curr_log_prob, curr_entropy, curr_normalized_entropy, curr_val, next_state = (
            policy.evaluate_actions_tokens(
                tokens_t,
                aux_t,
                numerical_t,
                actions_t,
                action_mask=action_masks_t,
                state=curr_state,
                is_warmup=is_warmup,
            )
        )

        if not torch.isfinite(curr_log_prob).all():
            non_finite_idx = (~torch.isfinite(curr_log_prob)).nonzero(as_tuple=True)[0]
            for idx in non_finite_idx:
                logging.error(
                    f"DEBUG: Non-finite log_prob at step {t}, episode element {idx}.\n"
                    f"  actions_t: {actions_t[idx].tolist()}\n"
                    f"  action_masks_t (p1): {action_masks_t[idx, 0].nonzero().squeeze(-1).tolist()}\n"
                    f"  action_masks_t (p2): {action_masks_t[idx, 1].nonzero().squeeze(-1).tolist()}\n"
                    f"  is_tp_t: {is_tp_t[idx].item()}\n"
                    f"  numerical_t[:, TOKEN_IDX_GLOBAL_FIELD_NUMERIC, :4]: {numerical_t[idx, TOKEN_IDX_GLOBAL_FIELD_NUMERIC, :4].tolist()}\n"
                    f"  old_log_prob: {old_log_probs_t[idx].item()}\n"
                    f"  curr_log_prob: {curr_log_prob[idx].item()}\n"
                )

        log_ratio = curr_log_prob - old_log_probs_t
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - config.clip_low, 1.0 + config.clip_high) * advantages_t

        step_policy_loss = -torch.min(surr1, surr2)
        step_value_loss = F.mse_loss(curr_val, returns_t, reduction="none")
        step_entropy_loss = -curr_normalized_entropy

        is_tp_mask = is_tp_t.reshape(-1)
        step_ent_coef = curr_ent_coef * torch.where(
            is_tp_mask, config.teampreview_entropy_mult, 1.0
        )

        step_loss = config.value_coef * step_value_loss

        if not is_warmup:
            step_loss = step_loss + step_policy_loss + step_ent_coef * step_entropy_loss

        step_loss = torch.where(
            is_tp_mask,
            step_loss * config.teampreview_loss_mult,
            step_loss,
        )

        total_loss = total_loss + step_loss.sum()
        total_steps += active_n

        if is_warmup:
            next_state = next_state.detach()

        if active_n < batch_size:
            state = torch.cat([next_state, state[active_n:]], dim=0)
        else:
            state = next_state

        with torch.no_grad():
            metrics["policy_loss"] += step_policy_loss.sum().item() if not is_warmup else 0.0
            metrics["value_loss"] += step_value_loss.sum().item()

            metrics["normalized_entropy"] += curr_normalized_entropy.sum().item()

            metrics["kl_div"] += ((ratio - 1) - log_ratio).sum().item() if not is_warmup else 0.0
            metrics["clip_frac"] += (
                ((ratio < 1 - config.clip_low) | (ratio > 1 + config.clip_high))
                .float()
                .sum()
                .item()
                if not is_warmup
                else 0.0
            )
            metrics["entropy_coef"] = curr_ent_coef

    return total_loss, metrics, total_steps


def ppo_update(
    episodes: list,
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    episode: int,
    shutdown_requested: bool = False,
) -> dict:
    policy.train()
    t0 = time.time()

    with torch.no_grad():
        all_returns = torch.cat([ep["returns"] for ep in episodes])
        all_values = torch.cat([ep["values"] for ep in episodes])
        var_y = torch.var(all_returns)
        if var_y > 1e-8:
            explained_var = 1.0 - torch.var(all_returns - all_values) / var_y
        else:
            explained_var = torch.tensor(0.0)
        explained_var = explained_var.item()

    tot_policy_loss = 0.0
    tot_value_loss = 0.0
    tot_normalized_entropy = 0.0
    tot_kl_div = 0.0
    tot_grad_norm = 0.0
    tot_clip_frac = 0.0
    tot_steps = 0
    num_updates = 0
    epochs_done = 0

    effective_batch_size = config.chunk_size * (
        (config.batch_size - config.chunk_size + 1) // config.chunk_size
    )  # round up to nearest chunk

    early_stop = False
    last_entropy_coef = config.entropy_coef
    for epoch_idx in range(config.ppo_epochs):
        if shutdown_requested or early_stop:
            break
        random.shuffle(episodes)

        epoch_steps = 0
        epoch_kl = 0.0

        for batch_start in range(0, len(episodes), effective_batch_size):
            if shutdown_requested:
                break

            minibatch = episodes[batch_start : batch_start + effective_batch_size]
            if not minibatch:
                continue

            optimizer.zero_grad(set_to_none=True)

            minibatch_steps = 0
            minibatch_kl = 0.0
            expected_minibatch_steps = sum(ep["length"] for ep in minibatch)
            for chunk_idx in range(0, len(minibatch), config.chunk_size):
                chunk = minibatch[chunk_idx : chunk_idx + config.chunk_size]
                chunk.sort(key=lambda ep: ep["length"], reverse=True)

                batch_loss, batch_metrics, batch_steps = _run_batched_ppo(
                    chunk, policy, config, policy.device, episode
                )

                tot_policy_loss += batch_metrics["policy_loss"]
                tot_value_loss += batch_metrics["value_loss"]
                tot_normalized_entropy += batch_metrics["normalized_entropy"]
                epoch_kl += batch_metrics["kl_div"]
                minibatch_kl += batch_metrics["kl_div"]
                tot_clip_frac += batch_metrics["clip_frac"]
                last_entropy_coef = batch_metrics["entropy_coef"]
                minibatch_steps += batch_steps

                if batch_steps > 0:
                    scaled_loss = batch_loss / expected_minibatch_steps
                    if torch.isfinite(scaled_loss):
                        scaled_loss.backward()
                    else:
                        logging.warning(
                            f"Non-finite chunk loss at episode {episode}, skipping backward "
                            f"for {batch_steps} steps (minibatch gradient will be undercounted)"
                        )

            if minibatch_steps > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.max_grad_norm
                )
                if torch.isfinite(grad_norm):
                    tot_grad_norm += grad_norm.item()
                    optimizer.step()
                    num_updates += 1
                else:
                    logging.warning("Non-finite grad norm, skipping update")
                    optimizer.zero_grad(set_to_none=True)

            epoch_steps += minibatch_steps

            should_stop, minibatch_mean_kl = _kl_exceeds_target(
                minibatch_kl, minibatch_steps, config.target_kl
            )
            if should_stop:
                logging.info(
                    f"Early stop at epoch {epoch_idx + 1}/{config.ppo_epochs}, "
                    f"batch {batch_start // effective_batch_size + 1} "
                    f"(minibatch KL={minibatch_mean_kl:.4f} > {config.target_kl:.4f})"
                )
                early_stop = True
                break

        tot_steps += epoch_steps
        tot_kl_div += epoch_kl
        epochs_done += 1

    if epochs_done == 0 or tot_steps == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "normalized_entropy": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
            "clip_fraction": 0.0,
            "entropy_coefficient": config.entropy_coef,
            "explained_variance": 0.0,
            "time": time.time() - t0,
        }

    return {
        "policy_loss": tot_policy_loss / tot_steps,
        "value_loss": tot_value_loss / tot_steps,
        "normalized_entropy": tot_normalized_entropy / tot_steps,
        "kl_divergence": tot_kl_div / tot_steps,
        "grad_norm": tot_grad_norm / num_updates if num_updates > 0 else 0.0,
        "clip_fraction": tot_clip_frac / tot_steps,
        "entropy_coefficient": last_entropy_coef,
        "explained_variance": explained_var,
        "time": time.time() - t0,
    }


def main():
    showdown_procs = []
    vec_env = None
    tb_writer = None

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE).to(device)

    optimizer = optim.AdamW(
        adamw_param_groups(policy, weight_decay=1e-4),
        lr=config.lr,
        eps=1e-6,
    )

    # to guarantee executor shutdown
    try:
        # build showdown once instead of potentially having multiple suprocesses
        # trying to build into dist at the same time
        logging.info("Building pokemon-showdown...")
        try:
            subprocess.run(
                ["node", "build"],
                cwd="pokemon-showdown",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            logging.info("Pokemon-showdown built successfully.")
        except Exception as e:
            logging.error(f"Failed to build pokemon-showdown: {e}")
            raise e

        # clean up other processes occupying the port
        for i in range(config.n_envs):
            port = 8000 + i
            try:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        # one showdown server per thread
        for i in range(config.n_envs):
            port = 8000 + i
            proc = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", "--skip-build", str(port)],
                cwd="pokemon-showdown",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            showdown_procs.append(proc)

        # wait and check for all showdown servers to be listening
        # rather than a flat timeout
        start_time = time.time()
        timeout = 30.0  # auto fail if not listening in these many secs
        pending_ports = [8000 + i for i in range(config.n_envs)]
        while pending_ports and (time.time() - start_time) < timeout:
            for port in list(pending_ports):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        pending_ports.remove(port)
                except OSError:
                    pass
            if pending_ports:
                time.sleep(0.5)

        if pending_ports:
            raise RuntimeError(f"Showdown servers failed to start on ports: {pending_ports}")
        logging.info("All showdown servers are ready.")

        envs = [SimEnv.build_env(env_id=i, server_port=8000 + i) for i in range(config.n_envs)]

        vec_env = ThreadVecEnv(envs)
        buffer = RolloutBuffer()
        scheduler = PPOScheduler(config)

        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        config.pool_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = SummaryWriter(log_dir="runs/ppo_training")

        start = load_checkpoint(config.checkpoint_path, policy, optimizer)
        if start is not None:
            logging.info(f"Resuming training from episode {start + 1}")
        else:
            seed_path = config.pool_dir / "seed_fuzzy_heuristic.pt"
            if seed_path.exists():
                logging.info(f"No checkpoint found. Seeding policy from {seed_path}")
                load_checkpoint(seed_path, policy)
            else:
                logging.info(
                    "No checkpoint or seed policy found. Starting from random initialization."
                )
            start = 0

        pool = OpponentPool.load_or_create(config.pool_dir, config)
        if len(pool) == 0:
            logging.info("Opponent pool empty, seeding with current policy as ep0")
            pool.add(policy, "ep0", 0.5)
            pool.save_state()

        logging.info(f"Opponent pool: {pool}")

        from src.train.rollout import create_trajectory_buffers

        vec_env.reset()
        state1 = initial_state(policy, config.n_envs, policy.device)
        state2 = initial_state(policy, config.n_envs, policy.device)
        trajectories1 = create_trajectory_buffers(config.n_envs)
        trajectories2 = create_trajectory_buffers(config.n_envs)
        env_opponents = ["self"] * config.n_envs
        active_pool_policies = {}

        # smoothed agent-vs-pool win rate for snapshot admission; a single
        # episode's pbt phase (~70 games) is too noisy to gate on
        pool_wr_ema = 0.5

        for episode in range(start, config.num_episodes):
            if shutdown_requested:
                logging.warning("Shutdown requested, saving checkpoint and exiting")
                save_checkpoint(config.checkpoint_path, episode, policy, optimizer)
                break

            lr = scheduler.lr(episode)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            config.entropy_coef = scheduler.entropy_coef(episode)

            buffer.reset()
            rollout_time = 0.0
            pool_wr = 0.0

            for target_mode in ["self_play", "pbt"]:
                policy.eval()

                t0_rollout = time.time()
                mode_pool_wr, state1, state2 = collect_rollouts(
                    vec_env,
                    policy,
                    buffer,
                    pool,
                    config,
                    active_pool_policies,
                    trajectories1,
                    trajectories2,
                    state1,
                    state2,
                    env_opponents,
                    target_mode,
                )
                rollout_time += time.time() - t0_rollout
                if target_mode == "pbt":
                    pool_wr = mode_pool_wr

            if not buffer.trajectories:
                logging.warning("No trajectories collected, skipping update")
                continue

            num_trajectories = len(buffer.trajectories)
            avg_traj_length = sum(ep["length"] for ep in buffer.trajectories) / num_trajectories
            avg_traj_time = rollout_time / num_trajectories

            rollout_data = buffer.get_batches(policy.device, config)
            stats = ppo_update(rollout_data, policy, optimizer, config, episode, shutdown_requested)

            alpha = config.pool_win_rate_smoothing
            pool_wr_ema = (1 - alpha) * pool_wr_ema + alpha * pool_wr

            if (episode + 1) % config.snapshot_interval == 0:
                snap_id = f"ep{episode + 1}"
                added = pool.add(policy, snap_id, pool_wr_ema)
                if added:
                    pool.save_state()
                    logging.info(f"Snapshot '{snap_id}' added to opponent pool. Pool: {pool}")
                else:
                    logging.info(
                        f"Snapshot '{snap_id}' not added to opponent pool "
                        f"(win rate EMA: {pool_wr_ema:.4f}). Pool: {pool}"
                    )

            current_lr = optimizer.param_groups[0]["lr"]
            is_warmup = episode < config.warmup_episodes
            tag = "Warmup" if is_warmup else "Train"

            tb_writer.add_scalar(f"{tag}/WinRate/Pool", pool_wr, episode + 1)
            tb_writer.add_scalar(f"{tag}/Loss/Policy", stats["policy_loss"], episode + 1)
            tb_writer.add_scalar(f"{tag}/Loss/Value", stats["value_loss"], episode + 1)
            tb_writer.add_scalar(
                f"{tag}/Loss/NormalizedEntropy", stats["normalized_entropy"], episode + 1
            )
            tb_writer.add_scalar(
                f"{tag}/Training/KL_Divergence", stats["kl_divergence"], episode + 1
            )
            tb_writer.add_scalar(f"{tag}/Training/GradNorm", stats["grad_norm"], episode + 1)
            tb_writer.add_scalar(
                f"{tag}/Training/ClipFraction", stats["clip_fraction"], episode + 1
            )
            tb_writer.add_scalar(
                f"{tag}/Training/ExplainedVariance", stats["explained_variance"], episode + 1
            )
            tb_writer.add_scalar(
                f"{tag}/Training/EntropyCoef", stats["entropy_coefficient"], episode + 1
            )
            tb_writer.add_scalar(f"{tag}/Training/LearningRate", current_lr, episode + 1)
            tb_writer.add_scalar(f"{tag}/Timing/Rollout", rollout_time, episode + 1)
            tb_writer.add_scalar(f"{tag}/Timing/Update", stats["time"], episode + 1)
            tb_writer.add_scalar(f"{tag}/Buffer/NumTrajectories", num_trajectories, episode + 1)
            tb_writer.add_scalar(f"{tag}/Buffer/AvgTrajectoryLength", avg_traj_length, episode + 1)
            tb_writer.add_scalar(f"{tag}/Buffer/AvgTrajectoryTime", avg_traj_time, episode + 1)

            # slightly shorter list of things logged to the screen.
            logging.info(
                f"Ep {episode + 1}/{config.num_episodes} ({tag[:1]}) | "
                f"Pi: {stats['policy_loss']:.4f} | "
                f"V: {stats['value_loss']:.4f} | "
                f"NormEnt: {stats['normalized_entropy']:.2%} | "
                f"Clip: {stats['clip_fraction']:.2%} | "
                f"KL: {stats['kl_divergence']:.4f} | "
                f"ATL: {avg_traj_length:.1f} | "
                f"ATT: {avg_traj_time:.2f}"
            )

            if (episode + 1) % 10 == 0:
                save_checkpoint(config.checkpoint_path, episode + 1, policy, optimizer)
                logging.info("Checkpoint saved.")

    finally:
        if vec_env is not None:
            vec_env.shutdown()
        if tb_writer is not None:
            tb_writer.close()

        if vec_env is not None:
            for env in vec_env.envs:
                try:
                    env.close()
                except Exception:
                    pass

        for proc in showdown_procs:
            proc.terminate()
            proc.wait()

        logging.info("Training loop terminated successfully.")


if __name__ == "__main__":
    main()
