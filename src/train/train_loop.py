import logging
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import observation_builder
from env import SimEnv
from lookups import ACT_SIZE, OBS_DIM
from policy import PolicyNet
from ppo_utils import (
    OpponentPool,
    RolloutBuffer,
    initial_state,
    load_checkpoint,
    load_config,
    save_checkpoint,
)

config = load_config()
_buffer_lock = threading.Lock()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler(sys.stdout)],
)

shutdown_requested = False


def handle_sigterm(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        logging.warning("Second shutdown signal received, forcing immediate exit...")
        os._exit(1)
    logging.warning("SIGTERM received, requesting shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigterm)

policy = PolicyNet(obs_dim=OBS_DIM, act_size=ACT_SIZE)
optimizer = optim.AdamW(policy.parameters(), lr=config.lr, eps=1e-6, weight_decay=1e-4)


def lr_lambda(episode):
    if episode < config.warmup_episodes:
        return 1.0
    decay_total = config.num_episodes - config.warmup_episodes
    if decay_total <= 0:
        return 1.0
    decay_progress = (episode - config.warmup_episodes) / decay_total
    end_factor = config.min_lr / config.lr
    return max(end_factor, 1.0 - (1.0 - end_factor) * decay_progress)


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

if config.compile_policy and policy.device.type == "cuda":
    policy = cast(PolicyNet, torch.compile(policy))


def state_to_cpu(state):
    cls, hg = state
    return cls.detach().cpu(), hg.detach().cpu()


def cat_states(state_a, state_b):
    return torch.cat([state_a[0], state_b[0]], dim=0), torch.cat([state_a[1], state_b[1]], dim=0)


def split_state(state, idx: int):
    cls, hg = state
    return cls[idx : idx + 1], hg[idx : idx + 1]


@torch.inference_mode()
def collect_rollout(
    env,
    buffer: RolloutBuffer,
    opponent_policy: PolicyNet,
    is_self_play: bool = False,
    episode: int = 0,
) -> tuple[bool, bool]:
    obs, _ = env.reset()
    agent1 = env.agent1.username
    agent2 = env.agent2.username

    traj1 = []
    traj2 = []

    state1 = initial_state(policy, 1, policy.device)
    if is_self_play:
        state2 = initial_state(policy, 1, policy.device)
    else:
        state2 = initial_state(opponent_policy, 1, opponent_policy.device)

    final_rewards = None

    while True:
        obs1 = obs[agent1].unsqueeze(0).to(policy.device, non_blocking=True)
        mask1 = (
            observation_builder.get_action_mask(env.battle1)
            .unsqueeze(0)
            .to(policy.device, non_blocking=True)
        )

        # store both sides of the game if self play
        if is_self_play:
            obs2 = obs[agent2].unsqueeze(0).to(policy.device, non_blocking=True)
            mask2 = (
                observation_builder.get_action_mask(env.battle2)
                .unsqueeze(0)
                .to(policy.device, non_blocking=True)
            )

            combined_obs = torch.cat([obs1, obs2], dim=0)
            combined_masks = torch.cat([mask1, mask2], dim=0)
            combined_state = cat_states(state1, state2)

            _, combined_log_probs, combined_actions, combined_values, next_combined_state = policy(
                combined_obs,
                combined_state,
                combined_masks,
                sample_actions=True,
            )

            action1 = combined_actions[0:1]
            action2 = combined_actions[1:2]
            log_probs1 = combined_log_probs[0:1]
            log_probs2 = combined_log_probs[1:2]
            values1 = combined_values[0:1]
            values2 = combined_values[1:2]
            next_state1 = split_state(next_combined_state, 0)
            next_state2 = split_state(next_combined_state, 1)
        else:
            obs2 = obs[agent2].unsqueeze(0).to(opponent_policy.device, non_blocking=True)
            mask2 = (
                observation_builder.get_action_mask(env.battle2)
                .unsqueeze(0)
                .to(opponent_policy.device, non_blocking=True)
            )

            _, log_probs1, action1, values1, next_state1 = policy(
                obs1,
                state1,
                mask1,
                sample_actions=True,
            )
            _, _, action2, _, next_state2 = opponent_policy(
                obs2,
                state2,
                mask2,
                sample_actions=True,
            )

        actions = {
            agent1: action1[0].cpu().numpy(),
            agent2: action2[0].cpu().numpy(),
        }

        is_tp1 = env.battle1.teampreview
        is_tp2 = env.battle2.teampreview if is_self_play else None

        if shutdown_requested:
            break
        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        done = bool(
            terminated[agent1] or truncated[agent1] or terminated[agent2] or truncated[agent2]
        )

        traj1.append(
            {
                "obs": obs1.cpu(),
                "actions": action1.cpu(),
                "log_probs": log_probs1.cpu(),
                "values": values1.cpu(),
                "rewards": torch.tensor([rewards[agent1]], dtype=torch.float32),
                "dones": torch.tensor([done], dtype=torch.float32),
                "action_masks": mask1.cpu(),
                "is_team_preview": torch.tensor([is_tp1], dtype=torch.bool),
            }
        )

        if is_self_play:
            traj2.append(
                {
                    "obs": obs2.cpu(),
                    "actions": action2.cpu(),
                    "log_probs": log_probs2.cpu(),
                    "values": values2.cpu(),
                    "rewards": torch.tensor([rewards[agent2]], dtype=torch.float32),
                    "dones": torch.tensor([done], dtype=torch.float32),
                    "action_masks": mask2.cpu(),
                    "is_team_preview": torch.tensor([is_tp2], dtype=torch.bool),
                }
            )

        if done or shutdown_requested:
            final_rewards = rewards
            break

        obs = next_obs
        state1 = next_state1
        state2 = next_state2

    if shutdown_requested:
        # If interrupted, we don't return a valid win/loss as the game didn't finish.
        return False, is_self_play

    with _buffer_lock:
        buffer.add_episode(traj1)
        if is_self_play:
            buffer.add_episode(traj2)

    winner = bool(final_rewards[agent1] > final_rewards[agent2])
    return winner, is_self_play


def collect_all_rollouts(
    envs, buffer: RolloutBuffer, executor, pool: OpponentPool, episode: int = 0
):
    env_queue = queue.Queue()
    for env in envs:
        env_queue.put(env)

    # Pre-sample one opponent per rollout slot.
    sampled_opponents = []
    for _ in range(config.rollouts_per_episode):
        if random.random() < config.self_play_prob:
            sampled_opponents.append((policy, "latest"))
        else:
            sampled_opponents.append(pool.sample())

    def worker(opponent_policy: PolicyNet, opponent_id: str):
        env = env_queue.get()
        try:
            won, was_self_play = collect_rollout(
                env,
                buffer,
                opponent_policy,
                is_self_play=(opponent_id == "latest"),
                episode=episode,
            )
            return opponent_id, won, was_self_play
        finally:
            env_queue.put(env)

    pool_wins = 0
    pool_total = 0

    futures = [
        executor.submit(worker, opp_policy, opp_id) for opp_policy, opp_id in sampled_opponents
    ]

    for f in as_completed(futures):
        if shutdown_requested:
            # Cancel all pending futures if possible
            for future in futures:
                future.cancel()
            break
        opp_id, won, was_self_play = f.result()
        if not was_self_play:
            pool_wins += int(won)
            pool_total += 1
            pool.update_win_rate(opp_id, won)

    return {
        "pool_win_rate": pool_wins / pool_total if pool_total > 0 else 0.0,
    }


def _run_batched_ppo(
    episodes: list[dict], device: torch.device, episode: int
) -> tuple[torch.Tensor, dict[str, float], int]:
    """
    Run PPO BPTT over a minibatch of variable-length episodes.
    Expects episodes to already be sorted by length so the active recurrent
    batch is always a contiguous prefix that shrinks as shorter games finish.
    """
    if not episodes:
        return (
            torch.tensor(0.0, device=device),
            {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "kl_div": 0.0,
                "clip_frac": 0.0,
            },
            0,
        )

    batch_size = len(episodes)
    lengths = torch.tensor([ep["length"] for ep in episodes], device=device)
    max_steps = int(lengths[0].item())

    obs = [ep["obs"] for ep in episodes]
    actions = [ep["actions"] for ep in episodes]
    old_log_probs = [ep["log_probs"] for ep in episodes]
    advantages = [ep["advantages"] for ep in episodes]
    returns = [ep["returns"] for ep in episodes]
    action_masks = [ep["action_masks"] for ep in episodes]
    is_tp = [ep["is_team_preview"] for ep in episodes]

    state = initial_state(policy, batch_size, device)
    total_loss = torch.tensor(0.0, device=device)
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
        "normalized_entropy": 0.0,
        "kl_div": 0.0,
        "clip_frac": 0.0,
        "entropy_coef": 0.0,
    }
    total_steps = 0

    # annealed entropy coefficient
    anneal_steps = config.num_episodes - config.entropy_anneal_start
    if episode <= config.entropy_anneal_start or anneal_steps <= 0:
        curr_ent_coef = config.entropy_coef
    else:
        progress = (episode - config.entropy_anneal_start) / anneal_steps
        curr_ent_coef = (
            config.entropy_coef - (config.entropy_coef - config.entropy_coef_floor) * progress
        )
        curr_ent_coef = max(curr_ent_coef, config.entropy_coef_floor)

    for t in range(max_steps):
        active_n = int((lengths > t).sum().item())
        if active_n == 0:
            break

        obs_t = torch.cat([ep_obs[t : t + 1] for ep_obs in obs[:active_n]], dim=0)
        actions_t = torch.cat([ep_actions[t : t + 1] for ep_actions in actions[:active_n]], dim=0)
        old_log_probs_t = torch.cat(
            [ep_log_probs[t : t + 1] for ep_log_probs in old_log_probs[:active_n]], dim=0
        )
        advantages_t = torch.cat(
            [ep_advantages[t : t + 1] for ep_advantages in advantages[:active_n]], dim=0
        )
        returns_t = torch.cat([ep_returns[t : t + 1] for ep_returns in returns[:active_n]], dim=0)
        action_masks_t = torch.cat(
            [ep_action_masks[t : t + 1] for ep_action_masks in action_masks[:active_n]], dim=0
        )
        is_tp_t = torch.cat([ep_is_tp[t : t + 1] for ep_is_tp in is_tp[:active_n]], dim=0)

        curr_state = (state[0][:active_n], state[1][:active_n])
        curr_log_prob, curr_entropy, curr_normalized_entropy, curr_val, next_state = (
            policy.evaluate_actions(
                obs_t,
                actions_t,
                action_masks_t,
                state=curr_state,
            )
        )

        log_ratio = curr_log_prob - old_log_probs_t
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages_t

        step_policy_loss = -torch.min(surr1, surr2)
        step_value_loss = F.mse_loss(curr_val, returns_t, reduction="none")
        step_entropy_loss = -curr_entropy

        is_tp_mask = is_tp_t.reshape(-1)
        step_ent_coef = curr_ent_coef * torch.where(
            is_tp_mask, config.teampreview_entropy_mult, 1.0
        )

        is_warmup = episode < config.warmup_episodes
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

        state = next_state

        with torch.no_grad():
            metrics["policy_loss"] += step_policy_loss.sum().item() if not is_warmup else 0.0
            metrics["value_loss"] += step_value_loss.sum().item()
            metrics["entropy_loss"] += step_entropy_loss.sum().item() if not is_warmup else 0.0

            metrics["normalized_entropy"] += curr_normalized_entropy.sum().item()

            # schulman kl approx (was having negative kl in early steps)
            metrics["kl_div"] += ((ratio - 1) - log_ratio).sum().item() if not is_warmup else 0.0
            metrics["clip_frac"] += (
                ((ratio - 1.0).abs() > config.clip_range).float().sum().item()
                if not is_warmup
                else 0.0
            )
            metrics["entropy_coef"] = curr_ent_coef  # same for all steps in minibatch

    return total_loss, metrics, total_steps


def ppo_update(episodes: list, episode: int) -> dict:
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
    tot_entropy_loss = 0.0
    tot_normalized_entropy = 0.0
    tot_kl_div = 0.0
    tot_grad_norm = 0.0
    tot_clip_frac = 0.0
    tot_steps = 0
    num_updates = 0
    epochs_done = 0

    early_stop = False
    for epoch_idx in range(config.ppo_epochs):
        if shutdown_requested or early_stop:
            break
        random.shuffle(episodes)

        epoch_steps = 0
        epoch_kl = 0.0

        for batch_start in range(0, len(episodes), config.batch_size):
            if shutdown_requested:
                break
            batch = episodes[batch_start : batch_start + config.batch_size]
            batch.sort(key=lambda ep: ep["length"], reverse=True)

            optimizer.zero_grad(set_to_none=True)

            batch_loss, batch_metrics, batch_steps = _run_batched_ppo(batch, policy.device, episode)

            tot_policy_loss += batch_metrics["policy_loss"]
            tot_value_loss += batch_metrics["value_loss"]
            tot_entropy_loss += batch_metrics["entropy_loss"]
            tot_normalized_entropy += batch_metrics["normalized_entropy"]
            epoch_kl += batch_metrics["kl_div"]
            tot_clip_frac += batch_metrics["clip_frac"]
            last_entropy_coef = batch_metrics["entropy_coef"]

            if batch_steps > 0:
                scaled_loss = batch_loss / batch_steps
                if not torch.isfinite(scaled_loss):
                    logging.warning(
                        "Skipping PPO batch with non-finite loss at episode %s, epoch %s, batch %s",
                        episode,
                        epoch_idx + 1,
                        batch_start // config.batch_size + 1,
                    )
                    continue

                scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    logging.warning(
                        "Skipping PPO batch with non-finite grad norm at episode %s, epoch %s, batch %s",
                        episode,
                        epoch_idx + 1,
                        batch_start // config.batch_size + 1,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                tot_grad_norm += grad_norm.item()
                optimizer.step()
                num_updates += 1

            epoch_steps += batch_steps

            if epoch_steps > 0:
                avg_kl = epoch_kl / epoch_steps
                if avg_kl > config.target_kl:
                    logging.info(
                        f"Early stop at epoch {epoch_idx + 1}/{config.ppo_epochs}, "
                        f"batch {batch_start // config.batch_size + 1} "
                        f"(KL={avg_kl:.4f} > {config.target_kl:.4f})"
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
            "entropy_loss": 0.0,
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
        "entropy_loss": tot_entropy_loss / tot_steps,
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
    envs = []
    executor = None
    tb_writer = None

    # to guarantee executor shutdown
    try:
        for i in range(config.num_envs):
            port = 8000 + i
            proc = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", str(port)],
                cwd="pokemon-showdown",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            showdown_procs.append(proc)

        # Give the servers a moment to start
        time.sleep(10)

        envs = [
            SimEnv.build_env(env_id=i, server_port=8000 + (i % config.num_envs))
            for i in range(config.n_jobs)
        ]
        buffer = RolloutBuffer()
        executor = ThreadPoolExecutor(max_workers=config.n_jobs)

        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        config.pool_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = SummaryWriter(log_dir="runs/ppo_training")

        start = load_checkpoint(config.checkpoint_path, policy, optimizer, scheduler)

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
            pool.add(policy, "ep0")
            pool.save_state()

        logging.info(f"Opponent pool: {pool}")

        for episode in range(start, config.num_episodes):
            if shutdown_requested:
                logging.warning("Shutdown requested, saving checkpoint and exiting")
                save_checkpoint(config.checkpoint_path, episode, policy, optimizer, scheduler)
                break

            buffer.reset()
            policy.eval()

            t0_rollout = time.time()
            rollout_stats = collect_all_rollouts(envs, buffer, executor, pool, episode=episode)
            rollout_time = time.time() - t0_rollout

            if not buffer.trajectories:
                logging.warning("No trajectories collected, skipping update")
                continue

            rollout_data = buffer.get_batches(policy.device, config)
            stats = ppo_update(rollout_data, episode)
            scheduler.step()

            if (episode + 1) % config.snapshot_interval == 0:
                snap_id = f"ep{episode + 1}"
                pool.add(policy, snap_id)
                pool.save_state()
                logging.info(f"Snapshot '{snap_id}' added to opponent pool. Pool: {pool}")

            current_lr = scheduler.get_last_lr()[0]
            is_warmup = episode < config.warmup_episodes
            tag = "Warmup" if is_warmup else "Train"

            tb_writer.add_scalar(f"{tag}/WinRate/Pool", rollout_stats["pool_win_rate"], episode + 1)
            tb_writer.add_scalar(f"{tag}/Loss/Policy", stats["policy_loss"], episode + 1)
            tb_writer.add_scalar(f"{tag}/Loss/Value", stats["value_loss"], episode + 1)
            tb_writer.add_scalar(f"{tag}/Loss/Entropy", stats["entropy_loss"], episode + 1)
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
            tb_writer.add_scalar(
                f"{tag}/Buffer/NumTrajectories", len(buffer.trajectories), episode + 1
            )

            logging.info(
                f"Ep {episode + 1}/{config.num_episodes} ({tag[:1]}) | "
                f"Pool WR: {rollout_stats['pool_win_rate']:.1%} | "
                f"P-Loss: {stats['policy_loss']:.4f} | "
                f"V-Loss: {stats['value_loss']:.4f} | "
                f"Entropy: {-stats['entropy_loss']:.4f} | "
                f"NormEnt: {stats['normalized_entropy']:.2%} | "
                f"Grad: {stats['grad_norm']:.2f} | "
                f"Clip: {stats['clip_fraction']:.2%} | "
                f"ExpVar: {stats['explained_variance']:.4f} | "
                f"KL: {stats['kl_divergence']:.4f}"
            )

            if (episode + 1) % 10 == 0:
                save_checkpoint(config.checkpoint_path, episode + 1, policy, optimizer, scheduler)
                logging.info("Checkpoint saved.")

    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if tb_writer is not None:
            tb_writer.close()
        for env in envs:
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
