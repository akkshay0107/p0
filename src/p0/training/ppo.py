"""Pure PPO objective and stateful optimizer lifecycle."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from p0.model.policy import EncodedObs, PolicyNet
from p0.model.structured_observation import (
    TOKEN_IDX_GLOBAL_FIELD,
    StructuredObservation,
    is_teampreview,
)
from p0.training.config import TrainingConfig
from p0.training.trajectory import TrajectoryBatch


def compute_ppo_objective(
    current_log_probs: torch.Tensor,
    current_values: torch.Tensor,
    normalized_entropy: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    team_preview: torch.Tensor,
    config: TrainingConfig,
    *,
    entropy_coefficient: float,
    critic_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-step total/policy/value losses, ratios, and log-ratios."""
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_low, 1.0 + config.clip_high) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    value_loss = F.mse_loss(current_values, returns, reduction="none")
    entropy_scale = entropy_coefficient * torch.where(
        team_preview.reshape(-1), config.teampreview_entropy_mult, 1.0
    )
    total = config.value_coef * value_loss
    if not critic_only:
        total = total + policy_loss - entropy_scale * normalized_entropy
    total = torch.where(
        team_preview.reshape(-1),
        total * config.teampreview_loss_mult,
        total,
    )
    return total, policy_loss, value_loss, ratio, log_ratio


class PPOUpdater:
    """Own optimizer/scaler state while delegating recurrent evaluation."""

    def __init__(
        self,
        policy: PolicyNet,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        config: TrainingConfig,
        *,
        cancel_requested: Callable[[], bool],
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.cancel_requested = cancel_requested

    def update(
        self,
        trajectories: Sequence[TrajectoryBatch],
        episode: int,
        entropy_coefficient: float,
    ) -> dict[str, Any]:
        return ppo_update(
            list(trajectories),
            self.policy,
            self.optimizer,
            self.scaler,
            self.config,
            episode,
            entropy_coefficient,
            cancel_requested=self.cancel_requested,
        )


def _kl_exceeds_target(kl_sum: float, steps: int, target_kl: float) -> tuple[bool, float]:
    mean_kl = kl_sum / steps if steps > 0 else 0.0
    return mean_kl > target_kl, mean_kl


def _run_batched_ppo(
    episodes: list[TrajectoryBatch],
    policy: PolicyNet,
    config: TrainingConfig,
    device: torch.device,
    episode: int,
    entropy_coef: float,
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
    lengths = [ep.length for ep in episodes]
    max_steps = lengths[0]
    is_warmup = episode < config.warmup_episodes

    all_obs = StructuredObservation.cat([ep.observations for ep in episodes], dim=0)
    all_action_masks = torch.cat([ep.action_masks for ep in episodes], dim=0).to(device)
    with autocast(device_type=device.type, enabled=config.enable_optim):
        all_enc = policy.encode(all_obs, all_action_masks)
    if is_warmup:
        all_enc = EncodedObs(
            all_enc.tokens.detach(),
            all_enc.aux.detach(),
            all_enc.numerical,
        )
    split_sizes = [ep.length for ep in episodes]
    tokens_list = torch.split(all_enc.tokens, split_sizes)
    aux_list = torch.split(all_enc.aux, split_sizes)
    numerical_list = torch.split(all_enc.numerical, split_sizes)

    # time major padding
    enc_p = EncodedObs(
        tokens=torch.nn.utils.rnn.pad_sequence(list(tokens_list)),
        aux=torch.nn.utils.rnn.pad_sequence(list(aux_list)),
        numerical=torch.nn.utils.rnn.pad_sequence(list(numerical_list)),
    )

    # pre-pack non-observation tensors for fast slicing [Batch, Time, ...]
    def pack(fields):
        return torch.nn.utils.rnn.pad_sequence(fields).to(device)

    actions_p = pack([ep.actions for ep in episodes])
    old_log_probs_p = pack([ep.log_probs for ep in episodes])
    advantages_p = pack([ep.advantages for ep in episodes])
    returns_p = pack([ep.returns for ep in episodes])
    action_masks_p = pack([ep.action_masks for ep in episodes])

    state = policy.initial_state(batch_size)
    total_loss = torch.tensor(0.0, device=device)
    # convert them to python floats once at the end
    metrics = {
        "policy_loss": torch.tensor(0.0, device=device),
        "value_loss": torch.tensor(0.0, device=device),
        "normalized_entropy": torch.tensor(0.0, device=device),
        "kl_div": torch.tensor(0.0, device=device),
        "clip_frac": torch.tensor(0.0, device=device),
        "entropy_coef": 0.0,
    }
    total_steps = 0
    curr_ent_coef = entropy_coef

    for t in range(max_steps):
        active_n = sum(1 for length in lengths if length > t)
        if active_n == 0:
            break

        enc_t = enc_p.step(active_n, t)
        actions_t = actions_p[t, :active_n]
        old_log_probs_t = old_log_probs_p[t, :active_n]
        advantages_t = advantages_p[t, :active_n]
        returns_t = returns_p[t, :active_n]
        action_masks_t = action_masks_p[t, :active_n]
        is_tp_t = is_teampreview(enc_t.numerical)

        curr_state = state[:active_n]
        with autocast(device_type=device.type, enabled=config.enable_optim):
            out = policy.evaluate(
                enc_t,
                action_masks_t,
                actions_t,
                curr_state,
                critic_only=is_warmup,
            )
            curr_log_prob = out.log_probs
            curr_normalized_entropy = out.norm_entropy
            curr_val = out.value
            next_state = out.state.to(torch.float32)

            if not torch.isfinite(curr_log_prob).all():
                non_finite_idx = (~torch.isfinite(curr_log_prob)).nonzero(as_tuple=True)[0]
                for idx in non_finite_idx:
                    logging.error(
                        f"DEBUG: Non-finite log_prob at step {t}, episode element {idx}.\n"
                        f"  actions_t: {actions_t[idx].tolist()}\n"
                        f"  action_masks_t (p1): {action_masks_t[idx, 0].nonzero().squeeze(-1).tolist()}\n"
                        f"  action_masks_t (p2): {action_masks_t[idx, 1].nonzero().squeeze(-1).tolist()}\n"
                        f"  is_tp_t: {is_tp_t[idx].item()}\n"
                        "  numerical_t[:, TOKEN_IDX_GLOBAL_FIELD, :4]: "
                        f"{enc_t.numerical[idx, TOKEN_IDX_GLOBAL_FIELD, :4].tolist()}\n"
                        f"  old_log_prob: {old_log_probs_t[idx].item()}\n"
                        f"  curr_log_prob: {curr_log_prob[idx].item()}\n"
                    )

            step_loss, step_policy_loss, step_value_loss, ratio, log_ratio = compute_ppo_objective(
                curr_log_prob,
                curr_val,
                curr_normalized_entropy,
                old_log_probs_t,
                advantages_t,
                returns_t,
                is_tp_t,
                config,
                entropy_coefficient=curr_ent_coef,
                critic_only=is_warmup,
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
            metrics["policy_loss"] += (
                step_policy_loss.sum() if not is_warmup else torch.tensor(0.0, device=device)
            )
            metrics["value_loss"] += step_value_loss.sum()

            metrics["normalized_entropy"] += curr_normalized_entropy.sum()

            metrics["kl_div"] += (
                ((ratio - 1) - log_ratio).sum()
                if not is_warmup
                else torch.tensor(0.0, device=device)
            )
            metrics["clip_frac"] += (
                ((ratio < 1 - config.clip_low) | (ratio > 1 + config.clip_high)).float().sum()
                if not is_warmup
                else torch.tensor(0.0, device=device)
            )
            metrics["entropy_coef"] = curr_ent_coef

    for k in ["policy_loss", "value_loss", "normalized_entropy", "kl_div", "clip_frac"]:
        metrics[k] = metrics[k].item()

    return total_loss, metrics, total_steps


def ppo_update(
    episodes: list[TrajectoryBatch],
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    episode: int,
    entropy_coef: float,
    cancel_requested: Callable[[], bool],
) -> dict:
    policy.train()
    t0 = time.time()

    with torch.no_grad():
        all_returns = torch.cat([ep.returns for ep in episodes if ep.returns is not None])
        all_values = torch.cat([ep.values for ep in episodes])
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
    num_skipped = 0
    epochs_done = 0

    effective_batch_size = config.chunk_size * (
        (config.batch_size + config.chunk_size - 1) // config.chunk_size
    )  # round up to nearest chunk

    last_entropy_coef = entropy_coef
    for epoch_idx in range(config.ppo_epochs):
        if cancel_requested():
            break
        random.shuffle(episodes)

        epoch_steps = 0
        epoch_kl = 0.0

        for batch_start in range(0, len(episodes), effective_batch_size):
            if cancel_requested():
                break

            minibatch = episodes[batch_start : batch_start + effective_batch_size]
            if not minibatch:
                continue

            optimizer.zero_grad(set_to_none=True)

            minibatch_steps = 0
            minibatch_kl = 0.0
            expected_minibatch_steps = sum(ep.length for ep in minibatch)
            should_skip = False
            cancelled = False
            minibatch_mean_kl = 0.0

            for chunk_idx in range(0, len(minibatch), config.chunk_size):
                if cancel_requested():
                    cancelled = True
                    break
                chunk = minibatch[chunk_idx : chunk_idx + config.chunk_size]
                chunk.sort(key=lambda ep: ep.length, reverse=True)

                batch_loss, batch_metrics, batch_steps = _run_batched_ppo(
                    chunk, policy, config, policy.device, episode, entropy_coef
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
                        scaler.scale(scaled_loss).backward()
                    else:
                        logging.warning(
                            f"Non-finite chunk loss at episode {episode}, skipping backward "
                            f"for {batch_steps} steps (minibatch gradient will be undercounted)"
                        )

                # Check KL early at the chunk level to save processing remaining chunks
                should_skip, minibatch_mean_kl = _kl_exceeds_target(
                    minibatch_kl, minibatch_steps, config.target_kl
                )
                if should_skip:
                    break

            if cancelled:
                optimizer.zero_grad(set_to_none=True)
                break
            if minibatch_steps > 0:
                if should_skip:
                    logging.info(
                        f"Skipping minibatch at epoch {epoch_idx + 1}/{config.ppo_epochs}, "
                        f"batch {batch_start // effective_batch_size + 1} "
                        f"(minibatch KL={minibatch_mean_kl:.4f} > {config.target_kl:.4f})"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    num_skipped += 1
                else:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), config.max_grad_norm
                    )
                    scale_before_update = scaler.get_scale()
                    if not torch.isfinite(grad_norm):
                        logging.warning(
                            "Non-finite grad norm detected; scaler will skip this step "
                            f"(loss scale={scale_before_update:.0f})"
                        )
                    else:
                        tot_grad_norm += grad_norm.item()
                    scaler.step(optimizer)
                    scaler.update()
                    if torch.isfinite(grad_norm):
                        num_updates += 1

            epoch_steps += minibatch_steps

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
            "entropy_coefficient": entropy_coef,
            "explained_variance": 0.0,
            "skipped_minibatches": num_skipped,
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
        "skipped_minibatches": num_skipped,
        "time": time.time() - t0,
    }
