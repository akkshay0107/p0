"""Pure PPO objective and stateless memory-window optimizer lifecycle."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from p0.model.architecture_contract import HISTORY_WINDOW, SERIES_SLOTS
from p0.model.cls_reducer import pack_history_tokens
from p0.model.policy import EncodedObs, PolicyNet
from p0.model.series_context import SeriesFeatures
from p0.model.structured_observation import StructuredObservation, is_teampreview
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.trajectory import TrajectoryBatch


def magnet_kl_per_step(live_logits: torch.Tensor, magnet_logits: torch.Tensor) -> torch.Tensor:
    """Return reverse KL between masked two-slot action distributions."""
    live_log_probs = F.log_softmax(live_logits.float(), dim=-1)
    magnet_log_probs = F.log_softmax(magnet_logits.float(), dim=-1)
    live_probs = live_log_probs.exp()
    terms = live_probs * (live_log_probs - magnet_log_probs)
    terms = torch.where(live_probs > 0, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1).sum(dim=-1)


def compute_ppo_objective(
    current_log_probs: torch.Tensor,
    current_values: torch.Tensor,
    normalized_entropy: torch.Tensor,
    magnet_kl: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    team_preview: torch.Tensor,
    config: TrainingConfig,
    *,
    alpha: float,
    critic_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-step total/policy/value losses, ratios, and log-ratios."""
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_low, 1.0 + config.clip_high) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    value_loss = F.mse_loss(current_values, returns, reduction="none")
    total = config.value_coef * value_loss
    if not critic_only:
        alpha_scale = alpha * torch.where(
            team_preview.reshape(-1), config.teampreview_alpha_mult, 1.0
        )
        total = total + policy_loss + alpha_scale * magnet_kl
        if config.residual_entropy_coef > 0.0:
            total = total - config.residual_entropy_coef * normalized_entropy
    total = torch.where(
        team_preview.reshape(-1),
        total * config.teampreview_loss_mult,
        total,
    )
    return total, policy_loss, value_loss, ratio, log_ratio


class PPOUpdater:
    """Own optimizer/scaler state while delegating memory-window evaluation."""

    def __init__(
        self,
        policy: PolicyNet,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        config: TrainingConfig,
        magnet: Magnet,
        *,
        cancel_requested: Callable[[], bool],
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.magnet = magnet
        self.cancel_requested = cancel_requested

    def update(
        self,
        trajectories: Sequence[TrajectoryBatch],
        episode: int,
        alpha: float,
    ) -> dict[str, Any]:
        return ppo_update(
            list(trajectories),
            self.policy,
            self.magnet,
            self.optimizer,
            self.scaler,
            self.config,
            episode,
            alpha,
            cancel_requested=self.cancel_requested,
        )


def _kl_exceeds_target(kl_sum: float, steps: int, target_kl: float) -> tuple[bool, float]:
    mean_kl = kl_sum / steps if steps > 0 else 0.0
    return mean_kl > target_kl, mean_kl


def _build_memory_inputs(
    policy: PolicyNet,
    encoded: EncodedObs,
    episodes: list[TrajectoryBatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-window and series inputs from one encoded trajectory batch."""
    dtype = encoded.tokens.dtype
    local_lists = []
    offset = 0
    for episode_item in episodes:
        local_lists.append(
            policy.local_history_tokens(
                encoded._replace(
                    tokens=encoded.tokens[offset : offset + episode_item.length],
                    aux=encoded.aux[offset : offset + episode_item.length],
                    numerical=encoded.numerical[offset : offset + episode_item.length],
                )
            )
        )
        offset += episode_item.length

    history_parts = []
    history_mask_parts = []
    history_age_parts = []
    for local_tokens in local_lists:
        for target in range(local_tokens.size(0)):
            left = max(0, target - HISTORY_WINDOW)
            packed, mask, ages = pack_history_tokens(local_tokens[left:target].unsqueeze(0))
            history_parts.append(packed[0])
            history_mask_parts.append(mask[0])
            history_age_parts.append(ages[0])
    history_tokens = torch.stack(history_parts)
    history_mask = torch.stack(history_mask_parts)
    history_age_ids = torch.stack(history_age_parts)

    encoded_series: list[torch.Tensor | None] = [None] * len(episodes)
    raw_series_indices = [
        index for index, episode_item in enumerate(episodes) if episode_item.series_features is not None
    ]
    if raw_series_indices:
        raw_feature_values: list[SeriesFeatures] = []
        for index in raw_series_indices:
            features = episodes[index].series_features
            if features is None:
                raise RuntimeError("series feature index construction drifted")
            raw_feature_values.append(features)
        raw_features = SeriesFeatures.stack(raw_feature_values)
        raw_encoded = policy.encode_series(raw_features)
        for batch_index, episode_index in enumerate(raw_series_indices):
            encoded_series[episode_index] = raw_encoded[batch_index]

    series_token_parts = []
    series_mask_parts = []
    for episode_index, episode_item in enumerate(episodes):
        encoded_series_item = encoded_series[episode_index]
        if encoded_series_item is not None:
            features = episode_item.series_features
            if features is None:
                raise RuntimeError("encoded series context is missing its source features")
            series_token_parts.append(
                encoded_series_item.to(device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(episode_item.length, -1, -1)
            )
            series_mask_parts.append(
                features.game_mask.to(device=device).unsqueeze(0).expand(episode_item.length, -1)
            )
        elif episode_item.series_tokens is None:
            series_token_parts.append(
                torch.zeros(
                    (episode_item.length, SERIES_SLOTS, policy.d_model),
                    device=device,
                    dtype=dtype,
                )
            )
            series_mask_parts.append(
                torch.zeros((episode_item.length, SERIES_SLOTS), device=device, dtype=torch.bool)
            )
        else:
            if episode_item.series_tokens.shape != (SERIES_SLOTS, policy.d_model):
                raise ValueError("Trajectory series_tokens do not match the policy contract")
            if episode_item.series_mask is None or episode_item.series_mask.shape != (SERIES_SLOTS,):
                raise ValueError("Trajectory series_mask does not match the policy contract")
            series_token_parts.append(
                episode_item.series_tokens.to(device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(episode_item.length, -1, -1)
            )
            series_mask_parts.append(
                episode_item.series_mask.to(device=device)
                .unsqueeze(0)
                .expand(episode_item.length, -1)
            )
    return (
        torch.cat(series_token_parts, dim=0),
        torch.cat(series_mask_parts, dim=0),
        history_tokens,
        history_mask,
        history_age_ids,
    )


def _run_batched_ppo(
    episodes: list[TrajectoryBatch],
    policy: PolicyNet,
    magnet: Magnet,
    config: TrainingConfig,
    device: torch.device,
    episode: int,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    """Evaluate one PPO minibatch with one reducer pass per decision."""
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

    is_warmup = episode < config.warmup_episodes

    all_obs = StructuredObservation.cat([ep.observations for ep in episodes], dim=0)
    all_action_masks = torch.cat([ep.action_masks for ep in episodes], dim=0).to(device)
    with autocast(device_type=device.type, enabled=config.enable_optim):
        all_enc = policy.encode(all_obs, all_action_masks)
    if is_warmup:
        all_enc = all_enc._replace(tokens=all_enc.tokens.detach(), aux=all_enc.aux.detach())

    actions = torch.cat([ep.actions for ep in episodes]).to(device)
    old_log_probs = torch.cat([ep.log_probs for ep in episodes]).to(device)
    advantages = torch.cat([ep.advantages for ep in episodes if ep.advantages is not None]).to(
        device
    )
    returns = torch.cat([ep.returns for ep in episodes if ep.returns is not None]).to(device)
    live_memory = _build_memory_inputs(policy, all_enc, episodes, device)
    magnet_enc: EncodedObs | None = None
    magnet_memory: tuple[torch.Tensor, ...] | None = None
    if not is_warmup:
        with torch.inference_mode(), autocast(
            device_type=device.type, enabled=config.enable_optim
        ):
            magnet_enc = magnet.policy.encode(all_obs, all_action_masks)
            magnet_memory = _build_memory_inputs(magnet.policy, magnet_enc, episodes, device)
    total_loss = torch.tensor(0.0, device=device)
    # convert them to python floats once at the end
    metrics: dict[str, Any] = {
        "policy_loss": torch.tensor(0.0, device=device),
        "value_loss": torch.tensor(0.0, device=device),
        "normalized_entropy": torch.tensor(0.0, device=device),
        "magnet_kl": torch.tensor(0.0, device=device),
        "kl_div": torch.tensor(0.0, device=device),
        "clip_frac": torch.tensor(0.0, device=device),
    }
    with autocast(device_type=device.type, enabled=config.enable_optim):
        out = policy.evaluate(
            all_enc,
            all_action_masks,
            actions,
            *live_memory,
            critic_only=is_warmup,
        )
        if not is_warmup:
            if magnet_enc is None or magnet_memory is None:
                raise RuntimeError("magnet evaluation inputs were not prepared")
            with torch.inference_mode():
                magnet_logits, _, _, _ = magnet.policy.actor.score(
                    magnet_enc,
                    all_action_masks,
                    actions,
                    *magnet_memory,
                )
            magnet_kl = magnet_kl_per_step(out.logits, magnet_logits)
        else:
            magnet_kl = torch.zeros_like(out.log_probs)
        step_loss, step_policy_loss, step_value_loss, ratio, log_ratio = compute_ppo_objective(
            out.log_probs,
            out.value,
            out.norm_entropy,
            magnet_kl,
            old_log_probs,
            advantages,
            returns,
            is_teampreview(all_enc.numerical),
            config,
            alpha=alpha,
            critic_only=is_warmup,
        )

    total_loss = step_loss.sum()
    total_steps = int(step_loss.numel())

    with torch.no_grad():
        metrics["policy_loss"] = (
            step_policy_loss.sum() if not is_warmup else torch.tensor(0.0, device=device)
        )
        metrics["value_loss"] = step_value_loss.sum()
        metrics["normalized_entropy"] = out.norm_entropy.sum()
        metrics["magnet_kl"] = magnet_kl.sum()
        metrics["kl_div"] = (
            ((ratio - 1) - log_ratio).sum() if not is_warmup else torch.tensor(0.0, device=device)
        )
        metrics["clip_frac"] = (
            ((ratio < 1 - config.clip_low) | (ratio > 1 + config.clip_high)).float().sum()
            if not is_warmup
            else torch.tensor(0.0, device=device)
        )

    for k in [
        "policy_loss",
        "value_loss",
        "normalized_entropy",
        "magnet_kl",
        "kl_div",
        "clip_frac",
    ]:
        metrics[k] = metrics[k].item()

    return total_loss, metrics, total_steps


def ppo_update(
    episodes: list[TrajectoryBatch],
    policy: PolicyNet,
    magnet: Magnet,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    episode: int,
    alpha: float,
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
    tot_magnet_kl = 0.0
    tot_kl_div = 0.0
    tot_grad_norm = 0.0
    tot_clip_frac = 0.0
    tot_steps = 0
    num_updates = 0
    num_skipped = 0
    epochs_done = 0

    effective_batch_size = config.minibatch_size * (
        (config.batch_size + config.minibatch_size - 1) // config.minibatch_size
    )

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

            for chunk_idx in range(0, len(minibatch), config.minibatch_size):
                if cancel_requested():
                    cancelled = True
                    break
                chunk = minibatch[chunk_idx : chunk_idx + config.minibatch_size]
                chunk.sort(key=lambda ep: ep.length, reverse=True)

                batch_loss, batch_metrics, batch_steps = _run_batched_ppo(
                    chunk, policy, magnet, config, policy.device, episode, alpha
                )

                tot_policy_loss += batch_metrics["policy_loss"]
                tot_value_loss += batch_metrics["value_loss"]
                tot_normalized_entropy += batch_metrics["normalized_entropy"]
                tot_magnet_kl += batch_metrics["magnet_kl"]
                epoch_kl += batch_metrics["kl_div"]
                minibatch_kl += batch_metrics["kl_div"]
                tot_clip_frac += batch_metrics["clip_frac"]
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
            "magnet_kl": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
            "clip_fraction": 0.0,
            "magnet_alpha": alpha,
            "explained_variance": 0.0,
            "skipped_minibatches": num_skipped,
            "time": time.time() - t0,
        }

    return {
        "policy_loss": tot_policy_loss / tot_steps,
        "value_loss": tot_value_loss / tot_steps,
        "normalized_entropy": tot_normalized_entropy / tot_steps,
        "magnet_kl": tot_magnet_kl / tot_steps,
        "kl_divergence": tot_kl_div / tot_steps,
        "grad_norm": tot_grad_norm / num_updates if num_updates > 0 else 0.0,
        "clip_fraction": tot_clip_frac / tot_steps,
        "magnet_alpha": alpha,
        "explained_variance": explained_var,
        "skipped_minibatches": num_skipped,
        "time": time.time() - t0,
    }
