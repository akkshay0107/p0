"""PPO loss computation and update loop."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast

from p0.model.architecture_contract import (
    HISTORY_WINDOW,
    MAX_PRIOR_GAMES,
    SERIES_SLOTS,
    SERIES_TOKENS_PER_GAME,
)
from p0.model.policy import EncodedObs, MemoryInputs, PolicyNet
from p0.model.structured_observation import StructuredObservation
from p0.training.config import TrainingConfig
from p0.training.magnet import Magnet
from p0.training.series_history import SeriesHistorySnapshot
from p0.training.trajectory import PreparedTrajectory
from p0.training.utils import OptimizationPrecision, select_optimization_precision

LOGGER = logging.getLogger(__name__)
KL_METRIC_INDEX = 4


def magnet_kl_per_step(live_logits: torch.Tensor, magnet_logits: torch.Tensor) -> torch.Tensor:
    """Return reverse KL between masked two-slot action distributions."""
    live_log_probs = F.log_softmax(live_logits.float(), dim=-1)
    magnet_log_probs = F.log_softmax(magnet_logits.float(), dim=-1)
    live_probs = live_log_probs.exp()

    safe_live_log_probs = torch.where(live_probs > 0, live_log_probs, 0.0)
    safe_magnet_log_probs = torch.where(live_probs > 0, magnet_log_probs, 0.0)

    terms = live_probs * (safe_live_log_probs - safe_magnet_log_probs)
    return terms.sum(dim=-1).sum(dim=-1)


def compute_ppo_objective(
    current_log_probs: torch.Tensor,
    current_values: torch.Tensor,
    normalized_entropy: torch.Tensor,
    magnet_kl: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: TrainingConfig,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-step total/policy/value losses, ratios, and log-ratios."""
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
    policy_loss = -torch.min(unclipped, clipped)

    value_loss = F.mse_loss(current_values, returns, reduction="none")
    total = policy_loss + config.value_coef * value_loss + alpha * magnet_kl
    total = total - config.entropy_coef * normalized_entropy

    return total, policy_loss, value_loss, ratio, log_ratio


def _kl_exceeds_target(kl_sum: Tensor, steps: int, target_kl: float) -> tuple[bool, float]:
    """Check if the mean KL divergence exceeds the target early-stopping threshold."""
    mean_kl_value = float((kl_sum / steps).detach().item())
    return mean_kl_value > target_kl, mean_kl_value


def _build_memory_inputs(
    policy: PolicyNet,
    encoded: EncodedObs,
    episodes: Sequence[PreparedTrajectory],
    device: torch.device,
) -> MemoryInputs:
    """Build current-game and prior-game history memory inputs."""
    dtype = encoded.tokens.dtype
    # Use local summaries as history inputs for subsequent decisions.
    # Do not reuse the memory-aware readout to avoid leaking prior context into itself.
    local_tokens = encoded.local_history_token
    lengths = torch.tensor([item.length for item in episodes], device=device, dtype=torch.long)
    starts = torch.cat((torch.zeros(1, device=device, dtype=torch.long), lengths.cumsum(0)[:-1]))
    targets = torch.arange(encoded.tokens.size(0), device=device)
    episode_starts = torch.repeat_interleave(starts, lengths)
    history_offsets = torch.arange(-HISTORY_WINDOW, 0, device=device)
    history_indices = targets[:, None] + history_offsets[None, :]
    history_mask = history_indices >= episode_starts[:, None]
    history_indices = history_indices.clamp_min(0)
    history_tokens = local_tokens[history_indices] * history_mask.unsqueeze(-1)
    decision_count = encoded.tokens.size(0)
    histories: list[Tensor] = []
    history_rows: dict[int, int] = {}
    episode_history_rows_data = [[-1] * MAX_PRIOR_GAMES for _ in episodes]
    for episode_index, episode in enumerate(episodes):
        snapshot: SeriesHistorySnapshot = episode.series_history
        if len(snapshot) > MAX_PRIOR_GAMES:
            raise ValueError(f"A trajectory cannot expose more than {MAX_PRIOR_GAMES} prior games")
        for game_index, history in enumerate(snapshot):
            identity = id(history)
            row = history_rows.get(identity)
            if row is None:
                row = len(histories)
                history_rows[identity] = row
                histories.append(history.detach().to(device="cpu", dtype=torch.float32))
            episode_history_rows_data[episode_index][game_index] = row

    if histories:
        history_lengths = torch.tensor([history.size(0) for history in histories], dtype=torch.long)
        padded_cpu = torch.nn.utils.rnn.pad_sequence(histories, batch_first=True)
        mask_cpu = torch.arange(padded_cpu.size(1)).unsqueeze(0) < history_lengths.unsqueeze(1)
        padded = padded_cpu.to(device=device, dtype=dtype)
        mask = mask_cpu.to(device=device)
        summaries = policy.series(padded, mask)

        episode_history_rows = torch.tensor(
            episode_history_rows_data, dtype=torch.long, device=device
        )
        valid = episode_history_rows >= 0
        selected = summaries[episode_history_rows.clamp_min(0)]
        selected = selected.masked_fill(~valid[:, :, None, None], 0.0)
        per_episode_tokens = selected.flatten(1, 2)
        per_episode_mask = torch.arange(SERIES_SLOTS, device=device).unsqueeze(0) < (
            valid.sum(dim=1, keepdim=True) * SERIES_TOKENS_PER_GAME
        )
        series_tokens = torch.repeat_interleave(per_episode_tokens, lengths, dim=0)
        series_mask = torch.repeat_interleave(per_episode_mask, lengths, dim=0)
    else:
        series_tokens = torch.zeros(
            (decision_count, SERIES_SLOTS, policy.d_model), device=device, dtype=dtype
        )
        series_mask = torch.zeros((decision_count, SERIES_SLOTS), device=device, dtype=torch.bool)
    return MemoryInputs(
        series_tokens=series_tokens,
        series_mask=series_mask,
        history_tokens=history_tokens,
        history_mask=history_mask,
    )


def _compute_magnet_logits(
    episodes: Sequence[PreparedTrajectory],
    magnet: Magnet,
    device: torch.device,
    precision: OptimizationPrecision,
) -> torch.Tensor:
    """Compute frozen-policy logits without retaining intermediate tensors."""
    observations = StructuredObservation.cat([episode.observations for episode in episodes], dim=0)
    action_masks = torch.cat([episode.action_masks for episode in episodes], dim=0)
    actions = torch.cat([episode.actions for episode in episodes], dim=0)

    with (
        torch.inference_mode(),
        autocast(
            device_type=device.type,
            enabled=precision.autocast,
            dtype=precision.dtype,
        ),
    ):
        encoded = magnet.policy.encode(observations, action_masks)
        memory = _build_memory_inputs(magnet.policy, encoded, episodes, device)
        logits = magnet.policy.action_logits(
            magnet.policy.prepare(encoded, memory), action_masks, actions
        )
    return logits.detach()


def _cached_magnet_logits(
    episodes: Sequence[PreparedTrajectory],
    magnet: Magnet,
    device: torch.device,
    precision: OptimizationPrecision,
    cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Return GPU-resident magnet logits, computing each trajectory once per update."""
    missing = [episode for episode in episodes if id(episode) not in cache]
    if missing:
        logits = _compute_magnet_logits(missing, magnet, device, precision)
        offset = 0
        for episode in missing:
            end = offset + episode.length
            cache[id(episode)] = logits[offset:end]
            offset = end

    return torch.cat([cache[id(episode)] for episode in episodes], dim=0)


def _run_batched_ppo(
    episodes: list[PreparedTrajectory],
    policy: PolicyNet,
    magnet: Magnet,
    config: TrainingConfig,
    device: torch.device,
    precision: OptimizationPrecision,
    alpha: float,
    magnet_cache: dict[int, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Tensor, int]:
    """
    Compute PPO losses and metrics for one minibatch.

    Arguments:
        episodes: Trajectories in the minibatch.
        policy: Policy network being trained.
        magnet: Frozen reference policy for KL regularization.
        config: Training configuration.
        device: Target compute device.
        precision: Mixed precision settings.
        alpha: Weight for magnet KL loss.
        magnet_cache: Cache of precomputed magnet logits.

    Returns:
        Total loss, metric tensor, and number of steps.
    """
    if not episodes:
        raise ValueError("A PPO minibatch cannot be empty")

    all_obs = StructuredObservation.cat([ep.observations for ep in episodes], dim=0)
    all_action_masks = torch.cat([ep.action_masks for ep in episodes], dim=0)
    actions = torch.cat([ep.actions for ep in episodes])
    old_log_probs = torch.cat([ep.log_probs for ep in episodes])
    advantages = torch.cat([ep.advantages for ep in episodes])
    returns = torch.cat([ep.returns for ep in episodes])

    if magnet_cache is None:
        magnet_logits = _compute_magnet_logits(episodes, magnet, device, precision)
    else:
        magnet_logits = _cached_magnet_logits(
            episodes,
            magnet,
            device,
            precision,
            magnet_cache,
        )

    with autocast(
        device_type=device.type,
        enabled=precision.autocast,
        dtype=precision.dtype,
    ):
        all_enc = policy.encode(all_obs, all_action_masks)
        live_memory = _build_memory_inputs(policy, all_enc, episodes, device)
        out = policy.evaluate(
            policy.prepare(all_enc, live_memory),
            all_action_masks,
            actions,
        )
        magnet_kl = magnet_kl_per_step(out.logits, magnet_logits)
        step_loss, step_policy_loss, step_value_loss, ratio, log_ratio = compute_ppo_objective(
            out.log_probs,
            out.value,
            out.norm_entropy,
            magnet_kl,
            old_log_probs,
            advantages,
            returns,
            config,
            alpha=alpha,
        )

    total_loss = step_loss.sum()
    total_steps = int(step_loss.numel())

    with torch.no_grad():
        metrics = torch.stack(
            (
                step_policy_loss.sum(),
                step_value_loss.sum(),
                out.norm_entropy.sum(),
                magnet_kl.sum(),
                ((ratio - 1) - log_ratio).sum(),
                ((ratio < 1 - config.clip_range) | (ratio > 1 + config.clip_range)).float().sum(),
            )
        )

    return total_loss, metrics, total_steps


def ppo_update(
    episodes: list[PreparedTrajectory],
    policy: PolicyNet,
    magnet: Magnet,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
    episode: int,
    alpha: float,
    cancel_requested: Callable[[], bool],
) -> dict[str, float | int]:
    """
    Apply PPO epochs to a collection of completed trajectories.

    Arguments:
        episodes: Prepared trajectories with returns and advantages.
        policy: Live policy to update.
        magnet: Frozen policy used for regularization.
        optimizer: Optimizer for the live policy.
        scaler: Automatic mixed-precision gradient scaler.
        config: PPO optimization settings.
        episode: Current training episode index.
        alpha: Active magnet regularization coefficient.
        cancel_requested: Callback polled for cooperative cancellation.

    Returns:
        Scalar optimization and stability metrics.
    """
    if not episodes:
        raise ValueError("PPO update requires at least one prepared trajectory")
    policy.train()

    explained_var = episodes[0].explained_variance
    if any(trajectory.explained_variance != explained_var for trajectory in episodes[1:]):
        raise ValueError("Prepared trajectories must share one explained-variance metric")

    metric_zero = torch.zeros((), device=policy.device)
    metric_totals = torch.zeros(6, device=policy.device)
    tot_grad_norm = metric_zero.clone()
    tot_steps = 0
    num_updates = 0
    magnet_cache: dict[int, torch.Tensor] = {}
    precision = select_optimization_precision(config.enable_optim, policy.device)
    ordered_episodes = episodes.copy()

    for epoch_idx in range(config.ppo_epochs):
        if cancel_requested():
            break
        random.Random(config.seed + episode * config.ppo_epochs + epoch_idx).shuffle(
            ordered_episodes
        )

        for batch_start in range(0, len(ordered_episodes), config.batch_size):
            if cancel_requested():
                break

            minibatch = ordered_episodes[batch_start : batch_start + config.batch_size]

            optimizer.zero_grad(set_to_none=True)

            minibatch_steps = 0
            minibatch_kl = metric_zero.clone()
            expected_minibatch_steps = sum(ep.length for ep in minibatch)
            should_skip = False
            cancelled = False
            non_finite_loss = False
            minibatch_mean_kl = 0.0

            for chunk_idx in range(0, len(minibatch), config.minibatch_size):
                if cancel_requested():
                    cancelled = True
                    break
                chunk = minibatch[chunk_idx : chunk_idx + config.minibatch_size]
                chunk.sort(key=lambda ep: ep.length, reverse=True)

                batch_loss, batch_metrics, batch_steps = _run_batched_ppo(
                    chunk,
                    policy,
                    magnet,
                    config,
                    policy.device,
                    precision,
                    alpha,
                    magnet_cache,
                )

                metric_totals += batch_metrics
                minibatch_kl += batch_metrics[KL_METRIC_INDEX]
                minibatch_steps += batch_steps

                scaled_loss = batch_loss / expected_minibatch_steps
                if bool(torch.isfinite(scaled_loss).item()):
                    scaler.scale(scaled_loss).backward()
                else:
                    LOGGER.warning(
                        f"Non-finite chunk loss at episode {episode}; "
                        "discarding the entire minibatch"
                    )
                    non_finite_loss = True
                    break

                # Preserve the original early-stop behavior: once the running
                # mean KL for this effective minibatch exceeds the target, do
                # not process any remaining chunks.
                should_skip, minibatch_mean_kl = _kl_exceeds_target(
                    minibatch_kl, minibatch_steps, config.target_kl
                )
                if should_skip:
                    break

            if cancelled:
                optimizer.zero_grad(set_to_none=True)
                break
            if should_skip or non_finite_loss:
                reason = (
                    "non-finite loss"
                    if non_finite_loss
                    else f"KL={minibatch_mean_kl:.4f} > {config.target_kl:.4f}"
                )
                LOGGER.info(
                    f"Skipping minibatch at epoch {epoch_idx + 1}/{config.ppo_epochs}, "
                    f"batch {batch_start // config.batch_size + 1} "
                    f"({reason})"
                )
                optimizer.zero_grad(set_to_none=True)
            else:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.max_grad_norm
                )
                scale_before_update = scaler.get_scale()
                grad_norm_finite = bool(torch.isfinite(grad_norm).item())
                if not grad_norm_finite:
                    LOGGER.warning(
                        "Non-finite grad norm detected; discarding this step "
                        f"(loss scale={scale_before_update:.0f})"
                    )
                else:
                    tot_grad_norm += grad_norm.detach()
                if grad_norm_finite:
                    scaler.step(optimizer)
                # GradScaler.update consumes the finite check recorded by
                # unscale_, including the rejected norm-overflow case.
                scaler.update()
                if grad_norm_finite:
                    num_updates += 1
            tot_steps += minibatch_steps

    # No subsequent work consumes these gradients. Clear the final minibatch
    # gradient storage before the next rollout begins.
    optimizer.zero_grad(set_to_none=True)

    if tot_steps == 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "normalized_entropy": 0.0,
            "magnet_kl": 0.0,
            "kl_divergence": 0.0,
            "grad_norm": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "optimizer_updates": 0,
        }

    grad_norm = tot_grad_norm / num_updates if num_updates > 0 else metric_zero
    metric_means = metric_totals / tot_steps
    summary = (
        torch.cat((metric_means[:5], grad_norm.unsqueeze(0), metric_means[5:]))
        .detach()
        .cpu()
        .tolist()
    )
    (
        policy_loss,
        value_loss,
        normalized_entropy,
        magnet_kl,
        kl_divergence,
        grad_norm_value,
        clip_fraction,
    ) = summary

    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "normalized_entropy": normalized_entropy,
        "magnet_kl": magnet_kl,
        "kl_divergence": kl_divergence,
        "grad_norm": grad_norm_value,
        "clip_fraction": clip_fraction,
        "explained_variance": explained_var,
        "optimizer_updates": num_updates,
    }
