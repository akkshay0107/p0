import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from policy import PolicyNet
from ppo_utils import initial_state

BATCH_SIZE = 8  # number of episodes per gradient update


class ReplayDataset(Dataset):
    def __init__(self, replays_dir: str):
        self.episodes = []
        path = Path(replays_dir)

        for replay_file in sorted(path.rglob("*.replay")):
            try:
                # each replay file is a shard (list of episodes)
                shard_data = torch.load(replay_file, weights_only=False)
                if isinstance(shard_data, list):
                    self.episodes.extend(shard_data)
            except Exception as e:
                print(f"could not load shard {replay_file}: {e}")

        print(f"loaded {len(self.episodes)} episodes from {path}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


def _run_episode(
    policy: PolicyNet,
    episode: list,
    device: torch.device,
    state=None,
) -> tuple[torch.Tensor, int, int]:
    """
    Run one episode with BPTT. Carries recurrent state step-to-step.
    Returns:
        loss:    total cross-entropy loss over the episode steps (scalar tensor, grad attached)
        correct: number of correctly predicted actions
        total:   total number of actions evaluated
    """
    if state is None:
        state = initial_state(policy, 1, device)

    loss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0

    for sample in episode:
        # Unsqueeze to add batch dimension (B=1) as expected by PolicyNet
        obs = sample["obs"].to(device, non_blocking=True).unsqueeze(0)
        mask = sample["mask"].to(device, non_blocking=True).unsqueeze(0)
        target = sample["action"].to(device, non_blocking=True).unsqueeze(0)

        log_prob, _, _, _, next_state = policy.evaluate_actions(obs, target, mask, state)
        loss -= log_prob.mean()

        with torch.no_grad():
            logits = policy.get_policy_masked_logits(obs, target, mask, state)
            preds = torch.stack(
                [logits[:, 0].argmax(dim=-1), logits[:, 1].argmax(dim=-1)],
                dim=-1,
            )
            correct += (preds == target).sum().item()
            total += target.numel()

        state = next_state

    return loss, correct, total


def _evaluate_episodes(
    policy: PolicyNet,
    episodes: list,
    device: torch.device,
) -> tuple[float, int, int]:
    total_loss = 0.0
    correct = 0
    total = 0
    tot_steps = 0

    with torch.inference_mode():
        for episode in episodes:
            ep_loss, ep_correct, ep_total = _run_episode(policy, episode, device)
            total_loss += ep_loss.item()
            correct += ep_correct
            total += ep_total
            tot_steps += len(episode)

    return total_loss / tot_steps, correct, total


def train_behavior_cloning(
    dataset,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    val_split_ratio: float = 0.2,
    policy: PolicyNet | None = None,
) -> PolicyNet | None:
    if len(dataset) == 0:
        print("No data available for training.")
        return None

    # Train / val split
    episodes = [dataset[i] for i in range(len(dataset)) if dataset[i]]
    if not episodes:
        print("No valid episodes found in dataset.")
        return None

    random.shuffle(episodes)
    val_size = min(int(round(val_split_ratio * len(episodes))), len(episodes) - 1)
    val_episodes = episodes[:val_size]
    train_episodes = episodes[val_size:]

    if policy is None:
        policy = PolicyNet()

    device = policy.device
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, eps=1e-5, weight_decay=1e-4
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        policy.train()
        random.shuffle(train_episodes)

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        num_updates = 0

        for batch_start in range(0, len(train_episodes), batch_size):
            batch = train_episodes[batch_start : batch_start + batch_size]
            optimizer.zero_grad(set_to_none=True)

            batch_loss = torch.tensor(0.0, device=device)
            total_steps = 0
            for episode in batch:
                ep_loss, correct, total = _run_episode(policy, episode, device)
                batch_loss += ep_loss
                total_steps += len(episode)
                train_correct += correct
                train_total += total

            (batch_loss / total_steps).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += (batch_loss / total_steps).item()
            num_updates += 1

        if train_total > 0:
            print(
                f"  Train  | loss: {train_loss_sum / num_updates:.4f} "
                f"| acc: {train_correct / train_total:.4f}"
            )

        if val_episodes:
            policy.eval()
            val_loss_avg, val_correct, val_total = _evaluate_episodes(policy, val_episodes, device)
            if val_total > 0:
                print(f"  Val    | loss: {val_loss_avg:.4f} | acc: {val_correct / val_total:.4f}")
        else:
            print("  Val    | skipped (no validation split)")

        print()

    return policy
