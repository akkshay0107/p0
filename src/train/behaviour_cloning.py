import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.model.policy import EncodedObs, PolicyNet
from src.model.structured_observation import StructuredObservation
from src.train.utils import adamw_param_groups

BATCH_SIZE = 32  # number of episodes per gradient update


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


def _run_batched_bc(
    policy: PolicyNet,
    episodes: list,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, int]:
    """
    Run Behavior Cloning BPTT over a minibatch of variable-length episodes.
    Returns (total_loss, correct_predictions, total_predictions, total_steps).
    """
    if not episodes:
        return torch.tensor(0.0, device=device), 0, 0, 0

    episodes = sorted(episodes, key=len, reverse=True)
    batch_size = len(episodes)
    lengths = [len(ep) for ep in episodes]
    max_steps = lengths[0]

    all_obs_tensors = []
    for ep in episodes:
        all_obs_tensors.append(
            StructuredObservation.cat([sample["obs"].unsqueeze(0) for sample in ep], dim=0)
        )
    all_masks_list = [
        torch.cat([sample["mask"].unsqueeze(0) for sample in ep], dim=0) for ep in episodes
    ]
    all_obs = StructuredObservation.cat(all_obs_tensors, dim=0).to(device)
    all_masks = torch.cat(all_masks_list, dim=0).to(device)
    all_enc = policy.encode(all_obs, all_masks)
    split_sizes = [len(ep) for ep in episodes]
    tokens_list = torch.split(all_enc.tokens, split_sizes)
    aux_list = torch.split(all_enc.aux, split_sizes)
    numerical_list = torch.split(all_enc.numerical, split_sizes)

    # time major padding
    enc_p = EncodedObs(
        tokens=torch.nn.utils.rnn.pad_sequence(list(tokens_list)),
        aux=torch.nn.utils.rnn.pad_sequence(list(aux_list)),
        numerical=torch.nn.utils.rnn.pad_sequence(list(numerical_list)),
    )

    def pack(fields):
        return torch.nn.utils.rnn.pad_sequence(fields).to(device)

    all_targets_list = [
        torch.cat([sample["action"].unsqueeze(0) for sample in ep], dim=0) for ep in episodes
    ]

    masks_p = pack(all_masks_list)
    targets_p = pack(all_targets_list)

    state = policy.initial_state(batch_size)
    total_loss = torch.tensor(0.0, device=device)
    correct = 0
    total = 0
    total_steps = 0

    for t in range(max_steps):
        active_n = sum(1 for length in lengths if length > t)
        if active_n == 0:
            break

        enc_t = enc_p.step(active_n, t)
        masks_t = masks_p[t, :active_n]
        targets_t = targets_p[t, :active_n]

        curr_state = state[:active_n]
        out = policy.evaluate(enc_t, masks_t, targets_t, curr_state)

        loss = -out.log_probs.sum()
        total_loss = total_loss + loss
        total_steps += active_n

        with torch.no_grad():
            preds = out.logits.argmax(dim=-1)
            correct += (preds == targets_t).sum().item()
            total += targets_t.numel()

        next_state = out.state
        if active_n < batch_size:
            state = torch.cat([next_state, state[active_n:]], dim=0)
        else:
            state = next_state

    return total_loss, int(correct), total, total_steps


def _evaluate_episodes(
    policy: PolicyNet,
    episodes: list,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, int, int]:
    total_loss = 0.0
    correct = 0
    total = 0
    tot_steps = 0

    with torch.inference_mode():
        for batch_start in range(0, len(episodes), batch_size):
            batch = episodes[batch_start : batch_start + batch_size]
            batch_loss, batch_correct, batch_total, batch_steps = _run_batched_bc(
                policy, batch, device
            )
            total_loss += batch_loss.item()
            correct += batch_correct
            total += batch_total
            tot_steps += batch_steps

    return total_loss / tot_steps if tot_steps > 0 else 0.0, correct, total


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
        adamw_param_groups(policy, weight_decay=1e-4),
        lr=learning_rate,
        eps=1e-5,
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

            batch_loss, correct, total, steps = _run_batched_bc(policy, batch, device)

            if steps > 0:
                scaled_loss = batch_loss / steps
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                train_correct += correct
                train_total += total
                train_loss_sum += scaled_loss.item()
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
