import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical

from cls_reducer import CLSReducer
from lookups import ACT_SIZE, OBS_DIM


class ValueHead(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if tracking gradients, scale the gradient flowing back to the backbone
        if x.requires_grad:
            x = x.detach() + self.scale * (x - x.detach())

        return self.net(x)

    @torch.no_grad()
    def _init_weights(self):
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                if i == len(self.net) - 1:
                    init.orthogonal_(module.weight, gain=0.1)
                else:
                    init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)


# Needs all inputs to be on the same device as the model
class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim=OBS_DIM,
        act_size=ACT_SIZE,
        d_model=512,
        nhead=8,
        nlayer=3,
        n_hg=4,
    ):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reducer = CLSReducer(
            self.seq_len, self.feat_dim, d_model, nhead, nlayer, 4 * d_model, n_hg
        )

        # act embedding for autoregressive conditioning (P(a2 | z, a1))
        d_act_emb = d_model // 4
        self.action_embedding = nn.Embedding(act_size, d_act_emb)

        # Policy Head 1 (Pokemon 1): P(a1 | z)
        self.policy_head1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, act_size),
        )

        # Policy Head 2 (Pokemon 2): P(a2 | z, a1_emb)
        self.policy_head2 = nn.Sequential(
            nn.Linear(d_model + d_act_emb, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, act_size),
        )

        # outputs scalar value from the state V(s)
        self.value_head = ValueHead(in_features=d_model, hidden_features=d_model // 2, scale=0.1)

        self.to(self.device)
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for module in self.reducer.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)

        init.normal_(self.action_embedding.weight, mean=0, std=0.02)

        for head in [self.policy_head1, self.policy_head2]:
            for i, module in enumerate(head):
                if isinstance(module, nn.Linear):
                    if i == len(head) - 1:
                        init.orthogonal_(module.weight, gain=0.1)
                    else:
                        init.orthogonal_(module.weight, gain=1.0)
                    init.zeros_(module.bias)

    def forward(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        action_mask: torch.Tensor | None = None,
        sample_actions: bool = True,
        actions: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        # Add batch dimension if missing
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        B, S, F = obs.shape

        if S != self.seq_len or F != self.feat_dim:
            raise ValueError(f"Got shape ({S}, {F}). Expected({self.seq_len}, {self.feat_dim})")

        is_tp = obs[:, 41, 18] > 0.5

        z, next_state = self.reducer(obs, state)
        value = self.value_head(z).squeeze(-1)

        logits1 = self.policy_head1(z)  # (B, act_size)

        # eval / bc (action provided)
        if actions is not None:
            a1 = actions[:, 0]
            a1_emb = self.action_embedding(a1)
            logits2 = self.policy_head2(torch.cat([z, a1_emb], dim=-1))
            logits = torch.stack([logits1, logits2], dim=1)

            if action_mask is not None:
                logits = self._apply_masks(logits, action_mask)
                logits = self._apply_sequential_masks(logits, a1, action_mask, is_tp)

            return logits, None, None, value, next_state

        # no sampling requested (returns p1 logits and placeholder p2)
        if not sample_actions:
            logits = torch.stack([logits1, torch.zeros_like(logits1)], dim=1)
            return logits, None, None, value, next_state

        # sampling mode
        if action_mask is not None:
            l1 = logits1.masked_fill(action_mask[:, 0] == 0, float("-inf"))
        else:
            l1 = logits1

        cat1 = Categorical(logits=l1)
        a1 = cat1.sample()
        log_prob1 = cat1.log_prob(a1)

        a1_emb = self.action_embedding(a1)
        logits2 = self.policy_head2(torch.cat([z, a1_emb], dim=-1))

        logits = torch.stack([logits1, logits2], dim=1)
        if action_mask is not None:
            logits = self._apply_masks(logits, action_mask)
            logits = self._apply_sequential_masks(logits, a1, action_mask, is_tp)

        cat2 = Categorical(logits=logits[:, 1])
        a2 = cat2.sample()
        log_prob2 = cat2.log_prob(a2)

        return (
            logits,
            log_prob1 + log_prob2,  # joint log prob
            torch.stack([a1, a2], -1),
            value,
            next_state,
        )

    def get_policy_masked_logits(
        self,
        obs: torch.Tensor,
        action_taken: torch.Tensor,
        action_mask: torch.Tensor | None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        logits, _, _, _, _ = self(
            obs, state, action_mask, sample_actions=False, actions=action_taken
        )
        return logits

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:
        logits, _, _, value, next_state = self(
            obs, state, action_mask, sample_actions=False, actions=actions
        )

        if action_mask is not None:
            # Compute support sizes for normalized entropy
            valid_count_1 = (logits[:, 0] > float("-inf")).sum(-1).float().clamp_min(1.0)
            valid_count_2 = (logits[:, 1] > float("-inf")).sum(-1).float().clamp_min(1.0)
            max_entropy = torch.log(valid_count_1) + torch.log(valid_count_2)
        else:
            max_entropy = (
                torch.log(torch.tensor(self.act_size, device=logits.device, dtype=torch.float32))
                * 2
            )

        cat1 = Categorical(logits=logits[:, 0])
        cat2 = Categorical(logits=logits[:, 1])

        log_prob1 = cat1.log_prob(actions[:, 0])
        log_prob2 = cat2.log_prob(actions[:, 1])
        log_prob = log_prob1 + log_prob2

        entropy = cat1.entropy() + cat2.entropy()

        normalized_entropy = torch.where(
            max_entropy > 0,
            entropy / max_entropy.clamp_min(torch.finfo(entropy.dtype).eps),
            torch.zeros_like(entropy),
        )

        return log_prob, entropy, normalized_entropy, value, next_state

    def _apply_masks(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        # Replace logits of illegal actions with -inf so they have zero probability
        mask = action_mask == 0
        return logits.masked_fill(mask, float("-inf"))

    def _apply_sequential_masks(
        self,
        logits: torch.Tensor,
        action1: torch.Tensor,
        action_mask: torch.Tensor,
        is_tp: torch.Tensor,
    ):
        is_tp = is_tp.bool()
        mask2 = action_mask[:, 1].clone().bool()

        # If Pokemon 1 switches to slot idx, Pokemon 2 cannot switch to the same slot
        switch_mask = (1 <= action1) & (action1 <= 6) & (~is_tp)
        mask2[switch_mask, action1[switch_mask]] = 0

        # If Pokemon 1 uses terastallize, Pokemon 2 cannot also tera
        tera_mask = (26 < action1) & (action1 <= 46) & (~is_tp)
        mask2[tera_mask, 27:47] = 0

        # If Pokemon 1 passes, Pokemon 2 cannot pass as well unless no valid moves left
        pass_mask = (action1 == 0) & (~is_tp)
        mask2[pass_mask, 0] = 0

        # Ensure all 4 selected Pokemon are unique (no overlap between Lead and Back)
        if is_tp.any():
            tp_indices = torch.where(is_tp)[0]
            tp_actions = action1[tp_indices]

            p1_1 = tp_actions // 6 + 1
            p2_1 = tp_actions % 6 + 1

            all_a = torch.arange(36, device=logits.device)
            p1_2 = all_a // 6 + 1
            p2_2 = all_a % 6 + 1

            overlap = (
                (p1_2.unsqueeze(0) == p1_1.unsqueeze(1))
                | (p1_2.unsqueeze(0) == p2_1.unsqueeze(1))
                | (p2_2.unsqueeze(0) == p1_1.unsqueeze(1))
                | (p2_2.unsqueeze(0) == p2_1.unsqueeze(1))
            )

            mask2[tp_indices.unsqueeze(1), all_a] &= ~overlap

        # If no valid action remains, force pass action to be valid for Pokemon 2
        no_valid = mask2.sum(-1) == 0
        mask2[no_valid, 0] = 1

        return torch.stack(
            [
                logits[:, 0],  # first row actions unchanged
                logits[:, 1].masked_fill(
                    ~mask2.bool(), float("-inf")
                ),  # masked row for second pokemon
            ],
            dim=1,
        )
