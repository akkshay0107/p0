from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical

from src.lookups import ACT_SIZE
from src.model.cls_reducer import CLSReducer
from src.model.fused_token_encoder import FusedTokenEncoder, as_obs_dict
from src.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH

# Type alias for the recurrent state
State = Tuple[torch.Tensor, torch.Tensor]


class PolicyHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_features),
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                # init output layer with minimal variance at the start
                gain = 0.01 if i == len(self.net) - 1 else 1.0
                init.orthogonal_(module.weight, gain=gain)
                init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorPolicy(nn.Module):
    """Stateful actor policy path."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayer: int,
        act_size: int,
        seq_len: int = SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.act_size = act_size

        self.reducer = CLSReducer(
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer,
            use_history=True,
        )

        # Autoregressive embedding for P(a2 | z, a1)
        d_act_emb = d_model // 4
        self.action_embedding = nn.Embedding(act_size, d_act_emb)
        init.normal_(self.action_embedding.weight, mean=0, std=0.02)

        self.head1 = PolicyHead(d_model, act_size)  # P(a1 | z)
        self.head2 = PolicyHead(d_model + d_act_emb, act_size)  # P(a2 | z, a1)

    def forward(
        self,
        tokens: torch.Tensor,
        state: Optional[State] = None,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        sample: bool = True,
        is_tp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], State]:
        B = tokens.size(0)
        device = tokens.device

        z, next_state = self.reducer(tokens, state)
        logits1 = self.head1(z)

        # action eval
        if actions is not None:
            a1 = actions[:, 0]
            a1_emb = self.action_embedding(a1)
            logits2 = self.head2(torch.cat([z, a1_emb], dim=-1))
            logits = torch.stack([logits1, logits2], dim=1)

            if action_mask is not None:
                logits = self._apply_sequential_masks(logits, a1, action_mask, is_tp)

            dist1 = Categorical(logits=logits[:, 0])
            dist2 = Categorical(logits=logits[:, 1])
            log_probs = dist1.log_prob(actions[:, 0]) + dist2.log_prob(actions[:, 1])

            return logits, log_probs, None, next_state

        # inference mode
        if not sample:
            # Return logits with placeholder for second action if not sampling or evaluating
            logits = torch.stack([logits1, torch.zeros_like(logits1)], dim=1)
            return logits, torch.zeros(B, device=device), None, next_state

        # Sampling with sequential constraints
        m1 = action_mask[:, 0] if action_mask is not None else None
        l1 = logits1 if m1 is None else logits1.masked_fill(m1 == 0, float("-inf"))

        dist1 = Categorical(logits=l1)
        a1 = dist1.sample()

        a1_emb = self.action_embedding(a1)
        logits2 = self.head2(torch.cat([z, a1_emb], dim=-1))

        logits = torch.stack([logits1, logits2], dim=1)
        if action_mask is not None:
            logits = self._apply_sequential_masks(logits, a1, action_mask, is_tp)

        dist2 = Categorical(logits=logits[:, 1])
        a2 = dist2.sample()

        log_probs = dist1.log_prob(a1) + dist2.log_prob(a2)
        sampled_actions = torch.stack([a1, a2], dim=-1)

        return logits, log_probs, sampled_actions, next_state

    def _apply_sequential_masks(
        self,
        logits: torch.Tensor,
        action1: torch.Tensor,
        action_mask: torch.Tensor,
        is_tp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = logits.size(0)
        device = logits.device

        # Apply base action masks
        logits = logits.masked_fill(action_mask == 0, float("-inf"))

        if is_tp is None:
            is_tp = torch.zeros(B, device=device, dtype=torch.bool)
        else:
            is_tp = is_tp.bool()

        mask2 = action_mask[:, 1].clone().bool()

        # If Pokemon 1 switches to slot idx, Pokemon 2 cannot switch to the same slot
        switch_mask = (1 <= action1) & (action1 <= 6) & (~is_tp)
        mask2[switch_mask, action1[switch_mask]] = 0

        # Only one Mega per turn.
        # Mega moves are 27-46.
        mega_mask = (action1 >= 27) & (action1 <= 46) & (~is_tp)
        mask2[mega_mask, 27:47] = False

        # If Pokemon 1 passes, Pokemon 2 cannot pass as well unless no valid moves left
        pass_mask = (action1 == 0) & (~is_tp)
        mask2[pass_mask, 0] = False

        # Ensure all 4 selected Pokemon are unique (no overlap between Lead and Back).
        # compute overlap for all B rows simultaneously, gate with is_tp.
        # eliminates the is_tp.any() GPU->CPU sync
        all_a = torch.arange(36, device=device)
        p1_1 = action1 // 6 + 1  # (B,) — meaningful only for tp rows
        p2_1 = action1 % 6 + 1  # (B,)
        p1_2 = all_a // 6 + 1  # (36,)
        p2_2 = all_a % 6 + 1  # (36,)
        tp_overlap = (
            (p1_2[None] == p1_1[:, None])
            | (p1_2[None] == p2_1[:, None])
            | (p2_2[None] == p1_1[:, None])
            | (p2_2[None] == p2_1[:, None])
        )  # (B, 36)
        mask2[:, :36] = mask2[:, :36] & ~(is_tp[:, None] & tp_overlap)

        # If no valid action remains, force pass action to be valid for Pokemon 2
        no_valid = mask2.sum(-1) == 0
        mask2[no_valid, 0] = True

        l2 = logits[:, 1].masked_fill(~mask2, float("-inf"))
        return torch.stack([logits[:, 0], l2], dim=1)


class ValueNet(nn.Module):
    """Stateless critic value path with internal head and gradient scaling."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayer: int,
        seq_len: int = SEQUENCE_LENGTH,
        hidden_dim: int = 1024,  # double that of policy heads
        scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.reducer = CLSReducer(
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer,
            use_history=False,
        )
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for i, module in enumerate(self.net):
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        z, _ = self.reducer(tokens, None)

        # scale the gradient flowing back to the trunk
        # may not be needed anymore given the deeper split
        if z.requires_grad and self.scale < 1.0:
            z = z.detach() + self.scale * (z - z.detach())

        return self.net(z).squeeze(-1)


class PolicyNet(nn.Module):
    """
    Refactored Pokemon Policy Network with explicit Actor/Critic split.
    Name kept the same to be consistent with training loop. Might be fixed later.

    The Policy path is stateful (recurrent), while the Value path is stateless.
    Both share a common FusedTokenEncoder for efficiency.
    """

    def __init__(
        self,
        obs_dim=(SEQUENCE_LENGTH, NUMERICAL_WIDTH),
        act_size=ACT_SIZE,
        d_model=768,
        nhead=8,
        nlayer=3,
        critic_nlayer: int | None = None,
    ):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size
        self.d_model = d_model

        # Shared structured front-end
        self.encoder = FusedTokenEncoder(d_model, nhead, d_model * 4)

        # Stateful Actor
        self.actor = ActorPolicy(d_model, nhead, nlayer, act_size, self.seq_len)

        # Stateless Critic
        critic_nlayer = critic_nlayer or nlayer
        self.critic = ValueNet(d_model, nhead, critic_nlayer, self.seq_len)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        obs: Any,
        state: Optional[State] = None,
        action_mask: Optional[torch.Tensor] = None,
        sample_actions: bool = True,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, State]:
        """
        Main forward pass. Matches original PolicyNet signature for compatibility.
        """
        obs_dict = as_obs_dict(obs)
        tokens = self.encoder(obs_dict)

        # avoid duplicate is_tp calculation
        numerical = obs_dict["numerical"]
        if numerical.dim() == 2:
            numerical = numerical.unsqueeze(0)
        is_tp = (numerical[:, 25, 2] > 0.5).to(self.device)

        if action_mask is not None:
            action_mask = action_mask.to(self.device)
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

        if actions is not None:
            actions = actions.to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

        return self.forward_tokens(tokens, is_tp, state, action_mask, sample_actions, actions)

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        is_tp: torch.Tensor,
        state: Optional[State] = None,
        action_mask: Optional[torch.Tensor] = None,
        sample_actions: bool = True,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, State]:
        """
        Forward pass using already encoded tokens.
        """
        logits, log_probs, sampled_actions, next_state = self.actor(
            tokens, state, action_mask, actions, sample_actions, is_tp
        )
        value = self.critic(tokens)

        return logits, log_probs, sampled_actions, value, next_state

    def evaluate_actions(
        self,
        obs: Any,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        state: Optional[State] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, State]:
        """
        Evaluate actions for PPO updates.
        Returns (log_prob, entropy, normalized_entropy, value, next_state).
        """
        obs_dict = as_obs_dict(obs)
        tokens = self.encoder(obs_dict)
        is_tp = (obs_dict["numerical"][:, 25, 2] > 0.5).to(self.device)

        return self.evaluate_actions_tokens(tokens, is_tp, actions, action_mask, state)

    def evaluate_actions_tokens(
        self,
        tokens: torch.Tensor,
        is_tp: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        state: Optional[State] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, State]:
        """
        Evaluate actions using already encoded tokens.
        """
        logits, log_prob, _, value, next_state = self.forward_tokens(
            tokens, is_tp, state, action_mask, sample_actions=False, actions=actions
        )

        dist1 = Categorical(logits=logits[:, 0])
        dist2 = Categorical(logits=logits[:, 1])
        entropy = dist1.entropy() + dist2.entropy()

        # normalized entropy (relative to valid action support)
        if action_mask is not None:
            v1 = (logits[:, 0] > float("-inf")).sum(-1).float().clamp_min(1.0)
            v2 = (logits[:, 1] > float("-inf")).sum(-1).float().clamp_min(1.0)
            max_entropy = torch.log(v1) + torch.log(v2)
        else:
            max_entropy = torch.log(torch.tensor(self.act_size, device=self.device).float()) * 2

        norm_entropy = torch.where(
            max_entropy > 0,
            entropy / max_entropy.clamp_min(1e-8),
            torch.zeros_like(entropy),
        )

        return log_prob, entropy, norm_entropy, value, next_state

    def get_policy_masked_logits(
        self,
        obs: Any,
        action_taken: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        state: Optional[State] = None,
    ):
        logits, _, _, _, _ = self(
            obs, state, action_mask, sample_actions=False, actions=action_taken
        )
        return logits
