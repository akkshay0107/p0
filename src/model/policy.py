from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical

from src.lookups import ACT_SIZE
from src.model.cls_reducer import CLSReducer
from src.model.fused_token_encoder import FusedTokenEncoder
from src.model.structured_observation import NUMERICAL_WIDTH, SEQUENCE_LENGTH, StructuredObservation

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

        # embedding components for P(a2 | z, a1)
        d_act_emb = d_model // 4

        self.pass_emb = nn.Parameter(torch.empty(d_act_emb))
        self.tp_meta_emb = nn.Parameter(torch.empty(d_act_emb))
        self.switch_meta_emb = nn.Parameter(torch.empty(d_act_emb))
        self.move_meta_emb = nn.Parameter(torch.empty(d_act_emb))
        self.mega_meta_emb = nn.Parameter(torch.empty(d_act_emb))

        self.target_ally_emb = nn.Parameter(torch.empty(d_act_emb))
        self.target_opp_emb = nn.Parameter(torch.empty(d_act_emb))
        self.target_self_multi_emb = nn.Parameter(torch.empty(d_act_emb))

        def make_proj() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, d_act_emb), nn.GELU(), nn.Linear(d_act_emb, d_act_emb)
            )

        self.actor_proj = make_proj()
        self.target_proj = make_proj()
        self.move_proj = make_proj()

        for p in [
            self.pass_emb,
            self.tp_meta_emb,
            self.switch_meta_emb,
            self.move_meta_emb,
            self.mega_meta_emb,
            self.target_ally_emb,
            self.target_opp_emb,
            self.target_self_multi_emb,
        ]:
            init.normal_(p, mean=0, std=0.02)

        for seq_module in [self.actor_proj, self.target_proj, self.move_proj]:
            for module in seq_module.modules():
                if isinstance(module, nn.Linear):
                    init.orthogonal_(module.weight, gain=1.0)
                    init.zeros_(module.bias)

        self.ctx_norm = nn.LayerNorm(d_act_emb)
        self.head1 = PolicyHead(d_model, act_size)  # P(a1 | z)
        self.head2 = PolicyHead(d_model + d_act_emb, act_size)  # P(a2 | z, a1)

    def _build_action_context(
        self,
        a1: torch.Tensor,
        is_tp: torch.Tensor,
        tokens: torch.Tensor,
        aux: torch.Tensor,
        numerical: torch.Tensor,
    ) -> torch.Tensor:
        B = a1.size(0)
        device = a1.device

        d_act_emb = self.pass_emb.size(0)
        ctx = torch.zeros(B, d_act_emb, device=device)

        tp_mask = is_tp.bool()
        if tp_mask.any():
            b_idx = tp_mask.nonzero(as_tuple=True)[0]
            a1_tp = a1[b_idx]

            idx_p1 = (a1_tp // 6) * 2 + 1
            idx_p2 = (a1_tp % 6) * 2 + 1

            tok_p1 = tokens[b_idx, idx_p1, :]
            tok_p2 = tokens[b_idx, idx_p2, :]

            ctx[b_idx] = self.actor_proj(tok_p1) + self.target_proj(tok_p2) + self.tp_meta_emb

        battle_mask = ~tp_mask
        if battle_mask.any():
            b_idx = battle_mask.nonzero(as_tuple=True)[0]
            a1_b = a1[b_idx]

            actor_tok = tokens[b_idx, 1, :]
            actor_emb = self.actor_proj(actor_tok)

            pass_mask = a1_b == 0
            if pass_mask.any():
                ctx[b_idx[pass_mask]] = actor_emb[pass_mask] + self.pass_emb

            switch_mask = (a1_b >= 1) & (a1_b <= 6)
            if switch_mask.any():
                s_idx = b_idx[switch_mask]
                slot_idx = a1_b[switch_mask]

                ally_indices = torch.tensor([1, 3, 5, 7, 9, 11], device=device)
                orig_ids = torch.round(numerical[s_idx.unsqueeze(-1), ally_indices, 26] * 6).long()

                matches = orig_ids == slot_idx.unsqueeze(-1)
                has_match = matches.any(dim=-1)
                match_idx = matches.float().argmax(dim=-1)
                actual_seq_idx = ally_indices[match_idx]

                incoming_tok = tokens[s_idx, actual_seq_idx, :]
                matched_ctx = (
                    actor_emb[switch_mask] + self.target_proj(incoming_tok) + self.switch_meta_emb
                )
                # fallback: if orig_idx lookup fails, omit target projection
                fallback_ctx = actor_emb[switch_mask] + self.switch_meta_emb
                ctx[s_idx] = torch.where(has_match.unsqueeze(-1), matched_ctx, fallback_ctx)

            move_mask = a1_b >= 7
            if move_mask.any():
                m_idx = b_idx[move_mask]
                a1_m = a1_b[move_mask]

                is_mega = a1_m >= 27
                move_idx = ((a1_m - 7) % 20) // 5
                target_idx = (a1_m - 7) % 5  # 0: ally2, 1: ally1, 2: multi, 3: opp1, 4: opp2

                meta_emb = torch.where(
                    is_mega.unsqueeze(-1), self.mega_meta_emb, self.move_meta_emb
                )
                move_emb = aux[m_idx, move_idx, :]

                seq_indices = torch.tensor([3, 1, 0, 13, 15], device=device)
                target_toks = tokens[m_idx, seq_indices[target_idx], :]
                target_toks_proj = self.target_proj(target_toks)

                target_toks_proj = torch.where(
                    (target_idx == 2).unsqueeze(-1),
                    torch.zeros_like(target_toks_proj),
                    target_toks_proj,
                )

                target_meta_embs = torch.stack(
                    [
                        self.target_ally_emb,
                        self.target_ally_emb,
                        self.target_self_multi_emb,
                        self.target_opp_emb,
                        self.target_opp_emb,
                    ]
                )

                target_ctx = target_toks_proj + target_meta_embs[target_idx]
                ctx[m_idx] = actor_emb[move_mask] + self.move_proj(move_emb) + target_ctx + meta_emb

        return self.ctx_norm(ctx)

    def forward(
        self,
        tokens: torch.Tensor,
        aux: torch.Tensor,
        numerical: torch.Tensor,
        state: Optional[State] = None,
        action_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        sample: bool = True,
        is_tp: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], State]:
        B = tokens.size(0)
        device = tokens.device

        z, next_state = self.reducer(tokens, state, padding_mask)
        logits1 = self.head1(z)

        # action eval
        if actions is not None:
            a1 = actions[:, 0]

            if is_tp is None:
                is_tp = torch.zeros(B, device=device, dtype=torch.bool)

            a1_emb = self._build_action_context(a1, is_tp, tokens, aux, numerical)
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

        if is_tp is None:
            is_tp = torch.zeros(B, device=device, dtype=torch.bool)

        a1_emb = self._build_action_context(a1, is_tp, tokens, aux, numerical)
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

        l1 = logits[:, 0].masked_fill(action_mask[:, 0] == 0, float("-inf"))
        l2 = logits[:, 1].masked_fill(~mask2, float("-inf"))
        return torch.stack([l1, l2], dim=1)


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

    def forward(
        self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # scale the gradient flowing back to the trunk
        # may not be needed anymore given the deeper split
        if tokens.requires_grad and self.scale < 1.0:
            tokens = tokens.detach() + self.scale * (tokens - tokens.detach())

        z, _ = self.reducer(tokens, None, padding_mask)
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

    def _get_padding_mask(self, numerical: torch.Tensor) -> torch.Tensor:
        B, S = numerical.shape[:2]
        padding_mask = torch.zeros(B, S, dtype=torch.bool, device=self.device)
        # pokemon numeric tokens are at indices 2, 4, 6..24
        idx_num = torch.arange(2, 25, 2, device=self.device)
        idx_sup = idx_num - 1  # mask out corresponding super token as well
        # numerical[..., 27] is the fainted flag
        is_fainted = numerical[:, idx_num, 27] > 0.5
        padding_mask[:, idx_num] = is_fainted
        padding_mask[:, idx_sup] = is_fainted
        return padding_mask

    def forward(
        self,
        obs: StructuredObservation,
        state: Optional[State] = None,
        action_mask: Optional[torch.Tensor] = None,
        sample_actions: bool = True,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, State]:
        """
        Main forward pass. Matches original PolicyNet signature for compatibility.
        """
        tokens, aux = self.encoder(obs, aux=True)

        # avoid duplicate is_tp calculation
        numerical = obs.numerical
        if numerical.dim() == 2:
            numerical = numerical.unsqueeze(0)
        is_tp = (numerical[:, 25, 2] > 0.5).to(self.device)
        padding_mask = self._get_padding_mask(numerical)

        if action_mask is not None:
            action_mask = action_mask.to(self.device)
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

        if actions is not None:
            actions = actions.to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

        return self.forward_tokens(
            tokens, aux, numerical, is_tp, state, action_mask, sample_actions, actions, padding_mask
        )

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        aux: torch.Tensor,
        numerical: torch.Tensor,
        is_tp: torch.Tensor,
        state: Optional[State] = None,
        action_mask: Optional[torch.Tensor] = None,
        sample_actions: bool = True,
        actions: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, State]:
        """
        Forward pass using already encoded tokens.
        """
        logits, log_probs, sampled_actions, next_state = self.actor(
            tokens, aux, numerical, state, action_mask, actions, sample_actions, is_tp, padding_mask
        )
        value = self.critic(tokens, padding_mask)

        return logits, log_probs, sampled_actions, value, next_state

    def evaluate_actions(
        self,
        obs: StructuredObservation,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        state: Optional[State] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, State]:
        """
        Evaluate actions for PPO updates.
        Returns (log_prob, entropy, normalized_entropy, value, next_state).
        """
        tokens, aux = self.encoder(obs, aux=True)
        numerical = obs.numerical
        if numerical.dim() == 2:
            numerical = numerical.unsqueeze(0)
        is_tp = (numerical[:, 25, 2] > 0.5).to(self.device)
        padding_mask = self._get_padding_mask(numerical)

        return self.evaluate_actions_tokens(
            tokens, aux, numerical, is_tp, actions, action_mask, state, padding_mask
        )

    def evaluate_actions_tokens(
        self,
        tokens: torch.Tensor,
        aux: torch.Tensor,
        numerical: torch.Tensor,
        is_tp: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        state: Optional[State] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, State]:
        """
        Evaluate actions using already encoded tokens.
        """
        logits, log_prob, _, value, next_state = self.forward_tokens(
            tokens,
            aux,
            numerical,
            is_tp,
            state,
            action_mask,
            sample_actions=False,
            actions=actions,
            padding_mask=padding_mask,
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
        obs: StructuredObservation,
        action_taken: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        state: Optional[State] = None,
    ):
        logits, _, _, _, _ = self(
            obs, state, action_mask, sample_actions=False, actions=action_taken
        )
        return logits
