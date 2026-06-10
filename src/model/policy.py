from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor  # just to shorten type defs
from torch.distributions import Categorical

from src.lookups import ACT_SIZE
from src.model.cls_reducer import CLSReducer
from src.model.fused_token_encoder import FusedTokenEncoder
from src.model.structured_observation import (
    ALLY_POKE_TOKENS,
    NUM_IDX_ORIG_IDX_RATIO,
    NUM_IDX_TEAM_PREVIEW,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TARGET_SEQ_INDICES,
    TOKEN_IDX_GLOBAL_FIELD_NUMERIC,
    SideId,
    StructuredObservation,
)


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

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ValueHead(nn.Module):
    """Stateful critic value path with gradient scaling."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 1024,  # double that of policy heads
        scale: float = 0.05,
    ):
        super().__init__()
        self.scale = scale
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

    def forward(self, z: Tensor) -> Tensor:
        # scale the gradient flowing back to the trunk
        if z.requires_grad and self.scale < 1.0:
            z = z.detach() + self.scale * (z - z.detach())

        return self.net(z).squeeze(-1)


class ActorPolicy(nn.Module):
    """Stateful actor policy path."""

    target_seq_indices: Tensor
    ally_poke_tokens: Tensor
    batch_indices: Tensor
    all_a: Tensor

    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayer: int,
        act_size: int,
        side_emb: nn.Embedding,
        seq_len: int = SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.act_size = act_size
        self.side_emb = side_emb

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
        self.target_self_multi_emb = nn.Parameter(torch.empty(d_act_emb))

        def make_proj(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))

        self.actor_proj = make_proj(d_model, d_act_emb)
        self.target_proj = make_proj(d_model, d_act_emb)
        self.move_proj = make_proj(d_model, d_act_emb)
        self.side_proj = nn.Linear(d_model, d_act_emb)

        for p in [
            self.pass_emb,
            self.tp_meta_emb,
            self.switch_meta_emb,
            self.move_meta_emb,
            self.mega_meta_emb,
            self.target_self_multi_emb,
        ]:
            init.normal_(p, mean=0, std=0.02)

        for seq_module in [self.actor_proj, self.target_proj, self.move_proj]:
            for module in seq_module.modules():
                if isinstance(module, nn.Linear):
                    init.orthogonal_(module.weight, gain=1.0)
                    init.zeros_(module.bias)

        init.orthogonal_(self.side_proj.weight, gain=1.0)
        init.zeros_(self.side_proj.bias)

        self.register_buffer(
            "target_seq_indices", torch.tensor(TARGET_SEQ_INDICES, dtype=torch.long)
        )
        self.register_buffer("ally_poke_tokens", torch.tensor(ALLY_POKE_TOKENS, dtype=torch.long))
        self.register_buffer("all_a", torch.arange(36, dtype=torch.long))

        self.ctx_norm = nn.LayerNorm(d_act_emb)
        self.head1 = PolicyHead(d_model, act_size)  # P(a1 | z)
        self.head2 = PolicyHead(d_model + d_act_emb, act_size)  # P(a2 | z, a1)

    def _build_action_context(
        self,
        a1: Tensor,
        tokens: Tensor,
        aux: Tensor,
        numerical: Tensor,
    ) -> Tensor:
        # slightly wasteful but avoids CPU checks for assigning to mask
        B = a1.size(0)
        device = a1.device
        batch_idx = torch.arange(B, device=device)

        is_tp = (numerical[:, TOKEN_IDX_GLOBAL_FIELD_NUMERIC, NUM_IDX_TEAM_PREVIEW] > 0.5).bool()
        is_pass = (a1 == 0) & ~is_tp
        is_switch = (a1 >= 1) & (a1 <= 6) & ~is_tp
        is_move = (a1 >= 7) & ~is_tp
        is_mega = (a1 >= 27) & (a1 <= 46) & ~is_tp

        # actor embedding always comes at slot 1 and 2
        # since we order it in the obs builder
        actor_emb = self.actor_proj(tokens[:, 1:3, :]).sum(dim=1)

        # context embeddings for team preview
        idx_p1 = (a1 // 6) * 2 + 1
        idx_p2 = (a1 % 6) * 2 + 1
        tok_p1_super = tokens[batch_idx, idx_p1, :]
        tok_p1_num = tokens[batch_idx, idx_p1 + 1, :]
        tok_p2_super = tokens[batch_idx, idx_p2, :]
        tok_p2_num = tokens[batch_idx, idx_p2 + 1, :]
        tok_leads = torch.stack([tok_p1_super, tok_p1_num, tok_p2_super, tok_p2_num], dim=1)

        # set embedding for the lead pokemon
        # manual deepset impl
        tp_ctx = nn.functional.gelu(self.actor_proj(tok_leads).sum(dim=1)) + self.tp_meta_emb

        pass_ctx = actor_emb + self.pass_emb

        # switch context embeds pokemon switching out + pokemon switching in
        slot_idx = a1.clamp(1, 6)
        ally_indices = self.ally_poke_tokens
        orig_ids = torch.round(numerical[:, ally_indices + 1, NUM_IDX_ORIG_IDX_RATIO] * 6).long()
        matches = orig_ids == slot_idx.unsqueeze(-1)
        valid_match = matches.any(dim=-1)
        match_idx = matches.float().argmax(dim=-1)

        # force a index error if switch invalid
        if (is_switch & ~valid_match).any():
            raise IndexError("Invalid switch action: no matching ally pokemon found.")

        actual_seq_idx = ally_indices[match_idx]
        incoming_tok_super = tokens[batch_idx, actual_seq_idx, :]
        incoming_tok_num = tokens[batch_idx, actual_seq_idx + 1, :]
        incoming_toks = torch.stack([incoming_tok_super, incoming_tok_num], dim=1)
        incoming_proj = self.target_proj(incoming_toks).sum(dim=1)
        switch_ctx = actor_emb + incoming_proj + self.switch_meta_emb

        # move context embedding
        # if move is single target => actor emb + target emb
        # else => actor emb + learned token for self / multi target
        # mega token added if mega too
        a1_m = a1.clamp_min(7) - 7
        move_idx = (a1_m % 20) // 5
        target_idx = a1_m % 5

        mapped_target = self.target_seq_indices[target_idx]
        target_toks_super = tokens[batch_idx, mapped_target, :]
        target_toks_num = tokens[batch_idx, mapped_target + 1, :]
        target_toks = torch.stack([target_toks_super, target_toks_num], dim=1)
        target_toks_proj = self.target_proj(target_toks).sum(dim=1)

        target_toks_proj = torch.where(
            (target_idx == 2).unsqueeze(-1),
            torch.zeros_like(target_toks_proj),
            target_toks_proj,
        )

        is_ally = target_idx < 2
        is_opp = target_idx > 2

        # project side embeddings from the shared encoder
        side_weights_proj = self.side_proj(self.side_emb.weight)
        ally_meta = side_weights_proj[int(SideId.ALLY)].unsqueeze(0)
        opp_meta = side_weights_proj[int(SideId.OPPONENT)].unsqueeze(0)
        self_meta = self.target_self_multi_emb.unsqueeze(0)

        target_meta = torch.where(
            is_ally.unsqueeze(-1),
            ally_meta,
            torch.where(is_opp.unsqueeze(-1), opp_meta, self_meta),
        )

        move_meta = torch.where(is_mega.unsqueeze(-1), self.mega_meta_emb, self.move_meta_emb)
        move_emb = aux[batch_idx, move_idx, :]
        move_ctx = actor_emb + self.move_proj(move_emb) + target_toks_proj + target_meta + move_meta

        # combine all contexts
        ctx = torch.zeros(B, self.pass_emb.size(0), device=device)
        ctx = torch.where(is_tp.unsqueeze(-1), tp_ctx, ctx)
        ctx = torch.where(is_pass.unsqueeze(-1), pass_ctx, ctx)
        ctx = torch.where(is_switch.unsqueeze(-1), switch_ctx, ctx)
        ctx = torch.where(is_move.unsqueeze(-1), move_ctx, ctx)

        return self.ctx_norm(ctx)

    def forward(
        self,
        tokens: Tensor,
        aux: Tensor,
        numerical: Tensor,
        state: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        sample: bool = True,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        B = tokens.size(0)
        device = tokens.device

        # compute metadata internally to simplify signature
        is_tp = (numerical[:, TOKEN_IDX_GLOBAL_FIELD_NUMERIC, NUM_IDX_TEAM_PREVIEW] > 0.5).bool()

        # fainted mons stay visible to attention instead of masking
        z, next_state = self.reducer(tokens, state, None)

        logits1 = self.head1(z)

        # action eval
        if actions is not None:
            a1 = actions[:, 0]
            a1_emb = self._build_action_context(a1, tokens, aux, numerical)
            logits2 = self.head2(torch.cat([z, a1_emb], dim=-1))
            logits = torch.stack([logits1, logits2], dim=1)

            if action_mask is not None:
                logits = self._apply_sequential_masks(logits, a1, action_mask, is_tp)

            dist1 = Categorical(logits=logits[:, 0])
            dist2 = Categorical(logits=logits[:, 1])
            log_probs = dist1.log_prob(actions[:, 0]) + dist2.log_prob(actions[:, 1])

            return logits, log_probs, None, next_state, z

        # inference mode
        if not sample:
            # Return logits with placeholder for second action if not sampling or evaluating
            logits = torch.stack([logits1, torch.zeros_like(logits1)], dim=1)
            return logits, torch.zeros(B, device=device), None, next_state, z

        # Sampling with sequential constraints
        m1 = action_mask[:, 0] if action_mask is not None else None
        l1 = logits1 if m1 is None else logits1.masked_fill(m1 == 0, float("-inf"))

        dist1 = Categorical(logits=l1)
        a1 = dist1.sample()

        a1_emb = self._build_action_context(a1, tokens, aux, numerical)
        logits2 = self.head2(torch.cat([z, a1_emb], dim=-1))

        logits = torch.stack([logits1, logits2], dim=1)
        if action_mask is not None:
            logits = self._apply_sequential_masks(logits, a1, action_mask, is_tp)

        dist2 = Categorical(logits=logits[:, 1])
        a2 = dist2.sample()

        log_probs = dist1.log_prob(a1) + dist2.log_prob(a2)
        sampled_actions = torch.stack([a1, a2], dim=-1)

        return logits, log_probs, sampled_actions, next_state, z

    def _apply_sequential_masks(
        self,
        logits: Tensor,
        action1: Tensor,
        action_mask: Tensor,
        is_tp: Tensor,
    ) -> Tensor:
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
        p1_1 = action1 // 6 + 1  # (B,) — meaningful only for tp rows
        p2_1 = action1 % 6 + 1  # (B,)
        p1_2 = self.all_a // 6 + 1  # (36,)
        p2_2 = self.all_a % 6 + 1  # (36,)
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
        nlayer=4,
    ):
        super().__init__()
        self.seq_len, self.feat_dim = obs_dim
        self.act_size = act_size
        self.d_model = d_model

        # shared backbone + policy head
        self.encoder = FusedTokenEncoder(d_model, nhead, d_model * 4)
        self.actor = ActorPolicy(
            d_model, nhead, nlayer, act_size, self.encoder.side_emb, self.seq_len + 1
        )

        # value head
        self.critic = ValueHead(d_model)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        obs: StructuredObservation,
        state: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        sample_actions: bool = True,
        actions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Main forward pass. Matches original PolicyNet signature for compatibility.
        """
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

        tokens, aux = self.encoder(obs, action_mask=action_mask, aux=True)

        numerical = obs.numerical
        if numerical.dim() == 2:
            numerical = numerical.unsqueeze(0)

        if actions is not None:
            actions = actions.to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)

        return self.forward_tokens(
            tokens, aux, numerical, state, action_mask, sample_actions, actions
        )

    def forward_tokens(
        self,
        tokens: Tensor,
        aux: Tensor,
        numerical: Tensor,
        state: Optional[Tensor] = None,
        action_mask: Optional[Tensor] = None,
        sample_actions: bool = True,
        actions: Optional[Tensor] = None,
        is_warmup: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Forward pass using already encoded tokens.
        """
        if tokens.size(-2) == self.seq_len:
            tokens = self.encoder.append_action_mask_token(tokens, action_mask)

        logits, log_probs, sampled_actions, next_state, z = self.actor(
            tokens, aux, numerical, state, action_mask, actions, sample_actions
        )
        if is_warmup:
            z = z.detach()
        value = self.critic(z)

        return logits, log_probs, sampled_actions, value, next_state

    def evaluate_actions(
        self,
        obs: StructuredObservation,
        actions: Tensor,
        action_mask: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Evaluate actions for PPO updates.
        Returns (log_prob, entropy, normalized_entropy, value, next_state).
        """
        if action_mask is not None:
            action_mask = action_mask.to(self.device)
            if action_mask.dim() == 2:
                action_mask = action_mask.unsqueeze(0)

        tokens, aux = self.encoder(obs, action_mask=action_mask, aux=True)
        numerical = obs.numerical
        if numerical.dim() == 2:
            numerical = numerical.unsqueeze(0)

        return self.evaluate_actions_tokens(tokens, aux, numerical, actions, action_mask, state)

    def evaluate_actions_tokens(
        self,
        tokens: Tensor,
        aux: Tensor,
        numerical: Tensor,
        actions: Tensor,
        action_mask: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
        is_warmup: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Evaluate actions using already encoded tokens.
        """
        logits, log_prob, _, value, next_state = self.forward_tokens(
            tokens,
            aux,
            numerical,
            state,
            action_mask,
            sample_actions=False,
            actions=actions,
            is_warmup=is_warmup,
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
        action_taken: Tensor,
        action_mask: Optional[Tensor],
        state: Optional[Tensor] = None,
    ):
        logits, _, _, _, _ = self(
            obs, state, action_mask, sample_actions=False, actions=action_taken
        )
        return logits
