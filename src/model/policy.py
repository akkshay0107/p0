from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.distributions import Categorical

from src.lookups import ACT_SIZE
from src.model.cls_reducer import CLSReducer
from src.model.fused_token_encoder import FusedTokenEncoder
from src.model.structured_observation import (
    ALLY_NUM_TOKENS,
    ALLY_POKE_TOKENS,
    NUM_IDX_ORIG_IDX_RATIO,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TARGET_SEQ_INDICES,
    TEAM_SIZE,
    TOKEN_IDX_CLS,
    StructuredObservation,
    is_teampreview,
)

# Only these entities are ever pointed at: the 6 allies (switch/TP/ally targets)
# and the 2 opponent actives (move targets). Opponent bench rows get no keys.
N_KEY_ENTITIES = TEAM_SIZE + 2

# Action Space Layout Constants
PASS_START = 0
PASS_END = 1
SWITCH_START = 1
SWITCH_END = 7
MOVE_START = 7
MOVE_END = 27
MEGA_START = 27
MEGA_END = 47
MEGA_STRUGGLE_START = 47
STRUGGLE_START = 48
TP_START = 0
TP_END = 36


class EncodedObs(NamedTuple):
    tokens: Tensor
    aux: Tensor
    numerical: Tensor

    def step(self, n: int, t: int) -> EncodedObs:
        return EncodedObs(
            tokens=self.tokens[t, :n],
            aux=self.aux[t, :n],
            numerical=self.numerical[t, :n],
        )


class ActOutput(NamedTuple):
    actions: Tensor
    log_probs: Tensor
    value: Tensor
    state: Tensor


class EvalOutput(NamedTuple):
    log_probs: Tensor
    entropy: Tensor
    norm_entropy: Tensor
    value: Tensor
    state: Tensor
    logits: Tensor


class ValueHead(nn.Module):
    """Feedforward critic head over the recurrent CLS summary."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 1024,
    ):
        super().__init__()
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
        return self.net(z).squeeze(-1)


class ActorPolicy(nn.Module):
    """Stateful actor policy path using Pointer Head."""

    target_entity_indices: Tensor
    ally_poke_entities: Tensor
    ally_num_tokens: Tensor
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
        self.d_model = d_model
        self.d_k = d_model // 4

        self.reducer = CLSReducer(
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer,
            use_history=True,
        )

        self.w_k_super = nn.Linear(d_model, self.d_k)
        self.w_k_num = nn.Linear(d_model, self.d_k)
        self.w_k_move = nn.Linear(d_model, self.d_k)

        # fused query projector for the 4 query types
        # switch, move, pass, teampreview (mega reuses q_move with mega_emb keys)
        self.q_proj1 = nn.Linear(d_model, 4 * self.d_k)
        self.q_proj2 = nn.Linear(d_model + self.d_k, 4 * self.d_k)

        self.joint_move_mlp = nn.Sequential(
            nn.Linear(3 * self.d_k, self.d_k), nn.GELU(), nn.Linear(self.d_k, 1)
        )
        self.joint_tp_mlp = nn.Sequential(
            nn.Linear(3 * self.d_k, self.d_k), nn.GELU(), nn.Linear(self.d_k, 1)
        )

        self.mega_emb = nn.Parameter(torch.empty(self.d_k))
        self.pass_key = nn.Parameter(torch.empty(self.d_k))
        self.struggle_key = nn.Parameter(torch.empty(self.d_k))
        self.target_self_key = nn.Parameter(torch.empty(self.d_k))
        self.pointer_temp = nn.Parameter(torch.tensor(0.01))

        # entity i is built from the pokemon token pair (1 + 2i, 2 + 2i); the
        # learned self key is appended after the N_KEY_ENTITIES real entities
        target_entities = [
            N_KEY_ENTITIES if t == TOKEN_IDX_CLS else (t - 1) // 2 for t in TARGET_SEQ_INDICES
        ]
        self.register_buffer(
            "target_entity_indices", torch.tensor(target_entities, dtype=torch.long)
        )
        self.register_buffer(
            "ally_poke_entities",
            torch.tensor([(t - 1) // 2 for t in ALLY_POKE_TOKENS], dtype=torch.long),
        )
        self.register_buffer("ally_num_tokens", torch.tensor(ALLY_NUM_TOKENS, dtype=torch.long))
        self.register_buffer("all_a", torch.arange(TP_END, dtype=torch.long))

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        init.normal_(self.mega_emb, std=0.02)
        init.normal_(self.pass_key, std=0.02)
        init.normal_(self.struggle_key, std=0.02)
        init.normal_(self.target_self_key, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.orthogonal_(module.weight, gain=1.0)
                init.zeros_(module.bias)

    def _compute_keys(self, tokens_ctx: Tensor) -> Tensor:
        B = tokens_ctx.size(0)
        k_tokens = tokens_ctx[:, : 2 * N_KEY_ENTITIES]
        k_super = self.w_k_super(k_tokens[:, 0::2])
        k_num = self.w_k_num(k_tokens[:, 1::2])
        k_entity = k_super + k_num

        k_self = self.target_self_key.unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
        k_entity_extended = torch.cat([k_entity, k_self], dim=1)

        return k_entity_extended

    def _compute_pointer_logits(
        self,
        z: Tensor,
        k_entity_extended: Tensor,
        aux_moves: Tensor,
        numerical: Tensor,
        head_idx: int,
        ctx_a1: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B = z.size(0)
        device = z.device

        k_moves = self.w_k_move(aux_moves)

        if head_idx == 0:
            q_all = self.q_proj1(z)
        else:
            assert ctx_a1 is not None, "ctx_a1 must be provided for head 2"
            z_ctx = torch.cat([z, ctx_a1], dim=-1)
            q_all = self.q_proj2(z_ctx)

        q_switch, q_move, q_pass, q_tp = torch.split(q_all, self.d_k, dim=-1)

        # one scratch column past act_size absorbs the switch scatter of empty
        # ally rows (orig ratio 0), which would otherwise land on the pass slot
        logits = torch.zeros((B, self.act_size + 1), device=device)
        action_keys = torch.zeros(B, self.act_size + 1, self.d_k, device=device)

        logits[:, PASS_START] = ((q_pass * self.pass_key).sum(dim=-1) / math.sqrt(self.d_k)).to(logits.dtype)
        action_keys[:, PASS_START] = self.pass_key.unsqueeze(0).expand(B, -1).to(action_keys.dtype)

        k_ally = k_entity_extended[:, self.ally_poke_entities, :]
        switch_scores = torch.einsum("bd,bnd->bn", q_switch, k_ally) / math.sqrt(self.d_k)
        orig_ids = torch.round(
            numerical[:, self.ally_num_tokens, NUM_IDX_ORIG_IDX_RATIO] * 6
        ).long()
        orig_ids = torch.where(orig_ids > 0, orig_ids, self.act_size)
        logits.scatter_(1, orig_ids, switch_scores.to(logits.dtype))
        action_keys.scatter_(1, orig_ids.unsqueeze(-1).expand(-1, -1, self.d_k), k_ally.to(action_keys.dtype))

        k_targets = k_entity_extended[:, self.target_entity_indices, :]
        k_moves_grid = k_moves.unsqueeze(2).expand(-1, -1, 5, -1).reshape(B, 20, self.d_k)
        k_targets_grid = k_targets.unsqueeze(1).expand(-1, 4, -1, -1).reshape(B, 20, self.d_k)
        q_move_grid = q_move.unsqueeze(1).expand(-1, 20, -1)

        joint_move_input = torch.cat([k_moves_grid, k_targets_grid, q_move_grid], dim=-1)
        move_scores = self.joint_move_mlp(joint_move_input).squeeze(-1)
        move_ctxs = k_moves_grid + k_targets_grid

        logits[:, MOVE_START:MOVE_END] = move_scores.to(logits.dtype)
        action_keys[:, MOVE_START:MOVE_END, :] = move_ctxs.to(action_keys.dtype)

        k_mega_moves_grid = (
            (k_moves + self.mega_emb).unsqueeze(2).expand(-1, -1, 5, -1).reshape(B, 20, self.d_k)
        )
        joint_mega_input = torch.cat([k_mega_moves_grid, k_targets_grid, q_move_grid], dim=-1)
        mega_scores = self.joint_move_mlp(joint_mega_input).squeeze(-1)
        mega_ctxs = k_mega_moves_grid + k_targets_grid

        logits[:, MEGA_START:MEGA_END] = mega_scores.to(logits.dtype)
        action_keys[:, MEGA_START:MEGA_END, :] = mega_ctxs.to(action_keys.dtype)

        mega_struggle_key = self.struggle_key + self.mega_emb
        logits[:, MEGA_STRUGGLE_START] = ((q_move * mega_struggle_key).sum(dim=-1) / math.sqrt(
            self.d_k
        )).to(logits.dtype)
        action_keys[:, MEGA_STRUGGLE_START] = mega_struggle_key.unsqueeze(0).expand(B, -1).to(action_keys.dtype)

        logits[:, STRUGGLE_START] = ((q_move * self.struggle_key).sum(dim=-1) / math.sqrt(self.d_k)).to(logits.dtype)
        action_keys[:, STRUGGLE_START] = self.struggle_key.unsqueeze(0).expand(B, -1).to(action_keys.dtype)

        is_tp = is_teampreview(numerical).unsqueeze(-1)

        k_lead_grid = k_ally.unsqueeze(2).expand(-1, -1, 6, -1).reshape(B, TP_END, self.d_k)
        k_back_grid = k_ally.unsqueeze(1).expand(-1, 6, -1, -1).reshape(B, TP_END, self.d_k)
        q_tp_grid = q_tp.unsqueeze(1).expand(-1, TP_END, -1)

        joint_tp_input = torch.cat([k_lead_grid, k_back_grid, q_tp_grid], dim=-1)
        tp_scores = self.joint_tp_mlp(joint_tp_input).squeeze(-1)

        tp_ctxs = k_lead_grid + k_back_grid

        logits[:, TP_START:TP_END] = torch.where(is_tp, tp_scores, logits[:, TP_START:TP_END]).to(logits.dtype)
        action_keys[:, TP_START:TP_END, :] = torch.where(
            is_tp.unsqueeze(-1), tp_ctxs, action_keys[:, TP_START:TP_END, :]
        ).to(action_keys.dtype)

        # prevent pointer temp from becoming negative / 0
        logits = logits[:, : self.act_size] * self.pointer_temp.clamp_min(1e-4)
        return logits, action_keys[:, : self.act_size]

    @staticmethod
    def _apply_top_p(logits: Tensor, top_p: float) -> Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        return torch.empty_like(logits).scatter(-1, sorted_indices, sorted_logits)

    def sample(
        self,
        enc: EncodedObs,
        action_mask: Tensor,
        state: Tensor,
        *,
        top_p: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z, next_state, tokens_ctx = self.reducer(enc.tokens, state, None)
        k_entity_extended = self._compute_keys(tokens_ctx)

        logits1, keys1 = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 0], enc.numerical, head_idx=0
        )
        logits1 = logits1.masked_fill(action_mask[:, 0] == 0, float("-inf"))
        sample_logits1 = self._apply_top_p(logits1, top_p) if top_p < 1.0 else logits1

        dist1 = Categorical(logits=sample_logits1)
        a1 = dist1.sample()

        batch_idx = torch.arange(a1.size(0), device=a1.device)
        ctx_a1 = keys1[batch_idx, a1]

        logits2, _ = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 1], enc.numerical, head_idx=1, ctx_a1=ctx_a1
        )

        logits = torch.stack([logits1, logits2], dim=1)
        logits = self._apply_sequential_masks(
            logits, a1, action_mask, is_teampreview(enc.numerical)
        )
        sample_logits2 = self._apply_top_p(logits[:, 1], top_p) if top_p < 1.0 else logits[:, 1]

        dist2 = Categorical(logits=sample_logits2)
        a2 = dist2.sample()
        log_probs = dist1.log_prob(a1) + dist2.log_prob(a2)
        actions = torch.stack([a1, a2], dim=-1)
        return actions, log_probs, next_state, z

    def score(
        self,
        enc: EncodedObs,
        action_mask: Tensor,
        actions: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z, next_state, tokens_ctx = self.reducer(enc.tokens, state, None)
        k_entity_extended = self._compute_keys(tokens_ctx)

        logits1, keys1 = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 0], enc.numerical, head_idx=0
        )
        a1 = actions[:, 0]

        batch_idx = torch.arange(a1.size(0), device=a1.device)
        ctx_a1 = keys1[batch_idx, a1]

        logits2, _ = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 1], enc.numerical, head_idx=1, ctx_a1=ctx_a1
        )

        logits = torch.stack([logits1, logits2], dim=1)
        logits = self._apply_sequential_masks(
            logits, a1, action_mask, is_teampreview(enc.numerical)
        )

        dist1 = Categorical(logits=logits[:, 0])
        dist2 = Categorical(logits=logits[:, 1])
        log_probs = dist1.log_prob(actions[:, 0]) + dist2.log_prob(actions[:, 1])
        return logits, log_probs, next_state, z

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
        # Mega moves are 27-46, plus 47 for Mega Struggle.
        mega_mask = (action1 >= 27) & (action1 <= 47) & (~is_tp)
        mask2[mega_mask, 27:48] = False

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

    Both heads read the stateful (recurrent) reducer output: the actor builds
    pointer logits from it, the critic is a feedforward head on its CLS.
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
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initial_state(self, batch_size: int) -> Tensor:
        return self.actor.reducer.hg_init.expand(batch_size, -1, -1).to(self.device)

    def encode(
        self,
        obs: StructuredObservation,
        action_mask: Tensor,
    ) -> EncodedObs:
        if obs.categorical.dim() != 3:
            raise ValueError("PolicyNet.encode expects a batched StructuredObservation.")
        tokens, aux = self.encoder(obs, action_mask)
        return EncodedObs(tokens=tokens, aux=aux, numerical=obs.numerical)

    def act(
        self,
        enc: EncodedObs,
        action_mask: Tensor,
        state: Tensor,
        *,
        top_p: float = 1.0,
    ) -> ActOutput:
        # NOTE: with top_p < 1.0 the returned log_probs are taken w.r.t. the
        # truncated sampling distribution, not the full policy, while `evaluate`
        # always scores against the full distribution. Rollouts collected for
        # PPO training must therefore use top_p=1.0 (the default) or the
        # importance ratios will be wrong top_p < 1.0 is for
        # evaluation/play only.
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
        actions, log_probs, next_state, z = self.actor.sample(enc, action_mask, state, top_p=top_p)
        return ActOutput(actions, log_probs, self.critic(z), next_state)

    def evaluate(
        self,
        enc: EncodedObs,
        action_mask: Tensor,
        actions: Tensor,
        state: Tensor,
        *,
        critic_only: bool = False,
    ) -> EvalOutput:
        logits, log_probs, next_state, z = self.actor.score(enc, action_mask, actions, state)
        if critic_only:
            z = z.detach()
        value = self.critic(z)

        dist1 = Categorical(logits=logits[:, 0])
        dist2 = Categorical(logits=logits[:, 1])
        entropy = dist1.entropy() + dist2.entropy()

        v1 = torch.isfinite(logits[:, 0]).sum(-1).float().clamp_min(1.0)
        v2 = torch.isfinite(logits[:, 1]).sum(-1).float().clamp_min(1.0)
        max_entropy = torch.log(v1) + torch.log(v2)

        norm_entropy = torch.where(
            max_entropy > 0,
            entropy / max_entropy.clamp_min(1e-8),
            torch.zeros_like(entropy),
        )

        return EvalOutput(log_probs, entropy, norm_entropy, value, next_state, logits)

    def act_obs(
        self,
        obs: StructuredObservation,
        action_mask: Tensor,
        state: Tensor,
        *,
        top_p: float = 1.0,
    ) -> ActOutput:
        return self.act(self.encode(obs, action_mask), action_mask, state, top_p=top_p)

    def evaluate_obs(
        self,
        obs: StructuredObservation,
        action_mask: Tensor,
        actions: Tensor,
        state: Tensor,
        *,
        critic_only: bool = False,
    ) -> EvalOutput:
        return self.evaluate(
            self.encode(obs, action_mask),
            action_mask,
            actions,
            state,
            critic_only=critic_only,
        )
