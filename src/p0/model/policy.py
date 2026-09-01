"""
Policy network: fixed memory reducer, fused token encoder, and the 49-action joint heads.

Defines ActorPolicy (stateless action sampling/scoring with sequential joint-action masks)
and PolicyNet (full actor+critic model). Shared by live play, rollouts, and behaviour-cloning
candidate marginalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from p0.battle.actions import (
    ACT_SIZE,
    MOVE_END,
    MOVE_START,
)
from p0.battle.actions import (
    FORCED_ACTION as STRUGGLE_START,
)
from p0.battle.actions import (
    MEGA_FORCED_ACTION as MEGA_STRUGGLE_START,
)
from p0.battle.actions import (
    MEGA_MOVE_END as MEGA_END,
)
from p0.battle.actions import (
    MEGA_MOVE_START as MEGA_START,
)
from p0.battle.actions import (
    PASS_ACTION as PASS_START,
)
from p0.model.architecture_contract import (
    HISTORY_WINDOW,
    SELF_TARGET_SENTINEL,
    SERIES_SLOTS,
)
from p0.model.cls_reducer import MemoryReducer, ReducerOutput
from p0.model.config import ModelConfig
from p0.model.fused_token_encoder import FusedTokenEncoder
from p0.model.resources import RuntimeResources
from p0.model.series_context import (
    DynamicSeriesResampler,
)
from p0.model.structured_observation import (
    ALLY_POKE_TOKENS,
    NUM_IDX_ORIG_IDX_RATIO,
    NUMERICAL_WIDTH,
    SEQUENCE_LENGTH,
    TARGET_SEQ_INDICES,
    TEAM_SIZE,
    StructuredObservation,
    is_teampreview,
)
from p0.model.swiglu_encoder import MODEL_INIT_STD, initialize_module

# Only these entities are ever pointed at: the 6 allies (switch/TP/ally targets)
# and the 2 opponent actives (move targets). Opponent bench rows get no keys.
N_KEY_ENTITIES = TEAM_SIZE + 2

TP_START = 0
TP_END = TEAM_SIZE**2


@dataclass(frozen=True, slots=True)
class EncodedObs:
    tokens: Tensor
    aux: Tensor
    numerical: Tensor
    phase: Tensor
    local_history_token: Tensor

    def __post_init__(self) -> None:
        batch_shape = self.tokens.shape[:-2]
        if self.phase.shape != batch_shape or self.phase.dtype != torch.bool:
            raise ValueError("phase must be a boolean tensor matching the encoded batch shape")
        if self.local_history_token.shape != (*batch_shape, self.tokens.size(-1)):
            raise ValueError("local_history_token must match the encoded batch and model width")

    def __getitem__(self, index) -> EncodedObs:
        return EncodedObs(
            tokens=self.tokens[index],
            aux=self.aux[index],
            numerical=self.numerical[index],
            phase=self.phase[index],
            local_history_token=self.local_history_token[index],
        )


@dataclass(frozen=True, slots=True)
class MemoryInputs:
    """Immutable series and battle-history inputs for one policy batch."""

    series_tokens: Tensor
    series_mask: Tensor
    history_tokens: Tensor
    history_mask: Tensor

    @classmethod
    def empty(
        cls,
        batch_size: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MemoryInputs:
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        return cls(
            series_tokens=torch.zeros(
                (batch_size, SERIES_SLOTS, d_model), device=device, dtype=dtype
            ),
            series_mask=torch.zeros((batch_size, SERIES_SLOTS), device=device, dtype=torch.bool),
            history_tokens=torch.zeros(
                (batch_size, HISTORY_WINDOW, d_model), device=device, dtype=dtype
            ),
            history_mask=torch.zeros((batch_size, HISTORY_WINDOW), device=device, dtype=torch.bool),
        )


@dataclass(frozen=True, slots=True)
class PreparedDecision:
    """Encoded observation and memory-aware readout for one policy decision."""

    encoded: EncodedObs
    reduced: ReducerOutput


class ActOutput(NamedTuple):
    """
    Outputs for one action decision.

    value is predicted from the post-memory cls readout. The
    history_token is the pre-memory h_t snapshot returned by the reducer
    for storage as the next turn's history.
    """

    actions: Tensor
    log_probs: Tensor
    value: Tensor
    history_token: Tensor


class EvalOutput(NamedTuple):
    """Outputs for evaluating an existing action decision."""

    log_probs: Tensor
    entropy: Tensor
    norm_entropy: Tensor
    value: Tensor
    history_token: Tensor
    logits: Tensor


def _require_matching_batch(reduced: ReducerOutput, enc: EncodedObs) -> None:
    """Reject a reduced batch that was not produced from these observations."""
    if reduced.cls.size(0) != enc.tokens.size(0):
        raise ValueError(
            f"Reduced batch of {reduced.cls.size(0)} does not match "
            f"{enc.tokens.size(0)} encoded observations"
        )


class ValueHead(nn.Module):
    """Linear critic over the normalized Transformer readout."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, 1),
        )
        initialize_module(self)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


class ActorPolicy(nn.Module):
    """Stateless actor policy path using the fixed memory reducer."""

    target_entity_indices: Tensor
    ally_poke_entities: Tensor
    ally_token_pos: Tensor
    batch_indices: Tensor
    all_a: Tensor

    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayer: int,
        act_size: int,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()
        self.act_size = act_size
        self.d_model = d_model
        self.d_k = d_model // 4

        self.reducer = MemoryReducer(
            d_model=d_model,
            nhead=nhead,
            nlayer=nlayer,
            dim_feedforward=dim_feedforward,
        )

        self.w_k_entity = nn.Linear(d_model, self.d_k)
        self.w_k_move = nn.Linear(d_model, self.d_k)
        self.entity_key_norm = nn.RMSNorm(self.d_k)

        # fused query projector for the 4 query types
        # switch, move, pass, and team preview, with mega reusing the move query
        self.q_proj1 = nn.Linear(d_model + self.d_k, 4 * self.d_k)
        self.q_proj2 = nn.Linear(d_model + 2 * self.d_k, 4 * self.d_k)

        self.move_target_proj = nn.Sequential(
            nn.Linear(2 * self.d_k, self.d_k),
            nn.RMSNorm(self.d_k),
        )
        self.tp_pair_proj = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.RMSNorm(self.d_k),
        )

        self.mega_emb = nn.Parameter(torch.empty(self.d_k))
        self.pass_key = nn.Parameter(torch.empty(self.d_k))
        self.struggle_key = nn.Parameter(torch.empty(self.d_k))
        self.target_self_key = nn.Parameter(torch.empty(self.d_k))
        self.tp_lead_role = nn.Parameter(torch.empty(self.d_k))
        self.tp_back_role = nn.Parameter(torch.empty(self.d_k))

        # Entity keys are emitted as the first twelve outputs of the reducer;
        # the learned self key is appended after the real target entities.
        target_entities = [
            N_KEY_ENTITIES if t == SELF_TARGET_SENTINEL else t for t in TARGET_SEQ_INDICES
        ]
        self.register_buffer(
            "target_entity_indices", torch.tensor(target_entities, dtype=torch.long)
        )
        self.register_buffer(
            "ally_poke_entities",
            torch.tensor(ALLY_POKE_TOKENS, dtype=torch.long),
        )
        self.register_buffer("ally_token_pos", torch.tensor(ALLY_POKE_TOKENS, dtype=torch.long))
        self.register_buffer("all_a", torch.arange(TP_END, dtype=torch.long))

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self) -> None:
        initialize_module(self)
        for key in (
            self.mega_emb,
            self.pass_key,
            self.struggle_key,
            self.target_self_key,
            self.tp_lead_role,
            self.tp_back_role,
        ):
            nn.init.normal_(key, std=MODEL_INIT_STD)
        self.reducer.encoder.reset_parameters()

    def _compute_keys(self, tokens_ctx: Tensor) -> Tensor:
        B = tokens_ctx.size(0)
        k_entity = self.entity_key_norm(self.w_k_entity(tokens_ctx[:, :N_KEY_ENTITIES]))

        k_self = self.target_self_key.unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
        k_entity_extended = torch.cat([k_entity, k_self], dim=1)

        return k_entity_extended

    def _compute_pointer_logits(
        self,
        z: Tensor,
        k_entity_extended: Tensor,
        aux_moves: Tensor,
        numerical: Tensor,
        phase: Tensor,
        head_idx: int,
        ctx_a1: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Build phase-aware queries and score every action with attention.

        Arguments:
          z: Reduced battle-state summaries, one per batch row.
          k_entity_extended: Projected entity keys plus the learned self-target key.
          aux_moves: Encoded move tokens for the decision slot being scored.
          numerical: Observation numerics used for switch-slot routing.
          phase: Immutable team-preview flags captured during observation encoding.
          head_idx: Zero for the first decision and one for the second decision.
          ctx_a1: The exact key scored for the selected first action; required by head one.

        Returns:
          Raw scaled-dot-product logits and their corresponding action keys.
        """
        if head_idx not in (0, 1):
            raise ValueError(f"head_idx must be 0 or 1, got {head_idx}")
        if head_idx == 1 and ctx_a1 is None:
            raise ValueError("ctx_a1 must be provided for head 2")

        B = z.size(0)
        device = z.device
        pointer_scale = math.sqrt(self.d_k)

        k_moves = self.w_k_move(aux_moves)
        k_ally = k_entity_extended[:, self.ally_poke_entities, :]
        is_tp = phase

        if head_idx == 0:
            decision_owner = torch.where(
                is_tp.unsqueeze(-1),
                self.tp_lead_role.unsqueeze(0),
                k_ally[:, 0],
            )
            q_all = self.q_proj1(torch.cat([z, decision_owner], dim=-1))
        else:
            assert ctx_a1 is not None
            decision_owner = torch.where(
                is_tp.unsqueeze(-1),
                self.tp_back_role.unsqueeze(0),
                k_ally[:, 1],
            )
            z_ctx = torch.cat([z, decision_owner, ctx_a1], dim=-1)
            q_all = self.q_proj2(z_ctx)

        q_switch, q_move, q_pass, q_tp = torch.split(q_all, self.d_k, dim=-1)

        # one scratch column past the action count absorbs the switch scatter of empty
        # ally rows (orig ratio 0), which would otherwise land on the pass slot
        logits = torch.zeros((B, self.act_size + 1), device=device)
        action_keys = torch.zeros(B, self.act_size + 1, self.d_k, device=device)

        logits[:, PASS_START] = ((q_pass * self.pass_key).sum(dim=-1) / pointer_scale).to(
            logits.dtype
        )
        action_keys[:, PASS_START] = self.pass_key.unsqueeze(0).expand(B, -1).to(action_keys.dtype)

        switch_scores = torch.einsum("bd,bnd->bn", q_switch, k_ally) / pointer_scale
        orig_ids = torch.round(numerical[:, self.ally_token_pos, NUM_IDX_ORIG_IDX_RATIO] * 6).long()
        orig_ids = torch.where(orig_ids > 0, orig_ids, self.act_size)
        logits.scatter_(1, orig_ids, switch_scores.to(logits.dtype))
        action_keys.scatter_(
            1, orig_ids.unsqueeze(-1).expand(-1, -1, self.d_k), k_ally.to(action_keys.dtype)
        )

        k_targets = k_entity_extended[:, self.target_entity_indices, :]
        # Broadcast grids remain views until concatenation materializes
        # each combined move-target record once.
        k_moves_grid = k_moves.unsqueeze(2).expand(-1, -1, 5, -1)
        k_targets_grid = k_targets.unsqueeze(1).expand(-1, 4, -1, -1)

        move_action_keys = self.move_target_proj(
            torch.cat([k_moves_grid, k_targets_grid], dim=-1).reshape(B, 20, 2 * self.d_k)
        )
        move_scores = torch.einsum("bd,bnd->bn", q_move, move_action_keys) / pointer_scale

        logits[:, MOVE_START:MOVE_END] = move_scores.to(logits.dtype)
        action_keys[:, MOVE_START:MOVE_END, :] = move_action_keys.to(action_keys.dtype)

        k_mega_moves_grid = (k_moves + self.mega_emb).unsqueeze(2).expand(-1, -1, 5, -1)
        mega_action_keys = self.move_target_proj(
            torch.cat([k_mega_moves_grid, k_targets_grid], dim=-1).reshape(B, 20, 2 * self.d_k)
        )
        mega_scores = torch.einsum("bd,bnd->bn", q_move, mega_action_keys) / pointer_scale

        logits[:, MEGA_START:MEGA_END] = mega_scores.to(logits.dtype)
        action_keys[:, MEGA_START:MEGA_END, :] = mega_action_keys.to(action_keys.dtype)

        mega_struggle_key = self.struggle_key + self.mega_emb
        logits[:, MEGA_STRUGGLE_START] = (
            (q_move * mega_struggle_key).sum(dim=-1) / pointer_scale
        ).to(logits.dtype)
        action_keys[:, MEGA_STRUGGLE_START] = (
            mega_struggle_key.unsqueeze(0).expand(B, -1).to(action_keys.dtype)
        )

        logits[:, STRUGGLE_START] = ((q_move * self.struggle_key).sum(dim=-1) / pointer_scale).to(
            logits.dtype
        )
        action_keys[:, STRUGGLE_START] = (
            self.struggle_key.unsqueeze(0).expand(B, -1).to(action_keys.dtype)
        )

        is_tp_column = is_tp.unsqueeze(-1)

        tp_pair_keys_grid = (k_ally.unsqueeze(2) + k_ally.unsqueeze(1)) / math.sqrt(2.0)
        tp_pair_keys = self.tp_pair_proj(tp_pair_keys_grid.reshape(B, TP_END, self.d_k))
        tp_scores = torch.einsum("bd,bnd->bn", q_tp, tp_pair_keys) / pointer_scale

        logits[:, TP_START:TP_END] = torch.where(
            is_tp_column, tp_scores, logits[:, TP_START:TP_END]
        ).to(logits.dtype)
        action_keys[:, TP_START:TP_END, :] = torch.where(
            is_tp_column.unsqueeze(-1), tp_pair_keys, action_keys[:, TP_START:TP_END, :]
        ).to(action_keys.dtype)

        return logits[:, : self.act_size], action_keys[:, : self.act_size]

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

    def sample_reduced(
        self,
        reduced: ReducerOutput,
        enc: EncodedObs,
        action_mask: Tensor,
        *,
        top_p: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample actions from a reducer output that was already computed."""
        _require_matching_batch(reduced, enc)
        z = reduced.cls
        k_entity_extended = self._compute_keys(reduced.pokemon)

        logits1, keys1 = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 0], enc.numerical, enc.phase, head_idx=0
        )
        logits1 = logits1.masked_fill(action_mask[:, 0] == 0, float("-inf"))
        sample_logits1 = self._apply_top_p(logits1, top_p) if top_p < 1.0 else logits1

        dist1 = Categorical(logits=sample_logits1)
        a1 = dist1.sample()

        batch_idx = torch.arange(a1.size(0), device=a1.device)
        ctx_a1 = keys1[batch_idx, a1]

        logits2, _ = self._compute_pointer_logits(
            z,
            k_entity_extended,
            enc.aux[:, 1],
            enc.numerical,
            enc.phase,
            head_idx=1,
            ctx_a1=ctx_a1,
        )

        logits = torch.stack([logits1, logits2], dim=1)
        logits = self._apply_sequential_masks(logits, a1, action_mask, enc.phase)
        sample_logits2 = self._apply_top_p(logits[:, 1], top_p) if top_p < 1.0 else logits[:, 1]

        dist2 = Categorical(logits=sample_logits2)
        a2 = dist2.sample()
        log_probs = dist1.log_prob(a1) + dist2.log_prob(a2)
        actions = torch.stack([a1, a2], dim=-1)
        return actions, log_probs, z, reduced.local_history_token

    @torch.no_grad()
    def greedy_reduced(
        self,
        reduced: ReducerOutput,
        enc: EncodedObs,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Choose one legal joint action from a batch that is already reduced."""
        _require_matching_batch(reduced, enc)
        z = reduced.cls
        k_entity_extended = self._compute_keys(reduced.pokemon)

        logits1, keys1 = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 0], enc.numerical, enc.phase, head_idx=0
        )
        logits1 = logits1.masked_fill(action_mask[:, 0] == 0, float("-inf"))
        a1 = torch.argmax(logits1, dim=-1)

        batch_idx = torch.arange(a1.size(0), device=a1.device)
        ctx_a1 = keys1[batch_idx, a1]
        logits2, _ = self._compute_pointer_logits(
            z,
            k_entity_extended,
            enc.aux[:, 1],
            enc.numerical,
            enc.phase,
            head_idx=1,
            ctx_a1=ctx_a1,
        )
        logits = self._apply_sequential_masks(
            torch.stack([logits1, logits2], dim=1),
            a1,
            action_mask,
            enc.phase,
        )
        a2 = torch.argmax(logits[:, 1], dim=-1)
        actions = torch.stack([a1, a2], dim=-1)
        log_probs = F.log_softmax(logits[:, 0], dim=-1).gather(1, a1.unsqueeze(1)).squeeze(
            1
        ) + F.log_softmax(logits[:, 1], dim=-1).gather(1, a2.unsqueeze(1)).squeeze(1)
        return actions, log_probs, z, reduced.local_history_token

    def score_reduced(
        self,
        reduced: ReducerOutput,
        enc: EncodedObs,
        action_mask: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Score actions from a reducer output that was already computed."""
        _require_matching_batch(reduced, enc)
        z = reduced.cls
        k_entity_extended = self._compute_keys(reduced.pokemon)

        logits1, keys1 = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 0], enc.numerical, enc.phase, head_idx=0
        )
        a1 = actions[:, 0]

        batch_idx = torch.arange(a1.size(0), device=a1.device)
        ctx_a1 = keys1[batch_idx, a1]

        logits2, _ = self._compute_pointer_logits(
            z,
            k_entity_extended,
            enc.aux[:, 1],
            enc.numerical,
            enc.phase,
            head_idx=1,
            ctx_a1=ctx_a1,
        )

        logits = torch.stack([logits1, logits2], dim=1)
        logits = self._apply_sequential_masks(logits, a1, action_mask, enc.phase)

        dist1 = Categorical(logits=logits[:, 0])
        dist2 = Categorical(logits=logits[:, 1])
        log_probs = dist1.log_prob(actions[:, 0]) + dist2.log_prob(actions[:, 1])
        return logits, log_probs, z, reduced.local_history_token

    def _validated_offsets(
        self,
        enc: EncodedObs,
        action_mask: Tensor,
        candidate_values: Tensor,
        candidate_offsets: Tensor,
    ) -> Tensor:
        """Check the ragged candidate encoding and align its offsets to the batch."""
        batch_size = enc.tokens.size(0)
        if candidate_values.dim() != 2 or candidate_values.shape[1] != 2:
            raise ValueError("candidate_values must have shape (candidates, 2)")

        if candidate_values.dtype != torch.long:
            raise ValueError("candidate_values must use torch.long action ids")

        if candidate_offsets.dim() != 1 or candidate_offsets.numel() != batch_size + 1:
            raise ValueError("candidate_offsets must have one boundary per observation")

        if action_mask.shape != (batch_size, 2, self.act_size):
            raise ValueError("action_mask shape does not match encoded observations")

        if candidate_values.device != enc.tokens.device or action_mask.device != enc.tokens.device:
            raise ValueError("candidate tensors and action_mask must share the encoded device")

        # Inspect source offsets before transfer so CPU-produced BC batches do not
        # pay for device synchronization during scalar contract checks.
        if candidate_offsets[0].item() != 0 or candidate_offsets[
            -1
        ].item() != candidate_values.size(0):
            raise ValueError("candidate_offsets must start at zero and end at candidate count")

        if bool(torch.any(candidate_offsets[1:] < candidate_offsets[:-1]).item()):
            raise ValueError("candidate_offsets must be nondecreasing")

        offsets = candidate_offsets.to(device=enc.tokens.device, dtype=torch.long)
        return offsets

    def score_reduced_candidates(
        self,
        reduced: ReducerOutput,
        enc: EncodedObs,
        action_mask: Tensor,
        candidate_values: Tensor,
        candidate_offsets: Tensor,
        *,
        validated: bool = False,
    ) -> Tensor:
        """
        Score ragged candidates against a batch that is already reduced.

        Setting validated=True bypasses every candidate-contract check. It is
        reserved for callers that validated the complete source batch before
        moving it to the policy device.
        """
        _require_matching_batch(reduced, enc)
        if validated:
            return self._score_reduced_unchecked(
                reduced,
                enc,
                action_mask,
                candidate_values,
                candidate_offsets,
            )

        offsets = self._validated_offsets(enc, action_mask, candidate_values, candidate_offsets)
        return self._score_reduced(reduced, enc, action_mask, candidate_values, offsets)

    def _score_reduced(
        self,
        reduced: ReducerOutput,
        enc: EncodedObs,
        action_mask: Tensor,
        candidate_values: Tensor,
        offsets: Tensor,
    ) -> Tensor:
        if candidate_values.numel() == 0:
            return candidate_values.new_empty((0,), dtype=enc.tokens.dtype)
        if torch.any((candidate_values < 0) | (candidate_values >= self.act_size)):
            raise ValueError("candidate action ids are outside the action contract")
        return self._score_reduced_unchecked(
            reduced,
            enc,
            action_mask,
            candidate_values,
            offsets,
        )

    def _score_reduced_unchecked(
        self,
        reduced: ReducerOutput,
        enc: EncodedObs,
        action_mask: Tensor,
        candidate_values: Tensor,
        offsets: Tensor,
    ) -> Tensor:
        batch_size = enc.tokens.size(0)
        if candidate_values.numel() == 0:
            return candidate_values.new_empty((0,), dtype=enc.tokens.dtype)
        z = reduced.cls
        k_entity_extended = self._compute_keys(reduced.pokemon)
        logits1, keys1 = self._compute_pointer_logits(
            z, k_entity_extended, enc.aux[:, 0], enc.numerical, enc.phase, head_idx=0
        )
        counts = offsets[1:] - offsets[:-1]
        candidate_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=enc.tokens.device), counts
        )
        first_actions = candidate_values[:, 0]
        second_actions = candidate_values[:, 1]
        candidate_ctx = keys1[candidate_batch, first_actions]
        logits2, _ = self._compute_pointer_logits(
            z[candidate_batch],
            k_entity_extended[candidate_batch],
            enc.aux[candidate_batch, 1],
            enc.numerical[candidate_batch],
            enc.phase[candidate_batch],
            head_idx=1,
            ctx_a1=candidate_ctx,
        )
        candidate_logits = torch.stack((logits1[candidate_batch], logits2), dim=1)
        candidate_logits = self._apply_sequential_masks(
            candidate_logits,
            first_actions,
            action_mask[candidate_batch],
            enc.phase[candidate_batch],
        )
        log_prob_first = F.log_softmax(candidate_logits[:, 0], dim=-1).gather(
            1, first_actions.unsqueeze(1)
        )
        log_prob_second = F.log_softmax(candidate_logits[:, 1], dim=-1).gather(
            1, second_actions.unsqueeze(1)
        )
        return (log_prob_first + log_prob_second).squeeze(1)

    def unmasked_first_slot_logits(self, reduced: ReducerOutput, enc: EncodedObs) -> Tensor:
        """
        First-slot logits before legality masking, for legality diagnostics.

        Reuses an already-reduced batch so the diagnostic costs one pointer-head pass
        rather than a second reducer pass.
        """
        logits, _ = self._compute_pointer_logits(
            reduced.cls,
            self._compute_keys(reduced.pokemon),
            enc.aux[:, 0],
            enc.numerical,
            enc.phase,
            head_idx=0,
        )
        return logits

    def _apply_sequential_masks(
        self,
        logits: Tensor,
        action1: Tensor,
        action_mask: Tensor,
        is_tp: Tensor,
    ) -> Tensor:
        mask2 = action_mask[:, 1].clone().bool()

        # Both active Pokémon cannot switch into the same team slot.
        switch_mask = (1 <= action1) & (action1 <= 6) & (~is_tp)
        mask2[switch_mask, action1[switch_mask]] = 0

        # A first-slot Mega action masks the full second-slot Mega range because
        # only one active Pokémon may Mega Evolve per turn.
        mega_mask = (action1 >= 27) & (action1 <= 47) & (~is_tp)
        mask2[mega_mask, 27:48] = False

        # A double pass is illegal unless the fallback below finds no other action.
        pass_mask = (action1 == 0) & (~is_tp)
        mask2[pass_mask, 0] = False

        # Team preview requires four unique Pokémon across the lead and back pairs.
        # Phase-gated vectorization avoids synchronizing on the team-preview flag.
        p1_1 = action1 // TEAM_SIZE  # (B,) — meaningful only for tp rows
        p2_1 = action1 % TEAM_SIZE  # (B,)
        p1_2 = self.all_a // TEAM_SIZE  # team preview left positions
        p2_2 = self.all_a % TEAM_SIZE  # team preview right positions
        tp_overlap = (
            (p1_2[None] == p1_1[:, None])
            | (p1_2[None] == p2_1[:, None])
            | (p2_2[None] == p1_1[:, None])
            | (p2_2[None] == p2_1[:, None])
        )  # batch by team preview action count
        mask2[:, :TP_END] = mask2[:, :TP_END] & ~(is_tp[:, None] & tp_overlap)

        # Forced-action edge cases need PASS_START as a total fallback.
        no_valid = mask2.sum(-1) == 0
        mask2[no_valid, 0] = True

        l1 = logits[:, 0].masked_fill(action_mask[:, 0] == 0, float("-inf"))
        l2 = logits[:, 1].masked_fill(~mask2, float("-inf"))
        return torch.stack([l1, l2], dim=1)


class PolicyNet(nn.Module):
    """Policy network with explicit immutable memory inputs."""

    def __init__(
        self,
        config: ModelConfig,
        resources: RuntimeResources,
    ):
        super().__init__()
        self.config = config
        self.resources = resources
        self.seq_len = SEQUENCE_LENGTH
        self.feat_dim = NUMERICAL_WIDTH
        self.act_size = ACT_SIZE
        self.d_model = config.d_model

        self.encoder = FusedTokenEncoder(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            self.resources,
        )
        self.actor = ActorPolicy(
            config.d_model,
            config.nhead,
            config.reducer_layers,
            ACT_SIZE,
            dim_feedforward=config.dim_feedforward,
        )

        self.critic = ValueHead(config.d_model)

        self.series = DynamicSeriesResampler(
            config.d_model,
            config.nhead,
            config.dim_feedforward,
            num_layers=1,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_series(self, histories) -> tuple[Tensor, Tensor]:
        """Encode completed game turn histories into series tokens and mask."""
        return self.series(histories)

    def encode(
        self,
        obs: StructuredObservation,
        action_mask: Tensor,
    ) -> EncodedObs:
        if obs.categorical.dim() != 3:
            raise ValueError("PolicyNet.encode expects a batched StructuredObservation.")
        tokens, aux = self.encoder(obs, action_mask)
        return EncodedObs(
            tokens=tokens,
            aux=aux,
            numerical=obs.numerical,
            phase=is_teampreview(obs.numerical).detach().clone(),
            local_history_token=self.actor.reducer.local_summary(tokens),
        )

    def prepare(self, encoded: EncodedObs, memory: MemoryInputs) -> PreparedDecision:
        """Build the memory-aware readout from one encoded observation batch."""
        reduced = self.actor.reducer.reduce(
            encoded.local_history_token,
            encoded.tokens,
            memory.series_tokens,
            memory.series_mask,
            memory.history_tokens,
            memory.history_mask,
        )
        return PreparedDecision(encoded=encoded, reduced=reduced)

    def act(
        self,
        prepared: PreparedDecision,
        action_mask: Tensor,
        *,
        top_p: float = 1.0,
        deterministic: bool = False,
    ) -> ActOutput:
        """Select a joint action from an already-prepared decision."""
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
        if deterministic:
            actions, log_probs, z, local_history = self.actor.greedy_reduced(
                prepared.reduced,
                prepared.encoded,
                action_mask,
            )
        else:
            # Truncated sampling probabilities are valid for evaluation only.
            # PPO collection must use top-p equal to one so old log probabilities
            # match the distribution evaluated during optimization.
            actions, log_probs, z, local_history = self.actor.sample_reduced(
                prepared.reduced,
                prepared.encoded,
                action_mask,
                top_p=top_p,
            )
        return ActOutput(actions, log_probs, self.critic(z), local_history)

    def evaluate(
        self,
        prepared: PreparedDecision,
        action_mask: Tensor,
        actions: Tensor,
    ) -> EvalOutput:
        """Evaluate actions from an already-prepared decision."""
        logits, log_probs, z, local_history = self.actor.score_reduced(
            prepared.reduced,
            prepared.encoded,
            action_mask,
            actions,
        )
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

        return EvalOutput(log_probs, entropy, norm_entropy, value, local_history, logits)

    def score_candidates(
        self,
        prepared: PreparedDecision,
        action_mask: Tensor,
        candidate_values: Tensor,
        candidate_offsets: Tensor,
        *,
        validated: bool = False,
    ) -> Tensor:
        """
        Score ragged joint-action candidates from one prepared decision batch.

        Setting validated=True delegates directly to unchecked scoring and is
        safe only after the caller validates the complete source batch.
        """
        return self.actor.score_reduced_candidates(
            prepared.reduced,
            prepared.encoded,
            action_mask,
            candidate_values,
            candidate_offsets,
            validated=validated,
        )

    def unmasked_first_slot_logits(self, prepared: PreparedDecision) -> Tensor:
        """Return first-slot logits for legality diagnostics."""
        return self.actor.unmasked_first_slot_logits(prepared.reduced, prepared.encoded)
