"""Tests for battle action legality constraints."""

from __future__ import annotations

import numpy as np

from p0.battle.actions import (
    ACT_SIZE,
)
from p0.battle.legality import (
    DecisionView,
    SlotDecision,
    action_mask,
    apply_joint_constraints,
    legal_actions,
    second_action_mask,
    validate_joint_action,
)


def _struggle_battle(can_mega: bool) -> DecisionView:
    return DecisionView(
        slots=(
            SlotDecision(
                forced_move=True,
                can_mega=can_mega,
            ),
            SlotDecision(),
        )
    )


class TestLegality:
    def test_scalar_joint_constraints_match_policy_vectorization(self) -> None:
        """Verify that scalar joint validation (validate_joint_action) matches vectorized second_action_mask."""
        view = DecisionView(
            slots=(
                SlotDecision(
                    switch_slots=(2, 3),
                    move_targets=((-2, 1, 2), (0,), (), (1,)),
                    can_mega=True,
                ),
                SlotDecision(
                    switch_slots=(2, 4),
                    move_targets=((-1, 1, 2), (), (0,), ()),
                    can_mega=False,
                ),
            )
        )
        mask1 = action_mask(view)[0]
        for a1 in range(ACT_SIZE):
            if not mask1[a1]:
                continue
            mask2 = second_action_mask(view, a1)
            for a2 in range(ACT_SIZE):
                assert mask2[a2] == validate_joint_action(view, a1, a2)

    def test_double_force_switch_with_single_available_switch(self) -> None:
        """Verify fallback behavior when both slots are forced to switch but only one replacement is available."""
        view = DecisionView(
            slots=(
                SlotDecision(switch_slots=(2,), force_switch=True),
                SlotDecision(switch_slots=(2,), force_switch=True),
            )
        )
        assert legal_actions(view, 0) == (3, 0)
        assert legal_actions(view, 1) == (3, 0)

    def test_apply_joint_constraints_fallback_and_error_resilience(self) -> None:
        """Verify joint constraint filtering eliminates duplicate switch targets across both slots."""
        preview = DecisionView(slots=(SlotDecision(), SlotDecision()), team_preview=True)
        mask = np.ones(ACT_SIZE, dtype=np.bool_)
        apply_joint_constraints(mask, preview, first=-1)
        assert not mask.any()

        view = DecisionView(
            slots=(
                SlotDecision(switch_slots=(2,)),
                SlotDecision(switch_slots=(2,)),
            )
        )
        slot1_mask = np.zeros(ACT_SIZE, dtype=np.bool_)
        slot1_mask[3] = True
        # If slot 0 took switch to slot 2 (action 3), slot 1 cannot take switch 3 and falls back to pass (0)
        apply_joint_constraints(slot1_mask, view, first=3)
        assert slot1_mask[0]
        assert not slot1_mask[3]

    def test_struggle_env_roundtrip(self) -> None:
        """Verify forced moves map to action 48 (standard forced move) or 47 (mega-forced move)."""
        view = _struggle_battle(can_mega=False)
        assert list(legal_actions(view, 0)) == [48]

        mega_view = _struggle_battle(can_mega=True)
        mask = list(legal_actions(mega_view, 0))
        assert 48 in mask and 47 in mask

    def test_unknown_legality_masks_are_supersets_of_the_proven_mask(self) -> None:
        """Verify unknown legality generates superset action mask."""
        known_decision = SlotDecision(
            switch_slots=(2, 3),
            move_targets=((-2, 1, 2), (), (), ()),
            can_mega=False,
            legality_known=True,
        )
        unknown_decision = SlotDecision(
            switch_slots=(2, 3),
            move_targets=((-2, 1, 2), (), (), ()),
            can_mega=False,
            legality_known=False,
        )
        view_known = DecisionView(slots=(known_decision, SlotDecision()))
        view_unknown = DecisionView(slots=(unknown_decision, SlotDecision()))

        mask_known = action_mask(view_known)[0]
        mask_unknown = action_mask(view_unknown)[0]

        assert mask_unknown.sum() >= mask_known.sum()
        assert (mask_known & ~mask_unknown).sum() == 0
