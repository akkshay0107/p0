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
    def test_joint_actions_reject_duplicate_switches_and_double_mega(self) -> None:
        """Verify joint validation and second action masks reject duplicate switches and suppress all Mega variants."""
        view = DecisionView(
            slots=(
                SlotDecision(
                    switch_slots=(2, 3), move_targets=((-2, 1), (), (), ()), can_mega=True
                ),
                SlotDecision(
                    switch_slots=(2, 4), move_targets=((-2, 1), (), (), ()), can_mega=True
                ),
            )
        )

        # Actions 3/4/5 switch to roster slots 2/3/4; 7/10 are self/opp move 0;
        # 27/30 are self/opp Mega move 0.
        assert validate_joint_action(view, 3, 5)
        assert not validate_joint_action(view, 3, 3)
        assert validate_joint_action(view, 27, 7)
        assert validate_joint_action(view, 30, 10)
        assert not validate_joint_action(view, 27, 27)
        assert not validate_joint_action(view, 27, 30)
        assert not validate_joint_action(view, 30, 27)
        assert not validate_joint_action(view, 30, 30)
        assert not validate_joint_action(view, 1, 5)

        # Switching permits all second slot options: remaining switch (5), normal moves (7, 10), and Mega moves (27, 30)
        assert set(np.flatnonzero(second_action_mask(view, 3))) == {5, 7, 10, 27, 30}
        # Both self-target (27) and opponent (30) Mega choices suppress the entire 27:48 Mega slice
        mask_self_mega = second_action_mask(view, 27)
        assert set(np.flatnonzero(mask_self_mega)) == {3, 5, 7, 10}
        assert not mask_self_mega[27:48].any()

        mask_opp_mega = second_action_mask(view, 30)
        assert set(np.flatnonzero(mask_opp_mega)) == {3, 5, 7, 10}
        assert not mask_opp_mega[27:48].any()

        # Forced-Mega (47) endpoint is also suppressed when the first slot uses a Mega action
        forced_view = DecisionView(
            slots=(
                SlotDecision(
                    switch_slots=(2, 3),
                    move_targets=((-2, 1), (), (), ()),
                    can_mega=True,
                    forced_move=True,
                ),
                SlotDecision(switch_slots=(2, 4), can_mega=True, forced_move=True),
            )
        )
        assert validate_joint_action(forced_view, 3, 47)
        assert not validate_joint_action(forced_view, 27, 47)
        assert not validate_joint_action(forced_view, 47, 47)
        assert validate_joint_action(forced_view, 47, 48)

        mask_switch = second_action_mask(forced_view, 3)
        assert set(np.flatnonzero(mask_switch)) == {5, 47, 48}

        mask_forced_mega = second_action_mask(forced_view, 47)
        assert set(np.flatnonzero(mask_forced_mega)) == {3, 5, 48}
        assert not mask_forced_mega[27:48].any()

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

    def test_trapped_and_inactive_slot_legality(self) -> None:
        """Verify that trapped slots cannot switch and inactive slots only switch or pass."""
        trapped_view = DecisionView(
            slots=(
                SlotDecision(switch_slots=(2, 3), trapped=True, active=True, move_targets=((-2,),)),
                SlotDecision(switch_slots=(2, 3), active=False),
            )
        )
        # Slot 0 is trapped: switch actions (1..6) must not be in legal actions
        slot0_actions = legal_actions(trapped_view, 0)
        assert all(not (1 <= a < 7) for a in slot0_actions)
        assert 7 in slot0_actions  # Move is still legal

        # Slot 1 is inactive: only switches (or pass) are legal
        slot1_actions = legal_actions(trapped_view, 1)
        assert all(1 <= a < 7 for a in slot1_actions)
