"""Conservative extraction of joint action evidence from visible protocol facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from p0.battle.actions import ACT_SIZE, ActionKind, SlotAction, encode_action
from p0.battle.legality import DecisionView, legal_actions, validate_joint_action
from p0.replays.schema import ActionEvidence, LabelKind, MaskProvenance


@dataclass(frozen=True, slots=True)
class ObservedAction:
    """An action observation with alternatives retained when the log is ambiguous."""

    action: int | None = None
    alternatives: tuple[int, ...] = ()
    exact: bool = True
    tag: str = ""

    def __post_init__(self) -> None:
        values = (
            self.alternatives
            if self.alternatives
            else (() if self.action is None else (self.action,))
        )
        if any(type(value) is not int or not 0 <= value < ACT_SIZE for value in values):
            raise ValueError("Observed action ids must be within the closed action contract")
        if self.action is not None and self.action not in values:
            raise ValueError("ObservedAction.action must be one of its alternatives")

    @property
    def candidates(self) -> tuple[int, ...]:
        if self.alternatives:
            return tuple(dict.fromkeys(self.alternatives))
        return () if self.action is None else (self.action,)


@dataclass(frozen=True, slots=True)
class EvidenceRequest:
    """Inputs to the candidate enumerator for one perspective decision."""

    view: DecisionView
    slots: tuple[ObservedAction | None, ObservedAction | None]
    tags: tuple[str, ...] = ()
    max_candidates: int = 256
    unknown: bool = False


def enumerate_joint_candidates(
    view: DecisionView,
    first: Iterable[int],
    second: Iterable[int],
    *,
    max_candidates: int = 256,
) -> tuple[tuple[int, int], ...]:
    """Apply scalar legality and sequential joint constraints in stable order."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    candidates: list[tuple[int, int]] = []
    for first_action in dict.fromkeys(first):
        if first_action not in legal_actions(view, 0):
            continue
        for second_action in dict.fromkeys(second):
            pair = (first_action, second_action)
            if validate_joint_action(view, *pair):
                candidates.append(pair)
                if len(candidates) > max_candidates:
                    return ()
    return tuple(candidates)


def _unknown(tags: Iterable[str], provenance: MaskProvenance) -> ActionEvidence:
    return ActionEvidence(
        label_kind=LabelKind.UNKNOWN,
        candidates=(),
        confidence=0.0,
        mask_provenance=provenance,
        tags=tuple(dict.fromkeys(tags)),
    )


def extract_action_evidence(request: EvidenceRequest) -> ActionEvidence:
    """Return exact, partial, or unknown evidence without inventing hidden orders."""
    if request.max_candidates < 1:
        raise ValueError("EvidenceRequest.max_candidates must be positive")
    tags = list(request.tags)
    if request.unknown:
        tags.append("unsupported")
        return _unknown(tags, MaskProvenance.CONSERVATIVE_RECONSTRUCTED)
    scalar: list[tuple[int, ...]] = []
    uncertain = False
    for slot in request.slots:
        if slot is None or not slot.candidates:
            scalar.append(tuple(legal_actions(request.view, len(scalar))))
            uncertain = True
            if (
                slot is None
                and not request.view.wait
                and not request.view.team_preview
                and request.view.slots[len(scalar) - 1].active
                and not request.view.slots[len(scalar) - 1].force_switch
                and scalar[-1] == (0,)
            ):
                tags.append("no_visible_legal_move")
                return _unknown(tags, MaskProvenance.CONSERVATIVE_RECONSTRUCTED)
            continue
        scalar.append(slot.candidates)
        uncertain |= not slot.exact or len(slot.candidates) != 1
        if slot.tag:
            tags.append(slot.tag)
    candidates = enumerate_joint_candidates(
        request.view, scalar[0], scalar[1], max_candidates=request.max_candidates
    )
    if not candidates:
        if any(len(values) > 0 for values in scalar):
            tags.append("candidate_cap_or_illegal")
        return _unknown(tags, MaskProvenance.CONSERVATIVE_RECONSTRUCTED)
    if len(candidates) == 1:
        return ActionEvidence(
            LabelKind.EXACT,
            candidates,
            1.0 if not uncertain else 0.5,
            MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
            tuple(dict.fromkeys(tags)),
        )
    confidence = 0.75 if not uncertain else 0.5
    return ActionEvidence(
        LabelKind.PARTIAL,
        candidates,
        confidence,
        MaskProvenance.CONSERVATIVE_RECONSTRUCTED,
        tuple(dict.fromkeys(tags)),
    )


def observed_move_action(
    *,
    move_slot: int | None,
    target: int | None,
    mega: bool = False,
    forced: bool = False,
    tag: str = "",
) -> ObservedAction:
    """Encode a protocol move, retaining target ambiguity as alternatives."""
    if forced:
        return ObservedAction(47 if mega else 48, exact=True, tag=tag)
    if move_slot is None or target is None:
        return ObservedAction(None, exact=False, tag=tag or "move_slot_or_target_unknown")
    return ObservedAction(
        encode_action(SlotAction(ActionKind.MOVE, move_slot=move_slot, target=target, mega=mega)),
        exact=True,
        tag=tag,
    )


def observed_switch_action(slot: int | None, *, tag: str = "") -> ObservedAction:
    if slot is None:
        return ObservedAction(None, exact=False, tag=tag or "switch_slot_unknown")
    return ObservedAction(1 + slot, exact=True, tag=tag)


__all__ = [
    "EvidenceRequest",
    "ObservedAction",
    "enumerate_joint_candidates",
    "extract_action_evidence",
    "observed_move_action",
    "observed_switch_action",
]
