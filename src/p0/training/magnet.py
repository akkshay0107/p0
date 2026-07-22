"""Frozen magnet policy for Magnetic Mirror Descent regularization.

MMD (Sokota et al., ICLR 2023) turns self-play into a game with a unique,
attracting quantal-response equilibrium by adding a reverse-KL penalty toward a
slowly-refreshed frozen copy of the policy — the magnet. This module owns that
frozen copy: it never trains, carries no optimizer state, and is refreshed by a
pure state-dict copy from the live policy on a fixed episode interval.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from p0.model.policy import PolicyNet


class Magnet:
    """A frozen deepcopy of the live policy supplying the MMD reverse-KL target."""

    def __init__(self, policy: PolicyNet) -> None:
        self.policy = deepcopy(policy).to(policy.device).eval()
        self.policy.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return self.policy.device

    def refresh(self, policy: PolicyNet) -> None:
        """Reload the magnet weights from the live policy (state-dict copy only).

        This never touches the live optimizer's moment buffers — a refresh is a
        pure copy into the frozen net, so no optimizer reset is ever needed.
        """
        self.policy.load_state_dict(policy.state_dict())
        self.policy.eval()
        self.policy.requires_grad_(False)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        self.policy.requires_grad_(False)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.policy.state_dict()
