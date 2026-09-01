from __future__ import annotations

import pytest
import torch

from p0.model.fused_token_encoder import DeepSetEncoder


class TestDeepSetEncoder:
    def test_encoding_is_permutation_invariant_and_trainable(self) -> None:
        torch.manual_seed(23)
        encoder = DeepSetEncoder(in_features=5, d_model=16, max_members=4)
        members = torch.randn(3, 4, 5, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        permutation = torch.tensor([2, 0, 3, 1])

        encoded = encoder(members, mask)
        permuted = encoder(members[:, permutation], mask[:, permutation])

        assert encoded.shape == (3, 16)
        torch.testing.assert_close(encoded, permuted)
        encoded.square().mean().backward()
        assert members.grad is not None
        assert torch.isfinite(members.grad).all()

    def test_encoding_preserves_cardinality_and_handles_empty_sets(self) -> None:
        torch.manual_seed(29)
        encoder = DeepSetEncoder(in_features=3, d_model=12, max_members=3)
        member = torch.tensor([[[0.5, -0.25, 1.0]]]).expand(1, 3, 3)
        one_member = torch.tensor([[True, False, False]])
        two_members = torch.tensor([[True, True, False]])
        empty = torch.zeros((1, 3), dtype=torch.bool)

        encoded_one = encoder(member, one_member)
        encoded_two = encoder(member, two_members)
        encoded_empty = encoder(member, empty)

        assert not torch.allclose(encoded_one, encoded_two)
        assert torch.isfinite(encoded_empty).all()

    @pytest.mark.parametrize(
        ("members", "mask"),
        (
            (torch.empty(2, 2, 5), torch.ones(2, 2, dtype=torch.bool)),
            (torch.empty(2, 4, 5), torch.ones(2, 3, dtype=torch.bool)),
            (torch.empty(2, 4, 5), torch.ones(2, 4)),
        ),
    )
    def test_encoding_rejects_invalid_set_contracts(
        self,
        members: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        encoder = DeepSetEncoder(in_features=5, d_model=16, max_members=4)

        with pytest.raises(ValueError):
            encoder(members, mask)
