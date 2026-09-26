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


class TestEventPositions:
    @pytest.mark.parametrize(
        ("endpoint", "board_row", "special_row"),
        [
            (0, 0, -1),
            (1, 1, -1),
            (2, 6, -1),
            (3, 7, -1),
            (4, -1, 0),
            (5, 13, -1),
            (6, 14, -1),
            (7, 12, -1),
            (8, -1, 1),
        ],
    )
    def test_event_addresses_share_board_parameters(
        self, endpoint: int, board_row: int, special_row: int
    ) -> None:
        from p0.battle.events import EventKind
        from p0.format_config import FORMAT
        from p0.model.fused_token_encoder import FusedTokenEncoder
        from p0.model.resources import default_runtime_resources
        from p0.model.structured_observation import StructuredObservation

        torch.manual_seed(41)
        encoder = FusedTokenEncoder(32, 4, 64, default_runtime_resources())
        obs = StructuredObservation.empty_batch(1)
        obs.spatial_cat[0, 0] = torch.tensor([EventKind.MOVE, endpoint, endpoint, 1, 0, 0, 0, 0, 0])
        mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)
        tokens, _ = encoder(obs, mask)
        board_grad = torch.autograd.grad(
            tokens[:, :15].square().sum(), encoder.entity_position_emb.weight, retain_graph=True
        )[0]
        event_grad, special_grad = torch.autograd.grad(
            tokens[:, -4:].square().sum(),
            (encoder.entity_position_emb.weight, encoder.event_special_position_emb.weight),
        )

        expected_rows = {0, 1, 6, 7}  # The four queries always read these board positions.
        if board_row >= 0:
            expected_rows.add(board_row)
            assert board_grad[board_row].abs().sum() > 0
        assert set(event_grad.abs().sum(-1).nonzero().flatten().tolist()) == expected_rows
        assert set(special_grad.abs().sum(-1).nonzero().flatten().tolist()) == (
            {special_row} if special_row >= 0 else set()
        )
        assert torch.isfinite(event_grad).all()

    def test_reversing_participants_and_condition_namespace_changes_events(self) -> None:
        from p0.battle.events import EventKind
        from p0.format_config import FORMAT
        from p0.model.fused_token_encoder import FusedTokenEncoder
        from p0.model.resources import default_runtime_resources
        from p0.model.structured_observation import StructuredObservation

        torch.manual_seed(43)
        encoder = FusedTokenEncoder(32, 4, 64, default_runtime_resources())
        obs = StructuredObservation.empty_batch(3)
        obs.spatial_cat[:, 0] = torch.tensor(
            [
                [EventKind.CONDITION_SET, 0, 2, 0, 0, 0, 0, 3, 1],
                [EventKind.CONDITION_SET, 2, 0, 0, 0, 0, 0, 3, 1],
                [EventKind.CONDITION_SET, 0, 2, 0, 0, 0, 0, 4, 1],
            ]
        )
        with torch.no_grad():
            tokens, _ = encoder(obs, torch.ones((3, 2, FORMAT.action_size), dtype=torch.bool))
        assert not torch.allclose(tokens[0, -4:], tokens[1, -4:])
        assert not torch.allclose(tokens[0, -4:], tokens[2, -4:])
