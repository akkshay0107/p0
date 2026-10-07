from __future__ import annotations

import pytest
import torch

from p0.battle.events import EventDetail, EventKind, EventPosition
from p0.format_config import (
    FORMAT,
)
from p0.model.architecture_contract import POOLED_EVENT_COUNT
from p0.model.fused_token_encoder import (
    DeepSetEncoder,
    FusedTokenEncoder,
)
from p0.model.resources import default_runtime_resources
from p0.model.structured_observation import (
    CAT_IDX_NATURE,
    NUM_IDX_SLOT_LEGALITY_UNKNOWN,
    TOKEN_IDX_ALLY_SIDE,
    StructuredObservation,
)

ABILITY_CATEGORICAL_INDEX = 1
ITEM_CATEGORICAL_INDEX = 2


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


class TestResourceFeatures:
    @pytest.mark.parametrize(
        ("table", "column", "first_id", "second_id"),
        (
            ("abilities", ABILITY_CATEGORICAL_INDEX, "intimidate", "defiant"),
            ("items", ITEM_CATEGORICAL_INDEX, "sitrusberry", "leftovers"),
            ("natures", CAT_IDX_NATURE, "adamant", "bold"),
        ),
    )
    def test_each_resource_id_changes_the_encoded_pokemon(
        self, table: str, column: int, first_id: str, second_id: str
    ) -> None:
        """Changing either ability, item, or nature alone must reach the Pokémon token."""
        resources = default_runtime_resources()
        encoder = FusedTokenEncoder(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            resources=resources,
        )
        encoder.eval()
        first = StructuredObservation.empty_batch(1)
        second = first.clone()
        first_token = (
            resources.tokenizer.natures[first_id]
            if table == "natures"
            else resources.tokenizer.id_for(table, first_id)
        )
        second_token = (
            resources.tokenizer.natures[second_id]
            if table == "natures"
            else resources.tokenizer.id_for(table, second_id)
        )
        first.categorical[0, 0, column] = first_token
        second.categorical[0, 0, column] = second_token
        action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)

        with torch.inference_mode():
            first_tokens, _ = encoder(first, action_mask)
            second_tokens, _ = encoder(second, action_mask)

        assert not torch.equal(first_tokens[:, 0], second_tokens[:, 0])


class TestEventTokens:
    def test_event_tokens_depend_on_record_order_and_positions(self) -> None:
        """Verify event tokens distinguish record order and source/target positions."""
        resources = default_runtime_resources()
        encoder = FusedTokenEncoder(32, 4, 64, resources)
        action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)

        def encode_record(row: int, target: int) -> torch.Tensor:
            obs = StructuredObservation.empty_batch(1)
            obs.spatial_cat[0, row] = torch.tensor(
                [EventKind.MOVE, EventPosition.OWN_LEFT, target, 10, EventDetail.NONE, 0, 0, 0, 0]
            )
            with torch.no_grad():
                tokens, _ = encoder(obs, action_mask)
            return tokens[0, -4:]

        first_at_left = encode_record(0, EventPosition.OPPONENT_LEFT)
        first_at_right = encode_record(0, EventPosition.OPPONENT_RIGHT)
        second_at_left = encode_record(1, EventPosition.OPPONENT_LEFT)

        assert not torch.allclose(first_at_left, first_at_right, atol=1e-6)
        assert not torch.allclose(first_at_left, second_at_left, atol=1e-6)

    def test_padding_records_do_not_change_event_tokens(self) -> None:
        """Verify rows marked EventKind.NONE are ignored whatever their other fields hold, with an active record negative control."""
        resources = default_runtime_resources()
        encoder = FusedTokenEncoder(32, 4, 64, resources)
        action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)

        obs = StructuredObservation.empty_batch(1)
        obs.spatial_cat[0, 0] = torch.tensor([EventKind.MOVE, 0, 2, 10, 0, 0, 0, 0, 0])
        noisy_padding = obs.clone()
        noisy_padding.spatial_cat[0, 1:, 1:] = 3
        noisy_padding.spatial_num[0, 1:] = 0.9

        with torch.no_grad():
            clean_tokens, _ = encoder(obs, action_mask)
            noisy_tokens, _ = encoder(noisy_padding, action_mask)

        # Padding noise is ignored
        torch.testing.assert_close(clean_tokens, noisy_tokens)

        # Negative control: mutating the real active record DOES change the tokens
        active_modified = obs.clone()
        active_modified.spatial_cat[0, 0, 3] = 99
        with torch.no_grad():
            modified_tokens, _ = encoder(active_modified, action_mask)
        assert not torch.allclose(clean_tokens[:, -4:], modified_tokens[:, -4:], atol=1e-6)

    def test_spatial_event_channels_remain_observable(self) -> None:
        """Verify FusedTokenEncoder responds to each categorical and numerical event record channel."""
        resources = default_runtime_resources()
        encoder = FusedTokenEncoder(32, 4, 64, resources)
        action_mask = torch.ones((1, 2, FORMAT.action_size), dtype=torch.bool)

        base = StructuredObservation.empty_batch(1)
        base.spatial_cat[0, 0] = torch.tensor(
            [
                EventKind.BOOST,
                EventPosition.OWN_LEFT,
                EventPosition.OPPONENT_LEFT,
                0,
                EventDetail.ATK,
                0,
                0,
                4,
                1,
            ]
        )
        base.spatial_num[0, 0] = torch.tensor([-1 / 6, 1.0])

        with torch.no_grad():
            base_tokens, _ = encoder(base, action_mask)
            base_events = base_tokens[:, -4:]

        # Categorical channels 0..8
        cat_variants = (
            (0, EventKind.DAMAGE),
            (1, EventPosition.NONE),
            (2, EventPosition.OPPONENT_RIGHT),
            (3, 10),
            (4, EventDetail.SPE),
            (5, 1),
            (6, 1),
            (7, 3),
            (8, 2),
        )
        for column, value in cat_variants:
            variant = base.clone()
            variant.spatial_cat[0, 0, column] = value
            with torch.no_grad():
                variant_tokens, _ = encoder(variant, action_mask)
            assert not torch.allclose(base_events, variant_tokens[:, -4:], atol=1e-6)

        # Numerical channels 0..1
        num_variants = (
            (0, -2 / 6),
            (1, 0.0),
        )
        for column, value in num_variants:
            variant = base.clone()
            variant.spatial_num[0, 0, column] = value
            with torch.no_grad():
                variant_tokens, _ = encoder(variant, action_mask)
            assert not torch.allclose(base_events, variant_tokens[:, -4:], atol=1e-6)

    def test_unknown_legality_gate_replaces_the_mask_it_cannot_prove(self) -> None:
        """Verify that when legality is unknown, the encoder substitutes learned unknown-gate embeddings in place of action masks."""
        encoder = FusedTokenEncoder(32, 4, 64, default_runtime_resources())
        encoder.eval()

        observations = StructuredObservation.empty_batch(2)
        gates = slice(NUM_IDX_SLOT_LEGALITY_UNKNOWN, NUM_IDX_SLOT_LEGALITY_UNKNOWN + 2)
        observations.numerical[1, TOKEN_IDX_ALLY_SIDE, gates] = 1.0

        mask = torch.zeros((2, 2, FORMAT.action_size), dtype=torch.bool)
        mask[:, :, :8] = True

        with torch.no_grad():
            tokens, _ = encoder(observations, mask)
            other_mask = mask.clone()
            other_mask[:, :, 8:16] = True
            other_tokens, _ = encoder(observations, other_mask)

        mask_token = -POOLED_EVENT_COUNT - 1
        proven, unknown = tokens[0, mask_token], tokens[1, mask_token]
        assert torch.isfinite(tokens).all()
        assert not torch.allclose(proven, unknown)
        assert not torch.allclose(other_tokens[0, mask_token], proven)
        # When legality gate is active, changing the input mask has no effect on encoded output
        torch.testing.assert_close(other_tokens[1, mask_token], unknown)
