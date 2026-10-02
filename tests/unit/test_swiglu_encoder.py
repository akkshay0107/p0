from __future__ import annotations

import pytest
import torch

from p0.model.swiglu_encoder import AttentionPool, SwiGLUTransformerEncoder


class TestAttentionPool:
    def test_pool_is_permutation_invariant_and_trainable(self) -> None:
        torch.manual_seed(17)
        pool = AttentionPool(d_model=32, nhead=4)
        source = torch.randn(3, 7, 32, requires_grad=True)

        pooled = pool(source)
        permuted = pool(source[:, torch.tensor([4, 1, 6, 0, 3, 5, 2])])

        assert pooled.shape == (3, 32)
        torch.testing.assert_close(pooled, permuted)
        pooled.square().mean().backward()
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert pool.query.grad is not None
        assert torch.isfinite(pool.query.grad).all()

    @pytest.mark.parametrize(
        "source",
        (
            torch.empty(2, 0, 32),
            torch.empty(2, 4, 16),
            torch.empty(2, 32),
        ),
    )
    def test_pool_rejects_invalid_source_shapes(self, source: torch.Tensor) -> None:
        pool = AttentionPool(d_model=32, nhead=4)

        with pytest.raises(ValueError, match="source must have shape"):
            pool(source)


class TestSwiGLUTransformerEncoder:
    @pytest.mark.parametrize(
        ("layers", "selected", "unreturned_row", "has_padded_key"),
        (
            (1, slice(3, None), 1, True),
            (3, slice(1, 7, 2), 2, True),
            (2, slice(None), 1, False),
        ),
    )
    def test_selected_outputs_preserve_full_context_and_gradients_under_concrete_masks(
        self, layers: int, selected: slice, unreturned_row: int, has_padded_key: bool
    ) -> None:
        torch.manual_seed(29)
        encoder = SwiGLUTransformerEncoder(16, 4, 64, layers).double()
        source = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
        padding = torch.zeros(2, 7, dtype=torch.bool)
        if has_padded_key:
            padding[:, 0] = True
        full = encoder(source, src_key_padding_mask=padding)[:, selected]
        partial = encoder(source, src_key_padding_mask=padding, output_slice=selected)
        weights = torch.randn_like(full)
        inputs = (source, *encoder.parameters())

        full_gradients = torch.autograd.grad((full * weights).sum(), inputs)
        partial_gradients = torch.autograd.grad((partial * weights).sum(), inputs)

        torch.testing.assert_close(partial, full, rtol=1e-7, atol=1e-10)
        for actual, expected in zip(partial_gradients, full_gradients, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-10)
        # Unreturned rows remain part of the context, while padded keys contribute nothing.
        assert torch.count_nonzero(partial_gradients[0][:, unreturned_row]) > 0
        if has_padded_key:
            assert torch.count_nonzero(partial_gradients[0][:, 0]) == 0
        else:
            assert torch.count_nonzero(partial_gradients[0][:, 0]) > 0

    def test_all_keys_excluded_returns_finite_output_with_gradients(self) -> None:
        torch.manual_seed(41)
        encoder = SwiGLUTransformerEncoder(16, 4, 64, 2)
        source = torch.randn(2, 7, 16, requires_grad=True)
        padding = torch.ones(2, 7, dtype=torch.bool)

        output = encoder(source, padding, output_slice=slice(4, 7))
        output.square().mean().backward()

        assert output.shape == (2, 3, 16)
        assert torch.isfinite(output).all()
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in encoder.parameters()
        )

    def test_mask_is_required_and_must_be_boolean(self) -> None:
        encoder = SwiGLUTransformerEncoder(16, 4, 64, 1)
        source = torch.randn(2, 7, 16)

        with pytest.raises(TypeError):
            encoder(source)
        with pytest.raises(ValueError, match="boolean tensor"):
            encoder(source, torch.zeros(2, 7, dtype=torch.uint8))

    @pytest.mark.heavy
    def test_selected_outputs_compile_without_graph_breaks(self) -> None:
        torch.manual_seed(31)
        encoder = SwiGLUTransformerEncoder(16, 4, 64, 2)
        compiled = torch.compile(encoder, backend="aot_eager", fullgraph=True)
        source = torch.randn(2, 7, 16, requires_grad=True)
        padding = torch.zeros(2, 7, dtype=torch.bool)
        padding[:, 0] = True

        eager = encoder(source, src_key_padding_mask=padding, output_slice=slice(4, 7))
        actual = compiled(source, src_key_padding_mask=padding, output_slice=slice(4, 7))
        eager_gradient = torch.autograd.grad(eager.sum(), source)[0]
        actual_gradient = torch.autograd.grad(actual.sum(), source)[0]

        torch.testing.assert_close(actual, eager)
        torch.testing.assert_close(actual_gradient, eager_gradient)

    def test_each_layer_owns_its_parameters(self) -> None:
        counts = [len(list(SwiGLUTransformerEncoder(16, 4, 64, n).parameters())) for n in (1, 2, 3)]

        # parameters() counts a module shared between layers only once.
        assert counts[2] - counts[1] == counts[1] - counts[0] > 0
