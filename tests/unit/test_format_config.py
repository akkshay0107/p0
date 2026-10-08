"""Cached tensors must use the active vocabulary and encoding."""

import pytest

from p0.format_config import active_runtime_contract, validate_artifact_runtime_contract


class TestFormatConfig:
    def test_active_runtime_reference_is_accepted(self) -> None:
        validate_artifact_runtime_contract(
            {"runtime_major": active_runtime_contract().major_sha256}
        )

    def test_incompatible_runtime_reference_requires_reconstruction(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            validate_artifact_runtime_contract({"runtime_major": "older-model"})
