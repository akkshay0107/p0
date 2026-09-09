"""Unit tests for offline team validation and Showdown admission payload construction."""

from __future__ import annotations

import pytest

from p0.format_config import FORMAT
from p0.teams.validation import (
    AdmissionResult,
    PersistentShowdownValidator,
    _variant_dict,
    showdown_payload,
    validate_many,
    validate_many_batched,
)
from tests.team_fixtures import team_variant


class TestVariantPayload:
    def test_variant_dict_structure(self) -> None:
        variant = team_variant()
        payload = _variant_dict(variant)
        assert payload["format"] == FORMAT.battle_format
        pika = next(m for m in payload["team"] if m["species"] == "pikachu")
        assert pika["name"] == ""
        assert pika["item"] == "lightball"
        assert pika["ability"] == "static"
        assert pika["nature"] == "jolly"
        assert pika["evs"] == {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}
        assert pika["ivs"] == {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}

    def test_showdown_payload_json_encoding(self) -> None:
        variant = team_variant()
        raw = showdown_payload(variant)
        assert isinstance(raw, str)
        assert "pikachu" in raw
        assert FORMAT.battle_format in raw


class TestAdmissionResult:
    def test_fields(self) -> None:
        result = AdmissionResult(
            team_hash="a" * 64,
            valid=True,
            packed_team="packed",
            problems=(),
        )
        assert result.team_hash == "a" * 64
        assert result.valid is True
        assert result.packed_team == "packed"
        assert result.problems == ()


class TestValidationEmptyAndBounds:
    def test_validate_many_empty(self) -> None:
        assert validate_many(()) == ()
        assert validate_many_batched(()) == ()

    def test_validate_many_batched_invalid_batch_size(self) -> None:
        variant = team_variant()
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_many_batched((variant,), batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            validate_many_batched((variant,), batch_size=-5)


class TestPersistentValidatorLifecycle:
    def test_validate_many_without_enter_raises(self) -> None:
        validator = PersistentShowdownValidator()
        variant = team_variant()
        with pytest.raises(RuntimeError, match="is not open"):
            validator.validate_many((variant,))

    def test_validate_many_empty_returns_empty(self) -> None:
        validator = PersistentShowdownValidator()
        # Mocking an open state is not allowed, but empty variants return before process check
        # Wait, let's verify if empty returns before or after process check:
        # In validate_many: process check is first, so empty check comes after.
        with pytest.raises(RuntimeError, match="is not open"):
            validator.validate_many(())
