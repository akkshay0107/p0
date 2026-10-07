"""Unit tests for offline team validation and Showdown admission payload construction."""

from __future__ import annotations

import json

import pytest

from p0.format_config import FORMAT
from p0.teams.validation import (
    showdown_payload,
    validate_many,
    validate_many_batched,
)
from tests.team_fixtures import team_variant


class TestVariantPayload:
    def test_showdown_payload_json_structure(self) -> None:
        variant = team_variant()
        raw = showdown_payload(variant)
        assert isinstance(raw, str)
        payload = json.loads(raw)
        assert payload["format"] == FORMAT.battle_format
        pika = next(m for m in payload["team"] if m["species"] == "pikachu")
        assert pika["name"] == ""
        assert pika["item"] == "lightball"
        assert pika["ability"] == "static"
        assert pika["nature"] == "jolly"
        assert pika["evs"] == {"hp": 2, "atk": 0, "def": 0, "spa": 32, "spd": 0, "spe": 32}
        assert pika["ivs"] == {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}


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
