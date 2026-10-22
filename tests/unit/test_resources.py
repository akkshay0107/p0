"""Tests for validated runtime resources."""

from __future__ import annotations

import json
from pathlib import Path

from p0.model.tokenizer import PokemonTokenizer

ROOT = Path(__file__).resolve().parents[2]


class TestRuntimeResources:
    def test_every_legal_content_key_resolves(self) -> None:
        """Verify all legal species, items, abilities, and moves in champions_dex resolve to known tokenizer IDs."""
        vocab = json.loads((ROOT / "data/vocab.json").read_text())
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        tokenizer_instance = PokemonTokenizer(vocab)
        for table in ("species", "items", "abilities", "moves"):
            for key in dex["legality"][table]:
                assert tokenizer_instance.resolve(table, key)[1] == "known", (table, key)

    def test_legality_inventory_uses_resolved_showdown_rules(self) -> None:
        """Verify the active legality whitelist includes legal items/moves and excludes banned content."""
        dex = json.loads((ROOT / "data/champions_dex.json").read_text())
        assert "pikachu" in dex["legality"]["species"]
        assert "protect" in dex["legality"]["moves"]
        assert "ababo" not in dex["legality"]["species"]
        assert "berserkgene" not in dex["legality"]["items"]
