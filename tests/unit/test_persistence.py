"""Tests for atomic persistence primitives."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
import torch

from p0.persistence import atomic_json_save, atomic_torch_save


class TestPersistence:
    def test_atomic_json_save_creates_parents_and_writes_sorted_json(self, tmp_path: Path) -> None:
        """Verify atomic_json_save creates missing parent directories and writes sorted, formatted JSON."""
        target = tmp_path / "nested" / "output.json"
        data = {"zebra": 1, "apple": 2}

        atomic_json_save(target, data)

        assert target.is_file()
        content = target.read_bytes()
        assert content.endswith(b"\n")
        assert orjson.loads(content) == data
        # Verify keys are sorted in output
        assert content.index(b"apple") < content.index(b"zebra")

    def test_atomic_json_save_replaces_existing_file(self, tmp_path: Path) -> None:
        """Verify atomic_json_save overwrites an existing file atomically."""
        target = tmp_path / "output.json"
        target.write_text('{"initial": true}\n', encoding="utf-8")

        atomic_json_save(target, {"updated": True})

        assert orjson.loads(target.read_bytes()) == {"updated": True}

    def test_atomic_json_save_fails_fast_on_unserializable_value(self, tmp_path: Path) -> None:
        """Verify atomic_json_save raises TypeError and does not leave temporary files on non-serializable objects."""
        target = tmp_path / "bad.json"

        with pytest.raises(TypeError):
            atomic_json_save(target, {"invalid": object()})

        assert not target.exists()
        # Verify no temporary files remain in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_atomic_torch_save_creates_parents_and_round_trips(self, tmp_path: Path) -> None:
        """Verify atomic_torch_save writes checkpoint data that loads identically."""
        target = tmp_path / "nested" / "model.pt"
        payload = {"weights": torch.tensor([1.0, 2.0, 3.0]), "step": 42}

        atomic_torch_save(target, payload)

        assert target.is_file()
        restored = torch.load(target, weights_only=True)
        assert restored["step"] == 42
        assert torch.equal(restored["weights"], payload["weights"])

    def test_atomic_torch_save_replaces_existing_file(self, tmp_path: Path) -> None:
        """Verify atomic_torch_save overwrites an existing checkpoint atomically."""
        target = tmp_path / "checkpoint.pt"
        atomic_torch_save(target, {"step": 1})
        atomic_torch_save(target, {"step": 2})

        assert torch.load(target, weights_only=True)["step"] == 2
