#!/usr/bin/env python3
"""CPU-only unit tests for I8 last.safetensors promotion (no GPU, no Docker)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops" / "docker" / "scripts"))

from image_toolkit_entrypoint import promote_last_safetensors  # noqa: E402
from image_toolkit_entrypoint import select_lora_for_eval  # noqa: E402


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


class PromoteLastTests(unittest.TestCase):
    def test_prefers_existing_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = _touch(root / "last.safetensors")
            _touch(root / "step-000100.safetensors")
            self.assertEqual(select_lora_for_eval(list(root.glob("*.safetensors"))), last)

    def test_picks_highest_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "name-000100.safetensors")
            hi = _touch(root / "name-000500.safetensors")
            self.assertEqual(select_lora_for_eval(sorted(root.glob("*.safetensors"))), hi)

    def test_promote_copies_to_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _touch(root / "adapter-000300.safetensors")
            out = promote_last_safetensors(root)
            self.assertIsNotNone(out)
            assert out is not None
            self.assertEqual(out.name, "last.safetensors")
            self.assertTrue(out.is_file())
            self.assertEqual(out.read_bytes(), src.read_bytes())


if __name__ == "__main__":
    unittest.main()
