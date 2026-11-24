#!/usr/bin/env python3
"""CPU-only unit tests for I8 last.safetensors promotion (no GPU, no Docker)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops" / "docker" / "scripts"))

from image_toolkit_entrypoint import apply_checkpoint_save_policy  # noqa: E402
from image_toolkit_entrypoint import apply_empty_prompt_share  # noqa: E402
from image_toolkit_entrypoint import empty_prompt_rate_from_env  # noqa: E402
from image_toolkit_entrypoint import promote_last_safetensors  # noqa: E402
from image_toolkit_entrypoint import select_lora_for_eval  # noqa: E402
from image_toolkit_entrypoint import steps_for_hours  # noqa: E402


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


class EmptyPromptAndSaveTests(unittest.TestCase):
    def test_default_empty_prompt_rate(self) -> None:
        import os

        os.environ.pop("EMPTY_PROMPT_RATE", None)
        self.assertAlmostEqual(empty_prompt_rate_from_env(), 0.45)

    def test_apply_empty_prompt_share(self) -> None:
        process = {"datasets": [{"folder_path": "/x", "caption_dropout_rate": 0.05}]}
        apply_empty_prompt_share(process, 0.45)
        self.assertEqual(process["datasets"][0]["caption_dropout_rate"], 0.45)

    def test_apply_empty_prompt_share_injects_missing(self) -> None:
        process = {"datasets": [{"folder_path": "/x"}]}
        apply_empty_prompt_share(process, 0.4)
        self.assertEqual(process["datasets"][0]["caption_dropout_rate"], 0.4)

    def test_checkpoint_save_policy(self) -> None:
        import os

        os.environ["SAVE_EVERY_STEPS"] = "100"
        os.environ["MAX_STEP_SAVES"] = "8"
        process: dict = {}
        apply_checkpoint_save_policy(process)
        self.assertEqual(process["save"]["save_every"], 100)
        self.assertEqual(process["save"]["max_step_saves_to_keep"], 8)

    def test_steps_for_hours(self) -> None:
        import os

        os.environ.pop("STEPS_PER_HOUR", None)
        os.environ.pop("MIN_TRAIN_STEPS", None)
        os.environ.pop("MAX_TRAIN_STEPS_CAP", None)
        self.assertEqual(steps_for_hours(1.0), 900)
        self.assertEqual(steps_for_hours(0.5), 450)
        self.assertEqual(steps_for_hours(0.2), 300)  # min clamp


class PromoteLastTests(unittest.TestCase):
    def test_prefers_late_numbered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "name-000100.safetensors")
            late = _touch(root / "name-000800.safetensors")
            self.assertEqual(select_lora_for_eval(sorted(root.glob("*.safetensors"))), late)

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
