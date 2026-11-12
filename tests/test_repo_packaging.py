"""Source-only packaging contract; no torch import needed."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepoPackagingTests(unittest.TestCase):
    def test_package_discovery_includes_compatibility_roots(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            data = tomllib.load(handle)
        include = data["tool"]["setuptools"]["packages"]["find"]["include"]
        for package in ("egodet*", "ego_det_v2*", "ego_joint_object_slot*", "stage1*"):
            self.assertIn(package, include)
        self.assertEqual(
            data["tool"]["setuptools"]["dynamic"]["dependencies"]["file"],
            ["requirements.txt"],
        )

    def test_inference_entrypoint_remains_separate_from_research_evaluator(self):
        path = ROOT / "egodet/infer.py"
        self.assertTrue(path.exists())
        ast.parse(path.read_text(encoding="utf-8"))
        source = path.read_text(encoding="utf-8")
        self.assertIn('checkpoint["model"]', source)
        self.assertIn('strict=True', source)
        self.assertNotIn("use_ema", source)
        self.assertTrue((ROOT / "ego_det_v2/launch/launch_eval_ego_det_v2.py").is_file())

    def test_local_demo_images_and_private_data_are_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for value in ("/test.jpg", "/outputs/", "/checkpoints/", "/weights/", "/data/"):
            self.assertIn(value, ignore)

    def test_obsolete_staging_manifest_not_published(self):
        for stale in ("docs/staging_file_manifest.json", "docs/GIT_SETUP_AND_NAMING.md",
                      "docs/NAMING_MIGRATION_PLAN.md", "tools/check_checkpoint_metadata.py"):
            self.assertFalse((ROOT / stale).exists(), stale)


if __name__ == "__main__":
    unittest.main()
