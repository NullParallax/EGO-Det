"""PyTorch-free checks for the path migration and legacy compatibility."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from ego_det_v2.paths import (
    EGOJointPaths,
    EgoDetPaths,
    collect_missing_assets,
    required_asset_items,
)
from ego_joint_object_slot.configs.paths import (
    EGOJointPaths as HistoricalPaths,
    required_asset_items as historical_required_asset_items,
)
from ego_det_v2.import_utils import setup_project_paths
from ego_joint_object_slot.utils.import_utils import (
    setup_project_paths as historical_setup_project_paths,
)

ROOT = Path(__file__).resolve().parents[1]


class PathCompatibilityTests(unittest.TestCase):
    def test_original_and_public_class_are_identical(self) -> None:
        self.assertIs(EgoDetPaths, EGOJointPaths)
        self.assertIs(HistoricalPaths, EGOJointPaths)
        self.assertIs(historical_required_asset_items, required_asset_items)
        self.assertIs(historical_setup_project_paths, setup_project_paths)

    def test_root_and_derived_paths_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            expected = {
                "root": root,
                "overlock_weights": root / "weights/overlock/overlock_t_in1k_224.pth",
                "coco_train_instances": root / "data/coco/annotations/instances_train2017.json",
                "coco_val_instances": root / "data/coco/annotations/instances_val2017.json",
                "coco_image_root": root / "data/coco/images",
                "output_root": root / "outputs/ego_joint_object_slot",
            }
            for cls in (EGOJointPaths, HistoricalPaths):
                item = cls.from_root(root)
                for field, value in expected.items():
                    self.assertEqual(getattr(item, field), value, field)
            item = EGOJointPaths.from_root(root)
            self.assertEqual(collect_missing_assets([("missing", root / "missing")]),
                             [("missing", root / "missing")])
            self.assertTrue(required_asset_items(item))

    def test_default_root_resolves_to_repository(self) -> None:
        # Only asserted when the user's EGO_JOINT_PROJECT_ROOT override is absent.
        import os
        if os.environ.get("EGO_JOINT_PROJECT_ROOT"):
            self.skipTest("Explicit EGO_JOINT_PROJECT_ROOT override is set")
        self.assertEqual(EGOJointPaths().root.resolve(), ROOT.resolve())

    def test_v2_no_longer_imports_historical_path_package(self) -> None:
        candidates = [
            ROOT / "ego_det_v2/models/backbone.py",
            ROOT / "ego_det_v2/models/ego_det_v2.py",
            ROOT / "ego_det_v2/launch/_common.py",
            ROOT / "ego_det_v2/data/coco_dataset.py",
        ]
        for file in candidates:
            tree = ast.parse(file.read_text(encoding="utf-8"))
            imports = [
                node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            ]
            self.assertFalse(
                any(name and name.startswith("ego_joint_object_slot") for name in imports),
                file.as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
