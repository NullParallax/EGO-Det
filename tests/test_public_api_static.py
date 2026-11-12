"""Validate the public EGO-Det aliases without importing PyTorch."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicApiLayoutTests(unittest.TestCase):
    def test_api_is_alias_only(self) -> None:
        source = (ROOT / "egodet" / "__init__.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        exported = {
            (node.module, alias.name, alias.asname)
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn(
            ("ego_det_v2.configs.config", "EgoDetV2Config", "EgoDetConfig"),
            exported,
        )
        self.assertIn(
            ("ego_det_v2.models.ego_det_v2", "EgoDetV2", "EgoDet"),
            exported,
        )
        self.assertFalse(
            any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body),
            "The public package must not duplicate or wrap the model implementation.",
        )
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.literal_eval(assignments[0].value), ["EgoDet", "EgoDetConfig"])

    def test_original_implementation_remains_available(self) -> None:
        original_model = (ROOT / "ego_det_v2" / "models" / "ego_det_v2.py")
        original_config = (ROOT / "ego_det_v2" / "configs" / "config.py")
        self.assertTrue(original_model.is_file())
        self.assertTrue(original_config.is_file())
        model_classes = {
            node.name for node in ast.parse(original_model.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef)
        }
        config_classes = {
            node.name for node in ast.parse(original_config.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("EgoDetV2", model_classes)
        self.assertIn("EgoDetV2Config", config_classes)


if __name__ == "__main__":
    unittest.main()
