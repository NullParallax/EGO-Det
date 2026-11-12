"""Static contract for the migrated OverLoCK adapter; no torch required."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


class BackboneAdapterLayoutTests(unittest.TestCase):
    def test_adapter_class_structure(self):
        tree = ast.parse(
            _read("ego_det_v2/backbones/overlock_adapter.py")
        )
        found = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "OverLoCKMultiScaleAdapterV6"
        ]
        self.assertEqual(len(found), 1)

        methods = {
            node.name: node
            for node in found[0].body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("__init__", methods)
        self.assertIn("forward", methods)

        returns = [
            node.value
            for node in ast.walk(methods["forward"])
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        ]
        self.assertEqual(len(returns), 1)

        output_keys = {
            key.value
            for key in returns[0].keys
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        }
        expected = {
            "feat8", "feat16", "feat32", "ctx32",
            "grid8", "grid16", "grid32",
        }
        self.assertTrue(expected.issubset(output_keys))

    def test_old_path_is_reexport_only(self):
        tree = ast.parse(_read("stage1/multiscale_adapter_v6.py"))
        self.assertFalse(any(isinstance(n, ast.ClassDef) for n in tree.body))
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
        self.assertTrue(any(
            n.module == "ego_det_v2.backbones.overlock_adapter" and
            any(a.name == "OverLoCKMultiScaleAdapterV6" for a in n.names)
            for n in imports
        ))

    def test_detector_imports_new_adapter(self):
        source = _read("ego_det_v2/models/backbone.py")
        self.assertIn(
            "from ego_det_v2.backbones.overlock_adapter import OverLoCKMultiScaleAdapterV6",
            source
        )
        self.assertNotIn(
            "from stage1.multiscale_adapter_v6 import OverLoCKMultiScaleAdapterV6",
            source
        )

    def test_third_party_license_and_attribution(self):
        license_text = _read("third_party/overlock/LICENSE")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)
        source = _read("stage1/overlock_official_nofrills.py")
        self.assertIn("https://github.com/LMMMEng/OverLoCK", source[:500])


if __name__ == "__main__":
    unittest.main()
