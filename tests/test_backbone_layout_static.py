"""Static contract for the migrated OverLoCK adapter; no torch required."""

import ast
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CLASS_AST_SHA256 = "94dbb962eeddab8a3a05a3efe4a93e62d548eed335183d29ba9dc398d7489645"


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


class BackboneAdapterLayoutTests(unittest.TestCase):
    def test_adapter_class_ast_frozen(self):
        tree = ast.parse(_read("ego_det_v2/backbones/overlock_adapter.py"))
        found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OverLoCKMultiScaleAdapterV6"]
        self.assertEqual(len(found), 1)
        self.assertEqual(
            hashlib.sha256(ast.dump(found[0], include_attributes=False).encode()).hexdigest(),
            EXPECTED_CLASS_AST_SHA256,
        )

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
