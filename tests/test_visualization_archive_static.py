"""Static invariant: historical visualization test is archived intact."""

import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "543bdaf151f97123d46b2f92b020fb6dcf082c14fb3493c41f1f19e0ef0a140b"


class VisualizationArchiveTests(unittest.TestCase):
    def test_legacy_test_preserved_verbatim(self):
        archived = ROOT / "archive/legacy_tests/paper_visualization_contracts.py"
        self.assertTrue(archived.exists())
        self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), EXPECTED_SHA256)

    def test_missing_implementation_not_in_active_test_tree(self):
        self.assertFalse((ROOT / "ego_det_v2/tests/test_paper_visualization.py").exists())
        self.assertFalse((ROOT / "ego_det_v2/paper_visualization").exists())


if __name__ == "__main__":
    unittest.main()
