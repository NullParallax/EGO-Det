from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


DEFAULT_PROJECT_ROOT = Path(os.environ.get("EGO_JOINT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def setup_project_paths(project_root: Optional[Path] = None, *, add_mmdet: bool = True) -> Path:
    """Insert project paths needed by research submodules.

    The default is the confirmed Linux project root. Local mirrors must be
    supplied explicitly through project_root or EGO_JOINT_PROJECT_ROOT.
    """

    root = Path(project_root) if project_root is not None else DEFAULT_PROJECT_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if add_mmdet:
        mmdet_root = root / "third_party" / "mmdetection-main"
        if mmdet_root.exists() and str(mmdet_root) not in sys.path:
            sys.path.insert(0, str(mmdet_root))
    return root
