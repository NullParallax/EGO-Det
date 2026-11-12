"""Compatibility imports for historical EGO-Det checkpoints and scripts.

The maintained implementation lives in :mod:`ego_det_v2.paths`.
"""

from ego_det_v2.paths import (
    DEFAULT_PROJECT_ROOT,
    EGOJointPaths,
    EgoDetPaths,
    _coerce_path,
    _append_optional_required,
    required_asset_items,
    collect_missing_assets,
    assert_required_assets,
)

__all__ = [
    "DEFAULT_PROJECT_ROOT",
    "EGOJointPaths",
    "EgoDetPaths",
    "required_asset_items",
    "collect_missing_assets",
    "assert_required_assets",
]
