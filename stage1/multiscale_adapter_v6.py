"""Backward-compatible import path for the EGO-Det OverLoCK adapter.

The adapter implementation now lives in ego_det_v2.backbones.overlock_adapter.
Keep this path for historical scripts and serialized module references.
"""

from ego_det_v2.backbones.overlock_adapter import OverLoCKMultiScaleAdapterV6

__all__ = ["OverLoCKMultiScaleAdapterV6"]
