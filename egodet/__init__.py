"""Public Python API for the paper-compatible EGO-Det implementation.

The original V2 classes remain unchanged to preserve checkpoint compatibility.
"""

from ego_det_v2.configs.config import EgoDetV2Config as EgoDetConfig
from ego_det_v2.models.ego_det_v2 import EgoDetV2 as EgoDet

__all__ = ["EgoDet", "EgoDetConfig"]
