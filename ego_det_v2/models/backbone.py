from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ego_det_v2.paths import EGOJointPaths
from ego_det_v2.import_utils import setup_project_paths

from ..configs.config import EgoDetV2Config


def _configure_overlock_checkpointing(
    backbone: nn.Module,
    enabled: bool,
) -> int:
    configured = 0
    for module in backbone.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = bool(enabled)
            configured += 1
    return configured


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and (channels // groups >= 2 or groups == 1):
            return groups
    return 1


def _smooth_block(channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(_group_count(channels), channels),
    )


def _resize_masks(pad_mask: torch.Tensor, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        name: F.interpolate(
            pad_mask[:, None].float(),
            size=value.shape[-2:],
            mode="nearest",
        )[:, 0].to(torch.bool)
        for name, value in features.items()
    }


class FakeBackbone(nn.Module):
    """Small trainable stride pyramid used only by CPU contracts."""

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        channels = int(config.d_model)
        hidden = max(16, min(64, channels))
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(_smooth_block(channels, stride=2), nn.ReLU(inplace=True))
        self.down5 = nn.Sequential(_smooth_block(channels, stride=2), nn.ReLU(inplace=True))
        self.down6 = nn.Sequential(_smooth_block(channels, stride=2), nn.ReLU(inplace=True))

    def forward(
        self,
        images: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        p3 = self.stem(images)
        p4 = self.down4(p3)
        p5 = self.down5(p4)
        p6 = self.down6(p5)
        features = {"p3": p3, "p4": p4, "p5": p5, "p6": p6}
        return features, _resize_masks(pad_mask, features)

    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        return {
            "stem": list(self.stem.parameters()),
            "backbone": list(self.down4.parameters()) + list(self.down5.parameters()) + list(self.down6.parameters()),
            "other": [],
        }


class OverLoCKFPNBackbone(nn.Module):
    """Trainable OverLoCK-tiny adapter with an independent P3-P6 FPN."""

    def __init__(
        self,
        config: EgoDetV2Config,
        paths: Optional[EGOJointPaths] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.paths = paths or EGOJointPaths()
        setup_project_paths(Path(self.paths.root), add_mmdet=False)
        try:
            from ego_det_v2.backbones.overlock_adapter import OverLoCKMultiScaleAdapterV6
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ego_det_v2.backbones.overlock_adapter is required for the formal EGO-Det V2 backbone"
            ) from exc

        weight_path = Path(self.paths.overlock_weights) if self.paths.overlock_weights is not None else None
        if weight_path is None or not weight_path.exists():
            raise FileNotFoundError(f"OverLoCK ImageNet weights are missing: {weight_path}")
        self.backbone = OverLoCKMultiScaleAdapterV6(
            pretrained=str(weight_path),
            use_gemm=False,
            deploy=False,
        )
        self.overlock_checkpoint_blocks = _configure_overlock_checkpointing(
            self.backbone,
            enabled=bool(config.overlock_gradient_checkpointing),
        )
        if config.overlock_gradient_checkpointing:
            if self.overlock_checkpoint_blocks <= 0:
                warnings.warn(
                    "OverLoCK gradient checkpointing was requested but no block "
                    "with a use_checkpoint attribute was found.",
                    RuntimeWarning,
                )
            else:
                print(
                    "[EgoDetV2 OverLoCK] gradient_checkpointing=True "
                    f"configured_blocks={self.overlock_checkpoint_blocks}",
                    flush=True,
                )
        else:
            print(
                "[EgoDetV2 OverLoCK] gradient_checkpointing=False",
                flush=True,
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

        channels = self.backbone.out_channels
        model_dim = int(config.d_model)
        self.lateral3 = nn.Conv2d(int(channels["feat8"]), model_dim, 1)
        self.lateral4 = nn.Conv2d(int(channels["feat16"]), model_dim, 1)
        self.lateral5 = nn.Conv2d(int(channels["feat32"]), model_dim, 1)
        self.smooth3 = _smooth_block(model_dim)
        self.smooth4 = _smooth_block(model_dim)
        self.smooth5 = _smooth_block(model_dim)
        self.p6 = _smooth_block(model_dim, stride=2)

    def train(self, mode: bool = True) -> "OverLoCKFPNBackbone":
        super().train(mode)
        self.backbone.train(mode)
        return self

    def forward(
        self,
        images: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        source = self.backbone(images)
        c3 = source["feat8"]
        c4 = source["feat16"]
        c5 = source["feat32"]
        lateral5 = self.lateral5(c5)
        lateral4 = self.lateral4(c4) + F.interpolate(
            lateral5, size=c4.shape[-2:], mode="nearest"
        )
        lateral3 = self.lateral3(c3) + F.interpolate(
            lateral4, size=c3.shape[-2:], mode="nearest"
        )
        p3 = self.smooth3(lateral3)
        p4 = self.smooth4(lateral4)
        p5 = self.smooth5(lateral5)
        p6 = self.p6(p5)
        features = {"p3": p3, "p4": p4, "p5": p5, "p6": p6}
        return features, _resize_masks(pad_mask, features)

    def param_groups(self) -> Dict[str, List[nn.Parameter]]:
        stem: List[nn.Parameter] = []
        backbone: List[nn.Parameter] = []
        stem_markers = ("patch_embed1", "blocks1")
        for name, parameter in self.backbone.named_parameters():
            if any(marker in name for marker in stem_markers):
                stem.append(parameter)
            else:
                backbone.append(parameter)
        if not stem:
            warnings.warn(
                "Could not identify OverLoCK stem parameters; merging them into the backbone LR group",
                RuntimeWarning,
            )
        other_modules = (
            self.lateral3,
            self.lateral4,
            self.lateral5,
            self.smooth3,
            self.smooth4,
            self.smooth5,
            self.p6,
        )
        other = [parameter for module in other_modules for parameter in module.parameters()]
        return {"stem": stem, "backbone": backbone, "other": other}
