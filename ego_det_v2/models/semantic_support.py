from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from ..configs.config import EgoDetV2Config
from .box_ops import box_area


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and (channels // groups >= 2 or groups == 1):
            return groups
    return 1


def _sine_position_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if channels % 4 != 0:
        raise ValueError("d_model must be divisible by 4 for the 2D sine encoding")
    quarter = channels // 4
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    frequency = torch.exp(
        torch.arange(quarter, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    x_phase = xx[..., None] * frequency
    y_phase = yy[..., None] * frequency
    encoded = torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()),
        dim=-1,
    )
    return encoded.reshape(height * width, channels).to(dtype=dtype)


class SemanticSupport(nn.Module):
    """Parameter-independent-from-geometry semantic maps and token bank."""

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        channels = int(config.d_model)
        tower: List[nn.Module] = []
        for _ in range(int(config.sem_convs)):
            tower.extend(
                (
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.semantic_tower = nn.Sequential(*tower)
        self.token_projections = nn.ModuleDict(
            {name: nn.Conv2d(channels, channels, 1) for name in config.token_levels}
        )
        self.token_norm = nn.LayerNorm(channels)
        self.roi_projection = nn.Linear(channels, channels)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        feature_masks: Dict[str, torch.Tensor],
        image_sizes: torch.Tensor,
    ) -> Dict[str, object]:
        semantic_maps = {
            "p3": self.semantic_tower(features["p3"]),
            "p4": self.semantic_tower(features["p4"]),
        }
        token_parts: List[torch.Tensor] = []
        mask_parts: List[torch.Tensor] = []
        for name in self.config.token_levels:
            projected = self.token_projections[name](features[name])
            batch, channels, height, width = projected.shape
            tokens = projected.flatten(2).transpose(1, 2)
            position = _sine_position_encoding(
                height,
                width,
                channels,
                device=projected.device,
                dtype=projected.dtype,
            )
            token_parts.append(tokens + position[None])
            mask_parts.append(feature_masks[name].flatten(1))

        p5 = features["p5"]
        valid = (~feature_masks["p5"]).to(p5.dtype)[:, None]
        global_token = (p5 * valid).sum(dim=(-2, -1)) / valid.sum(dim=(-2, -1)).clamp(min=1.0)
        token_parts.append(global_token[:, None])
        mask_parts.append(
            torch.zeros(
                (p5.shape[0], 1),
                dtype=torch.bool,
                device=p5.device,
            )
        )
        tokens = self.token_norm(torch.cat(token_parts, dim=1))
        token_mask = torch.cat(mask_parts, dim=1)
        return {
            "maps": semantic_maps,
            "tokens": tokens,
            "token_mask": token_mask,
            "image_sizes": image_sizes,
        }

    def retrieve(
        self,
        support: Dict[str, object],
        boxes_xyxy_abs: torch.Tensor,
    ) -> torch.Tensor:
        """RoI retrieve content; box coordinates are selection conditions only."""

        maps = support["maps"]
        assert isinstance(maps, dict)
        batch, count = boxes_xyxy_abs.shape[:2]
        channels = maps["p3"].shape[1]
        output = maps["p3"].new_zeros((batch * count, channels))
        if count == 0:
            return output.reshape(batch, count, channels)

        detached_boxes = boxes_xyxy_abs.detach().reshape(-1, 4)
        batch_indices = (
            torch.arange(batch, device=detached_boxes.device)[:, None]
            .expand(batch, count)
            .reshape(-1)
        )
        area_scale = torch.sqrt(box_area(detached_boxes).clamp(min=0))
        choose_p3 = area_scale < 112.0
        for name, choose, stride in (
            ("p3", choose_p3, 8),
            ("p4", ~choose_p3, 16),
        ):
            indices = torch.nonzero(choose, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            rois = torch.cat(
                (
                    batch_indices[indices, None].to(detached_boxes.dtype),
                    detached_boxes[indices],
                ),
                dim=1,
            )
            pooled = roi_align(
                maps[name],
                rois,
                output_size=int(self.config.roi_size),
                spatial_scale=1.0 / float(stride),
                sampling_ratio=-1,
                aligned=True,
            ).mean(dim=(-2, -1))
            projected = self.roi_projection(pooled)
            projected = projected.to(device=output.device, dtype=output.dtype)
            output.index_copy_(0, indices, projected)
        return output.reshape(batch, count, channels)
