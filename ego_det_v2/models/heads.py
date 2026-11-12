from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import (
    box_cxcywh_to_xyxy,
    clip_normalized_cxcywh,
    pairwise_box_iou,
)


def _compatible_heads(channels: int, preferred: int) -> int:
    for heads in range(min(preferred, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class SlotPredictionHead(nn.Module):
    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        channels = int(config.d_model)
        self.classifier = nn.Linear(channels, int(config.num_classes))
        self.quality = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        nn.init.constant_(self.classifier.bias, -math.log(99.0))

    def forward(self, s: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(s), self.quality(torch.cat((s, c), dim=-1))


class DirectSlotBoxHead(nn.Module):
    """Lightweight direct box regression from the two slot streams.

    The head predicts four logit-space offsets from the slot state.  The
    current slot box is used only as an iterative reference, and no proposal
    ownership, boundary aggregation, dustbin gate, or readout residual enters
    this path.
    """

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        channels = int(config.d_model)
        self.regressor = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 4),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def forward(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
        reference_boxes: torch.Tensor,
    ) -> torch.Tensor:
        offsets = self.regressor(torch.cat((s, c), dim=-1)).float()
        reference = reference_boxes.float().clamp(
            min=1.0e-5,
            max=1.0 - 1.0e-5,
        )
        refined = torch.sigmoid(torch.logit(reference) + offsets)
        return clip_normalized_cxcywh(refined)


class ExistenceHead(nn.Module):
    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        channels = int(config.d_model)
        self.heads = _compatible_heads(channels, 2)
        self.attention = nn.MultiheadAttention(channels, self.heads, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        self.exist = nn.Linear(channels, 1)
        nn.init.zeros_(self.exist.weight)
        nn.init.zeros_(self.exist.bias)
        self.theta = nn.Linear(channels, 1)
        nn.init.zeros_(self.theta.weight)
        minimum, maximum = config.gamma_range
        ratio = (float(config.gamma_init) - minimum) / (maximum - minimum)
        ratio = min(max(ratio, 1.0e-6), 1.0 - 1.0e-6)
        nn.init.constant_(self.theta.bias, math.log(ratio / (1.0 - ratio)))

    def forward(
        self,
        c: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        geometry_boxes = boxes.float()
        xyxy = box_cxcywh_to_xyxy(geometry_boxes)
        iou, _ = pairwise_box_iou(xyxy, xyxy)
        bias = torch.log(iou.clamp(min=1.0e-6))
        below_gate = iou < float(self.config.exist_iou_gate)
        diagonal = torch.eye(
            geometry_boxes.shape[1],
            dtype=torch.bool,
            device=geometry_boxes.device,
        )[None]
        below_gate = below_gate & ~diagonal
        bias = bias.masked_fill(below_gate, -torch.inf)
        attention_mask = (
            bias[:, None]
            .expand(
                geometry_boxes.shape[0],
                self.heads,
                geometry_boxes.shape[1],
                geometry_boxes.shape[1],
            )
            .reshape(
                geometry_boxes.shape[0] * self.heads,
                geometry_boxes.shape[1],
                geometry_boxes.shape[1],
            )
            .to(dtype=c.dtype)
        )
        attended, _ = self.attention(
            c,
            c,
            c,
            attn_mask=attention_mask,
            need_weights=False,
        )
        content = c + attended
        existence = self.exist(content)
        minimum, maximum = self.config.gamma_range
        gamma = float(minimum) + (float(maximum) - float(minimum)) * torch.sigmoid(
            self.theta(content)
        )
        return existence, gamma


def compose_scores(
    class_logits: torch.Tensor,
    quality_logits: torch.Tensor,
    existence_logits: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    base = torch.sqrt(
        torch.sigmoid(class_logits) * torch.sigmoid(quality_logits).clamp(min=1.0e-8)
    )
    return base * torch.sigmoid(existence_logits).clamp(min=1.0e-8).pow(gamma)
