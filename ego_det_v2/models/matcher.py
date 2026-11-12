from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def prepare_targets_for_matching_and_losses(
    targets: Sequence[Dict],
    *,
    device: torch.device,
) -> List[Dict]:
    """Move immutable target tensors once and cache normalized xyxy boxes."""

    prepared: List[Dict] = []
    for target in targets:
        item = dict(target)
        item["labels"] = target["labels"].to(
            device=device,
            dtype=torch.long,
        )
        item["boxes"] = target["boxes"].to(
            device=device,
            dtype=torch.float32,
        )
        if "boxes_xyxy_abs" in target:
            item["boxes_xyxy_abs"] = target["boxes_xyxy_abs"].to(
                device=device,
                dtype=torch.float32,
            )
        item["_boxes_xyxy_norm"] = box_cxcywh_to_xyxy(item["boxes"])
        prepared.append(item)
    return prepared


class HungarianMatcherV2(nn.Module):
    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.cost_class = float(config.set_cost_class)
        self.cost_bbox = float(config.set_cost_bbox)
        self.cost_giou = float(config.set_cost_giou)
        self.efficiency_mode = str(config.efficiency_mode)
        self.efficiency_ablation_level = int(
            config.efficiency_ablation_level
        )

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - environment contract
            raise RuntimeError("scipy is required for Hungarian matching") from exc

        # scipy/numpy cannot consume bfloat16. Matching is non-differentiable,
        # so force stable fp32 costs even under formal bf16 autocast.
        probabilities = outputs["pred_logits"].float().sigmoid()
        boxes = outputs["pred_boxes"].float()
        if probabilities.ndim != 3:
            raise ValueError(
                "pred_logits must have shape [B,S,C], "
                f"got {tuple(probabilities.shape)}"
            )
        if boxes.shape != (*probabilities.shape[:2], 4):
            raise ValueError(
                "pred_boxes must have shape [B,S,4] matching pred_logits, "
                f"got {tuple(boxes.shape)}"
            )
        if len(targets) != probabilities.shape[0]:
            raise ValueError(
                "targets must contain one entry per batch item, "
                f"got len(targets)={len(targets)}, B={probabilities.shape[0]}"
            )
        result: List[Tuple[torch.Tensor, torch.Tensor]] = []
        alpha = 0.25
        gamma = 2.0
        for batch_index, target in enumerate(targets):
            target_labels = target["labels"].to(
                device=probabilities.device,
                dtype=torch.long,
            )
            target_boxes = target["boxes"].to(
                device=boxes.device,
                dtype=boxes.dtype,
            )
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=boxes.device)
                result.append((empty, empty))
                continue

            probability = probabilities[batch_index].clamp(min=1.0e-8, max=1.0 - 1.0e-8)
            negative_cost = (1.0 - alpha) * probability.pow(gamma) * (
                -(1.0 - probability).log()
            )
            positive_cost = alpha * (1.0 - probability).pow(gamma) * (-probability.log())
            class_cost = positive_cost[:, target_labels] - negative_cost[:, target_labels]
            bbox_cost = torch.cdist(boxes[batch_index], target_boxes, p=1)
            target_xyxy = target.get("_boxes_xyxy_norm")
            if target_xyxy is None:
                target_xyxy = box_cxcywh_to_xyxy(target_boxes)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(boxes[batch_index]),
                target_xyxy,
            )
            cost = (
                self.cost_class * class_cost
                + self.cost_bbox * bbox_cost
                + self.cost_giou * giou_cost
            )
            if (
                self.efficiency_mode == "reference"
                or self.efficiency_ablation_level < 5
                or cost.device.type != "cuda"
            ) and not bool(torch.isfinite(cost).all()):
                raise FloatingPointError(
                    f"non-finite Hungarian cost matrix for batch {batch_index}"
                )
            pred_indices, target_indices = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )
            result.append(
                (
                    torch.as_tensor(pred_indices, dtype=torch.long, device=boxes.device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=boxes.device),
                )
            )
        return result
