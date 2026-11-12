from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..models.box_ops import aligned_box_iou, aligned_generalized_box_iou


def _train_assign_gt(level: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "train_assign_gt" in level:
        return level["train_assign_gt"]
    return level["assign_gt"]


def _validate_dense_level(
    level: Dict[str, torch.Tensor],
    level_index: int,
) -> None:
    required = {
        "obj_logits",
        "qual_logits",
        "boxes",
        "target_boxes",
        "valid",
    }
    missing = sorted(required.difference(level))
    if missing:
        raise KeyError(
            f"dense level {level_index} missing required keys: {missing}"
        )
    if "train_assign_gt" not in level and "assign_gt" not in level:
        raise KeyError(
            f"dense level {level_index} requires train_assign_gt or assign_gt"
        )

    obj_logits = level["obj_logits"]
    qual_logits = level["qual_logits"]
    boxes = level["boxes"]
    target_boxes = level["target_boxes"]
    assign_gt = _train_assign_gt(level)
    valid = level["valid"]

    if obj_logits.ndim != 3 or obj_logits.shape[-1] != 1:
        raise ValueError(
            f"dense level {level_index} obj_logits must be [B,N,1], "
            f"got {tuple(obj_logits.shape)}"
        )
    if qual_logits.ndim != 3 or qual_logits.shape[-1] != 1:
        raise ValueError(
            f"dense level {level_index} qual_logits must be [B,N,1], "
            f"got {tuple(qual_logits.shape)}"
        )

    batch_points = tuple(obj_logits.shape[:2])
    if tuple(qual_logits.shape[:2]) != batch_points:
        raise ValueError(
            f"dense level {level_index} qual_logits [B,N] must match obj_logits, "
            f"got {tuple(qual_logits.shape[:2])} vs {batch_points}"
        )
    expected_box_shape = (*batch_points, 4)
    if tuple(boxes.shape) != expected_box_shape:
        raise ValueError(
            f"dense level {level_index} boxes must be [B,N,4], "
            f"got {tuple(boxes.shape)}"
        )
    if tuple(target_boxes.shape) != expected_box_shape:
        raise ValueError(
            f"dense level {level_index} target_boxes must be [B,N,4], "
            f"got {tuple(target_boxes.shape)}"
        )
    if tuple(assign_gt.shape) != batch_points:
        raise ValueError(
            f"dense level {level_index} assign_gt must be [B,N], "
            f"got {tuple(assign_gt.shape)}"
        )
    if "train_assign_source" in level:
        source = level["train_assign_source"]
        if tuple(source.shape) != batch_points or source.dtype != torch.long:
            raise ValueError(
                f"dense level {level_index} train_assign_source must be "
                f"long [B,N], got {tuple(source.shape)} {source.dtype}"
            )
    if tuple(valid.shape) != batch_points:
        raise ValueError(
            f"dense level {level_index} valid must be [B,N], "
            f"got {tuple(valid.shape)}"
        )
    if valid.dtype != torch.bool:
        raise ValueError(
            f"dense level {level_index} valid must be bool, got {valid.dtype}"
        )


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.to(device=logits.device, dtype=torch.float32)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = bce * (1.0 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def dense_geometry_losses(dense: Dict[str, object]) -> Dict[str, torch.Tensor]:
    levels = dense["levels"]
    if not isinstance(levels, (list, tuple)) or len(levels) == 0:
        raise ValueError("dense['levels'] must be a non-empty sequence")
    for level_index, level in enumerate(levels):
        _validate_dense_level(level, level_index)

    reference = levels[0]["obj_logits"].float()
    total_positive = sum(
        int(((_train_assign_gt(level) >= 0) & level["valid"]).sum())
        for level in levels
    )
    native_positive = 0
    fallback_positive = 0
    legacy_promoted = 0
    for level in levels:
        valid = level["valid"]
        source = level.get("train_assign_source")
        if source is None:
            native_positive += int(
                ((_train_assign_gt(level) >= 0) & valid).sum()
            )
            continue
        native_positive += int(((source == 0) & valid).sum())
        fallback_positive += int(((source == 1) & valid).sum())
        legacy_promoted += int(((source == 2) & valid).sum())
    denominator = reference.new_tensor(
        float(max(total_positive, 1)),
        dtype=torch.float32,
    )
    objectness = reference.sum() * 0.0
    giou = objectness + sum(
        (level["boxes"].float().sum() * 0.0 for level in levels),
        reference.new_zeros(()),
    )
    quality = objectness + sum(
        (level["qual_logits"].float().sum() * 0.0 for level in levels),
        reference.new_zeros(()),
    )
    for level in levels:
        positive = (_train_assign_gt(level) >= 0) & level["valid"]
        objectness_logits = level["obj_logits"].squeeze(-1).float()
        targets = positive.to(
            device=objectness_logits.device,
            dtype=torch.float32,
        )
        focal = sigmoid_focal_loss(
            objectness_logits,
            targets,
            reduction="none",
        )
        objectness = objectness + (
            focal
            * level["valid"].to(
                device=objectness_logits.device,
                dtype=torch.float32,
            )
        ).sum()
        if positive.any():
            predicted = level["boxes"].float()[positive]
            target_boxes = level["target_boxes"].to(
                device=predicted.device,
                dtype=torch.float32,
            )[positive]
            giou = giou + (1.0 - aligned_generalized_box_iou(predicted, target_boxes)).sum()
            iou_target = aligned_box_iou(predicted, target_boxes).detach().float()
            qual_logits_2d = level["qual_logits"].squeeze(-1).float()
            quality = quality + F.binary_cross_entropy_with_logits(
                qual_logits_2d[positive],
                iou_target,
                reduction="sum",
            )
    return {
        "loss_dense_obj": objectness / denominator,
        "loss_dense_giou": giou / denominator,
        "loss_dense_quality": quality / denominator,
        "native_positive_count": reference.new_tensor(
            float(native_positive),
            dtype=torch.float32,
        ),
        "fallback_positive_count": reference.new_tensor(
            float(fallback_positive),
            dtype=torch.float32,
        ),
        "legacy_promoted_count": reference.new_tensor(
            float(legacy_promoted),
            dtype=torch.float32,
        ),
        "dense_positive_count": reference.new_tensor(
            float(total_positive),
            dtype=torch.float32,
        ),
    }
