from __future__ import annotations

from typing import Tuple

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w = 0.5 * w
    half_h = 0.5 * h
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1),
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def pairwise_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise IoU supporting ordinary [N,4] and batched [...,N,4] tensors."""

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area1[..., :, None] + area2[..., None, :] - intersection
    iou = intersection / union.clamp(min=1.0e-7)
    return iou, union


def aligned_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.shape != boxes2.shape:
        raise ValueError(f"aligned boxes must share shape, got {boxes1.shape} and {boxes2.shape}")
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1) + box_area(boxes2) - intersection
    return intersection / union.clamp(min=1.0e-7)


def aligned_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.shape != boxes2.shape:
        raise ValueError(f"aligned boxes must share shape, got {boxes1.shape} and {boxes2.shape}")
    lt_intersection = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb_intersection = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection_wh = (rb_intersection - lt_intersection).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = box_area(boxes1) + box_area(boxes2) - intersection
    iou = intersection / union.clamp(min=1.0e-7)
    lt_enclosing = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    rb_enclosing = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing_wh = (rb_enclosing - lt_enclosing).clamp(min=0)
    enclosing = enclosing_wh[..., 0] * enclosing_wh[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=1.0e-7)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = pairwise_box_iou(boxes1, boxes2)
    lt = torch.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    enclosing = (rb - lt).clamp(min=0)
    enclosing_area = enclosing[..., 0] * enclosing[..., 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1.0e-7)


def normalized_cxcywh_to_absolute_xyxy(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
) -> torch.Tensor:
    """Convert [B,N,4] normalized cxcywh to absolute xyxy."""

    xyxy = box_cxcywh_to_xyxy(boxes)
    scales = torch.stack(
        (image_sizes[:, 1], image_sizes[:, 0], image_sizes[:, 1], image_sizes[:, 0]),
        dim=-1,
    ).to(device=boxes.device, dtype=boxes.dtype)
    return xyxy * scales[:, None, :]


def absolute_xyxy_to_normalized_cxcywh(
    boxes: torch.Tensor,
    image_sizes: torch.Tensor,
) -> torch.Tensor:
    scales = torch.stack(
        (image_sizes[:, 1], image_sizes[:, 0], image_sizes[:, 1], image_sizes[:, 0]),
        dim=-1,
    ).to(device=boxes.device, dtype=boxes.dtype)
    return box_xyxy_to_cxcywh(boxes / scales[:, None, :].clamp(min=1.0))


def clip_normalized_cxcywh(boxes: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    xyxy = box_cxcywh_to_xyxy(boxes)
    x1y1 = xyxy[..., :2].clamp(min=eps, max=1.0 - 2.0 * eps)
    x2y2 = xyxy[..., 2:].clamp(min=2.0 * eps, max=1.0 - eps)
    lo = torch.minimum(x1y1, x2y2 - eps).clamp(min=eps)
    hi = torch.maximum(x2y2, lo + eps).clamp(max=1.0 - eps)
    return box_xyxy_to_cxcywh(torch.cat((lo, hi), dim=-1))
