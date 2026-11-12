from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from .dense_losses import sigmoid_focal_loss
from ..models.box_ops import (
    aligned_box_iou,
    aligned_generalized_box_iou,
    box_cxcywh_to_xyxy,
    pairwise_box_iou,
)


def flatten_matches(
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Flatten ragged Hungarian matches once for vectorized losses.

    M is the total number of matched pairs across the batch.  Target tensors
    are gathered while iterating over the small CPU-side batch list, after
    which every loss consumes contiguous ``[M,...]`` tensors.
    """

    if len(matches) != len(targets):
        raise ValueError(
            "matches and targets must have equal batch length, "
            f"got {len(matches)} and {len(targets)}"
        )
    batch_parts = []
    slot_parts = []
    gt_parts = []
    label_parts = []
    norm_box_parts = []
    abs_box_parts = []
    for batch_index, (slot_indices, gt_indices) in enumerate(matches):
        if slot_indices.ndim != 1 or gt_indices.ndim != 1:
            raise ValueError(
                "each match must contain 1D slot and GT index tensors, "
                f"got {tuple(slot_indices.shape)} and {tuple(gt_indices.shape)}"
            )
        if slot_indices.numel() != gt_indices.numel():
            raise ValueError(
                "each match must contain equally many slot and GT indices, "
                f"got {slot_indices.numel()} and {gt_indices.numel()}"
            )
        count = int(slot_indices.shape[0])
        if count == 0:
            continue
        slots = slot_indices.to(device=device, dtype=torch.long)
        gts = gt_indices.to(device=device, dtype=torch.long)
        norm_boxes = targets[batch_index]["boxes"].to(
            device=device,
            dtype=torch.float32,
        )
        labels = targets[batch_index].get("labels")
        if labels is None:
            labels_on_device = torch.zeros(
                norm_boxes.shape[0],
                dtype=torch.long,
                device=device,
            )
        else:
            labels_on_device = labels.to(device=device, dtype=torch.long)
        absolute = targets[batch_index].get("boxes_xyxy_abs")
        if absolute is None:
            absolute_on_device = box_cxcywh_to_xyxy(norm_boxes)
        else:
            absolute_on_device = absolute.to(
                device=device,
                dtype=torch.float32,
            )
        batch_parts.append(
            torch.full(
                (count,),
                batch_index,
                dtype=torch.long,
                device=device,
            )
        )
        slot_parts.append(slots)
        gt_parts.append(gts)
        label_parts.append(labels_on_device.index_select(0, gts))
        norm_box_parts.append(norm_boxes.index_select(0, gts))
        abs_box_parts.append(absolute_on_device.index_select(0, gts))

    if not batch_parts:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
        return {
            "batch_idx": empty_long,
            "slot_idx": empty_long,
            "gt_idx": empty_long,
            "flat_labels": empty_long,
            "flat_target_boxes_norm": empty_boxes,
            "flat_target_boxes_abs": empty_boxes,
        }
    return {
        "batch_idx": torch.cat(batch_parts),
        "slot_idx": torch.cat(slot_parts),
        "gt_idx": torch.cat(gt_parts),
        "flat_labels": torch.cat(label_parts),
        "flat_target_boxes_norm": torch.cat(norm_box_parts),
        "flat_target_boxes_abs": torch.cat(abs_box_parts),
    }


def slot_round_losses_reference(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    del num_classes
    logits = prediction["pred_logits"].float()
    predicted_boxes_all = prediction["pred_boxes"].float()
    quality_logits_all = prediction["pred_quality"].float()
    target_classes = torch.zeros(
        logits.shape,
        device=logits.device,
        dtype=torch.float32,
    )
    total_gt = sum(int(target["labels"].numel()) for target in targets)
    denominator = logits.new_tensor(float(max(total_gt, 1)), dtype=torch.float32)
    l1 = predicted_boxes_all.sum() * 0.0
    giou = predicted_boxes_all.sum() * 0.0
    quality = quality_logits_all.sum() * 0.0
    for batch_index, (slot_indices, gt_indices) in enumerate(matches):
        if slot_indices.numel() == 0:
            continue
        slot_indices = slot_indices.to(device=logits.device, dtype=torch.long)
        gt_indices = gt_indices.to(device=logits.device, dtype=torch.long)
        labels = (
            targets[batch_index]["labels"]
            .to(device=logits.device, dtype=torch.long)
            .index_select(0, gt_indices)
        )
        target_classes[batch_index, slot_indices, labels] = 1.0
        predicted_boxes = predicted_boxes_all[batch_index].index_select(0, slot_indices)
        target_boxes = targets[batch_index]["boxes"].to(
            device=predicted_boxes.device,
            dtype=torch.float32,
        ).index_select(0, gt_indices)
        l1 = l1 + F.l1_loss(predicted_boxes, target_boxes, reduction="sum")
        predicted_xyxy = box_cxcywh_to_xyxy(predicted_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = giou + (1.0 - aligned_generalized_box_iou(predicted_xyxy, target_xyxy)).sum()
        iou_target = aligned_box_iou(predicted_xyxy, target_xyxy).detach().float()
        quality = quality + F.binary_cross_entropy_with_logits(
            quality_logits_all[batch_index, slot_indices, 0],
            iou_target,
            reduction="sum",
        )
    classification = sigmoid_focal_loss(
        logits,
        target_classes,
        reduction="sum",
    ) / denominator
    return {
        "loss_cls": classification,
        "loss_l1": l1 / denominator,
        "loss_giou": giou / denominator,
        "loss_quality": quality / denominator,
    }


def existence_loss_reference(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    raw_logits = prediction["pred_exist"]
    predicted_boxes = prediction["pred_boxes"]
    if raw_logits.ndim != 3 or raw_logits.shape[-1] != 1:
        raise ValueError(
            "pred_exist must have shape [B,S,1], "
            f"got {tuple(raw_logits.shape)}"
        )
    if predicted_boxes.ndim != 3 or predicted_boxes.shape[-1] != 4:
        raise ValueError(
            "pred_boxes must have shape [B,S,4], "
            f"got {tuple(predicted_boxes.shape)}"
        )
    if tuple(predicted_boxes.shape[:2]) != tuple(raw_logits.shape[:2]):
        raise ValueError(
            "pred_boxes [B,S] must match pred_exist [B,S], "
            f"got {tuple(predicted_boxes.shape[:2])} vs {tuple(raw_logits.shape[:2])}"
        )
    batch, slots = raw_logits.shape[:2]
    if len(targets) != batch or len(matches) != batch:
        raise ValueError(
            "targets and matches must contain one entry per batch item, "
            f"got len(targets)={len(targets)}, len(matches)={len(matches)}, B={batch}"
        )

    logits = raw_logits[..., 0].float()
    boxes_fp32 = predicted_boxes.float()
    target_values = torch.zeros(
        logits.shape,
        device=logits.device,
        dtype=torch.float32,
    )
    positive = torch.zeros(logits.shape, device=logits.device, dtype=torch.bool)
    for batch_index, (slot_indices, gt_indices) in enumerate(matches):
        if slot_indices.ndim != 1 or gt_indices.ndim != 1:
            raise ValueError(
                "each match must contain 1D slot and GT index tensors, "
                f"got {tuple(slot_indices.shape)} and {tuple(gt_indices.shape)}"
            )
        if slot_indices.numel() != gt_indices.numel():
            raise ValueError(
                "each match must contain equally many slot and GT indices, "
                f"got {slot_indices.numel()} and {gt_indices.numel()}"
            )
        if slot_indices.numel() == 0:
            continue
        slot_indices = slot_indices.to(device=logits.device, dtype=torch.long)
        gt_indices = gt_indices.to(device=logits.device, dtype=torch.long)
        target_boxes = targets[batch_index]["boxes"]
        if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
            raise ValueError(
                f"targets[{batch_index}]['boxes'] must have shape [N,4], "
                f"got {tuple(target_boxes.shape)}"
            )
        if bool((slot_indices < 0).any()) or bool((slot_indices >= slots).any()):
            raise IndexError(
                f"matched slot index is outside [0,{slots}) for batch {batch_index}"
            )
        target_count = int(target_boxes.shape[0])
        if bool((gt_indices < 0).any()) or bool((gt_indices >= target_count).any()):
            raise IndexError(
                f"matched GT index is outside [0,{target_count}) for batch {batch_index}"
            )
        predicted = boxes_fp32[batch_index].index_select(0, slot_indices)
        target = target_boxes.to(
            device=predicted.device,
            dtype=torch.float32,
        ).index_select(0, gt_indices)
        iou = aligned_box_iou(
            box_cxcywh_to_xyxy(predicted),
            box_cxcywh_to_xyxy(target),
        ).detach().float().clamp(0.0, 1.0)
        if not bool(torch.isfinite(iou).all()):
            raise FloatingPointError(
                f"non-finite IoU target in existence_loss for batch {batch_index}"
            )
        target_values[batch_index].index_copy_(0, slot_indices, iou)
        positive[batch_index].index_fill_(0, slot_indices, True)
    raw = F.binary_cross_entropy_with_logits(logits, target_values, reduction="none")
    negative = ~positive
    zero = logits.sum() * 0.0
    positive_loss = raw[positive].mean() if positive.any() else zero
    negative_loss = raw[negative].mean() if negative.any() else zero
    # Separate normalization prevents roughly 300 unmatched slots from drowning
    # the handful of matched positives. The original positive weight of 2 is kept.
    return 2.0 * positive_loss + negative_loss


def duplicate_margin_loss_reference(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    margin: float,
) -> torch.Tensor:
    reference = prediction["pred_scores"].float()
    losses = []
    for batch_index, (slot_indices, gt_indices) in enumerate(matches):
        if slot_indices.numel() == 0:
            continue
        slot_indices = slot_indices.to(device=reference.device, dtype=torch.long)
        gt_indices = gt_indices.to(device=reference.device, dtype=torch.long)
        boxes = box_cxcywh_to_xyxy(prediction["pred_boxes"][batch_index].float())
        target_boxes = box_cxcywh_to_xyxy(
            targets[batch_index]["boxes"].to(device=boxes.device, dtype=torch.float32)
        )
        matched_slot_set = {int(value.item()) for value in slot_indices}
        for winner_tensor, gt_tensor in zip(slot_indices, gt_indices):
            winner = int(winner_tensor.item())
            gt_index = int(gt_tensor.item())
            label = int(targets[batch_index]["labels"][gt_index].item())
            iou = pairwise_box_iou(boxes, target_boxes[gt_index : gt_index + 1])[0][:, 0]
            candidates = torch.tensor(
                [
                    index not in matched_slot_set
                    for index in range(int(boxes.shape[0]))
                ],
                dtype=torch.bool,
                device=boxes.device,
            ) & (iou > 0.5)
            if candidates.any():
                competitor = reference[batch_index, candidates, label].max()
                winner_score = reference[batch_index, winner, label]
                losses.append(F.relu(float(margin) - winner_score + competitor))
    if not losses:
        return reference.sum() * 0.0
    return torch.stack(losses).mean()


def readout_gain_loss_reference(
    prediction: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    ownership: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Require improvement only over evidence actually visible to each slot.

    Comparing with the best assigned proposal in the whole bank can create an
    impossible target when that proposal is outside the slot's dynamic P-window.
    """

    reference = prediction["pred_boxes"].float()
    losses = []
    for batch_index, (slot_indices, gt_indices) in enumerate(matches):
        if slot_indices.numel() == 0:
            continue
        slot_indices = slot_indices.to(device=reference.device, dtype=torch.long)
        gt_indices = gt_indices.to(device=reference.device, dtype=torch.long)
        gt_abs = targets[batch_index]["boxes_xyxy_abs"].to(
            device=bank["boxes"].device,
            dtype=torch.float32,
        )
        predicted_norm = reference[batch_index].index_select(0, slot_indices)
        target_norm = targets[batch_index]["boxes"].to(
            device=predicted_norm.device,
            dtype=torch.float32,
        ).index_select(0, gt_indices)
        final_iou = aligned_box_iou(
            box_cxcywh_to_xyxy(predicted_norm),
            box_cxcywh_to_xyxy(target_norm),
        )
        for local_index, (slot_tensor, gt_tensor) in enumerate(zip(slot_indices, gt_indices)):
            slot_index = int(slot_tensor.item())
            gt_index = int(gt_tensor.item())
            candidate_indices = ownership["cand_idx"][batch_index, slot_index]
            candidate_valid = ownership["cand_valid"][batch_index, slot_index]
            candidate_boxes = bank["boxes"][batch_index, candidate_indices].float()
            candidate_assign = bank["gt_assign"][batch_index, candidate_indices]
            eligible = candidate_valid & (candidate_assign == gt_index)
            if not eligible.any():
                eligible = candidate_valid
            if not eligible.any():
                continue
            oracle = pairwise_box_iou(
                candidate_boxes[eligible],
                gt_abs[gt_index : gt_index + 1],
            )[0][:, 0].max().detach()
            losses.append(F.relu(oracle - final_iou[local_index]))
    if not losses:
        return reference.sum() * 0.0
    return torch.stack(losses).mean()


def denoising_round_losses_reference(
    prediction: Dict[str, torch.Tensor],
    meta: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    *,
    include_existence: bool,
) -> Dict[str, torch.Tensor]:
    logits = prediction["pred_logits"].float()
    valid = meta["valid"].to(device=logits.device, dtype=torch.bool)
    positive = meta["positive"].to(device=logits.device, dtype=torch.bool) & valid
    source_gt = meta["source_gt"].to(device=logits.device, dtype=torch.long)
    predicted_boxes_all = prediction["pred_boxes"].float()
    quality_logits_all = prediction["pred_quality"].float()
    class_target = torch.zeros(
        logits.shape,
        device=logits.device,
        dtype=torch.float32,
    )
    for batch_index, target in enumerate(targets):
        slots = torch.nonzero(positive[batch_index], as_tuple=False).flatten()
        if slots.numel() == 0:
            continue
        sources = source_gt[batch_index, slots]
        labels = target["labels"].to(device=logits.device, dtype=torch.long)[sources]
        class_target[batch_index, slots, labels] = 1.0
    class_loss_raw = sigmoid_focal_loss(logits, class_target, reduction="none").sum(dim=-1)
    valid_count = valid.sum().clamp(min=1).to(dtype=torch.float32)
    class_loss = (class_loss_raw * valid.float()).sum() / valid_count

    l1 = predicted_boxes_all.sum() * 0.0
    giou = predicted_boxes_all.sum() * 0.0
    quality = quality_logits_all.sum() * 0.0
    positive_count = positive.sum().clamp(min=1).to(dtype=torch.float32)
    for batch_index, target in enumerate(targets):
        slots = torch.nonzero(positive[batch_index], as_tuple=False).flatten()
        if slots.numel() == 0:
            continue
        sources = source_gt[batch_index, slots]
        target_boxes = target["boxes"].to(
            device=logits.device,
            dtype=torch.float32,
        )[sources]
        predicted_boxes = predicted_boxes_all[batch_index, slots]
        l1 = l1 + F.l1_loss(predicted_boxes, target_boxes, reduction="sum")
        predicted_xyxy = box_cxcywh_to_xyxy(predicted_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = giou + (1.0 - aligned_generalized_box_iou(predicted_xyxy, target_xyxy)).sum()
        iou = aligned_box_iou(predicted_xyxy, target_xyxy).detach()
        quality = quality + F.binary_cross_entropy_with_logits(
            quality_logits_all[batch_index, slots, 0],
            iou,
            reduction="sum",
        )
    result = {
        "loss_dn_cls": class_loss,
        "loss_dn_l1": l1 / positive_count,
        "loss_dn_giou": giou / positive_count,
        "loss_dn_quality": quality / positive_count,
    }
    if include_existence and "pred_exist" in prediction:
        exist_logits = prediction["pred_exist"][..., 0].float()
        exist_target = positive.float()
        exist = F.binary_cross_entropy_with_logits(
            exist_logits,
            exist_target,
            reduction="none",
        )
        result["loss_dn_exist"] = (exist * valid.float()).sum() / valid_count
    return result


def slot_round_losses_optimized(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_classes: int,
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    del num_classes
    logits = prediction["pred_logits"].float()
    predicted_boxes_all = prediction["pred_boxes"].float()
    quality_logits_all = prediction["pred_quality"].float()
    target_classes = torch.zeros_like(logits, dtype=torch.float32)
    total_gt = sum(int(target["labels"].shape[0]) for target in targets)
    denominator = logits.new_tensor(float(max(total_gt, 1)), dtype=torch.float32)
    flat = (
        flatten_matches(matches, targets, logits.device)
        if flat_matches is None
        else flat_matches
    )
    batch_idx = flat["batch_idx"]
    slot_idx = flat["slot_idx"]
    labels = flat["flat_labels"]

    target_classes[batch_idx, slot_idx, labels] = 1.0
    predicted_boxes = predicted_boxes_all[batch_idx, slot_idx]
    target_boxes = flat["flat_target_boxes_norm"]
    l1 = F.l1_loss(predicted_boxes, target_boxes, reduction="sum")
    predicted_xyxy = box_cxcywh_to_xyxy(predicted_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    giou = (
        1.0 - aligned_generalized_box_iou(predicted_xyxy, target_xyxy)
    ).sum()
    iou_target = aligned_box_iou(
        predicted_xyxy,
        target_xyxy,
    ).detach().float()
    quality = F.binary_cross_entropy_with_logits(
        quality_logits_all[batch_idx, slot_idx, 0],
        iou_target,
        reduction="sum",
    )
    classification = sigmoid_focal_loss(
        logits,
        target_classes,
        reduction="sum",
    ) / denominator
    return {
        "loss_cls": classification,
        "loss_l1": l1 / denominator,
        "loss_giou": giou / denominator,
        "loss_quality": quality / denominator,
    }


def existence_loss_optimized(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    raw_logits = prediction["pred_exist"]
    predicted_boxes = prediction["pred_boxes"]
    if raw_logits.ndim != 3 or raw_logits.shape[-1] != 1:
        raise ValueError(
            "pred_exist must have shape [B,S,1], "
            f"got {tuple(raw_logits.shape)}"
        )
    if predicted_boxes.ndim != 3 or predicted_boxes.shape[-1] != 4:
        raise ValueError(
            "pred_boxes must have shape [B,S,4], "
            f"got {tuple(predicted_boxes.shape)}"
        )
    if tuple(predicted_boxes.shape[:2]) != tuple(raw_logits.shape[:2]):
        raise ValueError(
            "pred_boxes [B,S] must match pred_exist [B,S], "
            f"got {tuple(predicted_boxes.shape[:2])} vs {tuple(raw_logits.shape[:2])}"
        )
    batch, slots = raw_logits.shape[:2]
    if len(targets) != batch or len(matches) != batch:
        raise ValueError(
            "targets and matches must contain one entry per batch item, "
            f"got len(targets)={len(targets)}, len(matches)={len(matches)}, B={batch}"
        )

    logits = raw_logits[..., 0].float()
    boxes_fp32 = predicted_boxes.float()
    target_values = torch.zeros_like(logits, dtype=torch.float32)
    positive = torch.zeros_like(logits, dtype=torch.bool)
    flat = (
        flatten_matches(matches, targets, logits.device)
        if flat_matches is None
        else flat_matches
    )
    batch_idx = flat["batch_idx"]
    slot_idx = flat["slot_idx"]
    predicted = boxes_fp32[batch_idx, slot_idx]
    target = flat["flat_target_boxes_norm"]
    iou = aligned_box_iou(
        box_cxcywh_to_xyxy(predicted),
        box_cxcywh_to_xyxy(target),
    ).detach().float().clamp(0.0, 1.0)
    target_values[batch_idx, slot_idx] = iou
    positive[batch_idx, slot_idx] = True

    raw = F.binary_cross_entropy_with_logits(
        logits,
        target_values,
        reduction="none",
    )
    negative = ~positive
    positive_loss = (
        raw * positive.to(raw.dtype)
    ).sum() / positive.sum().clamp(min=1).to(raw.dtype)
    negative_loss = (
        raw * negative.to(raw.dtype)
    ).sum() / negative.sum().clamp(min=1).to(raw.dtype)
    return 2.0 * positive_loss + negative_loss


def duplicate_margin_loss_optimized(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    margin: float,
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    reference = prediction["pred_scores"].float()
    flat = (
        flatten_matches(matches, targets, reference.device)
        if flat_matches is None
        else flat_matches
    )
    batch_idx = flat["batch_idx"]
    if batch_idx.numel() == 0:
        return reference.sum() * 0.0
    slot_idx = flat["slot_idx"]
    labels = flat["flat_labels"]
    batch, slots = reference.shape[:2]

    predicted_xyxy = box_cxcywh_to_xyxy(
        prediction["pred_boxes"].float()
    )
    per_match_boxes = predicted_xyxy.index_select(0, batch_idx)  # [M,S,4]
    target_xyxy = box_cxcywh_to_xyxy(
        flat["flat_target_boxes_norm"]
    )[:, None, :]  # [M,1,4]
    candidate_iou = pairwise_box_iou(
        per_match_boxes,
        target_xyxy,
    )[0][..., 0]  # [M,S]

    matched_slot_mask = torch.zeros(
        (batch, slots),
        dtype=torch.bool,
        device=reference.device,
    )
    matched_slot_mask[batch_idx, slot_idx] = True
    competitors = (
        ~matched_slot_mask.index_select(0, batch_idx)
        & (candidate_iou > 0.5)
    )
    label_scores = reference.index_select(0, batch_idx).gather(
        2,
        labels[:, None, None].expand(-1, slots, 1),
    ).squeeze(-1)
    competitor_scores = label_scores.masked_fill(
        ~competitors,
        -torch.inf,
    )
    best_competitor = competitor_scores.max(dim=-1).values
    winner_score = reference[batch_idx, slot_idx, labels]
    has_competitor = competitors.any(dim=-1)
    row_loss = F.relu(float(margin) - winner_score + best_competitor)
    return (
        row_loss * has_competitor.to(row_loss.dtype)
    ).sum() / has_competitor.sum().clamp(min=1).to(row_loss.dtype)


def readout_gain_loss_optimized(
    prediction: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    ownership: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    reference = prediction["pred_boxes"].float()
    flat = (
        flatten_matches(matches, targets, reference.device)
        if flat_matches is None
        else flat_matches
    )
    batch_idx = flat["batch_idx"]
    if batch_idx.numel() == 0:
        return reference.sum() * 0.0
    slot_idx = flat["slot_idx"]
    gt_idx = flat["gt_idx"]
    candidate_indices = ownership["cand_idx"][batch_idx, slot_idx]  # [M,P]
    candidate_valid = ownership["cand_valid"][batch_idx, slot_idx]  # [M,P]
    gather_batch = batch_idx[:, None].expand_as(candidate_indices)
    candidate_boxes = bank["boxes"][
        gather_batch,
        candidate_indices,
    ].float()  # [M,P,4]
    candidate_assign = bank["gt_assign"][
        gather_batch,
        candidate_indices,
    ]
    assigned_eligible = candidate_valid & (
        candidate_assign == gt_idx[:, None]
    )
    eligible = torch.where(
        assigned_eligible.any(dim=-1, keepdim=True),
        assigned_eligible,
        candidate_valid,
    )
    candidate_iou = aligned_box_iou(
        candidate_boxes,
        flat["flat_target_boxes_abs"][:, None, :].expand_as(candidate_boxes),
    )
    oracle = candidate_iou.masked_fill(~eligible, -torch.inf).max(
        dim=-1
    ).values.detach()
    final_iou = aligned_box_iou(
        box_cxcywh_to_xyxy(reference[batch_idx, slot_idx]),
        box_cxcywh_to_xyxy(flat["flat_target_boxes_norm"]),
    )
    has_eligible = eligible.any(dim=-1)
    row_loss = F.relu(oracle - final_iou)
    return (
        row_loss * has_eligible.to(row_loss.dtype)
    ).sum() / has_eligible.sum().clamp(min=1).to(row_loss.dtype)


def _padded_dn_targets(
    targets: Sequence[Dict],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_targets = max(
        (int(target["boxes"].shape[0]) for target in targets),
        default=0,
    )
    width = max(max_targets, 1)
    labels = torch.zeros(
        (len(targets), width),
        dtype=torch.long,
        device=device,
    )
    boxes = torch.zeros(
        (len(targets), width, 4),
        dtype=torch.float32,
        device=device,
    )
    for batch_index, target in enumerate(targets):
        count = int(target["boxes"].shape[0])
        if count == 0:
            continue
        labels[batch_index, :count] = target["labels"].to(
            device=device,
            dtype=torch.long,
        )
        boxes[batch_index, :count] = target["boxes"].to(
            device=device,
            dtype=torch.float32,
        )
    return labels, boxes


def denoising_round_losses_optimized(
    prediction: Dict[str, torch.Tensor],
    meta: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    *,
    include_existence: bool,
) -> Dict[str, torch.Tensor]:
    logits = prediction["pred_logits"].float()
    valid = meta["valid"].to(device=logits.device, dtype=torch.bool)
    positive = (
        meta["positive"].to(device=logits.device, dtype=torch.bool) & valid
    )
    source_gt = meta["source_gt"].to(device=logits.device, dtype=torch.long)
    predicted_boxes_all = prediction["pred_boxes"].float()
    quality_logits_all = prediction["pred_quality"].float()
    class_target = torch.zeros_like(logits, dtype=torch.float32)
    positive_positions = torch.nonzero(positive, as_tuple=False)
    batch_idx = positive_positions[:, 0]
    slot_idx = positive_positions[:, 1]
    sources = source_gt[batch_idx, slot_idx]
    padded_labels, padded_boxes = _padded_dn_targets(
        targets,
        device=logits.device,
    )
    labels = padded_labels[batch_idx, sources]
    target_boxes = padded_boxes[batch_idx, sources]
    class_target[batch_idx, slot_idx, labels] = 1.0

    class_loss_raw = sigmoid_focal_loss(
        logits,
        class_target,
        reduction="none",
    ).sum(dim=-1)
    valid_count = valid.sum().clamp(min=1).to(dtype=torch.float32)
    class_loss = (
        class_loss_raw * valid.to(class_loss_raw.dtype)
    ).sum() / valid_count

    predicted_boxes = predicted_boxes_all[batch_idx, slot_idx]
    l1 = F.l1_loss(predicted_boxes, target_boxes, reduction="sum")
    predicted_xyxy = box_cxcywh_to_xyxy(predicted_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)
    giou = (
        1.0 - aligned_generalized_box_iou(predicted_xyxy, target_xyxy)
    ).sum()
    iou = aligned_box_iou(predicted_xyxy, target_xyxy).detach()
    quality = F.binary_cross_entropy_with_logits(
        quality_logits_all[batch_idx, slot_idx, 0],
        iou,
        reduction="sum",
    )
    positive_count = positive.sum().clamp(min=1).to(dtype=torch.float32)
    result = {
        "loss_dn_cls": class_loss,
        "loss_dn_l1": l1 / positive_count,
        "loss_dn_giou": giou / positive_count,
        "loss_dn_quality": quality / positive_count,
    }
    if include_existence and "pred_exist" in prediction:
        exist_logits = prediction["pred_exist"][..., 0].float()
        exist = F.binary_cross_entropy_with_logits(
            exist_logits,
            positive.float(),
            reduction="none",
        )
        result["loss_dn_exist"] = (
            exist * valid.to(exist.dtype)
        ).sum() / valid_count
    return result


def slot_round_losses(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    num_classes: int,
    efficiency_mode: str = "reference",
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    function = (
        slot_round_losses_optimized
        if efficiency_mode == "optimized"
        else slot_round_losses_reference
    )
    if efficiency_mode not in {"reference", "optimized"}:
        raise ValueError("invalid efficiency_mode")
    if efficiency_mode == "optimized":
        return function(
            prediction,
            targets,
            matches,
            num_classes=num_classes,
            flat_matches=flat_matches,
        )
    return function(
        prediction,
        targets,
        matches,
        num_classes=num_classes,
    )


def existence_loss(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    efficiency_mode: str = "reference",
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if efficiency_mode == "optimized":
        return existence_loss_optimized(
            prediction,
            targets,
            matches,
            flat_matches=flat_matches,
        )
    if efficiency_mode != "reference":
        raise ValueError("invalid efficiency_mode")
    return existence_loss_reference(prediction, targets, matches)


def duplicate_margin_loss(
    prediction: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    margin: float,
    efficiency_mode: str = "reference",
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if efficiency_mode == "optimized":
        return duplicate_margin_loss_optimized(
            prediction,
            targets,
            matches,
            margin=margin,
            flat_matches=flat_matches,
        )
    if efficiency_mode != "reference":
        raise ValueError("invalid efficiency_mode")
    return duplicate_margin_loss_reference(
        prediction,
        targets,
        matches,
        margin=margin,
    )


def readout_gain_loss(
    prediction: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    ownership: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    efficiency_mode: str = "reference",
    flat_matches: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if efficiency_mode == "optimized":
        return readout_gain_loss_optimized(
            prediction,
            bank,
            ownership,
            targets,
            matches,
            flat_matches=flat_matches,
        )
    if efficiency_mode != "reference":
        raise ValueError("invalid efficiency_mode")
    return readout_gain_loss_reference(
        prediction,
        bank,
        ownership,
        targets,
        matches,
    )


def denoising_round_losses(
    prediction: Dict[str, torch.Tensor],
    meta: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
    *,
    include_existence: bool,
    efficiency_mode: str = "reference",
) -> Dict[str, torch.Tensor]:
    function = (
        denoising_round_losses_optimized
        if efficiency_mode == "optimized"
        else denoising_round_losses_reference
    )
    if efficiency_mode not in {"reference", "optimized"}:
        raise ValueError("invalid efficiency_mode")
    return function(
        prediction,
        meta,
        targets,
        include_existence=include_existence,
    )
