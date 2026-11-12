from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import pairwise_box_iou


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and (channels // groups >= 2 or groups == 1):
            return groups
    return 1


def _gather_2d(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather dimension 1 from [B,L,...] with [B,K] indices."""

    view_shape = list(indices.shape) + [1] * (values.ndim - 2)
    expand_shape = list(indices.shape) + list(values.shape[2:])
    gather_index = indices.view(*view_shape).expand(*expand_shape)
    return torch.gather(values, dim=1, index=gather_index)


@torch.no_grad()
def correct_assign_gt_with_current_iou(
    base_assign: torch.Tensor,
    proposal_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    *,
    promote_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the legacy dynamic dense assignment using decoded boxes.

    The FCOS assignment remains the prior:
      * positive locations stay positive, but switch to another GT when the
        current proposal overlaps that GT more;
      * background locations are promoted only for a high-IoU current proposal.

    MOD9A retains this function only for ``legacy_dynamic`` A/B contracts.
    Stable dense supervision must never call it.
    """

    corrected = base_assign.clone()
    assigned_iou = proposal_boxes.new_zeros(base_assign.shape, dtype=torch.float32)
    if proposal_boxes.numel() == 0 or gt_boxes.numel() == 0:
        corrected.fill_(-1)
        return corrected, assigned_iou

    ious, _ = pairwise_box_iou(proposal_boxes.float(), gt_boxes.float())
    best_iou, best_gt = ious.max(dim=-1)
    positive = base_assign >= 0
    if positive.any():
        safe_base = base_assign.clamp(min=0, max=max(int(gt_boxes.shape[0]) - 1, 0))
        base_iou = ious.gather(1, safe_base[:, None]).squeeze(1)
        switch = positive & (best_iou > base_iou + 1.0e-7)
        corrected[switch] = best_gt[switch]
        assigned_iou[positive] = torch.where(switch, best_iou, base_iou)[positive]

    promote = (~positive) & (best_iou >= float(promote_threshold))
    corrected[promote] = best_gt[promote]
    assigned_iou[promote] = best_iou[promote]
    return corrected, assigned_iou


@torch.no_grad()
def assign_proposal_identity_from_current_iou(
    proposal_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    *,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign only the current proposal identity, never a dense training target."""

    if proposal_boxes.ndim != 2 or proposal_boxes.shape[-1] != 4:
        raise ValueError(
            "proposal_boxes must be [N,4], got "
            f"{tuple(proposal_boxes.shape)}"
        )
    if gt_boxes.ndim != 2 or gt_boxes.shape[-1] != 4:
        raise ValueError(f"gt_boxes must be [G,4], got {tuple(gt_boxes.shape)}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("proposal identity threshold must be in [0, 1]")
    count = int(proposal_boxes.shape[0])
    identity = torch.full(
        (count,),
        -1,
        dtype=torch.long,
        device=proposal_boxes.device,
    )
    assigned_iou = torch.zeros(
        (count,),
        dtype=torch.float32,
        device=proposal_boxes.device,
    )
    if count == 0 or gt_boxes.numel() == 0:
        return identity, assigned_iou
    ious, _ = pairwise_box_iou(proposal_boxes.float(), gt_boxes.float())
    best_iou, best_gt = ious.max(dim=-1)
    assigned = best_iou >= float(threshold)
    identity[assigned] = best_gt[assigned]
    assigned_iou[assigned] = best_iou[assigned].float()
    return identity, assigned_iou


class DenseGeometryBranch(nn.Module):
    """Shared FCOS-style geometry tower plus differentiable-value proposal bank."""

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        channels = int(config.d_model)
        tower: List[nn.Module] = []
        for _ in range(int(config.tower_convs)):
            tower.extend(
                (
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.tower = nn.Sequential(*tower)
        self.obj_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.ltrb_head = nn.Conv2d(channels, 4, 3, padding=1)
        self.qual_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.desc_head = nn.Conv2d(channels, channels, 1)
        self.scales = nn.Parameter(torch.tensor(config.fpn_strides, dtype=torch.float32))
        prior_bias = -math.log(99.0)
        nn.init.constant_(self.obj_head.bias, prior_bias)
        nn.init.constant_(self.qual_head.bias, prior_bias)
        nn.init.zeros_(self.ltrb_head.bias)

    @staticmethod
    def _locations(
        height: int,
        width: int,
        stride: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * float(stride)
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * float(stride)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    @staticmethod
    def _decode(
        locations: torch.Tensor,
        distances: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        x = locations[:, 0][None]
        y = locations[:, 1][None]
        left, top, right, bottom = distances.unbind(-1)
        boxes = torch.stack((x - left, y - top, x + right, y + bottom), dim=-1)
        max_x = image_sizes[:, 1].to(device=boxes.device, dtype=boxes.dtype)[:, None]
        max_y = image_sizes[:, 0].to(device=boxes.device, dtype=boxes.dtype)[:, None]
        x1 = boxes[..., 0].clamp(min=0)
        y1 = boxes[..., 1].clamp(min=0)
        x2 = boxes[..., 2].clamp(min=0)
        y2 = boxes[..., 3].clamp(min=0)
        x1 = torch.minimum(x1, max_x)
        x2 = torch.minimum(x2, max_x)
        y1 = torch.minimum(y1, max_y)
        y2 = torch.minimum(y2, max_y)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    @torch.no_grad()
    def _fcos_assign(
        self,
        locations: torch.Tensor,
        gt_boxes: torch.Tensor,
        *,
        stride: int,
        scale_range: Tuple[float, float],
    ) -> torch.Tensor:
        count = int(locations.shape[0])
        if gt_boxes.numel() == 0:
            return torch.full((count,), -1, dtype=torch.long, device=locations.device)

        x = locations[:, 0, None]
        y = locations[:, 1, None]
        left = x - gt_boxes[None, :, 0]
        top = y - gt_boxes[None, :, 1]
        right = gt_boxes[None, :, 2] - x
        bottom = gt_boxes[None, :, 3] - y
        ltrb = torch.stack((left, top, right, bottom), dim=-1)
        inside = ltrb.min(dim=-1).values > 0

        centers = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
        radius = float(self.config.center_sampling_radius) * float(stride)
        in_center = (
            (x >= (centers[None, :, 0] - radius))
            & (x <= (centers[None, :, 0] + radius))
            & (y >= (centers[None, :, 1] - radius))
            & (y <= (centers[None, :, 1] + radius))
        )
        max_distance = ltrb.max(dim=-1).values
        minimum, maximum = scale_range
        in_scale = (max_distance >= float(minimum)) & (max_distance < float(maximum))
        eligible = inside & in_center & in_scale

        areas = (
            (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
            * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
        )
        costs = areas[None].expand(count, -1).clone()
        costs[~eligible] = torch.inf
        minimum_cost, indices = costs.min(dim=1)
        indices[~torch.isfinite(minimum_cost)] = -1
        return indices.to(torch.long)

    def _assign_level(
        self,
        locations: torch.Tensor,
        decoded_boxes: torch.Tensor,
        targets: Optional[Sequence[Dict]],
        *,
        stride: int,
        scale_range: Tuple[float, float],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, locations_count = decoded_boxes.shape[:2]
        train_assign = torch.full(
            (batch, locations_count),
            -1,
            dtype=torch.long,
            device=decoded_boxes.device,
        )
        proposal_assign = torch.full(
            (batch, locations_count),
            -1,
            dtype=torch.long,
            device=decoded_boxes.device,
        )
        proposal_assign_iou = torch.zeros(
            (batch, locations_count),
            device=decoded_boxes.device,
            dtype=torch.float32,
        )
        train_target_boxes = torch.zeros(
            (batch, locations_count, 4),
            device=decoded_boxes.device,
            dtype=torch.float32,
        )
        train_assign_source = torch.full(
            (batch, locations_count),
            -1,
            dtype=torch.long,
            device=decoded_boxes.device,
        )
        if targets is None:
            return (
                train_assign,
                proposal_assign,
                proposal_assign_iou,
                train_target_boxes,
                train_assign_source,
            )

        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes_xyxy_abs"].to(
                device=decoded_boxes.device,
                dtype=torch.float32,
            )
            base = self._fcos_assign(
                locations.float(),
                gt_boxes,
                stride=stride,
                scale_range=scale_range,
            )
            identity, identity_iou = assign_proposal_identity_from_current_iou(
                decoded_boxes[batch_index].detach().float(),
                gt_boxes,
                threshold=self.config.proposal_identity_iou_threshold,
            )
            proposal_assign[batch_index] = identity
            proposal_assign_iou[batch_index] = identity_iou
            if self.config.dense_assignment_mode == "legacy_dynamic":
                assigned, _ = correct_assign_gt_with_current_iou(
                    base,
                    decoded_boxes[batch_index].detach().float(),
                    gt_boxes,
                    promote_threshold=self.config.assign_iou_promote_threshold,
                )
                changed = assigned != base
                native_unchanged = (assigned >= 0) & (base >= 0) & ~changed
                dynamic = (assigned >= 0) & changed
                train_assign_source[batch_index, native_unchanged] = 0
                train_assign_source[batch_index, dynamic] = 2
            else:
                assigned = base
                train_assign_source[batch_index, assigned >= 0] = 0
            train_assign[batch_index] = assigned
            positive = assigned >= 0
            if positive.any():
                train_target_boxes[batch_index, positive] = gt_boxes[
                    assigned[positive]
                ]
        return (
            train_assign,
            proposal_assign,
            proposal_assign_iou,
            train_target_boxes,
            train_assign_source,
        )

    def _target_level_index(self, gt_box: torch.Tensor) -> int:
        width = max(float(gt_box[2] - gt_box[0]), 0.0)
        height = max(float(gt_box[3] - gt_box[1]), 0.0)
        center_max_distance = 0.5 * max(width, height)
        for level_index, (minimum, maximum) in enumerate(
            self.config.level_scale_ranges
        ):
            if float(minimum) <= center_max_distance < float(maximum):
                return level_index
        return len(self.config.level_scale_ranges) - 1

    @torch.no_grad()
    def _ensure_min_one_positive(
        self,
        levels: Sequence[Dict[str, torch.Tensor]],
        targets: Optional[Sequence[Dict]],
    ) -> Tuple[int, int]:
        """Add at most one non-stealing fallback for each globally uncovered GT."""

        if (
            targets is None
            or self.config.dense_assignment_mode != "stable_fcos"
            or not self.config.assign_ensure_min_one_positive
        ):
            return 0, 0
        success_count = 0
        failure_count = 0
        for batch_index, target in enumerate(targets):
            gt_boxes = target["boxes_xyxy_abs"].to(
                device=levels[0]["boxes"].device,
                dtype=torch.float32,
            )
            for gt_index in range(int(gt_boxes.shape[0])):
                already_assigned = any(
                    bool(
                        (
                            level["valid"][batch_index]
                            & (
                                level["train_assign_gt"][batch_index]
                                == gt_index
                            )
                        ).any()
                    )
                    for level in levels
                )
                if already_assigned:
                    continue
                level_index = self._target_level_index(gt_boxes[gt_index])
                level = levels[level_index]
                background = (
                    level["valid"][batch_index]
                    & (level["train_assign_gt"][batch_index] < 0)
                )
                candidate_indices = torch.nonzero(
                    background,
                    as_tuple=False,
                ).flatten()
                if candidate_indices.numel() == 0:
                    failure_count += 1
                    continue
                center = 0.5 * (
                    gt_boxes[gt_index, :2] + gt_boxes[gt_index, 2:]
                )
                locations = level["locations"].float()[candidate_indices]
                distances = (locations - center[None]).square().sum(dim=-1)
                chosen = candidate_indices[torch.argmin(distances)]
                level["train_assign_gt"][batch_index, chosen] = gt_index
                level["train_assign_source"][batch_index, chosen] = 1
                level["target_boxes"][batch_index, chosen] = gt_boxes[gt_index]
                success_count += 1
        return success_count, failure_count

    @torch.no_grad()
    def _select_indices(
        self,
        level_scores: Sequence[torch.Tensor],
        level_valid: Sequence[torch.Tensor],
        budget: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reserve each level's best evidence, then globally fill the budget."""

        batch = int(level_scores[0].shape[0])
        device = level_scores[0].device
        selected_indices = torch.zeros((batch, budget), dtype=torch.long, device=device)
        selected_valid = torch.zeros((batch, budget), dtype=torch.bool, device=device)
        selected_level_counts = torch.zeros(
            (batch, len(level_scores)), dtype=torch.long, device=device
        )
        offsets: List[int] = []
        running = 0
        for scores in level_scores:
            offsets.append(running)
            running += int(scores.shape[1])

        for batch_index in range(batch):
            reserved: List[torch.Tensor] = []
            remaining: List[torch.Tensor] = []
            for level_index, (scores, valid) in enumerate(zip(level_scores, level_valid)):
                local = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
                if local.numel() == 0:
                    continue
                pre_count = min(
                    int(local.numel()),
                    max(
                        int(self.config.bank_pre_topk_per_level),
                        int(self.config.bank_min_per_level),
                    ),
                )
                local_scores = scores[batch_index, local]
                order = torch.topk(local_scores, k=pre_count, sorted=True).indices
                local = local[order]
                global_indices = local + int(offsets[level_index])
                reserve_count = min(int(self.config.bank_min_per_level), int(global_indices.numel()))
                if reserve_count:
                    reserved.append(global_indices[:reserve_count])
                if reserve_count < int(global_indices.numel()):
                    remaining.append(global_indices[reserve_count:])

            reserve_tensor = (
                torch.cat(reserved)
                if reserved
                else torch.empty(0, dtype=torch.long, device=device)
            )
            remaining_budget = max(0, int(budget) - int(reserve_tensor.numel()))
            if remaining and remaining_budget:
                remainder = torch.cat(remaining)
                all_scores = torch.cat(level_scores, dim=1)[batch_index]
                count = min(remaining_budget, int(remainder.numel()))
                top = torch.topk(all_scores[remainder], k=count, sorted=True).indices
                remainder = remainder[top]
            else:
                remainder = torch.empty(0, dtype=torch.long, device=device)

            chosen = torch.cat((reserve_tensor, remainder))[:budget]
            if chosen.numel() > 0:
                # Keep bank prefix semantics meaningful for AR@100/300 while
                # preserving the quota membership.
                all_scores = torch.cat(level_scores, dim=1)[batch_index]
                chosen = chosen[torch.argsort(all_scores[chosen], descending=True)]
                count = int(chosen.numel())
                selected_indices[batch_index, :count] = chosen
                selected_valid[batch_index, :count] = True
                level_ids = []
                for level_index, scores in enumerate(level_scores):
                    start = offsets[level_index]
                    stop = start + int(scores.shape[1])
                    level_ids.append((chosen >= start) & (chosen < stop))
                for level_index, mask in enumerate(level_ids):
                    selected_level_counts[batch_index, level_index] = int(mask.sum())
        return selected_indices, selected_valid, selected_level_counts

    def _build_bank(self, levels: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        budget = (
            int(self.config.bank_topk_train)
            if self.training
            else int(self.config.bank_topk_eval)
        )
        level_scores = [
            torch.sigmoid(level["obj_logits"].squeeze(-1))
            * torch.sigmoid(level["qual_logits"].squeeze(-1))
            for level in levels
        ]
        level_valid = [level["valid"] for level in levels]
        with torch.no_grad():
            indices, selected_valid, level_counts = self._select_indices(
                [value.detach() for value in level_scores],
                level_valid,
                budget,
            )

        boxes = torch.cat([level["boxes"] for level in levels], dim=1)
        desc = torch.cat([level["desc"] for level in levels], dim=1)
        obj = torch.cat([level["obj_logits"] for level in levels], dim=1)
        qual = torch.cat([level["qual_logits"] for level in levels], dim=1)
        assign = torch.cat(
            [level["proposal_gt_assign"] for level in levels],
            dim=1,
        )
        assign_iou = torch.cat(
            [level["proposal_assign_iou"] for level in levels],
            dim=1,
        )
        level_ids = torch.cat(
            [
                torch.full(
                    level["proposal_gt_assign"].shape,
                    index,
                    dtype=torch.long,
                    device=assign.device,
                )
                for index, level in enumerate(levels)
            ],
            dim=1,
        )

        bank_boxes = _gather_2d(boxes, indices)
        bank_desc = _gather_2d(desc, indices)
        bank_obj = _gather_2d(obj, indices)
        bank_qual = _gather_2d(qual, indices)
        bank_assign = _gather_2d(assign, indices)
        bank_assign_iou = _gather_2d(assign_iou, indices)
        bank_level = _gather_2d(level_ids, indices)
        valid_float = selected_valid[..., None].to(bank_boxes.dtype)
        bank_boxes = bank_boxes * valid_float
        bank_desc = bank_desc * valid_float
        bank_obj = bank_obj * valid_float
        bank_qual = bank_qual * valid_float
        bank_assign = bank_assign.masked_fill(~selected_valid, -1)
        bank_assign_iou = bank_assign_iou * selected_valid.to(bank_assign_iou.dtype)
        bank_level = bank_level.masked_fill(~selected_valid, -1)
        return {
            "boxes": bank_boxes,
            "desc": bank_desc,
            "obj": bank_obj,
            "qual": bank_qual,
            "score": torch.sigmoid(bank_obj) * torch.sigmoid(bank_qual),
            "level": bank_level,
            "valid": selected_valid,
            "gt_assign": bank_assign,
            "assign_iou": bank_assign_iou,
            "selection_indices": indices,
            "level_counts": level_counts,
        }

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        feature_masks: Dict[str, torch.Tensor],
        image_sizes: torch.Tensor,
        targets: Optional[Sequence[Dict]] = None,
        *,
        profiler=None,
    ) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
        outputs: List[Dict[str, torch.Tensor]] = []
        for level_index, (name, stride, scale_range) in enumerate(
            zip(
                self.config.fpn_levels,
                self.config.fpn_strides,
                self.config.level_scale_ranges,
            )
        ):
            feature = self.tower(features[name])
            batch, _, height, width = feature.shape
            obj_logits = self.obj_head(feature).flatten(2).transpose(1, 2)
            qual_logits = self.qual_head(feature).flatten(2).transpose(1, 2)
            raw_ltrb = self.ltrb_head(feature).flatten(2).transpose(1, 2)
            desc = self.desc_head(feature).flatten(2).transpose(1, 2)
            scale = self.scales[level_index].clamp(min=1.0e-3)
            distances = scale * torch.exp(raw_ltrb.clamp(min=-8.0, max=8.0))
            locations = self._locations(
                height,
                width,
                int(stride),
                device=feature.device,
                dtype=feature.dtype,
            )
            boxes = self._decode(locations, distances, image_sizes)
            valid = ~feature_masks[name].flatten(1)
            (
                train_assign,
                proposal_assign,
                proposal_assign_iou,
                train_target_boxes,
                train_assign_source,
            ) = self._assign_level(
                locations,
                boxes,
                targets,
                stride=int(stride),
                scale_range=scale_range,
            )
            train_assign = train_assign.masked_fill(~valid, -1)
            proposal_assign = proposal_assign.masked_fill(~valid, -1)
            proposal_assign_iou = proposal_assign_iou * valid.to(
                proposal_assign_iou.dtype
            )
            train_target_boxes = train_target_boxes * valid[..., None].to(
                train_target_boxes.dtype
            )
            train_assign_source = train_assign_source.masked_fill(~valid, -1)
            outputs.append(
                {
                    "name": name,
                    "stride": torch.tensor(stride, device=feature.device),
                    "obj_logits": obj_logits,
                    "qual_logits": qual_logits,
                    "raw_ltrb": raw_ltrb,
                    "distances": distances,
                    "desc": desc,
                    "boxes": boxes,
                    "locations": locations,
                    "valid": valid,
                    # assign_gt remains a compatibility alias for dense losses.
                    "assign_gt": train_assign,
                    "train_assign_gt": train_assign,
                    "proposal_gt_assign": proposal_assign,
                    # assign_iou is now explicitly a proposal-identity alias.
                    "assign_iou": proposal_assign_iou,
                    "proposal_assign_iou": proposal_assign_iou,
                    "target_boxes": train_target_boxes,
                    "train_target_boxes": train_target_boxes,
                    "train_assign_source": train_assign_source,
                }
            )

        fallback_success, fallback_failure = self._ensure_min_one_positive(
            outputs,
            targets,
        )
        bank_context = (
            profiler.stage("bank_build")
            if profiler is not None
            else contextlib.nullcontext()
        )
        with bank_context:
            bank = self._build_bank(outputs)
        bank["image_sizes"] = image_sizes
        dense: Dict[str, object] = {
            "levels": outputs,
            "image_sizes": image_sizes,
            "assignment_mode": self.config.dense_assignment_mode,
            "fallback_success_count": torch.tensor(
                fallback_success,
                dtype=torch.float32,
                device=image_sizes.device,
            ),
            "fallback_failure_count": torch.tensor(
                fallback_failure,
                dtype=torch.float32,
                device=image_sizes.device,
            ),
        }
        return dense, bank


@torch.no_grad()
def oracle_recall(
    bank: Dict[str, torch.Tensor],
    targets: Sequence[Dict],
) -> Dict[str, float]:
    thresholds = torch.arange(0.50, 0.96, 0.05, device=bank["boxes"].device)
    ks = (100, 300, 1000)
    recall_sums = {key: 0.0 for key in ks}
    recall50 = 0.0
    recall75 = 0.0
    total_gt = 0
    for batch_index, target in enumerate(targets):
        gt = target["boxes_xyxy_abs"].to(
            device=bank["boxes"].device,
            dtype=torch.float32,
        )
        if gt.numel() == 0:
            continue
        valid = bank["valid"][batch_index]
        proposals = bank["boxes"][batch_index, valid].float()
        scores = bank["score"][batch_index, valid, 0].float()
        if proposals.numel() == 0:
            total_gt += int(gt.shape[0])
            continue
        order = torch.argsort(scores, descending=True)
        proposals = proposals[order]
        total_gt += int(gt.shape[0])
        for key in ks:
            chosen = proposals[: min(key, int(proposals.shape[0]))]
            maxima = pairwise_box_iou(gt, chosen)[0].max(dim=1).values
            recall_sums[key] += float((maxima[:, None] >= thresholds[None]).float().sum())
        chosen300 = proposals[: min(300, int(proposals.shape[0]))]
        maxima300 = pairwise_box_iou(gt, chosen300)[0].max(dim=1).values
        recall50 += float((maxima300 >= 0.5).sum())
        recall75 += float((maxima300 >= 0.75).sum())

    denominator = max(total_gt, 1)
    threshold_count = int(thresholds.numel())
    result = {
        f"AR@{key}": recall_sums[key] / float(denominator * threshold_count)
        for key in ks
    }
    result["Recall@0.5"] = recall50 / float(denominator)
    result["Recall@0.75"] = recall75 / float(denominator)
    return result
