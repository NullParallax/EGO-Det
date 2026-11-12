from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..models.box_ops import aligned_box_iou, pairwise_box_iou


def _best_iou_per_gt(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    if gt_boxes.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=gt_boxes.device)
    if proposals.numel() == 0:
        return torch.zeros(
            gt_boxes.shape[0],
            dtype=torch.float32,
            device=gt_boxes.device,
        )
    return pairwise_box_iou(gt_boxes.float(), proposals.float())[0].max(dim=1).values


@torch.no_grad()
def brute_force_proposal_metrics(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    ks: Sequence[int] = (100, 300, 1000, 1500),
    thresholds: Sequence[float] = tuple(
        round(0.50 + 0.05 * index, 2) for index in range(10)
    ),
) -> Dict[str, float]:
    """Independent proposal metrics used by diagnostics and synthetic contracts."""

    proposals = proposals.float()
    gt_boxes = gt_boxes.to(device=proposals.device, dtype=torch.float32)
    if scores is not None and proposals.numel() > 0:
        order = torch.argsort(scores.reshape(-1).float(), descending=True)
        proposals = proposals[order]
    result: Dict[str, float] = {}
    denominator = max(int(gt_boxes.shape[0]), 1)
    for k in ks:
        chosen = proposals[: min(int(k), int(proposals.shape[0]))]
        best = _best_iou_per_gt(chosen, gt_boxes)
        for threshold in thresholds:
            result[f"Recall@{int(k)}/{float(threshold):.2f}"] = (
                float((best >= float(threshold)).sum()) / denominator
            )
        result[f"AR@{int(k)}"] = sum(
            result[f"Recall@{int(k)}/{float(threshold):.2f}"]
            for threshold in thresholds
        ) / max(len(thresholds), 1)
    all_best = _best_iou_per_gt(proposals, gt_boxes)
    if all_best.numel() > 0:
        result["best_iou_mean"] = float(all_best.mean())
        result["best_iou_median"] = float(all_best.median())
        result["best_iou_min"] = float(all_best.min())
    else:
        result.update(
            {
                "best_iou_mean": 0.0,
                "best_iou_median": 0.0,
                "best_iou_min": 0.0,
            }
        )
    return result


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().cpu().reshape(-1)
    count = int(values.numel())
    if count == 0:
        return values
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    new_group = torch.ones(count, dtype=torch.bool)
    if count > 1:
        new_group[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = new_group.cumsum(dim=0) - 1
    counts = torch.bincount(group_ids)
    starts = counts.cumsum(dim=0) - counts
    average_ranks = starts.float() + 0.5 * (counts.float() - 1.0)
    ranks = torch.empty(count, dtype=torch.float32)
    ranks[order] = average_ranks[group_ids]
    return ranks


def _spearman(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    if not left or not right:
        return 0.0
    x = _rankdata(torch.cat([value.reshape(-1).cpu() for value in left]))
    y = _rankdata(torch.cat([value.reshape(-1).cpu() for value in right]))
    if x.numel() < 2 or y.numel() != x.numel():
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    return float((x * y).sum()) / denominator if denominator > 0.0 else 0.0


def _quantile(values: Sequence[torch.Tensor], q: float) -> float:
    if not values:
        return 0.0
    merged = torch.cat([value.detach().float().cpu().reshape(-1) for value in values])
    if merged.numel() == 0:
        return 0.0
    return float(torch.quantile(merged, float(q)))


class GeometryDiagnosticsAccumulator:
    """Accumulate assignment, decode, bank-stage, and ranking evidence."""

    def __init__(self, config: EgoDetV2Config) -> None:
        self.config = config
        self.levels: Dict[str, Dict[str, object]] = {
            str(name).upper(): {
                "num_locations": 0,
                "valid_locations": 0,
                "positive_locations": 0,
                "native_positive_locations": 0,
                "fallback_positive_locations": 0,
                "proposal_identity_assigned_locations": 0,
                "positive_per_gt": [],
            }
            for name in config.fpn_levels
        }
        self.total_gt = 0
        self.native_assignable_gt = 0
        self.train_assignable_gt = 0
        self.proposal_identity_covered_gt = 0
        self.valid_location_count = 0
        self.proposal_identity_assigned_locations = 0
        self.native_positive_locations = 0
        self.fallback_positive_locations = 0
        self.fallback_success_count = 0
        self.fallback_failure_count = 0
        self.zero_positive_gt_records: list[Dict[str, object]] = []
        self.positive_ious: list[torch.Tensor] = []
        self.stage_hits = {
            "dense_all_recall@0.5": 0,
            "dense_all_recall@0.75": 0,
            "after_per_level_pre_topk_recall@0.5": 0,
            "after_level_quota_recall@0.5": 0,
            "bank_top100_recall@0.5": 0,
            "bank_top300_recall@0.5": 0,
            "bank_top1000_recall@0.5": 0,
            "bank_top1500_recall@0.5": 0,
        }
        self.best_dense_iou: list[torch.Tensor] = []
        self.ranking_obj: list[torch.Tensor] = []
        self.ranking_qual: list[torch.Tensor] = []
        self.ranking_combined: list[torch.Tensor] = []
        self.ranking_iou: list[torch.Tensor] = []
        self.top_iou: Dict[int, list[torch.Tensor]] = {
            100: [],
            300: [],
            1000: [],
        }
        self.selected_level_counts = torch.zeros(
            len(config.fpn_levels),
            dtype=torch.long,
        )
        self.images = 0

    def _assigned_level(self, box: torch.Tensor) -> str:
        width = max(float(box[2] - box[0]), 0.0)
        height = max(float(box[3] - box[1]), 0.0)
        center_max_distance = 0.5 * max(width, height)
        for name, (minimum, maximum) in zip(
            self.config.fpn_levels,
            self.config.level_scale_ranges,
        ):
            if float(minimum) <= center_max_distance < float(maximum):
                return str(name).upper()
        return str(self.config.fpn_levels[-1]).upper()

    @torch.no_grad()
    def update(
        self,
        dense: Mapping[str, object],
        bank: Mapping[str, torch.Tensor],
        targets: Sequence[Mapping[str, object]],
    ) -> None:
        levels = list(dense["levels"])
        batch = len(targets)
        self.images += batch
        per_image_positive: list[torch.Tensor] = []
        per_image_native: list[torch.Tensor] = []
        per_image_proposal_identity: list[torch.Tensor] = []
        for batch_index, target in enumerate(targets):
            gt_count = int(target["boxes_xyxy_abs"].shape[0])
            for destination in (
                per_image_positive,
                per_image_native,
                per_image_proposal_identity,
            ):
                destination.append(
                    torch.zeros(
                        gt_count,
                        dtype=torch.long,
                        device=bank["boxes"].device,
                    )
                )
        fallback_failure = dense.get("fallback_failure_count", 0)
        fallback_success = dense.get("fallback_success_count", 0)
        self.fallback_success_count += int(
            fallback_success.detach().float().cpu()
            if torch.is_tensor(fallback_success)
            else fallback_success
        )
        self.fallback_failure_count += int(
            fallback_failure.detach().float().cpu()
            if torch.is_tensor(fallback_failure)
            else fallback_failure
        )

        for level_index, level in enumerate(levels):
            level_name = str(level["name"]).upper()
            stats = self.levels[level_name]
            valid = level["valid"].to(torch.bool)
            assign = (
                level["train_assign_gt"]
                if "train_assign_gt" in level
                else level["assign_gt"]
            ).to(torch.long)
            source = level.get("train_assign_source")
            if source is None:
                source = torch.where(
                    assign >= 0,
                    torch.zeros_like(assign),
                    torch.full_like(assign, -1),
                )
            proposal_assign = level.get("proposal_gt_assign", assign).to(
                torch.long
            )
            positive = valid & (assign >= 0)
            native_positive = valid & (source == 0)
            fallback_positive = valid & (source == 1)
            proposal_positive = valid & (proposal_assign >= 0)
            stats["num_locations"] = int(stats["num_locations"]) + int(valid.numel())
            stats["valid_locations"] = int(stats["valid_locations"]) + int(valid.sum())
            stats["positive_locations"] = int(stats["positive_locations"]) + int(
                positive.sum()
            )
            stats["native_positive_locations"] = int(
                stats["native_positive_locations"]
            ) + int(native_positive.sum())
            stats["fallback_positive_locations"] = int(
                stats["fallback_positive_locations"]
            ) + int(fallback_positive.sum())
            stats["proposal_identity_assigned_locations"] = int(
                stats["proposal_identity_assigned_locations"]
            ) + int(proposal_positive.sum())
            self.valid_location_count += int(valid.sum())
            self.native_positive_locations += int(native_positive.sum())
            self.fallback_positive_locations += int(fallback_positive.sum())
            self.proposal_identity_assigned_locations += int(
                proposal_positive.sum()
            )
            if positive.any():
                positive_iou = aligned_box_iou(
                    level["boxes"].float()[positive],
                    level["target_boxes"].float()[positive],
                )
                self.positive_ious.append(positive_iou.cpu())
            for batch_index, target in enumerate(targets):
                gt_count = int(target["boxes_xyxy_abs"].shape[0])
                counts = torch.bincount(
                    assign[batch_index, positive[batch_index]],
                    minlength=gt_count,
                ) if gt_count > 0 else torch.empty(
                    0,
                    dtype=torch.long,
                    device=assign.device,
                )
                stats["positive_per_gt"].extend(int(value) for value in counts.cpu())
                per_image_positive[batch_index] += counts
                native_counts = torch.bincount(
                    assign[batch_index, native_positive[batch_index]],
                    minlength=gt_count,
                ) if gt_count > 0 else counts
                proposal_counts = torch.bincount(
                    proposal_assign[
                        batch_index,
                        proposal_positive[batch_index],
                    ],
                    minlength=gt_count,
                ) if gt_count > 0 else counts
                per_image_native[batch_index] += native_counts
                per_image_proposal_identity[batch_index] += proposal_counts

        for batch_index, target in enumerate(targets):
            gt = target["boxes_xyxy_abs"].to(
                device=bank["boxes"].device,
                dtype=torch.float32,
            )
            gt_count = int(gt.shape[0])
            self.total_gt += gt_count
            if gt_count == 0:
                continue
            positive_counts = per_image_positive[batch_index]
            self.train_assignable_gt += int((positive_counts > 0).sum())
            self.native_assignable_gt += int(
                (per_image_native[batch_index] > 0).sum()
            )
            self.proposal_identity_covered_gt += int(
                (per_image_proposal_identity[batch_index] > 0).sum()
            )
            image_id_value = target.get("image_id", -1)
            image_id = (
                int(image_id_value.item())
                if torch.is_tensor(image_id_value)
                else int(image_id_value)
            )
            for gt_index in torch.nonzero(
                positive_counts == 0,
                as_tuple=False,
            ).flatten():
                index = int(gt_index)
                box = gt[index].detach().cpu()
                self.zero_positive_gt_records.append(
                    {
                        "image_id": image_id,
                        "gt_index": index,
                        "box_size": [
                            float(box[2] - box[0]),
                            float(box[3] - box[1]),
                        ],
                        "assigned_level": self._assigned_level(box),
                    }
                )

            level_boxes: list[torch.Tensor] = []
            level_obj: list[torch.Tensor] = []
            level_qual: list[torch.Tensor] = []
            level_combined: list[torch.Tensor] = []
            pretop_boxes: list[torch.Tensor] = []
            quota_boxes: list[torch.Tensor] = []
            for level in levels:
                valid = level["valid"][batch_index].to(torch.bool)
                boxes = level["boxes"][batch_index, valid].float()
                obj = torch.sigmoid(
                    level["obj_logits"][batch_index, valid, 0].float()
                )
                qual = torch.sigmoid(
                    level["qual_logits"][batch_index, valid, 0].float()
                )
                combined = obj * qual
                level_boxes.append(boxes)
                level_obj.append(obj)
                level_qual.append(qual)
                level_combined.append(combined)
                pre_count = min(
                    int(boxes.shape[0]),
                    max(
                        int(self.config.bank_pre_topk_per_level),
                        int(self.config.bank_min_per_level),
                    ),
                )
                if pre_count > 0:
                    order = torch.topk(combined, k=pre_count, sorted=True).indices
                    selected = boxes[order]
                    pretop_boxes.append(selected)
                    quota_boxes.append(
                        selected[: min(int(self.config.bank_min_per_level), pre_count)]
                    )

            all_boxes = torch.cat(level_boxes, dim=0)
            all_obj = torch.cat(level_obj, dim=0)
            all_qual = torch.cat(level_qual, dim=0)
            all_combined = torch.cat(level_combined, dim=0)
            pretop = (
                torch.cat(pretop_boxes, dim=0)
                if pretop_boxes
                else all_boxes[:0]
            )
            quota = (
                torch.cat(quota_boxes, dim=0)
                if quota_boxes
                else all_boxes[:0]
            )
            bank_valid = bank["valid"][batch_index].to(torch.bool)
            bank_boxes = bank["boxes"][batch_index, bank_valid].float()

            dense_best = _best_iou_per_gt(all_boxes, gt)
            pretop_best = _best_iou_per_gt(pretop, gt)
            quota_best = _best_iou_per_gt(quota, gt)
            self.best_dense_iou.append(dense_best.cpu())
            self.stage_hits["dense_all_recall@0.5"] += int(
                (dense_best >= 0.5).sum()
            )
            self.stage_hits["dense_all_recall@0.75"] += int(
                (dense_best >= 0.75).sum()
            )
            self.stage_hits["after_per_level_pre_topk_recall@0.5"] += int(
                (pretop_best >= 0.5).sum()
            )
            self.stage_hits["after_level_quota_recall@0.5"] += int(
                (quota_best >= 0.5).sum()
            )
            for k in (100, 300, 1000, 1500):
                best = _best_iou_per_gt(bank_boxes[:k], gt)
                self.stage_hits[f"bank_top{k}_recall@0.5"] += int(
                    (best >= 0.5).sum()
                )

            if all_boxes.numel() > 0:
                proposal_iou = pairwise_box_iou(all_boxes, gt)[0].max(dim=1).values
                self.ranking_obj.append(all_obj.cpu())
                self.ranking_qual.append(all_qual.cpu())
                self.ranking_combined.append(all_combined.cpu())
                self.ranking_iou.append(proposal_iou.cpu())
            if bank_boxes.numel() > 0:
                bank_iou = pairwise_box_iou(bank_boxes, gt)[0].max(dim=1).values
                for k in self.top_iou:
                    self.top_iou[k].append(bank_iou[:k].cpu())

        if "level_counts" in bank:
            self.selected_level_counts += bank["level_counts"].sum(dim=0).cpu()

    def compute(self) -> Dict[str, object]:
        result: Dict[str, object] = {}
        for name, stats in self.levels.items():
            positive_per_gt = list(stats["positive_per_gt"])
            result[name] = {
                "num_locations": int(stats["num_locations"]),
                "valid_locations": int(stats["valid_locations"]),
                "positive_locations": int(stats["positive_locations"]),
                "native_positive_locations": int(
                    stats["native_positive_locations"]
                ),
                "fallback_positive_locations": int(
                    stats["fallback_positive_locations"]
                ),
                "proposal_identity_assigned_locations": int(
                    stats["proposal_identity_assigned_locations"]
                ),
                "positive_per_gt_mean": (
                    sum(positive_per_gt) / len(positive_per_gt)
                    if positive_per_gt
                    else 0.0
                ),
                "positive_per_gt_min": min(positive_per_gt)
                if positive_per_gt
                else 0,
                "gt_with_zero_positive_on_level": sum(
                    int(value == 0) for value in positive_per_gt
                ),
            }
            # Temporary compatibility alias; the scoped name above is the
            # authoritative MOD9A metric.
            result[name]["gt_with_zero_positive"] = result[name][
                "gt_with_zero_positive_on_level"
            ]
        result["native_assignable_gt_ratio"] = (
            self.native_assignable_gt / max(self.total_gt, 1)
        )
        result["train_assignable_gt_ratio"] = (
            self.train_assignable_gt / max(self.total_gt, 1)
        )
        result["assignable_gt_ratio"] = result["train_assignable_gt_ratio"]
        result["native_positive_locations"] = self.native_positive_locations
        result["fallback_positive_locations"] = self.fallback_positive_locations
        result["proposal_identity_assigned_locations"] = (
            self.proposal_identity_assigned_locations
        )
        result["proposal_identity_assigned_ratio"] = (
            self.proposal_identity_assigned_locations
            / max(self.valid_location_count, 1)
        )
        result["proposal_identity_gt_coverage"] = (
            self.proposal_identity_covered_gt / max(self.total_gt, 1)
        )
        result["fallback_failure_count"] = self.fallback_failure_count
        result["fallback_success_count"] = self.fallback_success_count
        result["zero_positive_gt_records"] = self.zero_positive_gt_records
        result["positive_pred_iou_mean"] = _quantile(self.positive_ious, 0.5)
        result["positive_pred_iou_median"] = _quantile(self.positive_ious, 0.5)
        result["positive_pred_iou_p90"] = _quantile(self.positive_ious, 0.9)
        if self.positive_ious:
            positives = torch.cat(self.positive_ious)
            result["positive_pred_iou_mean"] = float(positives.mean())
            result["positive_pred_iou_fraction_ge_0.5"] = float(
                (positives >= 0.5).float().mean()
            )
            result["positive_pred_iou_fraction_ge_0.75"] = float(
                (positives >= 0.75).float().mean()
            )
        else:
            result["positive_pred_iou_fraction_ge_0.5"] = 0.0
            result["positive_pred_iou_fraction_ge_0.75"] = 0.0
        result["positive_pred_iou_fraction>=0.5"] = result[
            "positive_pred_iou_fraction_ge_0.5"
        ]
        result["positive_pred_iou_fraction>=0.75"] = result[
            "positive_pred_iou_fraction_ge_0.75"
        ]
        for name, hit_count in self.stage_hits.items():
            result[name] = hit_count / max(self.total_gt, 1)
        if self.best_dense_iou:
            best = torch.cat(self.best_dense_iou)
            result["best_iou_per_gt_mean"] = float(best.mean())
            result["best_iou_per_gt_median"] = float(best.median())
            result["best_iou_per_gt_min"] = float(best.min())
        else:
            result["best_iou_per_gt_mean"] = 0.0
            result["best_iou_per_gt_median"] = 0.0
            result["best_iou_per_gt_min"] = 0.0
        result["objectness_vs_iou_spearman"] = _spearman(
            self.ranking_obj,
            self.ranking_iou,
        )
        result["quality_vs_iou_spearman"] = _spearman(
            self.ranking_qual,
            self.ranking_iou,
        )
        result["combined_score_vs_iou_spearman"] = _spearman(
            self.ranking_combined,
            self.ranking_iou,
        )
        for k, values in self.top_iou.items():
            result[f"top{k}_mean_iou"] = (
                float(torch.cat(values).mean()) if values else 0.0
            )
        for index, name in enumerate(self.config.fpn_levels):
            count = int(self.selected_level_counts[index])
            result[f"{str(name).upper()}_selected_count"] = count
            result[f"{str(name).upper()}_selected_mean_per_image"] = (
                count / max(self.images, 1)
            )
        result["diagnostic_gt_count"] = self.total_gt
        result["diagnostic_image_count"] = self.images
        return result
