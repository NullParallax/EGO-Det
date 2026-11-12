from __future__ import annotations

import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - runtime dependency contract
    raise RuntimeError(
        "tqdm is required for EGO-Det V2 validation progress"
    ) from exc

from ..losses.ownership_losses import build_soft_ownership_targets
from ..models.box_ops import aligned_box_iou, box_cxcywh_to_xyxy, pairwise_box_iou
from ..models.dense_geometry_branch import oracle_recall
from ..models.matcher import HungarianMatcherV2
from .geometry_diagnostics import GeometryDiagnosticsAccumulator


def _evaluation_autocast(model, device: torch.device):
    config = model.config
    if device.type != "cuda" or not bool(config.amp):
        return contextlib.nullcontext()
    use_bf16 = (
        str(config.amp_dtype).lower() == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )


@torch.no_grad()
def matched_slot_metrics(
    outputs: Dict[str, object],
    targets: Sequence[Dict],
    matcher: HungarianMatcherV2,
) -> Dict[str, float]:
    prediction = {
        "pred_logits": outputs["pred_logits"].float(),
        "pred_boxes": outputs["pred_boxes"].float(),
    }
    matches = matcher(prediction, targets)
    correct = 0
    total = 0
    iou_sum = 0.0
    recalled = 0
    for batch_index, (slot_indices, gt_indices) in enumerate(matches):
        if slot_indices.numel() == 0:
            continue
        labels = targets[batch_index]["labels"].to(outputs["pred_logits"].device)[gt_indices]
        predicted_labels = (
            outputs["pred_logits"][batch_index, slot_indices]
            .float()
            .argmax(dim=-1)
        )
        correct += int((predicted_labels == labels).sum())
        predicted_boxes = box_cxcywh_to_xyxy(
            outputs["pred_boxes"][batch_index, slot_indices].float()
        )
        target_boxes = box_cxcywh_to_xyxy(
            targets[batch_index]["boxes"].to(
                device=predicted_boxes.device,
                dtype=torch.float32,
            )[gt_indices]
        )
        iou = aligned_box_iou(predicted_boxes, target_boxes)
        iou_sum += float(iou.sum())
        recalled += int((iou >= 0.5).sum())
        total += int(slot_indices.numel())
    denominator = max(total, 1)
    return {
        "matched_accuracy": correct / denominator,
        "matched_mean_iou": iou_sum / denominator,
        "matched_recall_50": recalled / denominator,
        "matched_count": float(total),
    }


def _postprocess_for_coco(
    outputs: Dict[str, object],
    targets: Sequence[Dict],
    contiguous_to_category_id: Dict[int, int],
    *,
    topk: int = 300,
) -> List[Dict[str, float | int | List[float]]]:
    scores = outputs["pred_scores"].float()
    boxes = outputs["pred_boxes"].float()
    classes = int(scores.shape[-1])
    results: List[Dict[str, float | int | List[float]]] = []
    for batch_index, target in enumerate(targets):
        flattened = scores[batch_index].flatten()
        count = min(int(topk), int(flattened.numel()))
        values, flat_indices = torch.topk(flattened, k=count)
        slots = torch.div(flat_indices, classes, rounding_mode="floor")
        labels = flat_indices % classes
        xyxy = box_cxcywh_to_xyxy(boxes[batch_index, slots])
        orig_h, orig_w = target["orig_size"]
        scale = xyxy.new_tensor([orig_w, orig_h, orig_w, orig_h])
        xyxy = xyxy * scale
        xywh = xyxy.clone()
        xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
        for score, label, box in zip(values, labels, xywh):
            results.append(
                {
                    "image_id": int(target["image_id"]),
                    "category_id": int(contiguous_to_category_id[int(label)]),
                    "bbox": [float(value) for value in box],
                    "score": float(score),
                }
            )
    return results


@torch.no_grad()
def _evaluate_impl(
    model,
    data_loader: Iterable,
    *,
    matcher: HungarianMatcherV2,
    device: torch.device,
    dataset=None,
    output_json: Optional[Path | str] = None,
    subset_size: int = 0,
    show_progress: bool = True,
    include_geometry_diagnostics: bool = False,
    geometry_only: bool = False,
) -> Dict[str, object]:
    model.eval()
    coco_results: List[Dict] = []
    metric_sums = defaultdict(float)
    metric_weight = 0.0
    oracle_sums = defaultdict(float)
    oracle_weight = 0.0
    batches = 0
    dustbin_sum = 0.0
    matched_dustbin_sum = 0.0
    matched_dustbin_weight = 0.0
    proposal_mass_sum = 0.0
    duplicate_sum = 0.0
    gain_sum = 0.0
    seen_images = 0
    geometry_accumulator = (
        GeometryDiagnosticsAccumulator(model.config)
        if include_geometry_diagnostics
        else None
    )
    try:
        loader_images = int(len(data_loader.dataset))
    except (AttributeError, TypeError):
        loader_images = 0
    if subset_size:
        progress_total = min(
            int(subset_size),
            loader_images if loader_images > 0 else int(subset_size),
        )
    else:
        progress_total = loader_images or None
    progress = tqdm(
        total=progress_total,
        desc="Val",
        unit="img",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )

    for images, pad_mask, targets in data_loader:
        if subset_size and seen_images >= int(subset_size):
            break
        if subset_size and seen_images + len(targets) > int(subset_size):
            keep = int(subset_size) - seen_images
            images = images[:keep]
            pad_mask = pad_mask[:keep]
            targets = targets[:keep]
        images = images.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        with _evaluation_autocast(model, device):
            outputs = (
                model.forward_geometry(images, pad_mask, targets)
                if geometry_only
                else model(images, pad_mask, targets)
            )
        if not geometry_only:
            metrics = matched_slot_metrics(outputs, targets, matcher)
            count = metrics["matched_count"]
            for name in ("matched_accuracy", "matched_mean_iou", "matched_recall_50"):
                metric_sums[name] += metrics[name] * count
            metric_weight += count

        oracle = oracle_recall(outputs["bank"], targets)
        batch_gt = float(sum(int(target["labels"].numel()) for target in targets))
        for name, value in oracle.items():
            oracle_sums[name] += value * batch_gt
        oracle_weight += batch_gt
        batches += 1
        if geometry_accumulator is not None:
            geometry_accumulator.update(outputs["dense"], outputs["bank"], targets)
        if not geometry_only:
            dustbin_sum += float(
                outputs["ownership"][-1]["w"][..., -1].float().mean()
            )
            proposal_mass_sum += float(
                outputs["ownership"][-1]["proposal_mass"].float().mean()
            )

        # Readout gain is a monitor: final matched IoU minus best owned proposal.
        if not geometry_only:
            final_matches = matcher(
                {
                    "pred_logits": outputs["pred_logits"],
                    "pred_boxes": outputs["pred_boxes"],
                },
                targets,
            )
            ownership_target, ownership_matched = build_soft_ownership_targets(
                outputs["ownership"][-1],
                outputs["bank"],
                final_matches,
                targets,
            )
            matched_dustbin_sum += float(
                (ownership_matched & (ownership_target[..., -1] > 0.5)).sum()
            )
            matched_dustbin_weight += float(ownership_matched.sum())
            batch_gain: List[float] = []
            batch_duplicates: List[float] = []
            for batch_index, (slot_indices, gt_indices) in enumerate(final_matches):
                if slot_indices.numel() == 0:
                    continue
                predicted_norm = outputs["pred_boxes"][
                    batch_index, slot_indices
                ].float()
                target_norm = targets[batch_index]["boxes"].to(
                    device=device,
                    dtype=torch.float32,
                )[gt_indices]
                final_iou = aligned_box_iou(
                    box_cxcywh_to_xyxy(predicted_norm),
                    box_cxcywh_to_xyxy(target_norm),
                )
                gt_abs = targets[batch_index]["boxes_xyxy_abs"].to(
                    device=device,
                    dtype=torch.float32,
                )
                for local, gt_tensor in enumerate(gt_indices):
                    gt_index = int(gt_tensor.item())
                    owned = (
                        outputs["bank"]["valid"][batch_index]
                        & (outputs["bank"]["gt_assign"][batch_index] == gt_index)
                    )
                    if owned.any():
                        oracle_iou = pairwise_box_iou(
                            outputs["bank"]["boxes"][batch_index, owned].float(),
                            gt_abs[gt_index : gt_index + 1],
                        )[0][:, 0].max()
                        batch_gain.append(float(final_iou[local] - oracle_iou))
                all_boxes = box_cxcywh_to_xyxy(
                    outputs["pred_boxes"][batch_index].float()
                )
                pairwise = pairwise_box_iou(all_boxes, all_boxes)[0]
                off_diagonal = ~torch.eye(
                    pairwise.shape[0], dtype=torch.bool, device=pairwise.device
                )
                batch_duplicates.append(
                    float((pairwise[off_diagonal] > 0.5).float().mean())
                )
            gain_sum += sum(batch_gain) / max(len(batch_gain), 1)
            duplicate_sum += sum(batch_duplicates) / max(len(batch_duplicates), 1)

        if dataset is not None and not geometry_only:
            coco_results.extend(
                _postprocess_for_coco(
                    outputs,
                    targets,
                    dataset.contiguous_to_category_id,
                )
            )
        seen_images += len(targets)
        predictions_per_image = (
            min(
                300,
                int(outputs["pred_scores"].shape[1])
                * int(outputs["pred_scores"].shape[2]),
            )
            if not geometry_only
            else int(outputs["bank"]["valid"].sum(dim=1).float().mean())
        )
        running_ar300 = (
            oracle_sums["AR@300"] / max(oracle_weight, 1.0)
        )
        progress.update(len(targets))
        progress.set_postfix(
            {
                "pred/img": predictions_per_image,
                "oracle_AR300": f"{running_ar300:.4f}",
            },
            refresh=True,
        )

    progress.close()

    result = {
        name: value / max(metric_weight, 1.0)
        for name, value in metric_sums.items()
    }
    result["matched_count"] = metric_weight
    for name, value in oracle_sums.items():
        result[f"bank_{name}"] = value / max(oracle_weight, 1.0)
    result["dustbin_mean"] = dustbin_sum / max(batches, 1)
    result["ownership_matched_dustbin_rate"] = (
        matched_dustbin_sum / max(matched_dustbin_weight, 1.0)
    )
    result["proposal_mass_mean"] = proposal_mass_sum / max(batches, 1)
    result["duplicate_rate"] = duplicate_sum / max(batches, 1)
    result["readout_gain"] = gain_sum / max(batches, 1)
    if geometry_accumulator is not None:
        result.update(geometry_accumulator.compute())

    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(coco_results), encoding="utf-8")

    if dataset is not None and coco_results:
        try:
            from pycocotools.cocoeval import COCOeval

            with contextlib.redirect_stdout(io.StringIO()):
                detections = dataset.coco.loadRes(coco_results)
                evaluator = COCOeval(dataset.coco, detections, "bbox")
                if subset_size:
                    evaluator.params.imgIds = sorted(
                        {int(item["image_id"]) for item in coco_results}
                    )
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            names = (
                "AP",
                "AP50",
                "AP75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "AR_1",
                "AR_10",
                "AR_100",
                "AR_small",
                "AR_medium",
                "AR_large",
            )
            result.update({name: float(value) for name, value in zip(names, evaluator.stats)})
        except Exception as exc:  # evaluation remains diagnostic, never training control
            result["coco_eval_error"] = str(exc)
    return result


@torch.no_grad()
def evaluate(
    model,
    data_loader: Iterable,
    **kwargs,
) -> Dict[str, object]:
    """Evaluate and always restore the caller's original train/eval mode."""

    was_training = bool(model.training)
    try:
        return _evaluate_impl(model, data_loader, **kwargs)
    finally:
        model.train(was_training)
