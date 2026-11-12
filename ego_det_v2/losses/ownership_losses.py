from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..models.box_ops import aligned_box_iou, pairwise_box_iou


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, slots, candidates = indices.shape
    trailing = values.shape[2:]
    expanded = values[:, None].expand(batch, slots, values.shape[1], *trailing)
    gather_index = indices.view(
        batch, slots, candidates, *([1] * len(trailing))
    ).expand(batch, slots, candidates, *trailing)
    return torch.gather(expanded, 2, gather_index)


@torch.no_grad()
def build_soft_ownership_targets_reference(
    ownership: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build normalized IoU×quality targets plus a dustbin dimension."""

    logits = ownership["logits"]
    indices = ownership["cand_idx"]
    candidate_valid = ownership["cand_valid"]
    bank_boxes = bank["boxes"]
    bank_quality = bank["qual"]
    bank_assign = bank["gt_assign"]

    if logits.ndim != 3:
        raise ValueError(
            f"ownership logits must have shape [B,S,P+1], got {tuple(logits.shape)}"
        )
    if indices.ndim != 3:
        raise ValueError(
            f"ownership cand_idx must have shape [B,S,P], got {tuple(indices.shape)}"
        )
    if tuple(candidate_valid.shape) != tuple(indices.shape):
        raise ValueError(
            "ownership cand_valid shape must equal cand_idx shape, "
            f"got {tuple(candidate_valid.shape)} vs {tuple(indices.shape)}"
        )
    if tuple(indices.shape[:2]) != tuple(logits.shape[:2]):
        raise ValueError(
            "ownership cand_idx [B,S] must match logits [B,S], "
            f"got {tuple(indices.shape[:2])} vs {tuple(logits.shape[:2])}"
        )
    if logits.shape[-1] != indices.shape[-1] + 1:
        raise ValueError(
            "ownership logits must contain exactly P proposal channels plus one dustbin, "
            f"got logits[-1]={logits.shape[-1]} and cand_idx[-1]={indices.shape[-1]}"
        )

    batch, slots, candidates_plus_dustbin = logits.shape
    candidates = candidates_plus_dustbin - 1
    if bank_boxes.ndim != 3 or bank_boxes.shape[-1] != 4:
        raise ValueError(
            f"bank boxes must have shape [B,K,4], got {tuple(bank_boxes.shape)}"
        )
    bank_batch, bank_size = bank_boxes.shape[:2]
    if tuple(bank_quality.shape) != (bank_batch, bank_size, 1):
        raise ValueError(
            "bank qual must have shape [B,K,1] consistent with bank boxes, "
            f"got {tuple(bank_quality.shape)} vs boxes {tuple(bank_boxes.shape)}"
        )
    if tuple(bank_assign.shape) != (bank_batch, bank_size):
        raise ValueError(
            "bank gt_assign must have shape [B,K] consistent with bank boxes, "
            f"got {tuple(bank_assign.shape)} vs boxes {tuple(bank_boxes.shape)}"
        )
    if bank_batch != batch:
        raise ValueError(
            f"bank batch size must match ownership batch size, got {bank_batch} vs {batch}"
        )
    if len(matches) != batch or len(targets) != batch:
        raise ValueError(
            "matches and targets must each contain one entry per batch item, "
            f"got len(matches)={len(matches)}, len(targets)={len(targets)}, B={batch}"
        )
    if indices.dtype != torch.long:
        raise ValueError(f"ownership cand_idx must be torch.long, got {indices.dtype}")
    if candidate_valid.dtype != torch.bool:
        raise ValueError(
            f"ownership cand_valid must be a bool tensor, got {candidate_valid.dtype}"
        )
    if indices.numel() > 0 and (
        bool((indices < 0).any()) or bool((indices >= bank_size).any())
    ):
        raise ValueError(
            f"ownership cand_idx contains an index outside bank range [0,{bank_size})"
        )

    soft_target = torch.zeros(
        logits.shape,
        device=logits.device,
        dtype=torch.float32,
    )
    matched_mask = torch.zeros((batch, slots), dtype=torch.bool, device=logits.device)
    gathered_assign = _gather_candidates(bank_assign[..., None], indices).squeeze(-1)
    gathered_boxes = _gather_candidates(bank_boxes.float(), indices)
    gathered_quality = torch.sigmoid(
        _gather_candidates(bank_quality.float(), indices).squeeze(-1)
    ).detach()

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
        matched_mask[batch_index, slot_indices] = True
        target_boxes = targets[batch_index]["boxes_xyxy_abs"].to(
            device=logits.device,
            dtype=torch.float32,
        )
        for slot_index_tensor, gt_index_tensor in zip(slot_indices, gt_indices):
            slot_index = int(slot_index_tensor)
            gt_index = int(gt_index_tensor)
            valid = candidate_valid[batch_index, slot_index]
            if not valid.any() or target_boxes.numel() == 0:
                soft_target[batch_index, slot_index, -1] = 1.0
                continue
            candidate_boxes = gathered_boxes[batch_index, slot_index]
            gt_box = target_boxes[gt_index : gt_index + 1]
            iou = pairwise_box_iou(candidate_boxes, gt_box)[0][:, 0]
            assignment_match = gathered_assign[batch_index, slot_index] == gt_index
            weight = (
                iou
                * gathered_quality[batch_index, slot_index]
                * assignment_match.to(iou.dtype)
                * valid.to(iou.dtype)
            )
            if float(weight.sum()) <= 0.0:
                # Current proposal identity should normally provide support. If the
                # local P-window misses it, choose the best current-IoU evidence
                # rather than silently converting a matched slot to dustbin.
                fallback = (
                    iou * gathered_quality[batch_index, slot_index] * valid.to(iou.dtype)
                )
                if float(fallback.max()) <= 0.0:
                    fallback = gathered_quality[batch_index, slot_index] * valid.to(iou.dtype)
                best = int(torch.argmax(fallback))
                weight[best] = 1.0
            soft_target[batch_index, slot_index, :candidates] = weight / weight.sum().clamp(
                min=1.0e-8
            )

    unmatched = ~matched_mask

    # soft_target has shape [B, S, P+1], while unmatched has shape [B, S].
    # Select the dustbin channel first so boolean indexing is performed between
    # tensors with exactly matching shapes. Do not overwrite the full dustbin
    # channel: matched slots without valid evidence may already route there.
    dustbin_target = soft_target[..., -1]
    dustbin_target.masked_fill_(unmatched, 1.0)

    row_mass = soft_target.sum(dim=-1)
    finite_rows = torch.isfinite(soft_target).all(dim=-1)
    nonnegative_rows = (soft_target >= 0.0).all(dim=-1)
    normalized_rows = torch.isclose(
        row_mass,
        torch.ones_like(row_mass),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    valid_rows = finite_rows & nonnegative_rows & normalized_rows
    if not bool(valid_rows.all()):
        bad_rows = (~valid_rows).nonzero(as_tuple=False).detach().cpu().tolist()
        finite_values = soft_target[torch.isfinite(soft_target)]
        if finite_values.numel() > 0:
            value_min = float(finite_values.min())
            value_max = float(finite_values.max())
        else:
            value_min = float("nan")
            value_max = float("nan")
        mass_min = float(row_mass.min())
        mass_max = float(row_mass.max())
        raise RuntimeError(
            "invalid soft ownership target: "
            f"bad_rows={bad_rows}, mass_range=[{mass_min},{mass_max}], "
            f"value_range=[{value_min},{value_max}]"
        )
    return soft_target, matched_mask


def _validate_ownership_shapes(
    ownership: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
) -> Tuple[int, int, int, int]:
    logits = ownership["logits"]
    indices = ownership["cand_idx"]
    candidate_valid = ownership["cand_valid"]
    bank_boxes = bank["boxes"]
    bank_quality = bank["qual"]
    bank_assign = bank["gt_assign"]
    if logits.ndim != 3:
        raise ValueError(
            f"ownership logits must have shape [B,S,P+1], got {tuple(logits.shape)}"
        )
    if indices.ndim != 3:
        raise ValueError(
            f"ownership cand_idx must have shape [B,S,P], got {tuple(indices.shape)}"
        )
    if tuple(candidate_valid.shape) != tuple(indices.shape):
        raise ValueError(
            "ownership cand_valid shape must equal cand_idx shape, "
            f"got {tuple(candidate_valid.shape)} vs {tuple(indices.shape)}"
        )
    if tuple(indices.shape[:2]) != tuple(logits.shape[:2]):
        raise ValueError(
            "ownership cand_idx [B,S] must match logits [B,S], "
            f"got {tuple(indices.shape[:2])} vs {tuple(logits.shape[:2])}"
        )
    if logits.shape[-1] != indices.shape[-1] + 1:
        raise ValueError(
            "ownership logits must contain exactly P proposal channels plus one dustbin, "
            f"got logits[-1]={logits.shape[-1]} and cand_idx[-1]={indices.shape[-1]}"
        )
    if bank_boxes.ndim != 3 or bank_boxes.shape[-1] != 4:
        raise ValueError(
            f"bank boxes must have shape [B,K,4], got {tuple(bank_boxes.shape)}"
        )
    batch, slots, candidates = indices.shape
    bank_batch, bank_size = bank_boxes.shape[:2]
    if tuple(bank_quality.shape) != (bank_batch, bank_size, 1):
        raise ValueError(
            "bank qual must have shape [B,K,1] consistent with bank boxes, "
            f"got {tuple(bank_quality.shape)} vs boxes {tuple(bank_boxes.shape)}"
        )
    if tuple(bank_assign.shape) != (bank_batch, bank_size):
        raise ValueError(
            "bank gt_assign must have shape [B,K] consistent with bank boxes, "
            f"got {tuple(bank_assign.shape)} vs boxes {tuple(bank_boxes.shape)}"
        )
    if bank_batch != batch:
        raise ValueError(
            f"bank batch size must match ownership batch size, got {bank_batch} vs {batch}"
        )
    if len(matches) != batch or len(targets) != batch:
        raise ValueError(
            "matches and targets must each contain one entry per batch item, "
            f"got len(matches)={len(matches)}, len(targets)={len(targets)}, B={batch}"
        )
    if indices.dtype != torch.long:
        raise ValueError(f"ownership cand_idx must be torch.long, got {indices.dtype}")
    if candidate_valid.dtype != torch.bool:
        raise ValueError(
            f"ownership cand_valid must be a bool tensor, got {candidate_valid.dtype}"
        )
    return batch, slots, candidates, bank_size


def _flatten_ownership_matches(
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_parts = []
    slot_parts = []
    gt_parts = []
    target_box_parts = []
    target_present_parts = []
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
        target_boxes = targets[batch_index]["boxes_xyxy_abs"].to(
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
        if target_boxes.numel() == 0:
            target_box_parts.append(
                torch.zeros((count, 4), dtype=torch.float32, device=device)
            )
            target_present_parts.append(
                torch.zeros(count, dtype=torch.bool, device=device)
            )
        else:
            target_box_parts.append(target_boxes.index_select(0, gts))
            target_present_parts.append(
                torch.ones(count, dtype=torch.bool, device=device)
            )
    if not batch_parts:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        return (
            empty_long,
            empty_long,
            empty_long,
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty(0, dtype=torch.bool, device=device),
        )
    return (
        torch.cat(batch_parts),
        torch.cat(slot_parts),
        torch.cat(gt_parts),
        torch.cat(target_box_parts),
        torch.cat(target_present_parts),
    )


@torch.no_grad()
def build_soft_ownership_targets_optimized(
    ownership: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized ownership targets for all matched slot rows.

    Shapes below use M for the total number of matched slots and P for the
    candidate window.  All candidate IoUs are computed as one aligned
    ``[M,P]`` operation; no CUDA value is converted to a Python scalar.
    """

    batch, slots, candidates, bank_size = _validate_ownership_shapes(
        ownership,
        bank,
        matches,
        targets,
    )
    logits = ownership["logits"]
    indices = ownership["cand_idx"]
    candidate_valid = ownership["cand_valid"]
    if indices.numel() > 0 and logits.device.type != "cuda":
        if bool((indices < 0).any()) or bool((indices >= bank_size).any()):
            raise ValueError(
                f"ownership cand_idx contains an index outside bank range [0,{bank_size})"
            )

    soft_target = torch.zeros(
        (*logits.shape[:2], candidates + 1),
        dtype=torch.float32,
        device=logits.device,
    )
    matched_mask = torch.zeros(
        (batch, slots),
        dtype=torch.bool,
        device=logits.device,
    )
    (
        batch_idx,
        slot_idx,
        gt_idx,
        target_boxes,
        target_present,
    ) = _flatten_ownership_matches(
        matches,
        targets,
        device=logits.device,
    )
    if batch_idx.numel() > 0:
        matched_mask[batch_idx, slot_idx] = True
        matched_candidate_indices = indices[batch_idx, slot_idx]  # [M,P]
        matched_valid = candidate_valid[batch_idx, slot_idx]  # [M,P]
        gather_batch = batch_idx[:, None].expand_as(matched_candidate_indices)
        candidate_boxes = bank["boxes"][
            gather_batch,
            matched_candidate_indices,
        ].float()  # [M,P,4]
        candidate_assign = bank["gt_assign"][
            gather_batch,
            matched_candidate_indices,
        ]  # [M,P]
        candidate_quality = torch.sigmoid(
            bank["qual"][
                gather_batch,
                matched_candidate_indices,
                0,
            ].float()
        ).detach()  # [M,P]
        expanded_targets = target_boxes[:, None, :].expand_as(candidate_boxes)
        candidate_iou = aligned_box_iou(
            candidate_boxes,
            expanded_targets,
        )  # [M,P]

        usable = matched_valid & target_present[:, None]
        assignment_match = candidate_assign == gt_idx[:, None]
        weight = (
            candidate_iou
            * candidate_quality
            * assignment_match.to(candidate_iou.dtype)
            * usable.to(candidate_iou.dtype)
        )
        needs_fallback = weight.sum(dim=-1) <= 0.0
        iou_quality_fallback = (
            candidate_iou
            * candidate_quality
            * usable.to(candidate_iou.dtype)
        )
        quality_fallback = candidate_quality * usable.to(candidate_iou.dtype)
        use_quality_only = iou_quality_fallback.max(dim=-1).values <= 0.0
        fallback = torch.where(
            use_quality_only[:, None],
            quality_fallback,
            iou_quality_fallback,
        )
        fallback_index = fallback.argmax(dim=-1)
        fallback_one_hot = F.one_hot(
            fallback_index,
            num_classes=candidates,
        ).to(weight.dtype)
        has_usable = usable.any(dim=-1)
        apply_fallback = needs_fallback & has_usable
        weight = torch.where(
            apply_fallback[:, None],
            fallback_one_hot,
            weight,
        )
        normalized = weight / weight.sum(dim=-1, keepdim=True).clamp(
            min=1.0e-8
        )
        soft_target[batch_idx, slot_idx, :candidates] = normalized
        route_to_dustbin = ~has_usable
        soft_target[batch_idx, slot_idx, -1] = route_to_dustbin.to(
            soft_target.dtype
        )

    soft_target[..., -1].masked_fill_(~matched_mask, 1.0)
    if soft_target.device.type != "cuda":
        row_mass = soft_target.sum(dim=-1)
        valid_rows = (
            torch.isfinite(soft_target).all(dim=-1)
            & (soft_target >= 0.0).all(dim=-1)
            & torch.isclose(
                row_mass,
                torch.ones_like(row_mass),
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )
        if not bool(valid_rows.all()):
            raise RuntimeError("invalid optimized soft ownership target")
    return soft_target, matched_mask


@torch.no_grad()
def build_soft_ownership_targets(
    ownership: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
    *,
    efficiency_mode: str = "reference",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if efficiency_mode == "optimized":
        return build_soft_ownership_targets_optimized(
            ownership,
            bank,
            matches,
            targets,
        )
    if efficiency_mode != "reference":
        raise ValueError(
            "efficiency_mode must be 'reference' or 'optimized'"
        )
    return build_soft_ownership_targets_reference(
        ownership,
        bank,
        matches,
        targets,
    )


def soft_ownership_losses(
    ownership: Dict[str, torch.Tensor],
    bank: Dict[str, torch.Tensor],
    matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    targets: Sequence[Dict],
    *,
    efficiency_mode: str = "reference",
) -> Dict[str, torch.Tensor]:
    target, matched = build_soft_ownership_targets(
        ownership,
        bank,
        matches,
        targets,
        efficiency_mode=efficiency_mode,
    )
    logits = ownership["logits"].float()
    target = target.to(device=logits.device, dtype=torch.float32)
    log_probability = F.log_softmax(logits, dim=-1)
    row_cross_entropy = -(target * log_probability).sum(dim=-1)
    matched_count = matched.sum().clamp(min=1)
    matched_denominator = matched_count.to(dtype=torch.float32)
    unmatched = ~matched
    unmatched_count = unmatched.sum().clamp(min=1)
    unmatched_denominator = unmatched_count.to(dtype=torch.float32)
    matched_loss = (
        row_cross_entropy * matched.to(row_cross_entropy.dtype)
    ).sum() / matched_denominator
    dustbin_loss = (
        row_cross_entropy * unmatched.to(row_cross_entropy.dtype)
    ).sum() / unmatched_denominator

    with torch.no_grad():
        proposal_supervised = matched & (target[..., :-1].sum(dim=-1) > 0.0)
        proposal_supervised_count = proposal_supervised.sum().clamp(min=1)
        if target.shape[-1] > 1:
            predicted = ownership["logits"][..., :-1].argmax(dim=-1)
            expected = target[..., :-1].argmax(dim=-1)
            correct = ((predicted == expected) & proposal_supervised).sum()
            accuracy = correct.float() / proposal_supervised_count.float()
        else:
            accuracy = target.new_zeros(())
        matched_dustbin_rate = (
            (matched & (target[..., -1] > 0.5)).sum().float()
            / matched_denominator
        )
        entropy = (
            -(target[..., :-1].clamp(min=1.0e-8).log() * target[..., :-1]).sum(dim=-1)
            * matched
        ).sum() / matched_denominator
    return {
        "loss_ownership": matched_loss,
        "loss_dustbin": dustbin_loss,
        "ownership_accuracy": accuracy,
        "ownership_target_entropy": entropy,
        "ownership_matched_dustbin_rate": matched_dustbin_rate,
    }
