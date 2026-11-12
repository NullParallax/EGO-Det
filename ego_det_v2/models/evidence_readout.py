from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import (
    absolute_xyxy_to_normalized_cxcywh,
    clip_normalized_cxcywh,
    normalized_cxcywh_to_absolute_xyxy,
    pairwise_box_iou,
)
from .slot_update import BoxPositionalEncoding, ZeroMLP


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, slots, candidates = indices.shape
    trailing = values.shape[2:]
    expanded_values = values[:, None].expand(batch, slots, values.shape[1], *trailing)
    gather_index = indices.view(
        batch, slots, candidates, *([1] * len(trailing))
    ).expand(batch, slots, candidates, *trailing)
    return torch.gather(expanded_values, dim=2, index=gather_index)


def _aligned_candidate_iou(current: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    current = current.float()
    candidates = candidates.float()
    current_expanded = current[:, :, None, :]
    lt = torch.maximum(current_expanded[..., :2], candidates[..., :2])
    rb = torch.minimum(current_expanded[..., 2:], candidates[..., 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    current_wh = (current_expanded[..., 2:] - current_expanded[..., :2]).clamp(min=0)
    candidate_wh = (candidates[..., 2:] - candidates[..., :2]).clamp(min=0)
    current_area = current_wh[..., 0] * current_wh[..., 1]
    candidate_area = candidate_wh[..., 0] * candidate_wh[..., 1]
    union = current_area + candidate_area - intersection
    return intersection / union.clamp(min=1.0e-7)


def apply_bounded_residual(
    b_hat_normalized: torch.Tensor,
    raw_residual: torch.Tensor,
) -> torch.Tensor:
    """Apply the contracted bounded residual in normalized cxcywh space."""

    b_hat_normalized = b_hat_normalized.float()
    raw_residual = raw_residual.float()
    dx, dy, dw, dh = raw_residual.unbind(-1)
    cx_hat, cy_hat, width_hat, height_hat = b_hat_normalized.unbind(-1)
    cx = cx_hat + 0.5 * torch.tanh(dx) * width_hat
    cy = cy_hat + 0.5 * torch.tanh(dy) * height_hat
    width = width_hat * torch.exp(0.35 * torch.tanh(dw))
    height = height_hat * torch.exp(0.35 * torch.tanh(dh))
    return clip_normalized_cxcywh(torch.stack((cx, cy, width, height), dim=-1))


class EvidenceReadoutRound(nn.Module):
    """Persistent-bank readout with candidate-complete feature paths."""

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        channels = int(config.d_model)
        level_dim = int(config.level_embed_dim)
        self.level_embedding = nn.Embedding(len(config.fpn_levels), level_dim)
        self.relative_geometry_dim = 6
        self.candidate_dim = channels + 2 + level_dim + self.relative_geometry_dim
        self.candidate_norm = nn.LayerNorm(self.candidate_dim)

        # Ownership consumes the complete candidate feature through both its
        # key and additive branch. Names remain explicit for contract checks.
        self.mlp_q = nn.Linear(channels, channels)
        self.mlp_k = nn.Sequential(
            nn.Linear(self.candidate_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )
        self.mlp_geo = nn.Sequential(
            nn.Linear(self.candidate_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
        )
        self.dustbin = nn.Linear(channels, 1)
        self.desc_update = ZeroMLP(channels, channels)

        # Side weighting directly consumes descriptor, obj/quality logits,
        # level embedding, and relative geometry as one feature.
        self.mlp_side = nn.Sequential(
            nn.Linear(self.candidate_dim, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 4),
        )
        self.box_pe = BoxPositionalEncoding(channels)
        # Residual input includes the ownership-pooled *raw* complete feature,
        # making obj/quality value gradients unambiguous.
        residual_input = channels + channels + self.candidate_dim + channels
        self.mlp_res = nn.Sequential(
            nn.Linear(residual_input, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 4),
        )

    @torch.no_grad()
    def select_neighborhood(
        self,
        boxes: torch.Tensor,
        bank: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        current_abs = normalized_cxcywh_to_absolute_xyxy(
            boxes.detach().float(),
            bank["image_sizes"],
        )
        bank_boxes = bank["boxes"].detach().float()
        batch, slots = current_abs.shape[:2]
        bank_count = int(bank_boxes.shape[1])
        requested = int(self.config.evidence_per_slot)
        actual = min(requested, max(bank_count, 1))
        if bank_count == 0:
            return (
                torch.zeros((batch, slots, requested), dtype=torch.long, device=boxes.device),
                torch.zeros((batch, slots, requested), dtype=torch.bool, device=boxes.device),
            )

        iou, _ = pairwise_box_iou(current_abs, bank_boxes)
        current_centers = 0.5 * (current_abs[..., :2] + current_abs[..., 2:])
        bank_centers = 0.5 * (bank_boxes[..., :2] + bank_boxes[..., 2:])
        distance = torch.cdist(current_centers.float(), bank_centers.float()).to(iou.dtype)
        image_diagonal = torch.sqrt(
            bank["image_sizes"][:, 0].float().square()
            + bank["image_sizes"][:, 1].float().square()
        ).clamp(min=1.0)
        distance = distance / image_diagonal[:, None, None].to(distance.dtype)
        rank = iou + (iou <= 0).to(iou.dtype) * (1.0e-3 / (1.0 + distance))
        rank = rank.masked_fill(~bank["valid"][:, None], -torch.inf)
        indices = torch.topk(rank, k=actual, dim=-1, sorted=True).indices
        candidate_valid = _gather_candidates(bank["valid"][..., None], indices).squeeze(-1)
        if actual < requested:
            padding = requested - actual
            pad_index = indices[..., :1].expand(batch, slots, padding)
            indices = torch.cat((indices, pad_index), dim=-1)
            candidate_valid = torch.cat(
                (
                    candidate_valid,
                    torch.zeros(
                        (batch, slots, padding),
                        dtype=torch.bool,
                        device=boxes.device,
                    ),
                ),
                dim=-1,
            )
        return indices, candidate_valid

    def _slot_levels(self, current_abs: torch.Tensor) -> torch.Tensor:
        width_height = (current_abs[..., 2:] - current_abs[..., :2]).clamp(min=1.0e-6)
        scale = torch.sqrt(width_height[..., 0] * width_height[..., 1])
        levels = torch.zeros_like(scale, dtype=torch.long)
        for level_index, (minimum, maximum) in enumerate(self.config.level_scale_ranges):
            mask = (scale >= float(minimum)) & (scale < float(maximum))
            levels[mask] = level_index
        return levels

    def _candidate_features(
        self,
        boxes: torch.Tensor,
        current_abs: torch.Tensor,
        bank: Dict[str, torch.Tensor],
        indices: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        candidate_boxes = _gather_candidates(bank["boxes"].float(), indices)
        descriptor = _gather_candidates(bank["desc"], indices)
        objectness = _gather_candidates(bank["obj"], indices)
        quality = _gather_candidates(bank["qual"], indices)
        level = _gather_candidates(bank["level"][..., None], indices).squeeze(-1)
        safe_level = level.clamp(min=0, max=len(self.config.fpn_levels) - 1)
        level_feature = self.level_embedding(safe_level)

        iou = _aligned_candidate_iou(current_abs, candidate_boxes)
        current_center = 0.5 * (current_abs[..., :2] + current_abs[..., 2:])
        candidate_center = 0.5 * (candidate_boxes[..., :2] + candidate_boxes[..., 2:])
        current_wh = (current_abs[..., 2:] - current_abs[..., :2]).clamp(min=1.0e-6)
        candidate_wh = (candidate_boxes[..., 2:] - candidate_boxes[..., :2]).clamp(min=1.0e-6)
        normalizer = torch.sqrt(current_wh[..., 0] * current_wh[..., 1]).clamp(min=1.0e-6)
        delta_center = (
            candidate_center - current_center[:, :, None]
        ) / normalizer[:, :, None, None]
        log_width = torch.log(candidate_wh[..., 0] / current_wh[:, :, None, 0])
        log_height = torch.log(candidate_wh[..., 1] / current_wh[:, :, None, 1])
        slot_level = self._slot_levels(current_abs)
        same_level = (level == slot_level[:, :, None]).to(torch.float32)
        relative_geometry = torch.cat(
            (
                iou[..., None],
                delta_center,
                log_width[..., None],
                log_height[..., None],
                same_level[..., None],
            ),
            dim=-1,
        )
        feature_dtype = descriptor.dtype
        raw_feature = torch.cat(
            (
                descriptor,
                objectness,
                quality,
                level_feature.to(dtype=feature_dtype),
                relative_geometry.to(dtype=feature_dtype),
            ),
            dim=-1,
        )
        raw_feature = raw_feature * candidate_valid[..., None].to(raw_feature.dtype)
        normalized_feature = self.candidate_norm(raw_feature)
        normalized_feature = normalized_feature * candidate_valid[..., None].to(normalized_feature.dtype)
        return normalized_feature, {
            "raw": raw_feature,
            "boxes": candidate_boxes,
            "desc": descriptor,
            "obj": objectness,
            "qual": quality,
            "level": level,
            "relative_geometry": relative_geometry,
        }

    def forward(
        self,
        c: torch.Tensor,
        s: torch.Tensor,
        boxes: torch.Tensor,
        bank: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del s  # semantic content is intentionally not a free geometry regressor.
        with torch.no_grad():
            candidate_indices, candidate_valid = self.select_neighborhood(boxes, bank)
        geometry_boxes = boxes.float()
        current_abs = normalized_cxcywh_to_absolute_xyxy(
            geometry_boxes,
            bank["image_sizes"],
        )
        candidate_feature, parts = self._candidate_features(
            geometry_boxes,
            current_abs,
            bank,
            candidate_indices,
            candidate_valid,
        )

        query = self.mlp_q(c)
        key = self.mlp_k(candidate_feature)
        proposal_logits = (
            (query[:, :, None] * key).sum(dim=-1) / math.sqrt(float(self.config.d_model))
            + self.mlp_geo(candidate_feature).squeeze(-1)
        )
        proposal_logits = proposal_logits.masked_fill(~candidate_valid, -1.0e4)
        dustbin_logit = self.dustbin(c)
        ownership_logits = torch.cat((proposal_logits, dustbin_logit), dim=-1)
        ownership = torch.softmax(ownership_logits, dim=-1)
        proposal_weights = (
            ownership[..., :-1].float()
            * candidate_valid.to(dtype=torch.float32)
        )
        proposal_mass = proposal_weights.sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)
        normalized_proposal_weight = proposal_weights / proposal_mass.clamp(min=1.0e-8)

        evidence_weights = (
            proposal_weights
            if self.config.readout_dustbin_gating
            else normalized_proposal_weight
        )
        evidence = (
            evidence_weights[..., None] * parts["desc"].float()
        ).sum(dim=2)
        c = c + self.desc_update(evidence)

        # Side selection uses the conditional distribution over proposal evidence,
        # while proposal_mass=(1-dustbin) controls whether geometry is accepted at all.
        # Without the explicit mass gate, adding log(w_p) before a softmax cancels
        # the dustbin scale and an abstaining slot still receives a full proposal box.
        if self.config.readout_mode == "whole_box":
            aggregated_abs = (
                normalized_proposal_weight.float()[..., None]
                * parts["boxes"].float()
            ).sum(dim=2)
        else:
            side_logits = self.mlp_side(candidate_feature)
            side_logits = side_logits + torch.log(
                normalized_proposal_weight.clamp(min=1.0e-8)
            )[..., None]
            side_logits = side_logits.masked_fill(
                ~candidate_valid[..., None],
                -1.0e4,
            )
            side_weights = torch.softmax(side_logits, dim=2)
            side_weights = side_weights * candidate_valid[..., None].to(
                side_weights.dtype
            )
            side_weights = side_weights / side_weights.sum(
                dim=2,
                keepdim=True,
            ).clamp(min=1.0e-8)
            raw_sides = (
                side_weights.float() * parts["boxes"].float()
            ).sum(dim=2)
            # `raw_sides` already contains [x1,y1,x2,y2], one independently
            # aggregated coordinate for each side.
            center = 0.5 * (raw_sides[..., :2] + raw_sides[..., 2:])
            half_size = 0.5 * (
                raw_sides[..., 2:] - raw_sides[..., :2]
            ).abs().clamp(min=1.0e-4)
            aggregated_abs = torch.cat(
                (center - half_size, center + half_size),
                dim=-1,
            )
        has_candidate = candidate_valid.any(dim=-1)
        learned_proposal_mass = (
            proposal_mass
            * has_candidate[..., None].to(dtype=torch.float32)
        )
        geometry_gate = (
            learned_proposal_mass
            if self.config.readout_dustbin_gating
            else has_candidate[..., None].to(dtype=torch.float32)
        )
        b_hat_abs = (
            geometry_gate * aggregated_abs
            + (1.0 - geometry_gate) * current_abs
        )

        b_hat = absolute_xyxy_to_normalized_cxcywh(b_hat_abs, bank["image_sizes"])
        pooled_complete_feature = (
            normalized_proposal_weight.float()[..., None] * parts["raw"].float()
        ).sum(dim=2)
        residual_input = torch.cat(
            (c, evidence, pooled_complete_feature, self.box_pe(b_hat)),
            dim=-1,
        )
        if self.config.readout_bounded_residual:
            raw_residual = self.mlp_res(residual_input)
            raw_residual = raw_residual.float() * geometry_gate
            updated_boxes = apply_bounded_residual(b_hat, raw_residual)
        else:
            updated_boxes = b_hat
        return {
            "b": updated_boxes,
            "c": c,
            "ownership_logits": ownership_logits,
            "w": ownership,
            "cand_idx": candidate_indices,
            "cand_valid": candidate_valid,
            "proposal_mass": learned_proposal_mass.squeeze(-1),
            "geometry_gate": geometry_gate.squeeze(-1),
            "b_hat": b_hat_abs,
            "b_hat_norm": b_hat,
        }
