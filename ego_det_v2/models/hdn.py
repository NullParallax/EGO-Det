from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import clip_normalized_cxcywh
from .slot_update import BoxPositionalEncoding


class HypothesisDenoising(nn.Module):
    """Training-only noisy hypotheses and explicit block attention masks."""

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        channels = int(config.d_model)
        self.label_embedding = nn.Embedding(config.num_classes + 1, channels)
        self.box_pe = BoxPositionalEncoding(channels)
        self.box_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    def _one_image(
        self,
        target: Dict,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        del dtype
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        labels = target["labels"].to(device=device)
        gt_count = int(boxes.shape[0])
        if gt_count == 0:
            return {
                "boxes": boxes.new_zeros((0, 4)),
                "labels": labels.new_zeros((0,)),
                "source_gt": labels.new_full((0,), -1),
                "positive": torch.zeros(0, dtype=torch.bool, device=device),
                "group": labels.new_full((0,), -1),
            }

        groups = max(1, min(10, int(self.config.hdn_max_pairs) // max(1, gt_count)))
        all_boxes: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_sources: List[torch.Tensor] = []
        all_positive: List[torch.Tensor] = []
        all_groups: List[torch.Tensor] = []
        source = torch.arange(gt_count, device=device, dtype=torch.long)
        scale = torch.stack((boxes[:, 2], boxes[:, 3], boxes[:, 2], boxes[:, 3]), dim=-1)
        for group_index in range(groups):
            positive_noise = (
                torch.rand_like(boxes) * 2.0 - 1.0
            ) * float(self.config.hdn_pos_box_noise) * scale
            positive_boxes = clip_normalized_cxcywh(boxes + positive_noise)

            lower, upper = self.config.hdn_neg_box_noise
            magnitude = lower + (upper - lower) * torch.rand_like(boxes)
            sign = torch.where(torch.rand_like(boxes) < 0.5, -1.0, 1.0)
            negative_boxes = clip_normalized_cxcywh(boxes + magnitude * sign * scale)

            positive_labels = labels.clone()
            noisy = torch.rand((gt_count,), device=device) < float(self.config.hdn_label_noise)
            if noisy.any():
                alternatives = torch.randint(
                    0,
                    int(self.config.num_classes) - 1,
                    (int(noisy.sum()),),
                    device=device,
                )
                original = positive_labels[noisy]
                alternatives = alternatives + (alternatives >= original).to(alternatives.dtype)
                positive_labels[noisy] = alternatives
            negative_labels = labels.clone()

            all_boxes.extend((positive_boxes, negative_boxes))
            all_labels.extend((positive_labels, negative_labels))
            all_sources.extend((source, source))
            all_positive.extend(
                (
                    torch.ones(gt_count, dtype=torch.bool, device=device),
                    torch.zeros(gt_count, dtype=torch.bool, device=device),
                )
            )
            all_groups.extend(
                (
                    torch.full((gt_count,), group_index, dtype=torch.long, device=device),
                    torch.full((gt_count,), group_index, dtype=torch.long, device=device),
                )
            )
        return {
            "boxes": torch.cat(all_boxes),
            "labels": torch.cat(all_labels),
            "source_gt": torch.cat(all_sources),
            "positive": torch.cat(all_positive),
            "group": torch.cat(all_groups),
        }

    @staticmethod
    def build_attention_mask(
        matching_count: int,
        group_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """True means blocked, with per-image matching/DN group structure."""

        batch, dn_count = group_ids.shape
        total = int(matching_count) + int(dn_count)
        mask = torch.zeros((batch, total, total), dtype=torch.bool, device=group_ids.device)
        if dn_count == 0:
            return mask
        # Matching queries cannot inspect any denoising key.
        mask[:, :matching_count, matching_count:] = True
        for batch_index in range(batch):
            groups = group_ids[batch_index]
            is_valid = valid[batch_index]
            for row in range(dn_count):
                absolute_row = matching_count + row
                if not bool(is_valid[row]):
                    mask[batch_index, absolute_row, matching_count:] = True
                    continue
                allowed = is_valid & (groups == groups[row])
                mask[batch_index, absolute_row, matching_count:] = ~allowed
            # Invalid DN columns are never visible.
            invalid_columns = torch.nonzero(~is_valid, as_tuple=False).flatten()
            if invalid_columns.numel() > 0:
                mask[
                    batch_index,
                    :,
                    matching_count + invalid_columns,
                ] = True
        return mask

    def forward(
        self,
        targets: Optional[Sequence[Dict]],
        *,
        matching_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if targets is None:
            return None
        items = [
            self._one_image(target, device=device, dtype=dtype)
            for target in targets
        ]
        maximum = max((int(item["boxes"].shape[0]) for item in items), default=0)
        batch = len(items)
        if maximum == 0:
            return None
        boxes = torch.zeros(
            (batch, maximum, 4),
            device=device,
            dtype=torch.float32,
        )
        labels = torch.full(
            (batch, maximum),
            int(self.config.num_classes),
            device=device,
            dtype=torch.long,
        )
        source_gt = torch.full((batch, maximum), -1, device=device, dtype=torch.long)
        positive = torch.zeros((batch, maximum), device=device, dtype=torch.bool)
        group = torch.full((batch, maximum), -1, device=device, dtype=torch.long)
        valid = torch.zeros((batch, maximum), device=device, dtype=torch.bool)
        for batch_index, item in enumerate(items):
            count = int(item["boxes"].shape[0])
            if count == 0:
                continue
            boxes[batch_index, :count] = item["boxes"]
            labels[batch_index, :count] = item["labels"]
            source_gt[batch_index, :count] = item["source_gt"]
            positive[batch_index, :count] = item["positive"]
            group[batch_index, :count] = item["group"]
            valid[batch_index, :count] = True

        content = self.label_embedding(labels) + self.box_mlp(self.box_pe(boxes))
        content = content.to(dtype=dtype)
        content = content * valid[..., None].to(content.dtype)
        attention_mask = self.build_attention_mask(matching_count, group, valid)
        return {
            "boxes": boxes,
            "content": content,
            "valid": valid,
            "source_gt": source_gt,
            "positive": positive,
            "group": group,
            "attention_mask": attention_mask,
        }
