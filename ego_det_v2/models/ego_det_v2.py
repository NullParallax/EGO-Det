from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ego_det_v2.paths import EGOJointPaths

from ..configs.config import EgoDetV2Config
from .backbone import FakeBackbone, OverLoCKFPNBackbone
from .box_ops import (
    box_cxcywh_to_xyxy,
    normalized_cxcywh_to_absolute_xyxy,
)
from .dense_geometry_branch import DenseGeometryBranch
from .evidence_readout import EvidenceReadoutRound
from .hdn import HypothesisDenoising
from .heads import (
    DirectSlotBoxHead,
    ExistenceHead,
    SlotPredictionHead,
    compose_scores,
)
from .semantic_support import SemanticSupport
from .slot_seeding import SlotSeeder
from .slot_update import SlotUpdateRound


def _profile_stage(profiler, name: str):
    return (
        profiler.stage(name)
        if profiler is not None
        else contextlib.nullcontext()
    )


class EgoDetV2(nn.Module):
    """End-to-end detector whose geometry is read exclusively from its bank."""

    def __init__(
        self,
        config: Optional[EgoDetV2Config] = None,
        *,
        paths: Optional[EGOJointPaths] = None,
        fake_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or EgoDetV2Config()
        self.paths = paths or EGOJointPaths()
        self.fake_backbone = bool(fake_backbone)
        self.backbone = (
            FakeBackbone(self.config)
            if self.fake_backbone
            else OverLoCKFPNBackbone(self.config, self.paths)
        )
        self.dense_geometry = DenseGeometryBranch(self.config)
        self.semantic_support = SemanticSupport(self.config)
        self.slot_seeding = SlotSeeder(self.config)
        self.hdn = HypothesisDenoising(self.config)
        self.slot_updates = nn.ModuleList(
            [SlotUpdateRound(self.config) for _ in range(self.config.slot_rounds)]
        )
        if self.config.readout_mode == "direct_regression":
            self.readouts = nn.ModuleList()
            self.direct_box_heads = nn.ModuleList(
                [
                    DirectSlotBoxHead(self.config)
                    for _ in range(self.config.slot_rounds)
                ]
            )
        else:
            self.readouts = nn.ModuleList(
                [
                    EvidenceReadoutRound(self.config)
                    for _ in range(self.config.slot_rounds)
                ]
            )
            self.direct_box_heads = nn.ModuleList()
        self.round_heads = nn.ModuleList(
            [SlotPredictionHead(self.config) for _ in range(self.config.slot_rounds)]
        )
        self.existence_head = ExistenceHead(self.config)
        if self.config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            frozen = trainable = 0
            for parameter in self.parameters():
                if parameter.requires_grad:
                    trainable += parameter.numel()
                else:
                    frozen += parameter.numel()
            print(
                "[EgoDetV2 backbone] freeze_backbone=True "
                f"frozen_numel={frozen} trainable_numel={trainable} "
                f"frozen_fraction={frozen / max(frozen + trainable, 1):.4f}",
                flush=True,
            )

    def train(self, mode: bool = True) -> "EgoDetV2":
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _image_sizes(
        pad_mask: torch.Tensor,
        targets: Optional[Sequence[Dict]],
    ) -> torch.Tensor:
        if targets is not None:
            sizes = [target.get("size") for target in targets]
            if all(size is not None for size in sizes):
                return torch.tensor(sizes, dtype=torch.long, device=pad_mask.device)
        valid = ~pad_mask
        heights = valid.any(dim=2).sum(dim=1)
        widths = valid.any(dim=1).sum(dim=1)
        return torch.stack((heights, widths), dim=-1)

    def backbone_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        return self.backbone.param_groups()

    def _forward_backbone(
        self,
        images: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        context = (
            torch.no_grad()
            if self.config.freeze_backbone
            else contextlib.nullcontext()
        )
        with context:
            return self.backbone(images, pad_mask)

    def _direct_readout_placeholder(
        self,
        boxes: torch.Tensor,
        bank: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Keep the output schema stable when evidence readout is ablated."""

        batch, slots = boxes.shape[:2]
        candidates = int(self.config.evidence_per_slot)
        candidate_indices = torch.zeros(
            (batch, slots, candidates),
            dtype=torch.long,
            device=boxes.device,
        )
        candidate_valid = torch.zeros(
            (batch, slots, candidates),
            dtype=torch.bool,
            device=boxes.device,
        )
        ownership_logits = boxes.new_full(
            (batch, slots, candidates + 1),
            -1.0e4,
        )
        ownership_logits[..., -1] = 0
        ownership = torch.zeros_like(ownership_logits)
        ownership[..., -1] = 1
        zeros = boxes.new_zeros((batch, slots))
        absolute = normalized_cxcywh_to_absolute_xyxy(
            boxes.float(),
            bank["image_sizes"],
        )
        return {
            "b": boxes,
            "ownership_logits": ownership_logits,
            "w": ownership,
            "cand_idx": candidate_indices,
            "cand_valid": candidate_valid,
            "proposal_mass": zeros,
            "geometry_gate": zeros,
            "b_hat": absolute,
            "b_hat_norm": boxes,
        }

    def forward_geometry(
        self,
        images: torch.Tensor,
        pad_mask: torch.Tensor,
        targets: Optional[Sequence[Dict]] = None,
        *,
        profiler=None,
    ) -> Dict[str, object]:
        """Geometry-only diagnostic path used by overfit8-A.

        It uses the same trainable backbone, FPN, dense tower, assignment
        correction, and bank selector as the formal forward; no alternate
        model or checkpoint is introduced.
        """

        image_sizes = self._image_sizes(pad_mask, targets)
        with _profile_stage(profiler, "backbone_fpn"):
            features, feature_masks = self._forward_backbone(images, pad_mask)
        with _profile_stage(profiler, "dense_geometry"):
            dense, bank = self.dense_geometry(
                features,
                feature_masks,
                image_sizes,
                targets,
                profiler=profiler,
            )
        return {"dense": dense, "bank": bank}

    def forward(
        self,
        images: torch.Tensor,
        pad_mask: torch.Tensor,
        targets: Optional[Sequence[Dict]] = None,
        *,
        epoch: int = 0,
        profiler=None,
    ) -> Dict[str, object]:
        del epoch  # loss schedules belong to the criterion, not model routing.
        image_sizes = self._image_sizes(pad_mask, targets)
        with _profile_stage(profiler, "backbone_fpn"):
            features, feature_masks = self._forward_backbone(images, pad_mask)
        with _profile_stage(profiler, "dense_geometry"):
            dense, bank = self.dense_geometry(
                features,
                feature_masks,
                image_sizes,
                targets,
                profiler=profiler,
            )
        with _profile_stage(profiler, "semantic_support"):
            semantic_state = self.semantic_support(
                features,
                feature_masks,
                image_sizes,
            )
        with _profile_stage(profiler, "slot_seeding"):
            boxes, s, c, seed_indices, seed_valid = self.slot_seeding(
                bank,
                self.semantic_support,
                semantic_state,
            )
        matching_count = int(self.config.num_slots)

        dn_meta = None
        if self.training and targets is not None:
            with _profile_stage(profiler, "hdn_build"):
                dn_meta = self.hdn(
                    targets,
                    matching_count=matching_count,
                    device=images.device,
                    dtype=c.dtype,
                )
        if dn_meta is not None:
            boxes = torch.cat((boxes, dn_meta["boxes"]), dim=1)
            s = torch.cat((s, dn_meta["content"]), dim=1)
            c = torch.cat((c, dn_meta["content"]), dim=1)
            slot_padding_mask = torch.cat(
                (
                    torch.zeros(
                        (images.shape[0], matching_count),
                        dtype=torch.bool,
                        device=images.device,
                    ),
                    ~dn_meta["valid"],
                ),
                dim=1,
            )
            attention_mask = dn_meta["attention_mask"]
        else:
            slot_padding_mask = None
            attention_mask = None

        all_round_predictions: List[Dict[str, torch.Tensor]] = []
        ownership_outputs: List[Dict[str, torch.Tensor]] = []
        readout_outputs: List[Dict[str, torch.Tensor]] = []
        for round_index, (update, head) in enumerate(
            zip(self.slot_updates, self.round_heads),
            start=1,
        ):
            with _profile_stage(
                profiler,
                f"slot_round_{round_index}_update",
            ):
                s, c = update(
                    s,
                    c,
                    boxes,
                    self.semantic_support,
                    semantic_state,
                    self_attention_mask=attention_mask,
                    slot_padding_mask=slot_padding_mask,
                )
            if self.config.readout_mode == "direct_regression":
                with _profile_stage(
                    profiler,
                    f"slot_round_{round_index}_direct_box",
                ):
                    boxes = self.direct_box_heads[round_index - 1](
                        s,
                        c,
                        boxes,
                    )
                    readout_result = self._direct_readout_placeholder(
                        boxes,
                        bank,
                    )
            else:
                with _profile_stage(
                    profiler,
                    f"slot_round_{round_index}_readout",
                ):
                    readout_result = self.readouts[round_index - 1](
                        c,
                        s,
                        boxes,
                        bank,
                    )
                    boxes = readout_result["b"]
                    c = readout_result["c"]
            with _profile_stage(
                profiler,
                f"slot_round_{round_index}_heads",
            ):
                class_logits, quality_logits = head(s, c)
            all_round_predictions.append(
                {
                    "pred_logits": class_logits,
                    "pred_boxes": boxes,
                    "pred_quality": quality_logits,
                }
            )
            ownership_outputs.append(
                {
                    "logits": readout_result["ownership_logits"][:, :matching_count],
                    "w": readout_result["w"][:, :matching_count],
                    "cand_idx": readout_result["cand_idx"][:, :matching_count],
                    "cand_valid": readout_result["cand_valid"][:, :matching_count],
                    "proposal_mass": readout_result["proposal_mass"][:, :matching_count],
                    "geometry_gate": readout_result["geometry_gate"][:, :matching_count],
                }
            )
            readout_outputs.append(
                {
                    "b_hat": readout_result["b_hat"][:, :matching_count],
                    "b_hat_norm": readout_result["b_hat_norm"][:, :matching_count],
                    "b": boxes[:, :matching_count],
                }
            )

        with _profile_stage(profiler, "existence_and_score"):
            final_content = c[:, :matching_count]
            final_boxes = boxes[:, :matching_count]
            final_exist, gamma = self.existence_head(
                final_content,
                final_boxes,
            )
            final_round = all_round_predictions[-1]
            final_logits = final_round["pred_logits"][:, :matching_count]
            final_quality = final_round["pred_quality"][:, :matching_count]
            final_scores = compose_scores(
                final_logits,
                final_quality,
                final_exist,
                gamma,
            )

        auxiliary = [
            {
                "pred_logits": prediction["pred_logits"][:, :matching_count],
                "pred_boxes": prediction["pred_boxes"][:, :matching_count],
                "pred_quality": prediction["pred_quality"][:, :matching_count],
            }
            for prediction in all_round_predictions[:-1]
        ]
        output: Dict[str, object] = {
            "pred_logits": final_logits,
            "pred_boxes": final_boxes,
            "pred_quality": final_quality,
            "pred_exist": final_exist,
            "pred_gamma": gamma,
            "pred_scores": final_scores,
            "aux": auxiliary,
            "dense": dense,
            "bank": bank,
            "ownership": ownership_outputs,
            "readout": readout_outputs,
            "seed": {
                "indices": seed_indices,
                "valid": seed_valid,
            },
            "dn": None,
        }

        if dn_meta is not None:
            dn_outputs: List[Dict[str, torch.Tensor]] = []
            for prediction in all_round_predictions:
                dn_outputs.append(
                    {
                        "pred_logits": prediction["pred_logits"][:, matching_count:],
                        "pred_boxes": prediction["pred_boxes"][:, matching_count:],
                        "pred_quality": prediction["pred_quality"][:, matching_count:],
                    }
                )
            dn_exist, _ = self.existence_head(
                c[:, matching_count:],
                boxes[:, matching_count:],
            )
            dn_outputs[-1]["pred_exist"] = dn_exist
            output["dn"] = {
                "outputs": dn_outputs,
                "meta": {
                    key: dn_meta[key]
                    for key in ("valid", "source_gt", "positive", "group")
                },
            }
        return output

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        pad_mask: torch.Tensor,
        *,
        original_sizes: Optional[Sequence[Tuple[int, int]]] = None,
        topk: int = 300,
    ) -> List[Dict[str, torch.Tensor]]:
        was_training = self.training
        self.eval()
        output = self.forward(images, pad_mask, targets=None)
        if was_training:
            self.train()
        scores = output["pred_scores"]
        boxes = output["pred_boxes"]
        batch, slots, classes = scores.shape
        if original_sizes is None:
            sizes_tensor = self._image_sizes(pad_mask, None)
            original_sizes = [
                (int(size[0]), int(size[1])) for size in sizes_tensor
            ]
        results: List[Dict[str, torch.Tensor]] = []
        for batch_index in range(batch):
            flattened = scores[batch_index].flatten()
            count = min(int(topk), int(flattened.numel()))
            values, flat_indices = torch.topk(flattened, k=count)
            slot_indices = torch.div(flat_indices, classes, rounding_mode="floor")
            labels = flat_indices % classes
            selected_boxes = box_cxcywh_to_xyxy(boxes[batch_index, slot_indices])
            height, width = original_sizes[batch_index]
            scale = selected_boxes.new_tensor([width, height, width, height])
            selected_boxes = selected_boxes * scale
            result = {
                "scores": values,
                "labels": labels,
                "boxes": selected_boxes,
                "slot_indices": slot_indices,
            }
            if self.config.eval_nms_diagnostic:
                from torchvision.ops import nms

                keep = nms(selected_boxes, values, iou_threshold=0.6)
                result["diagnostic_nms_keep"] = keep
            results.append(result)
        return results
