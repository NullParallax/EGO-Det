from __future__ import annotations

import contextlib
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from ..models.matcher import (
    HungarianMatcherV2,
    prepare_targets_for_matching_and_losses,
)
from .dense_losses import dense_geometry_losses
from .ownership_losses import soft_ownership_losses
from .slot_losses import (
    denoising_round_losses,
    duplicate_margin_loss,
    existence_loss,
    flatten_matches,
    readout_gain_loss,
    slot_round_losses,
)


def _profile_stage(profiler, name: str):
    return (
        profiler.stage(name)
        if profiler is not None
        else contextlib.nullcontext()
    )


class EgoDetV2Criterion(nn.Module):
    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        self.matcher = HungarianMatcherV2(config)

    def _weight_for(self, key: str, *, ownership_scale: float, extras_enabled: bool) -> float:
        config = self.config
        if key.startswith("loss_dense_obj"):
            return config.w_dense_obj
        if key.startswith("loss_dense_giou"):
            return config.w_dense_giou
        if key.startswith("loss_dense_quality"):
            return config.w_dense_quality
        if key.startswith("loss_ownership"):
            return config.w_ownership * ownership_scale
        if key.startswith("loss_dustbin"):
            return config.w_dustbin * ownership_scale
        if key.startswith("loss_cls") or key.startswith("loss_dn_cls"):
            return config.w_cls
        if key.startswith("loss_l1") or key.startswith("loss_dn_l1"):
            return config.w_l1
        if key.startswith("loss_giou") or key.startswith("loss_dn_giou"):
            return config.w_giou
        if key.startswith("loss_quality") or key.startswith("loss_dn_quality"):
            return config.w_quality
        if key.startswith("loss_exist") or key.startswith("loss_dn_exist"):
            return config.w_exist
        if key.startswith("loss_duplicate"):
            return config.w_dup if extras_enabled else 0.0
        if key.startswith("loss_gain"):
            return config.w_gain if extras_enabled else 0.0
        return 0.0

    def weighted_loss_partitions(
        self,
        losses: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        epoch_progress: float,
    ) -> Dict[str, torch.Tensor]:
        """Partition the exact weighted total for gradient-source diagnostics."""

        ownership_scale = (
            1.0
            if int(epoch) >= int(self.config.ownership_warmup_epochs)
            else min(max(float(epoch_progress), 0.0), 1.0)
        )
        extras_enabled = int(epoch) >= int(self.config.dup_gain_start_epoch)
        reference = next(
            value
            for name, value in losses.items()
            if name.startswith("loss_") and name != "loss_total"
        )
        partitions = {
            name: reference.sum() * 0.0
            for name in ("dense", "ownership", "slot")
        }
        for name, value in losses.items():
            if not name.startswith("loss_") or name == "loss_total":
                continue
            weighted = value * self._weight_for(
                name,
                ownership_scale=ownership_scale,
                extras_enabled=extras_enabled,
            )
            if name.startswith("loss_dense"):
                partition = "dense"
            elif name.startswith(("loss_ownership", "loss_dustbin")):
                partition = "ownership"
            else:
                partition = "slot"
            partitions[partition] = partitions[partition] + weighted
        return partitions

    def forward(
        self,
        outputs: Dict[str, object],
        targets: Sequence[Dict],
        *,
        epoch: int = 0,
        epoch_progress: float = 1.0,
        profiler=None,
    ) -> Dict[str, torch.Tensor]:
        reference = outputs.get("pred_logits")
        if not isinstance(reference, torch.Tensor):
            raise TypeError("outputs['pred_logits'] must be a tensor")
        if reference.device.type == "cuda":
            loss_context = torch.autocast(device_type="cuda", enabled=False)
        else:
            loss_context = contextlib.nullcontext()
        with loss_context:
            result = self._forward_fp32(
                outputs,
                targets,
                epoch=epoch,
                epoch_progress=epoch_progress,
                profiler=profiler,
            )
        for name, value in result.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"criterion output {name!r} must be a tensor")
            if value.ndim != 0:
                raise ValueError(
                    f"criterion output {name!r} must be scalar, got {tuple(value.shape)}"
                )
            if value.dtype != torch.float32:
                result[name] = value.float()
        return result

    def _forward_fp32(
        self,
        outputs: Dict[str, object],
        targets: Sequence[Dict],
        *,
        epoch: int,
        epoch_progress: float,
        profiler=None,
    ) -> Dict[str, torch.Tensor]:
        efficiency_mode = self.config.efficiency_mode
        optimized_level = (
            int(self.config.efficiency_ablation_level)
            if efficiency_mode == "optimized"
            else 0
        )
        ownership_mode = (
            "optimized" if optimized_level >= 3 else "reference"
        )
        slot_loss_mode = (
            "optimized" if optimized_level >= 4 else "reference"
        )
        loss_targets = (
            prepare_targets_for_matching_and_losses(
                targets,
                device=outputs["pred_logits"].device,
            )
            if optimized_level >= 5
            else targets
        )
        predictions = list(outputs["aux"]) + [
            {
                "pred_logits": outputs["pred_logits"],
                "pred_boxes": outputs["pred_boxes"],
                "pred_quality": outputs["pred_quality"],
                "pred_exist": outputs["pred_exist"],
                "pred_scores": outputs["pred_scores"],
            }
        ]
        with _profile_stage(profiler, "hungarian_match_all_rounds"):
            matches = [
                self.matcher(prediction, loss_targets)
                for prediction in predictions
            ]
        with _profile_stage(profiler, "slot_losses_all_rounds"):
            flattened_matches = (
                [
                    flatten_matches(
                        round_matches,
                        loss_targets,
                        outputs["pred_logits"].device,
                    )
                    for round_matches in matches
                ]
                if optimized_level >= 4
                else [None] * len(matches)
            )
        with _profile_stage(profiler, "dense_loss"):
            losses: Dict[str, torch.Tensor] = dense_geometry_losses(
                outputs["dense"]
            )
        metric_values: Dict[str, torch.Tensor] = {}

        for round_index, (
            prediction,
            round_matches,
            round_flat_matches,
        ) in enumerate(
            zip(predictions, matches, flattened_matches),
            start=1,
        ):
            with _profile_stage(profiler, "slot_losses_all_rounds"):
                round_losses = slot_round_losses(
                    prediction,
                    loss_targets,
                    round_matches,
                    num_classes=self.config.num_classes,
                    efficiency_mode=slot_loss_mode,
                    flat_matches=round_flat_matches,
                )
            for name, value in round_losses.items():
                losses[f"{name}_r{round_index}"] = value
            with _profile_stage(
                profiler,
                "ownership_target_and_loss_all_rounds",
            ):
                if self.config.readout_mode == "direct_regression":
                    zero = prediction["pred_boxes"].float().sum() * 0.0
                    own = {
                        "loss_ownership": zero,
                        "loss_dustbin": zero,
                        "ownership_accuracy": zero.detach(),
                        "ownership_target_entropy": zero.detach(),
                        "ownership_matched_dustbin_rate": zero.detach(),
                    }
                else:
                    own = soft_ownership_losses(
                        outputs["ownership"][round_index - 1],
                        outputs["bank"],
                        round_matches,
                        loss_targets,
                        efficiency_mode=ownership_mode,
                    )
            losses[f"loss_ownership_r{round_index}"] = own["loss_ownership"]
            losses[f"loss_dustbin_r{round_index}"] = own["loss_dustbin"]
            metric_values[f"ownership_accuracy_r{round_index}"] = own["ownership_accuracy"]
            metric_values[f"ownership_target_entropy_r{round_index}"] = own[
                "ownership_target_entropy"
            ]
            metric_values[f"ownership_matched_dustbin_rate_r{round_index}"] = own[
                "ownership_matched_dustbin_rate"
            ]

        final_prediction = predictions[-1]
        with _profile_stage(profiler, "slot_losses_all_rounds"):
            losses["loss_exist"] = existence_loss(
                final_prediction,
                loss_targets,
                matches[-1],
                efficiency_mode=slot_loss_mode,
                flat_matches=flattened_matches[-1],
            )
        extras_enabled = int(epoch) >= int(self.config.dup_gain_start_epoch)
        if optimized_level >= 1 and not extras_enabled:
            zero = final_prediction["pred_boxes"].float().sum() * 0.0
            with _profile_stage(profiler, "duplicate_loss"):
                losses["loss_duplicate"] = zero
            with _profile_stage(profiler, "gain_loss"):
                losses["loss_gain"] = zero
        else:
            with _profile_stage(profiler, "duplicate_loss"):
                losses["loss_duplicate"] = duplicate_margin_loss(
                    final_prediction,
                    loss_targets,
                    matches[-1],
                    margin=self.config.dup_margin,
                    efficiency_mode=slot_loss_mode,
                    flat_matches=flattened_matches[-1],
                )
            with _profile_stage(profiler, "gain_loss"):
                if self.config.readout_mode == "direct_regression":
                    losses["loss_gain"] = (
                        final_prediction["pred_boxes"].float().sum() * 0.0
                    )
                else:
                    losses["loss_gain"] = readout_gain_loss(
                        final_prediction,
                        outputs["bank"],
                        outputs["ownership"][-1],
                        loss_targets,
                        matches[-1],
                        efficiency_mode=slot_loss_mode,
                        flat_matches=flattened_matches[-1],
                    )

        dn = outputs.get("dn")
        if dn is not None:
            for round_index, prediction in enumerate(dn["outputs"], start=1):
                with _profile_stage(profiler, "dn_losses"):
                    dn_losses = denoising_round_losses(
                        prediction,
                        dn["meta"],
                        loss_targets,
                        include_existence=round_index == len(dn["outputs"]),
                        efficiency_mode=slot_loss_mode,
                    )
                for name, value in dn_losses.items():
                    losses[f"{name}_r{round_index}"] = value

        if int(epoch) >= int(self.config.ownership_warmup_epochs):
            ownership_scale = 1.0
        else:
            ownership_scale = min(max(float(epoch_progress), 0.0), 1.0)
        with _profile_stage(profiler, "criterion_total"):
            total = next(iter(losses.values())).sum() * 0.0
            for name, value in losses.items():
                total = total + value * self._weight_for(
                    name,
                    ownership_scale=ownership_scale,
                    extras_enabled=extras_enabled,
                )
            losses["loss_total"] = total
            losses.update(metric_values)
        return losses
