from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch


OPTIMIZER_GROUP_NAMES = (
    "backbone_stem",
    "backbone",
    "fpn",
    "geometry_tower",
    "geometry_obj_head",
    "geometry_box_head",
    "geometry_quality_head",
    "semantic_tower",
    "slot_modules",
    "readout",
    "prediction_heads",
    "hdn",
)


def append_jsonl(path: Optional[Path | str], record: Mapping[str, object]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def optimizer_lr_summary(
    optimizer: torch.optim.Optimizer,
    *,
    require_names: bool = False,
) -> Dict[str, float]:
    """Return reader-facing LRs without assuming param_groups[0] is representative."""

    values: Dict[str, list[float]] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if not name:
            if require_names:
                raise ValueError(
                    "every optimizer parameter group must have a non-empty name"
                )
            fallback = float(group["lr"])
            return {
                "lr_stem": fallback,
                "lr_backbone": fallback,
                "lr_fpn": fallback,
                "lr_geometry": fallback,
                "lr_slot": fallback,
                "lr_heads": fallback,
            }
        values.setdefault(name, []).append(float(group["lr"]))
    for name, group_values in values.items():
        if max(group_values) - min(group_values) > 1.0e-15:
            raise ValueError(f"optimizer subgroups for {name!r} have inconsistent LRs")

    def read(name: str) -> float:
        return values.get(name, [0.0])[0]

    geometry_values = [
        read("geometry_tower"),
        read("geometry_obj_head"),
        read("geometry_box_head"),
        read("geometry_quality_head"),
    ]
    nonzero_geometry = [value for value in geometry_values if value > 0.0]
    if nonzero_geometry and max(nonzero_geometry) - min(nonzero_geometry) > 1.0e-15:
        raise ValueError("geometry optimizer groups must share one diagnostic LR")
    return {
        "lr_stem": read("backbone_stem"),
        "lr_backbone": read("backbone"),
        "lr_fpn": read("fpn"),
        "lr_geometry": nonzero_geometry[0] if nonzero_geometry else 0.0,
        "lr_slot": read("slot_modules"),
        "lr_heads": read("prediction_heads"),
    }


def _sum_of_squares(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for value in tensors:
        total += float(value.detach().float().square().sum().cpu())
    return total


def _audit_name(logical_name: str) -> Optional[str]:
    if logical_name in {"backbone_stem", "backbone"}:
        return "backbone"
    if logical_name == "fpn":
        return "fpn"
    if logical_name == "geometry_tower":
        return "geometry_tower"
    if logical_name == "geometry_obj_head":
        return "geometry_obj_head"
    if logical_name == "geometry_box_head":
        return "geometry_box_head"
    if logical_name == "geometry_quality_head":
        return "geometry_quality_head"
    if logical_name == "semantic_tower":
        return "semantic_tower"
    if logical_name == "slot_modules":
        return "slot_updater"
    if logical_name == "readout":
        return "readout"
    if logical_name == "prediction_heads":
        return "class_exist_heads"
    return None


class ParameterUpdateAuditor:
    """Measure gradients and real parameter deltas at sparse optimizer steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        jsonl_path: Optional[Path | str] = None,
    ) -> None:
        grouped: Dict[str, list[torch.nn.Parameter]] = {}
        seen: set[int] = set()
        for group in optimizer.param_groups:
            audit_name = _audit_name(str(group.get("name", "")))
            if audit_name is None:
                continue
            for parameter in group["params"]:
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                grouped.setdefault(audit_name, []).append(parameter)
        self.groups = grouped
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self._snapshot: Dict[int, torch.Tensor] = {}
        self._before: Dict[str, Dict[str, float]] = {}
        self._zero_audits: Dict[str, int] = {}

    def capture_before_step(self) -> None:
        self._snapshot = {}
        self._before = {}
        for name, parameters in self.groups.items():
            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            gradients = [
                parameter.grad
                for parameter in trainable
                if parameter.grad is not None
            ]
            self._before[name] = {
                "grad_norm": math.sqrt(_sum_of_squares(gradients)),
                "parameter_norm": math.sqrt(_sum_of_squares(trainable)),
            }
            for parameter in trainable:
                # CPU snapshots avoid perturbing the CUDA peak-memory measurement.
                self._snapshot[id(parameter)] = parameter.detach().float().cpu().clone()

    def capture_after_step(
        self,
        *,
        optimizer_step: int,
        phase: str,
    ) -> Dict[str, object]:
        modules: Dict[str, Dict[str, float]] = {}
        alerts: list[str] = []
        expected_active = {
            "backbone",
            "fpn",
            "geometry_tower",
            "geometry_obj_head",
            "geometry_box_head",
            "geometry_quality_head",
        }
        if phase == "full":
            expected_active.update(
                {
                    "semantic_tower",
                    "slot_updater",
                    "readout",
                    "class_exist_heads",
                }
            )
        for name, parameters in self.groups.items():
            update_squared = 0.0
            for parameter in parameters:
                before = self._snapshot.get(id(parameter))
                if before is None:
                    continue
                after = parameter.detach().float().cpu()
                update_squared += float((after - before).square().sum())
            update_norm = math.sqrt(update_squared)
            before_metrics = self._before.get(
                name,
                {"grad_norm": 0.0, "parameter_norm": 0.0},
            )
            parameter_norm = float(before_metrics["parameter_norm"])
            modules[name] = {
                "grad_norm": float(before_metrics["grad_norm"]),
                "parameter_norm": parameter_norm,
                "update_norm": update_norm,
                "relative_update_norm": update_norm / max(parameter_norm, 1.0e-12),
            }
            if name in expected_active:
                if update_norm == 0.0:
                    self._zero_audits[name] = self._zero_audits.get(name, 0) + 1
                else:
                    self._zero_audits[name] = 0
                if self._zero_audits[name] >= 2:
                    alerts.append(
                        f"{name} has zero parameter update for at least 50 optimizer steps"
                    )
        record: Dict[str, object] = {
            "kind": "parameter_update_audit",
            "optimizer_step": int(optimizer_step),
            "phase": str(phase),
            "modules": modules,
            "alerts": alerts,
        }
        append_jsonl(self.jsonl_path, record)
        self._snapshot = {}
        self._before = {}
        return record


def _geometry_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        if not str(group.get("name", "")).startswith("geometry_"):
            continue
        for parameter in group["params"]:
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


def _autograd_values(
    loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
) -> tuple[Optional[torch.Tensor], ...]:
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )


def measure_geometry_gradient_competition(
    partitions: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_step: int,
    stage_optimizer_step: Optional[int] = None,
    phase: str = "full",
    jsonl_path: Optional[Path | str] = None,
) -> Dict[str, float | int | str]:
    """Measure loss-source gradients without changing the training gradients."""

    parameters = _geometry_parameters(optimizer)
    dense = _autograd_values(partitions["dense"], parameters)
    slot = _autograd_values(partitions["slot"], parameters)
    ownership = _autograd_values(partitions["ownership"], parameters)

    def norm(values: Sequence[Optional[torch.Tensor]]) -> float:
        return math.sqrt(
            _sum_of_squares(value for value in values if value is not None)
        )

    dense_norm = norm(dense)
    slot_norm = norm(slot)
    ownership_norm = norm(ownership)
    dot = 0.0
    for dense_value, slot_value in zip(dense, slot):
        if dense_value is not None and slot_value is not None:
            dot += float(
                (dense_value.detach().float() * slot_value.detach().float())
                .sum()
                .cpu()
            )
    denominator = dense_norm * slot_norm
    record: Dict[str, float | int | str] = {
        "kind": "geometry_gradient_competition",
        "optimizer_step": int(optimizer_step),
        "stage_optimizer_step": int(
            optimizer_step
            if stage_optimizer_step is None
            else stage_optimizer_step
        ),
        "phase": str(phase),
        "geometry_grad_from_dense": dense_norm,
        "geometry_grad_from_slot": slot_norm,
        "geometry_grad_from_ownership": ownership_norm,
        "slot_to_dense_grad_ratio": slot_norm / max(dense_norm, 1.0e-12),
        "ownership_to_dense_grad_ratio": ownership_norm / max(dense_norm, 1.0e-12),
        "cosine_dense_vs_slot": dot / denominator if denominator > 0.0 else 0.0,
    }
    append_jsonl(jsonl_path, record)
    return record


def backward_with_slot_geometry_scale(
    partitions: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    divisor: float,
    scale: float,
    scaler=None,
) -> None:
    """Scale only slot/readout gradients entering geometry, preserving all other paths."""

    geometry_parameters = _geometry_parameters(optimizer)

    def backward(value: torch.Tensor, *, retain_graph: bool) -> None:
        scaled = value.float() / float(divisor)
        if scaler is not None:
            scaler.scale(scaled).backward(retain_graph=retain_graph)
        else:
            scaled.backward(retain_graph=retain_graph)

    backward(partitions["dense"], retain_graph=True)
    backward(partitions["ownership"], retain_graph=True)
    hooks = [
        parameter.register_hook(lambda gradient: gradient * float(scale))
        for parameter in geometry_parameters
    ]
    try:
        backward(partitions["slot"], retain_graph=False)
    finally:
        for hook in hooks:
            hook.remove()
