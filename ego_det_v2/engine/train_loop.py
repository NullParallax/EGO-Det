from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - runtime dependency contract
    raise RuntimeError(
        "tqdm is required for EGO-Det V2 training and evaluation progress"
    ) from exc

from ..configs.config import EgoDetV2Config
from ..losses.dense_losses import dense_geometry_losses
from ..models.dense_geometry_branch import oracle_recall
from .ema import ModelEMA
from .evaluator import evaluate
from .diagnostics import (
    OPTIMIZER_GROUP_NAMES,
    ParameterUpdateAuditor,
    append_jsonl,
    backward_with_slot_geometry_scale,
    measure_geometry_gradient_competition,
    optimizer_lr_summary,
)


def set_runtime_defaults(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True


def _optimizer_logical_name(
    parameter_name: str,
    parameter_id: int,
    *,
    stem_ids: set[int],
    backbone_ids: set[int],
    fpn_ids: set[int],
) -> str:
    if parameter_id in stem_ids:
        return "backbone_stem"
    if parameter_id in backbone_ids:
        return "backbone"
    if parameter_id in fpn_ids:
        return "fpn"
    if parameter_name.startswith(
        (
            "dense_geometry.tower.",
            "dense_geometry.desc_head.",
            "dense_geometry.scales",
        )
    ):
        return "geometry_tower"
    if parameter_name.startswith("dense_geometry.obj_head."):
        return "geometry_obj_head"
    if parameter_name.startswith("dense_geometry.ltrb_head."):
        return "geometry_box_head"
    if parameter_name.startswith("dense_geometry.qual_head."):
        return "geometry_quality_head"
    if parameter_name.startswith("semantic_support."):
        return "semantic_tower"
    if parameter_name.startswith(("slot_seeding.", "slot_updates.")):
        return "slot_modules"
    if parameter_name.startswith("readouts."):
        return "readout"
    if parameter_name.startswith(
        ("round_heads.", "direct_box_heads.", "existence_head.")
    ):
        return "prediction_heads"
    if parameter_name.startswith("hdn."):
        return "hdn"
    raise ValueError(f"unclassified trainable parameter: {parameter_name}")


def build_optimizer(
    model,
    config: EgoDetV2Config,
    *,
    base_lr: Optional[float] = None,
    geometry_lr: Optional[float] = None,
    fpn_lr: Optional[float] = None,
    backbone_lr: Optional[float] = None,
    backbone_stem_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    print_groups: bool = True,
) -> torch.optim.Optimizer:
    backbone_groups = model.backbone_param_groups()
    stem_ids = {id(parameter) for parameter in backbone_groups["stem"]}
    backbone_ids = {id(parameter) for parameter in backbone_groups["backbone"]}
    fpn_ids = {id(parameter) for parameter in backbone_groups.get("other", [])}
    effective_base_lr = float(config.lr if base_lr is None else base_lr)
    effective_geometry_lr = float(
        effective_base_lr if geometry_lr is None else geometry_lr
    )
    effective_fpn_lr = float(effective_base_lr if fpn_lr is None else fpn_lr)
    effective_backbone_lr = float(
        config.backbone_lr if backbone_lr is None else backbone_lr
    )
    effective_stem_lr = float(
        config.backbone_stem_lr
        if backbone_stem_lr is None
        else backbone_stem_lr
    )
    effective_weight_decay = float(
        config.weight_decay if weight_decay is None else weight_decay
    )
    learning_rates = {
        "backbone_stem": effective_stem_lr,
        "backbone": effective_backbone_lr,
        "fpn": effective_fpn_lr,
        "geometry_tower": effective_geometry_lr,
        "geometry_obj_head": effective_geometry_lr,
        "geometry_box_head": effective_geometry_lr,
        "geometry_quality_head": effective_geometry_lr,
        "semantic_tower": effective_base_lr,
        "slot_modules": effective_base_lr,
        "readout": effective_base_lr,
        "prediction_heads": effective_base_lr,
        "hdn": effective_base_lr,
    }
    buckets: Dict[tuple[str, float], list[torch.nn.Parameter]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        logical_name = _optimizer_logical_name(
            name,
            id(parameter),
            stem_ids=stem_ids,
            backbone_ids=backbone_ids,
            fpn_ids=fpn_ids,
        )
        lower_name = name.lower()
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in lower_name
            or "embedding" in lower_name
        )
        decay = 0.0 if no_decay else effective_weight_decay
        buckets.setdefault((logical_name, decay), []).append(parameter)

    optimizer_groups = []
    for logical_name in OPTIMIZER_GROUP_NAMES:
        matching = [
            (decay, parameters)
            for (name, decay), parameters in buckets.items()
            if name == logical_name
        ]
        if not matching:
            matching = [(0.0, [])]
        for decay, parameters in matching:
            optimizer_groups.append(
                {
                    "name": logical_name,
                    "params": parameters,
                    "lr": learning_rates[logical_name],
                    "weight_decay": decay,
                }
            )
        if print_groups:
            trainable_numel = sum(
                parameter.numel()
                for _, parameters in matching
                for parameter in parameters
                if parameter.requires_grad
            )
            print(
                "[EgoDetV2 optimizer group] "
                f"name={logical_name} "
                f"trainable_numel={trainable_numel} "
                f"initial_lr={learning_rates[logical_name]:.9g}",
                flush=True,
            )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=effective_base_lr,
        weight_decay=effective_weight_decay,
    )
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise AssertionError("a trainable parameter appears in multiple optimizer groups")
    if set(optimizer_ids) != seen:
        raise AssertionError("optimizer groups do not cover every trainable parameter")
    return optimizer


def optimizer_step_warmup_scale(optimizer_step: int, warmup_steps: int) -> float:
    warmup = min(
        1.0,
        (int(optimizer_step) + 1) / max(int(warmup_steps), 1),
    )
    return 0.01 + 0.99 * warmup


def build_optimizer_step_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up on optimizer-step units; callers step it after optimizer.step()."""

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: optimizer_step_warmup_scale(step, int(warmup_steps)),
    )


def _model_autocast_context(
    config: EgoDetV2Config,
    device: torch.device,
):
    """Enable autocast only on CUDA.

    Older PyTorch releases validate the requested CPU autocast dtype even when
    ``enabled=False``. Returning a true no-op context on CPU avoids accidentally
    presenting ``torch.float16`` to AutocastCPU.
    """

    if device.type != "cuda" or not config.amp:
        return contextlib.nullcontext()
    use_bf16 = (
        str(config.amp_dtype).lower() == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _autocast_context(config: EgoDetV2Config, device: torch.device):
    context = _model_autocast_context(config, device)
    if device.type != "cuda" or not config.amp:
        return context, None
    use_bf16 = (
        str(config.amp_dtype).lower() == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    # torch.cuda.amp.GradScaler is available in the contracted PyTorch 2.1.2;
    # the newer torch.amp.GradScaler("cuda") spelling is not.
    scaler = None if use_bf16 else torch.cuda.amp.GradScaler()
    return context, scaler


def _move_targets(targets: Sequence[Dict], device: torch.device) -> list[Dict]:
    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in target.items()
            }
        )
    return moved


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_optimizer_step: int,
    micro_step: int,
    config: EgoDetV2Config,
    ema: Optional[ModelEMA],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch),
            # Keep global_step as an optimizer-step compatibility alias while
            # persisting the unambiguous MOD7 counters.
            "global_step": int(global_optimizer_step),
            "global_optimizer_step": int(global_optimizer_step),
            "micro_step": int(micro_step),
            "config": config.to_dict(),
            "ema": ema.state_dict() if ema is not None else None,
        },
        path,
    )


def load_checkpoint(
    path: Path | str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema: Optional[ModelEMA] = None,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    global_optimizer_step = int(
        checkpoint.get(
            "global_optimizer_step",
            checkpoint.get("global_step", 0),
        )
    )
    micro_step = int(checkpoint.get("micro_step", 0))
    return int(checkpoint.get("epoch", 0)), global_optimizer_step, micro_step


def _loss_summary(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    def sum_prefix(*prefixes: str) -> float:
        return sum(
            float(value.detach())
            for key, value in losses.items()
            if any(key.startswith(prefix) for prefix in prefixes)
        )

    matched_dustbin_rates = [
        float(value.detach())
        for key, value in losses.items()
        if key.startswith("ownership_matched_dustbin_rate_r")
    ]
    return {
        "loss": float(losses["loss_total"].detach()),
        "dense": sum_prefix("loss_dense"),
        "own": sum_prefix("loss_ownership", "loss_dustbin"),
        "cls": sum_prefix("loss_cls", "loss_dn_cls"),
        "l1": sum_prefix("loss_l1", "loss_dn_l1"),
        "giou": sum_prefix("loss_giou", "loss_dn_giou"),
        "quality": sum_prefix("loss_quality", "loss_dn_quality"),
        "box": sum_prefix("loss_l1", "loss_giou", "loss_dn_l1", "loss_dn_giou"),
        "exist": sum_prefix("loss_exist", "loss_dn_exist"),
        "gain": float(losses.get("loss_gain", losses["loss_total"] * 0).detach()),
        "ownership_matched_dustbin_rate": (
            matched_dustbin_rates[-1] if matched_dustbin_rates else 0.0
        ),
    }


def _loss_debug_values(losses: Dict[str, torch.Tensor]) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for name, value in losses.items():
        if torch.is_tensor(value) and value.numel() == 1:
            values[name] = float(value.detach().float().cpu())
        elif torch.is_tensor(value):
            values[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            values[name] = str(value)
    return values


def _output_dtype_summary(outputs: Dict[str, object]) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for name in (
        "pred_logits",
        "pred_boxes",
        "pred_quality",
        "pred_exist",
        "pred_scores",
    ):
        value = outputs.get(name)
        if torch.is_tensor(value):
            summary[name] = str(value.dtype)
    dense = outputs.get("dense")
    if isinstance(dense, dict):
        levels = dense.get("levels")
        if isinstance(levels, (list, tuple)) and levels:
            for name in ("obj_logits", "qual_logits", "boxes", "target_boxes"):
                value = levels[0].get(name)
                if torch.is_tensor(value):
                    summary[f"dense.level0.{name}"] = str(value.dtype)
    bank = outputs.get("bank")
    if isinstance(bank, dict):
        for name in ("boxes", "desc", "obj", "qual", "score"):
            value = bank.get(name)
            if torch.is_tensor(value):
                summary[f"bank.{name}"] = str(value.dtype)
    ownership = outputs.get("ownership")
    if isinstance(ownership, (list, tuple)) and ownership:
        for name in ("logits", "w", "proposal_mass"):
            value = ownership[-1].get(name)
            if torch.is_tensor(value):
                summary[f"ownership.{name}"] = str(value.dtype)
    return summary


def _criterion_dtype_failure(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "dtypes match",
            "expected scalar type",
            "dtype",
        )
    )


def _finite_loss_status(
    loss: torch.Tensor,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    message: str,
) -> Optional[bool]:
    """Check immediately in reference mode or asynchronously on CUDA."""

    finite = torch.isfinite(loss.detach()).all()
    if (
        config.efficiency_mode == "optimized"
        and int(config.efficiency_ablation_level) >= 6
        and device.type == "cuda"
        and hasattr(torch, "_assert_async")
    ):
        torch._assert_async(finite, message)
        return None
    return bool(finite)


def _cuda_memory_gb(device: torch.device) -> tuple[float, float, float]:
    if device.type != "cuda":
        return 0.0, 0.0, 0.0
    divisor = float(1024**3)
    return (
        torch.cuda.memory_allocated(device) / divisor,
        torch.cuda.memory_reserved(device) / divisor,
        torch.cuda.max_memory_allocated(device) / divisor,
    )


def run_fixed_steps(
    model,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    steps: int,
    epoch: int = 0,
    geometry_only: bool = False,
    ema: Optional[ModelEMA] = None,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    global_step_offset: int = 0,
    diagnostic_jsonl: Optional[Path | str] = None,
    parameter_auditor: Optional[ParameterUpdateAuditor] = None,
    gradient_competition_steps: Sequence[int] = (),
    gradient_competition_jsonl: Optional[Path | str] = None,
    slot_to_geometry_grad_scale: float = 1.0,
    optimizer_step_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, float]:
    """Run a bounded number of *optimizer* steps for engineering diagnostics."""

    requested_steps = int(steps)
    if requested_steps < 1:
        raise ValueError("steps must be >= 1 optimizer step")
    model.train()
    _, scaler = _autocast_context(config, device)
    iterator = iter(data_loader)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss_history = []
    peak_memory = 0
    optimizer_steps = 0
    micro_step = 0
    accumulation_count = 0
    logged_first_batch = False
    phase = "geometry" if geometry_only else "full"
    description = progress_desc or ("Geometry" if geometry_only else "Full")
    progress = tqdm(
        total=requested_steps,
        desc=description,
        unit="opt-step",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    images = None
    try:
        while optimizer_steps < requested_steps:
            try:
                images, pad_mask, targets = next(iterator)
            except StopIteration:
                iterator = iter(data_loader)
                try:
                    images, pad_mask, targets = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("fixed-step data loader is empty") from exc
            images = images.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            targets = _move_targets(targets, device)
            if not logged_first_batch:
                progress.write(
                    "[EgoDetV2 fixed-step input] "
                    f"phase={phase} "
                    f"images={tuple(images.shape)} "
                    f"batch_size={images.shape[0]} "
                    f"amp={config.amp} "
                    f"amp_dtype={config.amp_dtype} "
                    f"grad_accum={config.grad_accum_steps}"
                )
                logged_first_batch = True

            with _model_autocast_context(config, device):
                if geometry_only:
                    outputs = model.forward_geometry(images, pad_mask, targets)
                else:
                    outputs = model(images, pad_mask, targets, epoch=epoch)

            if geometry_only:
                losses = dense_geometry_losses(outputs["dense"])
                loss = (
                    config.w_dense_obj * losses["loss_dense_obj"]
                    + config.w_dense_giou * losses["loss_dense_giou"]
                    + config.w_dense_quality * losses["loss_dense_quality"]
                )
            else:
                try:
                    losses = criterion(
                        outputs,
                        targets,
                        epoch=epoch,
                        epoch_progress=(optimizer_steps + 1) / requested_steps,
                    )
                except RuntimeError as exc:
                    if _criterion_dtype_failure(exc):
                        progress.write(
                            "[EgoDetV2 criterion dtype error] "
                            + json.dumps(
                                _output_dtype_summary(outputs),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                    raise
                loss = losses["loss_total"]

            finite_status = _finite_loss_status(
                loss,
                config=config,
                device=device,
                message=f"non-finite {phase} loss",
            )
            if finite_status is False:
                progress.write(
                    "[EgoDetV2 non-finite loss] "
                    f"phase={phase} micro_step={micro_step + 1} "
                    f"optimizer_step={optimizer_steps} "
                    + json.dumps(
                        _loss_debug_values(losses),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                raise FloatingPointError(
                    f"non-finite {phase} loss before backward at "
                    f"micro_step={micro_step + 1}, optimizer_step={optimizer_steps}"
                )

            next_global_optimizer_step = (
                int(global_step_offset) + optimizer_steps + 1
            )
            next_stage_optimizer_step = optimizer_steps + 1
            partitions = None
            if not geometry_only and (
                next_stage_optimizer_step in set(gradient_competition_steps)
                or float(slot_to_geometry_grad_scale) != 1.0
            ):
                partitions = criterion.weighted_loss_partitions(
                    losses,
                    epoch=epoch,
                    epoch_progress=(optimizer_steps + 1) / requested_steps,
                )
            if (
                not geometry_only
                and accumulation_count == 0
                and next_stage_optimizer_step in set(gradient_competition_steps)
                and partitions is not None
            ):
                competition = measure_geometry_gradient_competition(
                    partitions,
                    optimizer,
                    optimizer_step=next_global_optimizer_step,
                    stage_optimizer_step=next_stage_optimizer_step,
                    phase=phase,
                    jsonl_path=gradient_competition_jsonl,
                )
                progress.write(
                    "[EgoDetV2 geometry gradient competition] "
                    + json.dumps(competition, ensure_ascii=False, sort_keys=True)
                )

            scaled_loss = loss.float() / float(config.grad_accum_steps)
            loss_history.append(float(loss.detach().float()))
            if (
                not geometry_only
                and float(slot_to_geometry_grad_scale) != 1.0
                and partitions is not None
            ):
                backward_with_slot_geometry_scale(
                    partitions,
                    optimizer,
                    divisor=float(config.grad_accum_steps),
                    scale=float(slot_to_geometry_grad_scale),
                    scaler=scaler,
                )
            elif scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1
            accumulation_count += 1

            if accumulation_count >= int(config.grad_accum_steps):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                audit_this_step = (
                    parameter_auditor is not None
                    and next_global_optimizer_step % 25 == 0
                )
                if audit_this_step:
                    parameter_auditor.capture_before_step()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config.clip_max_norm),
                )
                used_lr_values = optimizer_lr_summary(optimizer)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                audit_record = None
                if audit_this_step:
                    audit_record = parameter_auditor.capture_after_step(
                        optimizer_step=next_global_optimizer_step,
                        phase=phase,
                    )
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model)

                lr_values = optimizer_lr_summary(optimizer)
                if next_global_optimizer_step % 25 == 0:
                    lr_record: Dict[str, object] = {
                        "kind": "optimizer_lr",
                        "optimizer_step": next_global_optimizer_step,
                        "phase": phase,
                        **used_lr_values,
                        **{
                            f"next_{name}": value
                            for name, value in lr_values.items()
                        },
                    }
                    append_jsonl(diagnostic_jsonl, lr_record)
                    progress.write(
                        "[EgoDetV2 optimizer LR] "
                        + json.dumps(lr_record, ensure_ascii=False, sort_keys=True)
                    )
                    if audit_record is not None:
                        progress.write(
                            "[EgoDetV2 parameter update audit] "
                            + json.dumps(
                                audit_record,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )

                allocated, _, current_peak = _cuda_memory_gb(device)
                peak_memory = max(
                    peak_memory,
                    int(current_peak * (1024**3)),
                )
                elapsed_now = max(time.perf_counter() - start, 1.0e-8)
                if geometry_only:
                    summary = {
                        "loss": float(loss.detach().float()),
                        "dense": sum(
                            float(value.detach().float())
                            for name, value in losses.items()
                            if name.startswith("loss_dense")
                        ),
                    }
                else:
                    summary = _loss_summary(losses)
                postfix = {
                    name: f"{value:.4g}"
                    for name, value in summary.items()
                    if name
                    in {
                        "loss",
                        "dense",
                        "own",
                        "cls",
                        "box",
                        "exist",
                    }
                }
                postfix.update(
                    {
                        "lr_geometry": (
                            f"{used_lr_values['lr_geometry']:.3g}"
                        ),
                        "mem_GB": f"{allocated:.2f}",
                        "it/s": f"{optimizer_steps / elapsed_now:.3g}",
                    }
                )
                progress.update(1)
                progress.set_postfix(postfix, refresh=True)
                # Do not retain the completed training outputs while a
                # potentially large periodic evaluation is running.
                del outputs, losses, loss, scaled_loss, partitions
                if optimizer_step_callback is not None:
                    optimizer_step_callback(
                        optimizer_steps,
                        next_global_optimizer_step,
                    )
    except torch.cuda.OutOfMemoryError:
        allocated, reserved, peak = _cuda_memory_gb(device)
        input_shape = tuple(images.shape) if torch.is_tensor(images) else None
        progress.write(
            "[EgoDetV2 OOM] "
            f"phase={phase} "
            f"input_shape={input_shape} "
            f"batch_size={input_shape[0] if input_shape else config.batch_size} "
            f"checkpointing={config.overlock_gradient_checkpointing} "
            f"allocated_gb={allocated:.2f} "
            f"reserved_gb={reserved:.2f} "
            f"peak_gb={peak:.2f} "
            f"optimizer_step={optimizer_steps} "
            f"micro_step={micro_step}"
        )
        raise
    finally:
        progress.close()

    elapsed = max(time.perf_counter() - start, 1.0e-8)
    window = max(1, min(25, len(loss_history) // 4 if len(loss_history) >= 4 else 1))
    first_window = sum(loss_history[:window]) / max(window, 1) if loss_history else math.nan
    last_window = sum(loss_history[-window:]) / max(window, 1) if loss_history else math.nan
    result = {
        "micro_step": float(micro_step),
        "optimizer_step": float(optimizer_steps),
        "global_optimizer_step": float(optimizer_steps),
        "first_loss": float(loss_history[0] if loss_history else math.nan),
        "last_loss": float(loss_history[-1] if loss_history else math.nan),
        "first_window_loss": float(first_window),
        "last_window_loss": float(last_window),
        "nonfinite": 0.0,
        "throughput_micro_it_s": float(micro_step / elapsed),
        "throughput_optimizer_step_s": float(optimizer_steps / elapsed),
        "peak_memory_gb": float(peak_memory / (1024**3)),
    }
    result.update(optimizer_lr_summary(optimizer))
    return result


def _should_run_full_eval(
    config: EgoDetV2Config,
    completed_epoch: int,
) -> bool:
    return (
        not config.full_eval_last_epoch_only
        or int(completed_epoch) == int(config.epochs)
    )


def fit(
    model,
    criterion,
    train_loader: Iterable,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    val_loader: Optional[Iterable] = None,
    val_dataset=None,
    resume: Optional[Path | str] = None,
    smoke_steps: int = 0,
    show_progress: bool = True,
) -> Dict[str, float]:
    set_runtime_defaults(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = build_optimizer(model, config)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / config.grad_accum_steps))

    def lr_lambda(step: int) -> float:
        warmup_scale = optimizer_step_warmup_scale(step, int(config.warmup_steps))
        epoch_at_step = step // steps_per_epoch
        drop = 0.1 if epoch_at_step >= int(config.lr_drop_epoch) else 1.0
        return warmup_scale * drop

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    _, scaler = _autocast_context(config, device)
    ema = ModelEMA(model, config.ema_decay) if config.use_ema else None
    start_epoch = 0
    global_optimizer_step = 0
    total_micro_step = 0
    if resume:
        start_epoch, global_optimizer_step, total_micro_step = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )

    model.to(device)
    criterion.to(device)
    train_log = output_dir / "train_log.jsonl"
    eval_log = output_dir / "eval_history.jsonl"
    last_quick_eval_step = -1
    last_summary: Dict[str, float] = {}
    optimizer_steps_this_run = 0
    for epoch in range(start_epoch, int(config.epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_count = 0
        epoch_optimizer_step = 0
        epoch_start = time.perf_counter()
        progress = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{int(config.epochs)}",
            unit="opt-step",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        images = None
        try:
            for batch_index, (images, pad_mask, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                pad_mask = pad_mask.to(device, non_blocking=True)
                targets = _move_targets(targets, device)
                with _model_autocast_context(config, device):
                    outputs = model(images, pad_mask, targets, epoch=epoch)
                try:
                    losses = criterion(
                        outputs,
                        targets,
                        epoch=epoch,
                        epoch_progress=(
                            epoch_optimizer_step + 1
                        ) / max(steps_per_epoch, 1),
                    )
                except RuntimeError as exc:
                    if _criterion_dtype_failure(exc):
                        progress.write(
                            "[EgoDetV2 criterion dtype error] "
                            + json.dumps(
                                _output_dtype_summary(outputs),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                    raise

                total_loss = losses["loss_total"]
                finite_status = _finite_loss_status(
                    total_loss,
                    config=config,
                    device=device,
                    message="non-finite training loss",
                )
                if finite_status is False:
                    progress.write(
                        "[EgoDetV2 non-finite loss] "
                        f"epoch={epoch} batch={batch_index} "
                        f"micro_step={total_micro_step + 1} "
                        f"optimizer_step={epoch_optimizer_step} "
                        f"global_optimizer_step={global_optimizer_step} "
                        + json.dumps(
                            _loss_debug_values(losses),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    raise FloatingPointError(
                        "non-finite training loss before backward at "
                        f"epoch={epoch}, batch={batch_index}, "
                        f"micro_step={total_micro_step + 1}, "
                        f"global_optimizer_step={global_optimizer_step}"
                    )

                loss = total_loss.float() / float(config.grad_accum_steps)
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_micro_step += 1
                accumulation_count += 1
                should_step = (
                    accumulation_count >= int(config.grad_accum_steps)
                    or batch_index + 1 == len(train_loader)
                )
                if should_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config.clip_max_norm),
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_count = 0
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                    epoch_optimizer_step += 1
                    global_optimizer_step += 1
                    optimizer_steps_this_run += 1

                    last_summary = _loss_summary(losses)
                    allocated, _, _ = _cuda_memory_gb(device)
                    elapsed_now = max(time.perf_counter() - epoch_start, 1.0e-8)
                    postfix = {
                        name: f"{value:.4g}"
                        for name, value in last_summary.items()
                        if name
                        in {
                            "loss",
                            "dense",
                            "own",
                            "cls",
                            "l1",
                            "giou",
                            "quality",
                            "exist",
                            "gain",
                        }
                    }
                    postfix.update(
                        {
                            "lr_geometry": (
                                f"{optimizer_lr_summary(optimizer)['lr_geometry']:.3g}"
                            ),
                            "mem": f"{allocated:.2f}GB",
                            "it/s": f"{epoch_optimizer_step / elapsed_now:.3g}",
                        }
                    )
                    progress.update(1)
                    progress.set_postfix(postfix, refresh=True)

                if total_micro_step % max(int(config.log_every_steps), 1) == 0:
                    log_record = _loss_summary(losses)
                    log_record.update(
                        {
                            "epoch": epoch,
                            "micro_step": total_micro_step,
                            "optimizer_step": epoch_optimizer_step,
                            "global_optimizer_step": global_optimizer_step,
                            **optimizer_lr_summary(optimizer),
                        }
                    )
                    oracle_cadence = int(
                        config.oracle_monitor_every_optimizer_steps
                    )
                    oracle_due = (
                        oracle_cadence == -1
                        or (
                            oracle_cadence > 0
                            and should_step
                            and global_optimizer_step % oracle_cadence == 0
                        )
                    )
                    if oracle_due:
                        log_record["oracleAR300_fast"] = oracle_recall(
                            outputs["bank"],
                            targets,
                        )["AR@300"]
                    progress.write(
                        "[EgoDetV2] "
                        + " ".join(
                            (
                                f"{name}={value:.5g}"
                                if isinstance(value, float)
                                else f"{name}={value}"
                            )
                            for name, value in log_record.items()
                        )
                    )
                    with train_log.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(log_record, ensure_ascii=False) + "\n"
                        )

                if (
                    should_step
                    and int(config.ckpt_every_steps) > 0
                    and global_optimizer_step % int(config.ckpt_every_steps) == 0
                ):
                    _save_checkpoint(
                        output_dir / "latest.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_optimizer_step=global_optimizer_step,
                        micro_step=total_micro_step,
                        config=config,
                        ema=ema,
                    )
                if (
                    val_loader is not None
                    and should_step
                    and int(config.quick_eval_every_steps) > 0
                    and global_optimizer_step != last_quick_eval_step
                    and global_optimizer_step
                    % int(config.quick_eval_every_steps)
                    == 0
                ):
                    evaluation_model = ema.module if ema is not None else model
                    quick_metrics = evaluate(
                        evaluation_model,
                        val_loader,
                        matcher=criterion.matcher,
                        device=device,
                        dataset=val_dataset,
                        subset_size=500,
                        show_progress=show_progress,
                    )
                    quick_metrics.update(
                        {
                            "kind": "quick500",
                            "epoch": epoch,
                            "global_optimizer_step": global_optimizer_step,
                        }
                    )
                    with eval_log.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(quick_metrics, ensure_ascii=False) + "\n"
                        )
                    progress.write(
                        "[EgoDetV2:quick500] "
                        + json.dumps(quick_metrics, ensure_ascii=False)
                    )
                    last_quick_eval_step = global_optimizer_step
                    model.train()

                if (
                    should_step
                    and smoke_steps
                    and optimizer_steps_this_run >= int(smoke_steps)
                ):
                    _save_checkpoint(
                        output_dir / "latest.pth",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_optimizer_step=global_optimizer_step,
                        micro_step=total_micro_step,
                        config=config,
                        ema=ema,
                    )
                    return last_summary
        except torch.cuda.OutOfMemoryError:
            allocated, reserved, peak = _cuda_memory_gb(device)
            input_shape = tuple(images.shape) if torch.is_tensor(images) else None
            progress.write(
                "[EgoDetV2 OOM] "
                f"phase=train "
                f"input_shape={input_shape} "
                f"batch_size={input_shape[0] if input_shape else config.batch_size} "
                f"checkpointing={config.overlock_gradient_checkpointing} "
                f"allocated_gb={allocated:.2f} "
                f"reserved_gb={reserved:.2f} "
                f"peak_gb={peak:.2f} "
                f"optimizer_step={epoch_optimizer_step} "
                f"global_optimizer_step={global_optimizer_step} "
                f"micro_step={total_micro_step}"
            )
            raise
        finally:
            progress.close()

        _save_checkpoint(
            output_dir / f"epoch_{epoch + 1}.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_optimizer_step=global_optimizer_step,
            micro_step=total_micro_step,
            config=config,
            ema=ema,
        )
        _save_checkpoint(
            output_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_optimizer_step=global_optimizer_step,
            micro_step=total_micro_step,
            config=config,
            ema=ema,
        )
        if val_loader is not None and _should_run_full_eval(config, epoch + 1):
            evaluation_model = ema.module if ema is not None else model
            metrics = evaluate(
                evaluation_model,
                val_loader,
                matcher=criterion.matcher,
                device=device,
                dataset=val_dataset,
                show_progress=show_progress,
            )
            metrics.update(
                {
                    "epoch": epoch + 1,
                    "global_optimizer_step": global_optimizer_step,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
            )
            with eval_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            tqdm.write(
                "[EgoDetV2:eval] "
                + json.dumps(metrics, ensure_ascii=False)
            )
    return last_summary
