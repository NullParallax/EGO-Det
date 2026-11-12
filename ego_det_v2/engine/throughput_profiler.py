from __future__ import annotations

import contextlib
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Mapping, Optional, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..models.dense_geometry_branch import oracle_recall
from .train_loop import (
    _autocast_context,
    _model_autocast_context,
    _move_targets,
    build_optimizer,
    build_optimizer_step_scheduler,
)


PROFILE_STAGE_NAMES = (
    "data_wait",
    "host_to_device",
    "backbone_fpn",
    "dense_geometry",
    "bank_build",
    "semantic_support",
    "slot_seeding",
    "hdn_build",
    "slot_round_1_update",
    "slot_round_1_readout",
    "slot_round_1_heads",
    "slot_round_2_update",
    "slot_round_2_readout",
    "slot_round_2_heads",
    "slot_round_3_update",
    "slot_round_3_readout",
    "slot_round_3_heads",
    "existence_and_score",
    "hungarian_match_all_rounds",
    "dense_loss",
    "ownership_target_and_loss_all_rounds",
    "slot_losses_all_rounds",
    "dn_losses",
    "duplicate_loss",
    "gain_loss",
    "criterion_total",
    "backward",
    "gradient_clip",
    "optimizer_step",
    "scheduler_step",
    "logging_and_oracle_monitor",
    "total_micro_step",
    "total_optimizer_step",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    converted = [float(value) for value in values]
    return {
        "count": len(converted),
        "mean": float(statistics.fmean(converted)),
        "p50": float(_percentile(converted, 0.50)),
        "p90": float(_percentile(converted, 0.90)),
        "min": float(min(converted)),
        "max": float(max(converted)),
    }


def _worktree_identity() -> Dict[str, str]:
    root = Path.cwd().resolve()
    package_root = Path(__file__).resolve().parents[1]
    source_hasher = hashlib.sha256()
    for source_path in sorted(package_root.rglob("*.py")):
        relative = source_path.relative_to(package_root).as_posix()
        source_hasher.update(relative.encode("utf-8"))
        source_hasher.update(b"\0")
        source_hasher.update(source_path.read_bytes())
        source_hasher.update(b"\0")
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    return {
        "root": str(root),
        "git_revision": revision,
        "ego_det_v2_python_source_sha256": source_hasher.hexdigest(),
        "modification": "MOD10A",
    }


class ThroughputProfiler:
    """Low-overhead stage collector enabled only by the profiler launcher.

    Stage contexts record CPU dispatch wall time and CUDA events without a
    per-stage synchronize.  A single synchronization at the end of each
    measured micro-step resolves every pending CUDA event.  Consequently
    ordinary training pays no profiler synchronization cost.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        trace_profiler=None,
    ) -> None:
        self.device = device
        self.trace_profiler = trace_profiler
        self._cpu_ms: DefaultDict[str, list[float]] = defaultdict(list)
        self._cuda_ms: DefaultDict[str, list[float]] = defaultdict(list)
        self._pending = []
        self._micro_start_cpu: Optional[float] = None
        self._micro_start_event = None
        self._optimizer_start_cpu: Optional[float] = None
        self._optimizer_start_event = None

    def _event(self):
        if self.device.type != "cuda":
            return None
        return torch.cuda.Event(enable_timing=True)

    @contextlib.contextmanager
    def stage(self, name: str):
        start_cpu = time.perf_counter()
        start_event = self._event()
        end_event = self._event()
        if start_event is not None:
            start_event.record()
        record = torch.autograd.profiler.record_function(name)
        with record:
            yield
        if end_event is not None:
            end_event.record()
        cpu_ms = (time.perf_counter() - start_cpu) * 1000.0
        self._cpu_ms[name].append(float(cpu_ms))
        if start_event is not None:
            self._pending.append((name, start_event, end_event))

    def begin_micro_step(self) -> None:
        if self._micro_start_cpu is not None:
            raise RuntimeError("a profiled micro-step is already active")
        self._micro_start_cpu = time.perf_counter()
        self._micro_start_event = self._event()
        if self._micro_start_event is not None:
            self._micro_start_event.record()

    def end_micro_step(self) -> None:
        if self._micro_start_cpu is None:
            raise RuntimeError("no profiled micro-step is active")
        end_event = self._event()
        if end_event is not None:
            end_event.record()
            self._pending.append(
                ("total_micro_step", self._micro_start_event, end_event)
            )
            torch.cuda.synchronize(self.device)
        total_wall_ms = (
            time.perf_counter() - self._micro_start_cpu
        ) * 1000.0
        self._cpu_ms["total_micro_step"].append(float(total_wall_ms))
        self._resolve_pending()
        self._micro_start_cpu = None
        self._micro_start_event = None

    def begin_optimizer_step(self) -> None:
        if self._optimizer_start_cpu is not None:
            raise RuntimeError("an optimizer-step window is already active")
        self._optimizer_start_cpu = time.perf_counter()
        self._optimizer_start_event = self._event()
        if self._optimizer_start_event is not None:
            self._optimizer_start_event.record()

    def end_optimizer_step(self) -> None:
        if self._optimizer_start_cpu is None:
            raise RuntimeError("no optimizer-step window is active")
        end_event = self._event()
        if end_event is not None:
            end_event.record()
            self._pending.append(
                (
                    "total_optimizer_step",
                    self._optimizer_start_event,
                    end_event,
                )
            )
        wall_ms = (
            time.perf_counter() - self._optimizer_start_cpu
        ) * 1000.0
        self._cpu_ms["total_optimizer_step"].append(float(wall_ms))
        self._optimizer_start_cpu = None
        self._optimizer_start_event = None

    def _resolve_pending(self) -> None:
        for name, start_event, end_event in self._pending:
            self._cuda_ms[name].append(
                float(start_event.elapsed_time(end_event))
            )
        self._pending.clear()

    def stage_summary(self) -> Dict[str, Dict[str, Dict[str, float | int]]]:
        result = {}
        for name in PROFILE_STAGE_NAMES:
            cpu = self._cpu_ms.get(name, [])
            cuda = self._cuda_ms.get(name, [])
            if cuda:
                wall = [
                    max(
                        float(cpu[min(index, len(cpu) - 1)]) if cpu else 0.0,
                        float(value),
                    )
                    for index, value in enumerate(cuda)
                ]
            else:
                wall = list(cpu)
            result[name] = {
                "cpu_dispatch_wall_ms": _distribution(cpu),
                "cuda_ms": _distribution(cuda),
                "wall_ms": _distribution(wall),
            }
        return result


def _top_operations(
    trace_profiler,
    *,
    sort_field: str,
    limit: int = 20,
) -> list[Dict[str, float | int | str]]:
    if trace_profiler is None:
        return []
    events = list(trace_profiler.key_averages())

    def value(event) -> float:
        if sort_field == "self_device_time_total":
            return float(
                getattr(
                    event,
                    "self_device_time_total",
                    getattr(event, "self_cuda_time_total", 0.0),
                )
                or 0.0
            )
        return float(getattr(event, sort_field, 0.0) or 0.0)

    events.sort(key=value, reverse=True)
    rows = []
    for event in events[: int(limit)]:
        rows.append(
            {
                "name": str(event.key),
                "calls": int(event.count),
                "self_cpu_ms": float(
                    getattr(event, "self_cpu_time_total", 0.0) / 1000.0
                ),
                "cpu_total_ms": float(
                    getattr(event, "cpu_time_total", 0.0) / 1000.0
                ),
                "self_cuda_ms": float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                    / 1000.0
                ),
                "cuda_total_ms": float(
                    getattr(
                        event,
                        "device_time_total",
                        getattr(event, "cuda_time_total", 0.0),
                    )
                    / 1000.0
                ),
            }
        )
    return rows


def _next_batch(iterator, data_loader: Iterable):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(data_loader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise RuntimeError("throughput data loader is empty") from exc


def _json_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        flattened = value.detach().cpu().reshape(-1)
        if flattened.numel() == 1:
            return flattened.item()
        return flattened.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _batch_observation(
    images: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> Dict[str, object]:
    """Capture cheap CPU-side evidence that A/B runs saw identical batches."""

    return {
        "image_ids": [
            _json_scalar(target.get("image_id")) for target in targets
        ],
        "padded_image_shape": [int(value) for value in images.shape],
        "target_sizes": [
            _json_scalar(target.get("size")) for target in targets
        ],
        "ground_truth_counts": [
            int(target["labels"].shape[0]) for target in targets
        ],
    }


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _data_source_identity(data_loader: Iterable) -> Dict[str, object]:
    dataset = getattr(data_loader, "dataset", None)
    image_ids = [
        _json_scalar(value)
        for value in getattr(dataset, "image_ids", [])
    ]
    return {
        "dataset_class": (
            type(dataset).__name__ if dataset is not None else "unavailable"
        ),
        "dataset_length": (
            int(len(dataset)) if dataset is not None else None
        ),
        "split": str(getattr(dataset, "split", "unavailable")),
        "annotation_file": str(
            getattr(dataset, "annotation_file", "unavailable")
        ),
        "image_root": str(
            getattr(dataset, "image_root", "unavailable")
        ),
        "dataset_image_ids_sha256": _stable_digest(image_ids),
    }


def _environment_identity(device: torch.device) -> Dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
    }


def _measurement_data_contract(
    data_loader: Iterable,
    *,
    warmup_batches: Sequence[Mapping[str, object]],
    measured_batches: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    warmup = list(warmup_batches)
    measured = list(measured_batches)
    return {
        "source": _data_source_identity(data_loader),
        "warmup_batches": warmup,
        "measured_batches": measured,
        "warmup_sequence_sha256": _stable_digest(warmup),
        "measured_sequence_sha256": _stable_digest(measured),
    }


def _async_finite_assert(
    loss: torch.Tensor,
    device: torch.device,
    config: EgoDetV2Config,
) -> None:
    finite = torch.isfinite(loss.detach()).all()
    if (
        config.efficiency_mode == "optimized"
        and int(config.efficiency_ablation_level) >= 6
        and device.type == "cuda"
        and hasattr(torch, "_assert_async")
    ):
        torch._assert_async(finite, "non-finite loss in throughput run")
        return
    if not bool(finite):
        raise FloatingPointError("non-finite loss in throughput run")


def _run_warmup(
    model,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    micro_steps: int,
) -> tuple[object, list[Dict[str, object]]]:
    iterator = iter(data_loader)
    observations: list[Dict[str, object]] = []
    _, scaler = _autocast_context(config, device)
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    for _ in range(int(micro_steps)):
        (images, pad_mask, targets), iterator = _next_batch(
            iterator,
            data_loader,
        )
        observations.append(_batch_observation(images, targets))
        images = images.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        targets = _move_targets(targets, device)
        with _model_autocast_context(config, device):
            outputs = model(images, pad_mask, targets, epoch=0)
        losses = criterion(outputs, targets, epoch=0, epoch_progress=0.0)
        loss = losses["loss_total"].float() / float(config.grad_accum_steps)
        _async_finite_assert(loss, device, config)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accumulation += 1
        if accumulation >= int(config.grad_accum_steps):
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
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
    optimizer.zero_grad(set_to_none=True)
    return iterator, observations


def _prepare_training_run(
    model,
    criterion,
    data_loader: Iterable,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    warmup_micro_steps: int,
) -> tuple[
    torch.optim.Optimizer,
    object,
    object,
    list[Dict[str, object]],
    object,
]:
    if int(warmup_micro_steps) < 0:
        raise ValueError("warmup_micro_steps must be >= 0")
    model.train()
    criterion.train()
    optimizer = build_optimizer(model, config)
    scheduler = build_optimizer_step_scheduler(
        optimizer,
        warmup_steps=int(config.warmup_steps),
    )
    iterator, warmup_observations = _run_warmup(
        model,
        criterion,
        data_loader,
        optimizer,
        scheduler,
        config=config,
        device=device,
        micro_steps=int(warmup_micro_steps),
    )
    optimizer.zero_grad(set_to_none=True)
    _, measurement_scaler = _autocast_context(config, device)
    return (
        optimizer,
        scheduler,
        iterator,
        warmup_observations,
        measurement_scaler,
    )


def _validate_measurement_window(
    *,
    config: EgoDetV2Config,
    measure_micro_steps: int,
    minimum_micro_steps: int,
) -> None:
    if int(measure_micro_steps) < int(minimum_micro_steps):
        raise ValueError(
            "measure_micro_steps must be >= "
            f"{int(minimum_micro_steps)} for this mode"
        )
    if int(measure_micro_steps) % int(config.grad_accum_steps) != 0:
        raise ValueError(
            "measure_micro_steps must be divisible by grad_accum_steps"
        )


def _oracle_monitor_due(
    config: EgoDetV2Config,
    *,
    measured_micro_step: int,
    measured_optimizer_steps: int,
    should_step: bool,
) -> bool:
    cadence = int(config.oracle_monitor_every_optimizer_steps)
    if cadence == -1:
        return (
            (int(measured_micro_step) + 1)
            % max(int(config.log_every_steps), 1)
            == 0
        )
    return (
        cadence > 0
        and bool(should_step)
        and int(measured_optimizer_steps) % cadence == 0
    )


def _memory_result(
    device: torch.device,
    *,
    gate_eligible: bool,
) -> Dict[str, object]:
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "measurement_source": (
            "torch.cuda.max_memory_allocated/max_memory_reserved"
            if device.type == "cuda"
            else "CUDA peak API unavailable on CPU"
        ),
        "gate_eligible": bool(gate_eligible),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_allocated / float(1024**3),
        "peak_reserved_gib": peak_reserved / float(1024**3),
    }


def _decision_eligibility(eligible: bool) -> Dict[str, bool]:
    return {
        "hard_2x_gate": bool(eligible),
        "five_images_per_second_target": bool(eligible),
        "checkpointing_ab": bool(eligible),
        "batch_ab": bool(eligible),
        "memory_margin": bool(eligible),
    }


def benchmark_training_throughput(
    model,
    criterion,
    data_loader: Iterable,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    warmup_micro_steps: int,
    measure_micro_steps: int,
) -> Dict[str, object]:
    """Measure real throughput without trace instrumentation.

    The measured CUDA interval has exactly two explicit synchronization
    boundaries: one immediately before ``perf_counter`` starts and one
    immediately before it stops.  There is no per-micro-step synchronization
    and no torch.profiler or record_function activity in this path.
    """

    _validate_measurement_window(
        config=config,
        measure_micro_steps=int(measure_micro_steps),
        minimum_micro_steps=40,
    )
    (
        optimizer,
        scheduler,
        iterator,
        warmup_observations,
        measurement_scaler,
    ) = _prepare_training_run(
        model,
        criterion,
        data_loader,
        config=config,
        device=device,
        warmup_micro_steps=int(warmup_micro_steps),
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    measured_optimizer_steps = 0
    measured_images = 0
    accumulation = 0
    measured_observations: list[Dict[str, object]] = []
    measurement_start = time.perf_counter()
    for measured_micro_step in range(int(measure_micro_steps)):
        (images, pad_mask, targets), iterator = _next_batch(
            iterator,
            data_loader,
        )
        measured_images += int(images.shape[0])
        measured_observations.append(_batch_observation(images, targets))
        images = images.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)
        targets = _move_targets(targets, device)
        with _model_autocast_context(config, device):
            outputs = model(images, pad_mask, targets, epoch=0)
        losses = criterion(
            outputs,
            targets,
            epoch=0,
            epoch_progress=0.0,
        )
        scaled_loss = losses["loss_total"].float() / float(
            config.grad_accum_steps
        )
        _async_finite_assert(scaled_loss, device, config)
        if measurement_scaler is not None:
            measurement_scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        accumulation += 1
        should_step = accumulation >= int(config.grad_accum_steps)
        if should_step:
            if measurement_scaler is not None:
                measurement_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config.clip_max_norm),
            )
            if measurement_scaler is not None:
                measurement_scaler.step(optimizer)
                measurement_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulation = 0
            measured_optimizer_steps += 1

        if _oracle_monitor_due(
            config,
            measured_micro_step=measured_micro_step,
            measured_optimizer_steps=measured_optimizer_steps,
            should_step=should_step,
        ):
            oracle_recall(outputs["bank"], targets)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measurement_end = time.perf_counter()
    measurement_wall = max(
        measurement_end - measurement_start,
        1.0e-12,
    )
    batch_size = int(config.batch_size)
    return {
        "status": "PASS",
        "version": "MOD10A",
        "result_kind": "benchmark",
        "gate_eligible": True,
        "memory_margin_eligible": True,
        "decision_eligibility": _decision_eligibility(True),
        "worktree": _worktree_identity(),
        "environment": _environment_identity(device),
        "config": config.to_dict(),
        "data_contract": _measurement_data_contract(
            data_loader,
            warmup_batches=warmup_observations,
            measured_batches=measured_observations,
        ),
        "benchmark_contract": {
            "warmup_micro_steps": int(warmup_micro_steps),
            "measure_micro_steps": int(measure_micro_steps),
            "minimum_measure_micro_steps": 40,
            "batch_size": batch_size,
            "grad_accum_steps": int(config.grad_accum_steps),
            "effective_batch_size": (
                batch_size * int(config.grad_accum_steps)
            ),
            "oracle_monitor_every_optimizer_steps": int(
                config.oracle_monitor_every_optimizer_steps
            ),
            "log_every_steps": int(config.log_every_steps),
            "torch_profiler_enabled": False,
            "record_shapes": False,
            "profile_memory": False,
            "per_micro_step_cuda_synchronize": False,
            "cuda_boundary_synchronize_count": (
                2 if device.type == "cuda" else 0
            ),
            "clock": "time.perf_counter between CUDA boundary syncs",
        },
        "benchmark": {
            "measured_images": int(measured_images),
            "measured_optimizer_steps": int(measured_optimizer_steps),
            "measurement_wall_seconds": float(measurement_wall),
        },
        "throughput": {
            "images_per_second": float(
                measured_images / measurement_wall
            ),
            "micro_steps_per_second": float(
                int(measure_micro_steps) / measurement_wall
            ),
            "optimizer_steps_per_second": float(
                measured_optimizer_steps / measurement_wall
            ),
        },
        "memory": _memory_result(device, gate_eligible=True),
    }


def trace_training_profile(
    model,
    criterion,
    data_loader: Iterable,
    *,
    config: EgoDetV2Config,
    device: torch.device,
    warmup_micro_steps: int,
    measure_micro_steps: int,
    trace_path: Path,
) -> Dict[str, object]:
    """Collect detailed stage timing and a Chrome trace for attribution only."""

    _validate_measurement_window(
        config=config,
        measure_micro_steps=int(measure_micro_steps),
        minimum_micro_steps=1,
    )
    (
        optimizer,
        scheduler,
        iterator,
        warmup_observations,
        measurement_scaler,
    ) = _prepare_training_run(
        model,
        criterion,
        data_loader,
        config=config,
        device=device,
        warmup_micro_steps=int(warmup_micro_steps),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    measured_optimizer_steps = 0
    measured_images = 0
    accumulation = 0
    measured_observations: list[Dict[str, object]] = []
    measurement_start = 0.0
    measurement_end = 0.0
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as trace:
        profiler = ThroughputProfiler(
            device=device,
            trace_profiler=trace,
        )
        measurement_start = time.perf_counter()
        for measured_micro_step in range(int(measure_micro_steps)):
            if accumulation == 0:
                profiler.begin_optimizer_step()
            profiler.begin_micro_step()
            with profiler.stage("data_wait"):
                (images, pad_mask, targets), iterator = _next_batch(
                    iterator,
                    data_loader,
                )
                measured_images += int(images.shape[0])
                measured_observations.append(
                    _batch_observation(images, targets)
                )
            with profiler.stage("host_to_device"):
                images = images.to(device, non_blocking=True)
                pad_mask = pad_mask.to(device, non_blocking=True)
                targets = _move_targets(targets, device)
            with _model_autocast_context(config, device):
                outputs = model(
                    images,
                    pad_mask,
                    targets,
                    epoch=0,
                    profiler=profiler,
                )
            losses = criterion(
                outputs,
                targets,
                epoch=0,
                epoch_progress=0.0,
                profiler=profiler,
            )
            scaled_loss = losses["loss_total"].float() / float(
                config.grad_accum_steps
            )
            _async_finite_assert(scaled_loss, device, config)
            with profiler.stage("backward"):
                if measurement_scaler is not None:
                    measurement_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            accumulation += 1
            should_step = accumulation >= int(config.grad_accum_steps)
            if should_step:
                with profiler.stage("gradient_clip"):
                    if measurement_scaler is not None:
                        measurement_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config.clip_max_norm),
                    )
                with profiler.stage("optimizer_step"):
                    if measurement_scaler is not None:
                        measurement_scaler.step(optimizer)
                        measurement_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                with profiler.stage("scheduler_step"):
                    scheduler.step()
                accumulation = 0
                measured_optimizer_steps += 1

            with profiler.stage("logging_and_oracle_monitor"):
                if _oracle_monitor_due(
                    config,
                    measured_micro_step=measured_micro_step,
                    measured_optimizer_steps=measured_optimizer_steps,
                    should_step=should_step,
                ):
                    oracle_recall(outputs["bank"], targets)
            if should_step:
                profiler.end_optimizer_step()
            profiler.end_micro_step()
            trace.step()
        measurement_end = time.perf_counter()
    measurement_wall = max(
        measurement_end - measurement_start,
        1.0e-12,
    )
    trace.export_chrome_trace(str(trace_path))

    return {
        "status": "PASS",
        "version": "MOD10A",
        "result_kind": "trace_profile",
        "gate_eligible": False,
        "memory_margin_eligible": False,
        "decision_eligibility": _decision_eligibility(False),
        "worktree": _worktree_identity(),
        "environment": _environment_identity(device),
        "config": config.to_dict(),
        "data_contract": _measurement_data_contract(
            data_loader,
            warmup_batches=warmup_observations,
            measured_batches=measured_observations,
        ),
        "trace_contract": {
            "warmup_micro_steps": int(warmup_micro_steps),
            "measure_micro_steps": int(measure_micro_steps),
            "measured_images": int(measured_images),
            "measured_optimizer_steps": int(measured_optimizer_steps),
            "batch_size": int(config.batch_size),
            "grad_accum_steps": int(config.grad_accum_steps),
            "oracle_monitor_every_optimizer_steps": int(
                config.oracle_monitor_every_optimizer_steps
            ),
            "log_every_steps": int(config.log_every_steps),
            "torch_profiler_enabled": True,
            "record_shapes": True,
            "profile_memory": True,
            "per_micro_step_cuda_event_resolution": True,
            "chrome_trace_path": str(trace_path),
            "usage": "hotspot attribution only",
        },
        "trace_observation": {
            "instrumented_wall_seconds": float(measurement_wall),
            "warning": (
                "instrumented trace timing is not benchmark throughput"
            ),
        },
        "memory": _memory_result(device, gate_eligible=False),
        "timing_method": {
            "stage_cpu": (
                "host dispatch wall time with record_function instrumentation"
            ),
            "stage_cuda": (
                "CUDA event elapsed time resolved once per micro-step"
            ),
            "total_wall": (
                "instrumented perf_counter including profiler synchronization"
            ),
            "gate_eligible": False,
        },
        "stages": profiler.stage_summary(),
        "top_cpu_ops": _top_operations(
            trace,
            sort_field="self_cpu_time_total",
            limit=20,
        ),
        "top_cuda_ops": _top_operations(
            trace,
            sort_field=(
                "self_device_time_total"
                if device.type == "cuda"
                else "self_cpu_time_total"
            ),
            limit=20,
        ),
    }


def build_benchmark_comparison(
    reference: Mapping[str, object],
    optimized: Mapping[str, object],
    *,
    reference_path: str,
    optimized_path: str,
) -> Dict[str, object]:
    """Compare only uninstrumented benchmark results.

    Passing a trace-profile payload is a contract violation, not a soft
    fallback.  This prevents instrumented timings from reaching the 2x,
    5 images/s, or memory-margin decisions.
    """

    for name, payload in (
        ("reference", reference),
        ("optimized", optimized),
    ):
        if payload.get("result_kind") != "benchmark":
            raise ValueError(
                f"{name} input is not a benchmark JSON"
            )
        if not bool(payload.get("gate_eligible")):
            raise ValueError(
                f"{name} benchmark is marked gate_eligible=false"
            )

    mismatches: list[str] = []
    if reference.get("status") != "PASS":
        mismatches.append("reference status is not PASS")
    if optimized.get("status") != "PASS":
        mismatches.append("optimized status is not PASS")

    reference_config = dict(reference.get("config", {}))
    optimized_config = dict(optimized.get("config", {}))
    if reference_config.get("efficiency_mode") != "reference":
        mismatches.append("reference benchmark mode is not reference")
    if optimized_config.get("efficiency_mode") != "optimized":
        mismatches.append("optimized benchmark mode is not optimized")
    reference_config.pop("efficiency_mode", None)
    optimized_config.pop("efficiency_mode", None)
    if reference_config != optimized_config:
        mismatches.append("config differs beyond efficiency_mode")

    if reference.get("benchmark_contract") != optimized.get(
        "benchmark_contract"
    ):
        mismatches.append("benchmark contract differs")
    if reference.get("data_contract") != optimized.get("data_contract"):
        mismatches.append("data or observed batch sequence differs")
    if reference.get("environment") != optimized.get("environment"):
        mismatches.append("runtime environment differs")
    if reference.get("worktree") != optimized.get("worktree"):
        mismatches.append("worktree identity differs")

    comparable = not mismatches
    reference_images_s = float(
        dict(reference.get("throughput", {})).get(
            "images_per_second",
            0.0,
        )
    )
    optimized_images_s = float(
        dict(optimized.get("throughput", {})).get(
            "images_per_second",
            0.0,
        )
    )
    speedup = optimized_images_s / max(reference_images_s, 1.0e-12)
    return {
        "version": "MOD10A",
        "source_policy": "benchmark_json_only",
        "reference_benchmark": str(reference_path),
        "optimized_benchmark": str(optimized_path),
        "comparable_contract": bool(comparable),
        "comparison_mismatches": mismatches,
        "reference_images_per_second": reference_images_s,
        "optimized_images_per_second": optimized_images_s,
        "speedup": float(speedup),
        "hard_2x_gate_passed": bool(comparable and speedup >= 2.0),
        "recommended_5_images_per_second_passed": bool(
            comparable and optimized_images_s >= 5.0
        ),
        "memory_margin_source": "benchmark CUDA peak API only",
        "reference_memory": reference.get("memory", {}),
        "optimized_memory": optimized.get("memory", {}),
    }
