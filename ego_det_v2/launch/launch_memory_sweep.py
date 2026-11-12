from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..engine.train_loop import (
    build_optimizer,
    build_optimizer_step_scheduler,
    run_fixed_steps,
    set_runtime_defaults,
)
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, resolve_paths


def memory_sweep_cases() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (batch_size, short_side, max_size)
        for batch_size in (1, 2, 4)
        for short_side in (512, 608)
        for max_size in (768, 896)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EGO-Det V2 one-step geometry/full CUDA memory sweep"
    )
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        default="outputs/ego_det_v2/memory_sweep_mod8.json",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable one-step progress bars",
    )
    parser.add_argument(
        "--efficiency-mode",
        choices=("reference", "optimized"),
        default="reference",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).lower()
    )


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device}")
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    """Initialize the CUDA context, clear reusable cache, then reset peaks.

    PyTorch 2.1 can raise ``Invalid device argument ... did you call init?``
    when ``reset_peak_memory_stats`` is the first CUDA operation in a process.
    Creating a one-element tensor on the selected device forces context
    initialization before the memory-stat API is used.
    """
    if device.type != "cuda":
        return
    device_index = _cuda_device_index(device)
    torch.cuda.set_device(device_index)
    context_probe = torch.empty(1, device=device, dtype=torch.float32)
    torch.cuda.synchronize(device_index)
    del context_probe
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)


def _run_case(
    *,
    batch_size: int,
    short_side: int,
    max_size: int,
    image_ids: Sequence[int],
    paths,
    args: argparse.Namespace,
    device: torch.device,
    case_state: Dict[str, object],
) -> Dict[str, object]:
    config = replace(
        EgoDetV2Config(),
        output_dir=str(Path(args.output).parent),
        num_workers=0,
        batch_size=int(batch_size),
        grad_accum_steps=1,
        train_short_sides=(int(short_side),),
        train_max_size=int(max_size),
        val_short_side=int(short_side),
        val_max_size=int(max_size),
        random_crop_prob=0.0,
        hflip_prob=0.0,
        warmup_steps=25,
        lr=3.0e-4,
        backbone_lr=1.0e-5,
        backbone_stem_lr=1.0e-6,
        weight_decay=0.0,
        overlock_gradient_checkpointing=True,
        efficiency_mode=str(args.efficiency_mode),
    )
    loader, _ = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        image_ids=image_ids,
        training_override=False,
        shuffle=False,
        batch_size=batch_size,
    )
    first_images, _, _ = next(iter(loader))
    input_shape = [int(value) for value in first_images.shape]
    case_state["input_shape"] = input_shape
    del first_images
    if device.type == "cuda":
        # Reset before model construction as well, so an OOM during .to(device)
        # is attributed to this case instead of inheriting the prior case peak.
        _reset_cuda_peak_memory_stats(device)
    model = EgoDetV2(config, paths=paths).to(device)
    criterion = EgoDetV2Criterion(config).to(device)
    optimizer = build_optimizer(
        model,
        config,
        base_lr=3.0e-4,
        geometry_lr=3.0e-4,
        fpn_lr=1.0e-4,
        backbone_lr=1.0e-5,
        backbone_stem_lr=1.0e-6,
        weight_decay=0.0,
    )
    scheduler = build_optimizer_step_scheduler(optimizer, warmup_steps=25)

    if device.type == "cuda":
        _reset_cuda_peak_memory_stats(device)
    geometry_start = time.perf_counter()
    geometry_metrics = run_fixed_steps(
        model,
        criterion,
        loader,
        optimizer,
        config=config,
        device=device,
        steps=1,
        geometry_only=True,
        scheduler=scheduler,
        show_progress=not args.no_progress,
        progress_desc=f"Sweep b{batch_size} {short_side}/{max_size} geometry",
    )
    geometry_seconds = time.perf_counter() - geometry_start
    case_state["geometry_step_seconds"] = geometry_seconds
    geometry_peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    geometry_peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )

    if device.type == "cuda":
        _reset_cuda_peak_memory_stats(device)
    full_start = time.perf_counter()
    full_metrics = run_fixed_steps(
        model,
        criterion,
        loader,
        optimizer,
        config=config,
        device=device,
        steps=1,
        epoch=2,
        geometry_only=False,
        scheduler=scheduler,
        global_step_offset=1,
        show_progress=not args.no_progress,
        progress_desc=f"Sweep b{batch_size} {short_side}/{max_size} full",
    )
    full_seconds = time.perf_counter() - full_start
    case_state["full_step_seconds"] = full_seconds
    full_peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    full_peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )
    total_seconds = max(geometry_seconds + full_seconds, 1.0e-9)
    peak_allocated = max(geometry_peak_allocated, full_peak_allocated)
    peak_reserved = max(geometry_peak_reserved, full_peak_reserved)
    return {
        "batch_size": int(batch_size),
        "short_side": int(short_side),
        "max_size": int(max_size),
        "status": "PASS",
        "oom": False,
        "input_shape": input_shape,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gb": peak_allocated / float(1024**3),
        "peak_reserved_gb": peak_reserved / float(1024**3),
        "geometry_step_seconds": geometry_seconds,
        "full_step_seconds": full_seconds,
        "samples_per_second": (2.0 * batch_size) / total_seconds,
        "geometry_metrics": geometry_metrics,
        "full_metrics": full_metrics,
    }


def main() -> None:
    args = parse_args()
    print(
        f"[EgoDetV2 efficiency] mode={args.efficiency_mode}",
        flush=True,
    )
    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "launch_memory_sweep requires a CUDA device for meaningful memory evidence"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no memory-sweep result was fabricated")
    set_runtime_defaults(EgoDetV2Config().seed)
    paths = resolve_paths(args.project_root)
    discovery_config = replace(
        EgoDetV2Config(),
        num_workers=0,
        batch_size=1,
        val_short_side=512,
        val_max_size=768,
        random_crop_prob=0.0,
        hflip_prob=0.0,
        efficiency_mode=str(args.efficiency_mode),
    )
    discovery_loader, discovery_dataset = make_loader(
        "train",
        paths=paths,
        config=discovery_config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        training_override=False,
        shuffle=False,
        batch_size=1,
    )
    del discovery_loader
    image_ids = [int(value) for value in discovery_dataset.image_ids[:8]]
    if len(image_ids) != 8:
        raise RuntimeError(f"memory sweep requires eight images, found {len(image_ids)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[Dict[str, object]] = []
    for batch_size, short_side, max_size in memory_sweep_cases():
        case = {
            "batch_size": batch_size,
            "short_side": short_side,
            "max_size": max_size,
        }
        try:
            result = _run_case(
                batch_size=batch_size,
                short_side=short_side,
                max_size=max_size,
                image_ids=image_ids,
                paths=paths,
                args=args,
                device=device,
                case_state=case,
            )
        except BaseException as exc:
            if not _is_oom(exc):
                raise
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            result = {
                **case,
                "status": "OOM",
                "oom": True,
                "input_shape": case.get("input_shape"),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "peak_allocated_gb": peak_allocated / float(1024**3),
                "peak_reserved_gb": peak_reserved / float(1024**3),
                "geometry_step_seconds": case.get("geometry_step_seconds"),
                "full_step_seconds": case.get("full_step_seconds"),
                "samples_per_second": 0.0,
                "error": str(exc),
            }
        results.append(result)
        candidate = next(
            (
                item
                for item in results
                if item["batch_size"] == 4
                and item["short_side"] == 512
                and item["max_size"] == 768
            ),
            None,
        )
        payload = {
            "device": str(device),
            "image_ids": image_ids,
            "cases": results,
            "candidate_batch4_512_768_proven": bool(
                candidate is not None
                and candidate.get("status") == "PASS"
                and not candidate.get("oom", False)
            ),
            "batch4_at_672_was_tested": False,
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            "[EgoDetV2 memory sweep] "
            + json.dumps(result, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
    print(output.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
