from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..engine.throughput_profiler import (
    benchmark_training_throughput,
    build_benchmark_comparison,
    trace_training_profile,
)
from ..engine.train_loop import set_runtime_defaults
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, print_config, resolve_paths


def _boolean(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark formal EGO-Det V2 throughput or collect a trace"
        )
    )
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--image-root")
    parser.add_argument(
        "--output-dir",
        default="outputs/ego_det_v2/throughput_profile",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--image-short-max", type=int, default=608)
    parser.add_argument("--warmup-micro-steps", type=int, default=10)
    parser.add_argument("--measure-micro-steps", type=int, default=40)
    parser.add_argument(
        "--profile-kind",
        choices=("benchmark", "trace"),
        default="benchmark",
        help=(
            "benchmark measures gate-eligible throughput without profiler "
            "instrumentation; trace collects instrumented hotspot evidence"
        ),
    )
    parser.add_argument(
        "--efficiency-mode",
        "--mode",
        dest="efficiency_mode",
        choices=("reference", "optimized"),
        default="reference",
    )
    parser.add_argument(
        "--overlock-checkpointing",
        type=_boolean,
        default=True,
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=6,
        help=(
            "launcher-only cumulative optimized ablation level 0..6; "
            "formal optimized mode uses 6"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--oracle-monitor-every-optimizer-steps",
        type=int,
        default=-1,
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.batch_size * args.grad_accum != 16:
        raise ValueError(
            "throughput A/B requires effective batch 16: "
            "--batch-size * --grad-accum must equal 16"
        )
    if args.image_short_max < 128:
        raise ValueError("--image-short-max must be >= 128")
    if args.warmup_micro_steps < 0:
        raise ValueError("--warmup-micro-steps must be >= 0")
    minimum_measure_steps = 40 if args.profile_kind == "benchmark" else 1
    if args.measure_micro_steps < minimum_measure_steps:
        raise ValueError(
            "--measure-micro-steps must be >= "
            f"{minimum_measure_steps} for --profile-kind {args.profile_kind}"
        )
    if args.measure_micro_steps % args.grad_accum != 0:
        raise ValueError(
            "--measure-micro-steps must be divisible by --grad-accum "
            "so every measured optimizer-step window is complete"
        )
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if not 0 <= args.optimization_level <= 6:
        raise ValueError("--optimization-level must be in [0, 6]")
    if args.oracle_monitor_every_optimizer_steps < -1:
        raise ValueError(
            "--oracle-monitor-every-optimizer-steps must be -1, 0, or positive"
        )
    return args


def profile_output_paths(
    output_dir: Path,
    *,
    efficiency_mode: str,
    optimization_level: int,
    profile_kind: str,
) -> tuple[Path, Optional[Path]]:
    stem = str(efficiency_mode)
    if efficiency_mode == "optimized" and int(optimization_level) < 6:
        stem = f"optimized_level{int(optimization_level)}"
    if profile_kind == "benchmark":
        return output_dir / f"{stem}_benchmark.json", None
    if profile_kind == "trace":
        return (
            output_dir / f"{stem}_trace_profile.json",
            output_dir / f"{stem}_chrome_trace.json",
        )
    raise ValueError(f"unsupported profile_kind: {profile_kind!r}")


def main() -> None:
    args = parse_args()
    base = EgoDetV2Config()
    short_sides = tuple(
        side
        for side in base.train_short_sides
        if side <= int(args.image_short_max)
    )
    if not short_sides:
        short_sides = (int(args.image_short_max),)
    updates = {
        "output_dir": str(args.output_dir),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum),
        "train_short_sides": short_sides,
        "efficiency_mode": str(args.efficiency_mode),
        "efficiency_ablation_level": int(args.optimization_level),
        "overlock_gradient_checkpointing": bool(
            args.overlock_checkpointing
        ),
        "oracle_monitor_every_optimizer_steps": int(
            args.oracle_monitor_every_optimizer_steps
        ),
    }
    if args.num_workers is not None:
        updates["num_workers"] = int(args.num_workers)
    config = replace(base, **updates)
    print_config(config)
    print(
        "[EgoDetV2 throughput profile] "
        f"profile_kind={args.profile_kind} "
        f"mode={config.efficiency_mode} "
        f"optimization_level={config.efficiency_ablation_level} "
        f"checkpointing={config.overlock_gradient_checkpointing} "
        f"warmup_micro_steps={args.warmup_micro_steps} "
        f"measure_micro_steps={args.measure_micro_steps}",
        flush=True,
    )
    set_runtime_defaults(config.seed)
    paths = resolve_paths(args.project_root)
    loader, _ = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        shuffle=True,
    )
    device = choose_device(args.device)
    model = EgoDetV2(config, paths=paths).to(device)
    criterion = EgoDetV2Criterion(config).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path, chrome_trace_path = profile_output_paths(
        output_dir,
        efficiency_mode=config.efficiency_mode,
        optimization_level=int(config.efficiency_ablation_level),
        profile_kind=str(args.profile_kind),
    )
    try:
        if args.profile_kind == "benchmark":
            result = benchmark_training_throughput(
                model,
                criterion,
                loader,
                config=config,
                device=device,
                warmup_micro_steps=int(args.warmup_micro_steps),
                measure_micro_steps=int(args.measure_micro_steps),
            )
        else:
            if chrome_trace_path is None:
                raise AssertionError("trace mode requires a Chrome trace path")
            result = trace_training_profile(
                model,
                criterion,
                loader,
                config=config,
                device=device,
                warmup_micro_steps=int(args.warmup_micro_steps),
                measure_micro_steps=int(args.measure_micro_steps),
                trace_path=chrome_trace_path,
            )
    except torch.cuda.OutOfMemoryError as exc:
        allocated = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        reserved = (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else 0
        )
        result = {
            "status": "OOM",
            "version": "MOD10A",
            "result_kind": (
                "benchmark"
                if args.profile_kind == "benchmark"
                else "trace_profile"
            ),
            "gate_eligible": args.profile_kind == "benchmark",
            "memory_margin_eligible": args.profile_kind == "benchmark",
            "decision_eligibility": {
                "hard_2x_gate": args.profile_kind == "benchmark",
                "five_images_per_second_target": (
                    args.profile_kind == "benchmark"
                ),
                "checkpointing_ab": args.profile_kind == "benchmark",
                "batch_ab": args.profile_kind == "benchmark",
                "memory_margin": args.profile_kind == "benchmark",
            },
            "error": str(exc),
            "config": config.to_dict(),
            f"{args.profile_kind}_contract": {
                "warmup_micro_steps": int(args.warmup_micro_steps),
                "measure_micro_steps": int(args.measure_micro_steps),
            },
            "memory": {
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
            },
        }
        if (
            chrome_trace_path is not None
            and not chrome_trace_path.exists()
        ):
            chrome_trace_path.write_text(
                json.dumps(
                    {"status": "OOM", "trace_available": False},
                    indent=2,
                ),
                encoding="utf-8",
            )
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if (
        args.profile_kind == "benchmark"
        and result.get("status") == "PASS"
        and (
            config.efficiency_mode == "reference"
            or int(config.efficiency_ablation_level) == 6
        )
    ):
        reference_path = output_dir / "reference_benchmark.json"
        optimized_path = output_dir / "optimized_benchmark.json"
        if reference_path.is_file() and optimized_path.is_file():
            reference = json.loads(
                reference_path.read_text(encoding="utf-8")
            )
            optimized = json.loads(
                optimized_path.read_text(encoding="utf-8")
            )
            comparison = build_benchmark_comparison(
                reference,
                optimized,
                reference_path=str(reference_path),
                optimized_path=str(optimized_path),
            )
            (output_dir / "throughput_comparison.json").write_text(
                json.dumps(
                    comparison,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                "[EgoDetV2 throughput comparison] "
                + json.dumps(comparison, ensure_ascii=False),
                flush=True,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if result.get("status") != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
