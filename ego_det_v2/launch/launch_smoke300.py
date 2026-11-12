from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..engine.train_loop import build_optimizer, run_fixed_steps, set_runtime_defaults
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, print_config, resolve_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EGO-Det V2 real-data 300-step smoke")
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir", default="outputs/ego_det_v2/smoke300")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--image-short-max", type=int, default=512)
    parser.add_argument(
        "--efficiency-mode",
        choices=("reference", "optimized"),
        default="reference",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--overlock-checkpointing",
        dest="overlock_checkpointing",
        action="store_true",
    )
    checkpoint_group.add_argument(
        "--no-overlock-checkpointing",
        dest="overlock_checkpointing",
        action="store_false",
    )
    parser.set_defaults(overlock_checkpointing=True)
    parser.add_argument("--device")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress bars",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.image_short_max < 128:
        raise ValueError("--image-short-max must be >= 128")
    return args


def main() -> None:
    args = parse_args()
    base = EgoDetV2Config()
    allowed = tuple(
        side for side in base.train_short_sides
        if side <= int(args.image_short_max)
    )
    if not allowed:
        allowed = (int(args.image_short_max),)
    config = replace(
        base,
        output_dir=args.output_dir,
        train_short_sides=allowed,
        train_max_size=min(base.train_max_size, 768),
        batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum),
        overlock_gradient_checkpointing=bool(args.overlock_checkpointing),
        efficiency_mode=args.efficiency_mode,
    )
    print_config(config)
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
    optimizer = build_optimizer(model, config)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    metrics = run_fixed_steps(
        model,
        criterion,
        loader,
        optimizer,
        config=config,
        device=device,
        steps=args.steps,
        show_progress=not args.no_progress,
        progress_desc="Smoke",
    )
    # Throughput is deliberately present only as a recorded field.
    metrics["throughput_contract_gate"] = False
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke300_result.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    passed = (
        not bool(metrics["nonfinite"])
        and metrics["last_window_loss"] < metrics["first_window_loss"]
        and (metrics["peak_memory_gb"] == 0.0 or metrics["peak_memory_gb"] < 21.0)
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
