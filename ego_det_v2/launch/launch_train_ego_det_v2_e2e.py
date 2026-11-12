from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional, Sequence

from ..configs.config import EgoDetV2Config
from ..engine.train_loop import fit, set_runtime_defaults
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, print_config, resolve_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal EGO-Det V2 end-to-end training")
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--val-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-drop-epoch", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--image-short-max", type=int)
    parser.add_argument(
        "--efficiency-mode",
        choices=("reference", "optimized"),
        default=None,
    )
    parser.add_argument(
        "--oracle-monitor-every-optimizer-steps",
        type=int,
        default=None,
    )
    parser.add_argument("--bank-topk", type=int)
    parser.add_argument("--evidence-per-slot", type=int)
    parser.add_argument("--use-ema", action="store_true")
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
    parser.set_defaults(overlock_checkpointing=None)
    parser.add_argument("--device")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress bars while retaining JSONL logs",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum is not None and args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.image_short_max is not None and args.image_short_max < 128:
        raise ValueError("--image-short-max must be >= 128")
    if (
        args.oracle_monitor_every_optimizer_steps is not None
        and args.oracle_monitor_every_optimizer_steps < -1
    ):
        raise ValueError(
            "--oracle-monitor-every-optimizer-steps must be -1, 0, or positive"
        )
    return args


def main() -> None:
    args = parse_args()
    config = EgoDetV2Config()
    updates = {}
    for argument, field in (
        ("batch_size", "batch_size"),
        ("grad_accum", "grad_accum_steps"),
        ("epochs", "epochs"),
        ("lr_drop_epoch", "lr_drop_epoch"),
        ("output_dir", "output_dir"),
        ("bank_topk", "bank_topk_train"),
        ("evidence_per_slot", "evidence_per_slot"),
    ):
        value = getattr(args, argument)
        if value is not None:
            updates[field] = value
    if args.image_short_max is not None:
        retained = tuple(
            value for value in config.train_short_sides if value <= args.image_short_max
        )
        updates["train_short_sides"] = retained or (int(args.image_short_max),)
    if args.efficiency_mode is not None:
        updates["efficiency_mode"] = args.efficiency_mode
    if args.oracle_monitor_every_optimizer_steps is not None:
        updates["oracle_monitor_every_optimizer_steps"] = int(
            args.oracle_monitor_every_optimizer_steps
        )
    if args.use_ema:
        updates["use_ema"] = True
    if args.overlock_checkpointing is not None:
        updates["overlock_gradient_checkpointing"] = bool(
            args.overlock_checkpointing
        )
    config = replace(config, **updates)
    print_config(config)
    set_runtime_defaults(config.seed)

    paths = resolve_paths(args.project_root)
    train_loader, _ = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        shuffle=True,
    )
    val_loader, val_dataset = make_loader(
        "val",
        paths=paths,
        config=config,
        annotation_file=args.val_annotations,
        image_root=args.image_root,
        shuffle=False,
    )
    device = choose_device(args.device)
    model = EgoDetV2(config, paths=paths).to(device)
    criterion = EgoDetV2Criterion(config).to(device)
    fit(
        model,
        criterion,
        train_loader,
        config=config,
        device=device,
        val_loader=val_loader,
        val_dataset=val_dataset,
        resume=args.resume,
        smoke_steps=args.smoke_steps,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
