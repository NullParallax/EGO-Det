from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from ..configs.config import EgoDetV2Config
from ..configs.project import (
    load_settings, model_from_settings, path_from_settings, resolve_root,
)
from dataclasses import replace as dataclass_replace
from ..engine.train_loop import fit, set_runtime_defaults
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, print_config, resolve_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formal EGO-Det V2 end-to-end training")
    parser.add_argument("--config", help="YAML configuration (default: configs/egodet.yaml)")
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
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use-ema", dest="use_ema", action="store_true", default=None)
    ema_group.add_argument("--no-ema", dest="use_ema", action="store_false")
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
    parser.add_argument("--smoke-steps", type=int, default=None)
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
    settings = load_settings(args.config)
    config = model_from_settings(settings)
    root = resolve_root(args.project_root)
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
    if args.use_ema is not None:
        updates["use_ema"] = bool(args.use_ema)
    if args.overlock_checkpointing is not None:
        updates["overlock_gradient_checkpointing"] = bool(
            args.overlock_checkpointing
        )
    config = replace(config, **updates)
    set_runtime_defaults(config.seed)

    paths = resolve_paths(str(root))
    pretrained = path_from_settings(settings, "overlock_weights", root)
    if pretrained is not None:
        paths = dataclass_replace(paths, overlock_weights=pretrained)
    train_annotation = path_from_settings(settings, "train_annotations", root, args.train_annotations)
    val_annotation = path_from_settings(settings, "val_annotations", root, args.val_annotations)
    images = path_from_settings(settings, "image_root", root, args.image_root)
    # Resolve the portable YAML run destination against the chosen project root.
    if args.output_dir is None:
        output = Path(config.output_dir).expanduser()
        config = replace(config, output_dir=str((root / output).resolve() if not output.is_absolute() else output))
    print_config(config)
    train_loader, _ = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=train_annotation,
        image_root=images,
        shuffle=True,
    )
    val_loader, val_dataset = make_loader(
        "val",
        paths=paths,
        config=config,
        annotation_file=val_annotation,
        image_root=images,
        shuffle=False,
    )
    training = settings["training"]
    device = choose_device(args.device or training.get("device"))
    resume = args.resume if args.resume is not None else training.get("resume")
    if resume:
        resume = str((root / resume).resolve()) if not Path(resume).is_absolute() else str(resume)
    smoke_steps = args.smoke_steps if args.smoke_steps is not None else int(training.get("smoke_steps", 0))
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
        resume=resume,
        smoke_steps=smoke_steps,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
