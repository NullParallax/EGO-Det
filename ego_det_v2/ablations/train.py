from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..engine.train_loop import fit, set_runtime_defaults
from ..launch._common import (
    choose_device,
    make_loader,
    print_config,
    resolve_paths,
)
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from .registry import (
    ABLATION_TABLES,
    INSTRUMENTATION_FIELDS,
    METHOD_FIELDS,
    build_variant_config,
    config_fingerprint,
    control_fingerprint,
    get_ablation,
)


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    partial.replace(path)
    return path


def _checkpoint_config(
    checkpoint: Mapping[str, object],
    path: Path,
) -> EgoDetV2Config:
    values = checkpoint.get("config", {})
    if not isinstance(values, Mapping):
        raise ValueError(f"checkpoint config is not a mapping: {path}")
    allowed = EgoDetV2Config.__dataclass_fields__
    return EgoDetV2Config(
        **{key: value for key, value in values.items() if key in allowed}
    )


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"reference manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"reference manifest is not a JSON object: {path}")
    return payload


def _config_from_mapping(
    values: Mapping[str, object],
    *,
    source: Path,
) -> EgoDetV2Config:
    allowed = EgoDetV2Config.__dataclass_fields__
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(
            f"reference config has unknown fields at {source}: {unknown}"
        )
    return EgoDetV2Config(**dict(values))


def _load_reference_manifest(
    path: Path,
) -> tuple[EgoDetV2Config, dict[str, object]]:
    manifest = _read_json_object(path)
    if manifest.get("variant") != "full":
        raise RuntimeError(
            "--reference-manifest must describe the canonical full run; "
            f"got variant={manifest.get('variant')!r} at {path}"
        )
    values = manifest.get("resolved_config")
    if not isinstance(values, Mapping):
        raise ValueError(f"reference manifest has no resolved_config: {path}")
    config = _config_from_mapping(values, source=path)
    expected = str(manifest.get("config_fingerprint", ""))
    observed = config_fingerprint(config, exclude=("output_dir",))
    if expected and expected != observed:
        raise RuntimeError(
            "reference manifest config fingerprint is invalid; do not edit "
            f"resolved_config by hand: {path}"
        )
    default_full = EgoDetV2Config().to_dict()
    values_now = config.to_dict()
    method_mismatches = {
        key: (default_full[key], values_now[key])
        for key in METHOD_FIELDS
        if default_full[key] != values_now[key]
    }
    if method_mismatches:
        raise RuntimeError(
            "reference manifest is not the canonical full method: "
            + json.dumps(
                method_mismatches,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if manifest.get("status") not in {
        "imported_complete",
        "training_complete",
    }:
        raise RuntimeError(
            "reference full run is not complete; register the finished full "
            f"run first: status={manifest.get('status')!r}"
        )
    return config, manifest


def _validate_resume(
    path: Path,
    config: EgoDetV2Config,
    *,
    allow_non_epoch_resume: bool,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    epoch_name = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
    safe_epoch_boundary = (
        epoch_name is not None
        and int(epoch_name.group(1)) == checkpoint_epoch
        and checkpoint_epoch >= 1
    )
    if not safe_epoch_boundary and not allow_non_epoch_resume:
        raise RuntimeError(
            "controlled ablations may resume only from an epoch-boundary "
            "checkpoint named epoch_N.pth with matching metadata. The core "
            "trainer cannot restore a mid-epoch dataloader position; use the "
            "last completed epoch_N.pth or explicitly pass "
            "--allow-non-epoch-resume for engineering-only recovery."
        )
    previous = _checkpoint_config(checkpoint, path)
    previous_values = previous.to_dict()
    current_values = config.to_dict()
    mismatches = {
        key: (previous_values[key], current_values[key])
        for key in previous_values
        if key != "output_dir" and previous_values[key] != current_values[key]
    }
    if mismatches:
        raise RuntimeError(
            "resume checkpoint does not match this controlled ablation: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _runtime_updates(
    args: argparse.Namespace,
    base: EgoDetV2Config,
    output_dir: Path,
) -> dict[str, object]:
    updates: dict[str, object] = {"output_dir": str(output_dir)}
    for argument, field in (
        ("batch_size", "batch_size"),
        ("grad_accum", "grad_accum_steps"),
        ("epochs", "epochs"),
        ("lr_drop_epoch", "lr_drop_epoch"),
        ("train_max_size", "train_max_size"),
        ("seed", "seed"),
        ("num_workers", "num_workers"),
        ("efficiency_mode", "efficiency_mode"),
        ("full_eval_last_epoch_only", "full_eval_last_epoch_only"),
        (
            "oracle_monitor_every_optimizer_steps",
            "oracle_monitor_every_optimizer_steps",
        ),
        ("quick_eval_every_steps", "quick_eval_every_steps"),
        ("checkpoint_every_steps", "ckpt_every_steps"),
        ("log_every_steps", "log_every_steps"),
    ):
        value = getattr(args, argument)
        if value is not None:
            updates[field] = value
    if args.train_short_sides is not None:
        updates["train_short_sides"] = tuple(args.train_short_sides)
    if args.image_short_max is not None:
        retained = tuple(
            value
            for value in base.train_short_sides
            if value <= int(args.image_short_max)
        )
        updates["train_short_sides"] = retained or (
            int(args.image_short_max),
        )
    if args.use_ema:
        updates["use_ema"] = True
    if args.overlock_checkpointing is not None:
        updates["overlock_gradient_checkpointing"] = bool(
            args.overlock_checkpointing
        )
    if args.freeze_backbone is not None:
        updates["freeze_backbone"] = bool(args.freeze_backbone)
    return updates


def _validate_reference_updates(
    base: EgoDetV2Config,
    updates: Mapping[str, object],
) -> None:
    base_values = base.to_dict()
    safe = INSTRUMENTATION_FIELDS | {"output_dir"}
    mismatches = {
        key: (base_values[key], value)
        for key, value in updates.items()
        if key not in safe and base_values[key] != value
    }
    if mismatches:
        raise RuntimeError(
            "runtime flags would change the completed full model's controlled "
            "training configuration. Remove these flags and inherit the "
            "reference manifest: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _normal_path(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _data_contract(
    args: argparse.Namespace,
    paths: object,
) -> dict[str, str]:
    return {
        "project_root": _normal_path(paths.root),
        "train_annotations": _normal_path(
            args.train_annotations or paths.coco_train_instances
        ),
        "val_annotations": _normal_path(
            args.val_annotations or paths.coco_val_instances
        ),
        "image_root": _normal_path(args.image_root or paths.coco_image_root),
    }


def _table_memberships(variant: str) -> list[str]:
    return [
        key
        for key, table in ABLATION_TABLES.items()
        if any(row.variant == variant for row in table.rows)
    ]


def _parse_train_short_sides(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--train-short-sides must be a comma-separated integer list"
        )
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--train-short-sides must be a comma-separated integer list"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one controlled EGO-Det component ablation"
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/ego_det_v2/ablations",
    )
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--val-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--device")
    parser.add_argument(
        "--reference-manifest",
        help=(
            "manifest produced by 'register-full'; all training-effective "
            "settings are inherited from the completed full run"
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-drop-epoch", type=int)
    parser.add_argument(
        "--train-short-sides",
        type=_parse_train_short_sides,
        help="comma-separated explicit training short sides, e.g. 448,480,512",
    )
    parser.add_argument("--train-max-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-short-max", type=int)
    parser.add_argument(
        "--efficiency-mode",
        choices=("reference", "optimized"),
    )
    parser.add_argument(
        "--oracle-monitor-every-optimizer-steps",
        type=int,
    )
    parser.add_argument("--quick-eval-every-steps", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int)
    parser.add_argument("--log-every-steps", type=int)
    parser.add_argument("--use-ema", action="store_true")
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--overlock-checkpointing",
        dest="overlock_checkpointing",
        action="store_true",
    )
    checkpointing.add_argument(
        "--no-overlock-checkpointing",
        dest="overlock_checkpointing",
        action="store_false",
    )
    parser.set_defaults(overlock_checkpointing=None)
    backbone_freezing = parser.add_mutually_exclusive_group()
    backbone_freezing.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
    )
    backbone_freezing.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
    )
    parser.set_defaults(freeze_backbone=None)
    parser.add_argument(
        "--full-eval-last-epoch-only",
        action="store_true",
        default=None,
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--allow-non-epoch-resume",
        action="store_true",
        help=(
            "allow latest/mid-epoch recovery even though exact dataloader "
            "position cannot be restored; do not use for reported tables"
        ),
    )
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    get_ablation(args.variant)
    for name in ("batch_size", "grad_accum", "epochs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.lr_drop_epoch is not None and args.lr_drop_epoch < 1:
        raise ValueError("--lr-drop-epoch must be >= 1")
    short_sides = args.train_short_sides
    if short_sides is not None:
        if not short_sides or any(side <= 0 for side in short_sides):
            raise ValueError("--train-short-sides values must be positive")
        if tuple(short_sides) != tuple(sorted(set(short_sides))):
            raise ValueError(
                "--train-short-sides must be strictly increasing without duplicates"
            )
        if args.image_short_max is not None:
            raise ValueError(
                "--train-short-sides and --image-short-max are mutually exclusive"
            )
    if args.train_max_size is not None:
        if args.train_max_size <= 0:
            raise ValueError("--train-max-size must be positive")
        effective_short_sides = (
            short_sides or EgoDetV2Config().train_short_sides
        )
        if args.train_max_size < max(effective_short_sides):
            raise ValueError(
                "--train-max-size must be >= max(--train-short-sides)"
            )
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.image_short_max is not None and args.image_short_max < 128:
        raise ValueError("--image-short-max must be >= 128")
    if args.smoke_steps < 0:
        raise ValueError("--smoke-steps must be >= 0")
    cadence = args.oracle_monitor_every_optimizer_steps
    if cadence is not None and cadence < -1:
        raise ValueError(
            "--oracle-monitor-every-optimizer-steps must be -1, 0, or positive"
        )
    for name in ("quick_eval_every_steps", "checkpoint_every_steps"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.log_every_steps is not None and args.log_every_steps < 1:
        raise ValueError("--log-every-steps must be >= 1")


def train_variant(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    spec = get_ablation(args.variant)
    if (
        spec.key != "full"
        and not args.reference_manifest
        and not args.dry_run
        and not args.smoke_steps
    ):
        raise RuntimeError(
            "formal non-full ablations require --reference-manifest from "
            "'register-full'; this prevents batch/scale/implementation drift"
        )
    output_dir = Path(args.output_root) / spec.key
    occupied = [
        path
        for path in (
            output_dir / "latest.pth",
            output_dir / "train_log.jsonl",
            output_dir / "eval_history.jsonl",
        )
        if path.exists()
    ]
    if occupied and not args.resume:
        raise RuntimeError(
            "ablation output already contains training state; use --resume "
            "with the matching checkpoint or choose a new --output-root: "
            + ", ".join(str(path) for path in occupied)
        )

    reference_manifest: dict[str, object] | None = None
    reference_path: Path | None = None
    if args.reference_manifest:
        reference_path = Path(args.reference_manifest).expanduser().resolve()
        base, reference_manifest = _load_reference_manifest(reference_path)
    else:
        base = EgoDetV2Config()
    runtime_updates = _runtime_updates(args, base, output_dir)
    if reference_manifest is not None:
        _validate_reference_updates(base, runtime_updates)
    config = build_variant_config(
        spec.key,
        base=base,
        extra_updates=runtime_updates,
    )
    if args.resume:
        _validate_resume(
            Path(args.resume),
            config,
            allow_non_epoch_resume=args.allow_non_epoch_resume,
        )

    paths = resolve_paths(args.project_root)
    data = _data_contract(args, paths)
    if reference_manifest is not None:
        reference_data = reference_manifest.get("data")
        if not isinstance(reference_data, Mapping):
            raise RuntimeError(
                f"reference manifest has no data contract: {reference_path}"
            )
        reference_data_normalized = {
            key: _normal_path(value)
            for key, value in reference_data.items()
            if key in data
        }
        if reference_data_normalized != data:
            raise RuntimeError(
                "ablation data paths do not match the registered full run: "
                + json.dumps(
                    {
                        "reference": reference_data_normalized,
                        "current": data,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": 2,
        "variant": spec.key,
        "label": spec.label,
        "description": spec.description,
        "tables": _table_memberships(spec.key),
        "status": "planned" if args.dry_run else "running",
        "created_at_utc": now,
        "updated_at_utc": now,
        "independent_initialization": args.resume is None,
        "resume": str(Path(args.resume)) if args.resume else None,
        "non_epoch_resume_allowed": bool(args.allow_non_epoch_resume),
        "pretrained_backbone": str(paths.overlock_weights),
        "reference_full": (
            {
                "manifest": str(reference_path),
                "source_checkpoint": reference_manifest.get(
                    "source_checkpoint"
                ),
                "source_eval_history": reference_manifest.get(
                    "source_eval_history"
                ),
            }
            if reference_manifest is not None
            else None
        ),
        "method_overrides": spec.override_dict(),
        "method_fields": {
            key: config.to_dict()[key]
            for key in sorted(METHOD_FIELDS)
        },
        "control_fingerprint": control_fingerprint(config),
        "config_fingerprint": config_fingerprint(
            config,
            exclude=("output_dir",),
        ),
        "resolved_config": config.to_dict(),
        "data": data,
    }
    manifest_path = _write_json(
        output_dir / "ablation_manifest.json",
        manifest,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return manifest

    print_config(config)
    set_runtime_defaults(config.seed)
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
    try:
        final_summary = fit(
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
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        raise
    manifest["status"] = (
        "smoke_complete" if args.smoke_steps else "training_complete"
    )
    manifest["final_training_summary"] = final_summary
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = train_variant(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
