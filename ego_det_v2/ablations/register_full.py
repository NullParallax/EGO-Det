from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..launch._common import resolve_paths
from .registry import (
    METHOD_FIELDS,
    config_fingerprint,
    control_fingerprint,
)
from .summarize import _read_jsonl, _select_evaluation
from .train import (
    _checkpoint_config,
    _data_contract,
    _table_memberships,
    _write_json,
)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    partial.replace(path)
    return path


def _validate_full_method(config: EgoDetV2Config) -> None:
    defaults = EgoDetV2Config().to_dict()
    values = config.to_dict()
    mismatches = {
        key: (defaults[key], values[key])
        for key in METHOD_FIELDS
        if defaults[key] != values[key]
    }
    if mismatches:
        raise RuntimeError(
            "source checkpoint is not the canonical full EGO-Det method: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Register a completed full EGO-Det run as the shared ablation "
            "control without retraining it"
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument(
        "--checkpoint",
        help="default: <source-run>/epoch_<configured epochs>.pth",
    )
    parser.add_argument(
        "--eval-history",
        help="default: <source-run>/eval_history.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/ego_det_v2/ablations",
    )
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--val-annotations")
    parser.add_argument("--image-root")
    parser.add_argument(
        "--expected-ap",
        type=float,
        help=(
            "optional percentage-scale AP guard, e.g. 45.3; tolerance is "
            "controlled by --ap-tolerance"
        ),
    )
    parser.add_argument("--ap-tolerance", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    return parser


def register_full(args: argparse.Namespace) -> dict[str, object]:
    source_run = Path(args.source_run).expanduser().resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f"full source run not found: {source_run}")
    eval_history = Path(
        args.eval_history or source_run / "eval_history.jsonl"
    ).expanduser().resolve()
    if not eval_history.exists():
        raise FileNotFoundError(f"full eval history not found: {eval_history}")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    else:
        candidates = []
        for path in source_run.glob("epoch_*.pth"):
            match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError(
                f"no epoch_N.pth checkpoint found under {source_run}"
            )
        checkpoint_path = max(candidates, key=lambda item: item[0])[1]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"full checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
    config = _checkpoint_config(checkpoint, checkpoint_path)
    _validate_full_method(config)
    expected_epoch = int(config.epochs)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch != expected_epoch:
        raise RuntimeError(
            "full checkpoint is not the configured final epoch: "
            f"checkpoint epoch={checkpoint_epoch}, config epochs={expected_epoch}"
        )
    if "model" not in checkpoint:
        raise RuntimeError(f"checkpoint has no raw model state: {checkpoint_path}")
    if bool(config.use_ema):
        raise RuntimeError(
            "registered full config uses EMA, but this ablation suite is for "
            "the confirmed raw-model control"
        )

    evaluation, status = _select_evaluation(
        _read_jsonl(eval_history),
        expected_epoch=expected_epoch,
        allow_incomplete=False,
    )
    if evaluation is None or status != "complete":
        raise RuntimeError(
            f"no full-val evaluation found for epoch {expected_epoch}: "
            f"{eval_history}"
        )
    ap_percent = 100.0 * float(evaluation["AP"])
    if args.expected_ap is not None and abs(ap_percent - args.expected_ap) > float(
        args.ap_tolerance
    ):
        raise RuntimeError(
            "registered full AP does not match --expected-ap: "
            f"observed={ap_percent:.4f}, expected={args.expected_ap:.4f}, "
            f"tolerance={args.ap_tolerance:.4f}"
        )

    output_dir = Path(args.output_root).expanduser().resolve() / "full"
    manifest_path = output_dir / "ablation_manifest.json"
    history_path = output_dir / "eval_history.jsonl"
    occupied = [path for path in (manifest_path, history_path) if path.exists()]
    if occupied and not args.force:
        raise RuntimeError(
            "registered full output already exists; inspect it or pass --force: "
            + ", ".join(str(path) for path in occupied)
        )

    config = replace(config, output_dir=str(output_dir))
    paths = resolve_paths(args.project_root)
    data = _data_contract(args, paths)
    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": 2,
        "variant": "full",
        "label": "EGO-Det",
        "description": (
            "Completed full EGO-Det run reused as the shared ablation control."
        ),
        "tables": _table_memberships("full"),
        "status": "imported_complete",
        "created_at_utc": now,
        "updated_at_utc": now,
        "independent_initialization": True,
        "evaluation_state": "raw_model",
        "source_run": str(source_run),
        "source_checkpoint": str(checkpoint_path),
        "source_eval_history": str(eval_history),
        "source_train_log": (
            str((source_run / "train_log.jsonl").resolve())
            if (source_run / "train_log.jsonl").exists()
            else None
        ),
        "checkpoint": {
            "epoch": checkpoint_epoch,
            "global_step": checkpoint.get("global_step"),
            "global_optimizer_step": checkpoint.get(
                "global_optimizer_step"
            ),
            "micro_step": checkpoint.get("micro_step"),
            "size_bytes": checkpoint_path.stat().st_size,
            "ema_state_present": checkpoint.get("ema") is not None,
        },
        "method_overrides": {},
        "method_fields": {
            key: config.to_dict()[key] for key in sorted(METHOD_FIELDS)
        },
        "control_fingerprint": control_fingerprint(config),
        "config_fingerprint": config_fingerprint(
            config,
            exclude=("output_dir",),
        ),
        "resolved_config": config.to_dict(),
        "data": data,
        "final_metrics": {
            key: evaluation.get(key)
            for key in (
                "AP",
                "AP50",
                "AP75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "matched_mean_iou",
                "duplicate_rate",
            )
        },
    }
    _write_jsonl(history_path, [evaluation])
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.ap_tolerance < 0:
        raise ValueError("--ap-tolerance must be >= 0")
    result = register_full(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
