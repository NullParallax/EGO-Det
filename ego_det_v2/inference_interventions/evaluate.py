from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from ..ablations.registry import (
    build_variant_config,
    config_fingerprint,
)
from ..ablations.train import (
    _checkpoint_config,
    _load_reference_manifest,
    _normal_path,
    _write_json,
)
from ..engine.evaluator import evaluate
from ..launch._common import choose_device, make_loader, resolve_paths
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from .registry import InterventionSpec, get_intervention


STUDY_TYPE = "inference_time_component_intervention"


class FrozenGeometryReadout(nn.Module):
    """Preserve evidence/content updates while freezing round-wise geometry."""

    def __init__(self, readout: nn.Module) -> None:
        super().__init__()
        self.readout = readout

    def forward(
        self,
        content: torch.Tensor,
        semantic: torch.Tensor,
        boxes: torch.Tensor,
        bank: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        result = dict(self.readout(content, semantic, boxes, bank))
        result["b"] = boxes
        return result


def apply_runtime_transform(model: EgoDetV2, spec: InterventionSpec) -> None:
    if spec.runtime_transform == "none":
        return
    if spec.runtime_transform == "freeze_readout_geometry":
        if not model.readouts:
            raise RuntimeError(
                "frozen seed geometry requires the trained evidence readouts"
            )
        model.readouts = nn.ModuleList(
            [FrozenGeometryReadout(readout) for readout in model.readouts]
        )
        return
    raise RuntimeError(
        f"unsupported runtime transform: {spec.runtime_transform}"
    )


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _reference_checkpoint(
    manifest: Mapping[str, object],
    override: str | None,
) -> Path:
    value = override or manifest.get("source_checkpoint")
    if not value:
        raise RuntimeError(
            "reference manifest has no source_checkpoint; pass --checkpoint"
        )
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"full raw checkpoint not found: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one inference-only EGO-Det component intervention from "
            "the completed epoch-12 raw checkpoint"
        )
    )
    parser.add_argument("--variant", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--output-root",
        default="outputs/ego_det_v2/inference_interventions_raw_epoch12",
    )
    parser.add_argument("--project-root")
    parser.add_argument("--val-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--device")
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> InterventionSpec:
    spec = get_intervention(args.variant)
    if args.subset_size < 0:
        raise ValueError("--subset-size must be >= 0")
    return spec


def evaluate_intervention(args: argparse.Namespace) -> dict[str, object]:
    spec = _validate_args(args)
    reference_path = Path(args.reference_manifest).expanduser().resolve()
    reference_config, reference_manifest = _load_reference_manifest(
        reference_path
    )
    checkpoint_path = _reference_checkpoint(reference_manifest, args.checkpoint)
    output_dir = Path(args.output_root).expanduser().resolve() / spec.key
    manifest_path = output_dir / "intervention_manifest.json"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "coco_predictions.json"
    occupied = [
        path
        for path in (manifest_path, metrics_path, predictions_path)
        if path.exists()
    ]
    if occupied and not args.force:
        raise RuntimeError(
            "intervention output already exists; inspect it or pass --force: "
            + ", ".join(str(path) for path in occupied)
        )

    paths = resolve_paths(args.project_root)
    data = {
        "project_root": _normal_path(paths.root),
        "val_annotations": _normal_path(
            args.val_annotations or paths.coco_val_instances
        ),
        "image_root": _normal_path(args.image_root or paths.coco_image_root),
    }
    reference_data = reference_manifest.get("data")
    if not isinstance(reference_data, Mapping):
        raise RuntimeError(
            f"reference manifest has no data contract: {reference_path}"
        )
    expected_data = {
        key: _normal_path(reference_data[key])
        for key in data
        if key in reference_data
    }
    if expected_data != data:
        raise RuntimeError(
            "intervention data paths do not match the registered full run: "
            + json.dumps(
                {"reference": expected_data, "current": data},
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    now = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "study_type": STUDY_TYPE,
        "variant": spec.key,
        "label": spec.label,
        "description": spec.description,
        "status": "planned" if args.dry_run else "running",
        "created_at_utc": now,
        "updated_at_utc": now,
        "evaluation_state": "raw_model",
        "training_performed": False,
        "independently_trained_variant": False,
        "reference_manifest": str(reference_path),
        "source_checkpoint": str(checkpoint_path),
        "expected_epoch": int(reference_config.epochs),
        "config_variant": spec.config_variant,
        "config_overrides": spec.config_overrides(),
        "runtime_transform": spec.runtime_transform,
        "direct_slot_regression": False,
        "subset_size": int(args.subset_size),
        "evaluation_scope": (
            "full_coco_val2017"
            if int(args.subset_size) == 0
            else f"diagnostic_subset_{int(args.subset_size)}"
        ),
        "save_predictions": bool(args.save_predictions),
        "data": data,
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        return manifest

    started = time.perf_counter()
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        if checkpoint_epoch != int(reference_config.epochs):
            raise RuntimeError(
                "checkpoint is not the registered final epoch: "
                f"checkpoint={checkpoint_epoch}, "
                f"expected={reference_config.epochs}"
            )
        checkpoint_config = _checkpoint_config(checkpoint, checkpoint_path)
        observed_fingerprint = config_fingerprint(
            checkpoint_config,
            exclude=("output_dir",),
        )
        expected_fingerprint = config_fingerprint(
            reference_config,
            exclude=("output_dir",),
        )
        if observed_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "checkpoint config does not match the registered full control"
            )
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise KeyError("raw model state is absent from the checkpoint")

        config = build_variant_config(
            spec.config_variant,
            base=checkpoint_config,
        )
        model = EgoDetV2(config, paths=paths)
        model.load_state_dict(state, strict=True)
        apply_runtime_transform(model, spec)
        device = choose_device(args.device)
        model.to(device)
        criterion = EgoDetV2Criterion(config)
        loader, dataset = make_loader(
            "val",
            paths=paths,
            config=config,
            annotation_file=args.val_annotations,
            image_root=args.image_root,
            shuffle=False,
        )
        output_json = predictions_path if args.save_predictions else None
        metrics = evaluate(
            model,
            loader,
            matcher=criterion.matcher,
            device=device,
            dataset=dataset,
            output_json=output_json,
            subset_size=int(args.subset_size),
            show_progress=not args.no_progress,
        )
        if not _finite(metrics.get("AP")):
            raise RuntimeError(
                "COCO evaluation did not produce a finite AP value: "
                f"{metrics.get('coco_eval_error', 'unknown error')}"
            )
        elapsed = time.perf_counter() - started
        metrics_payload: dict[str, object] = {
            "schema_version": 1,
            "study_type": STUDY_TYPE,
            "variant": spec.key,
            "label": spec.label,
            "checkpoint_epoch": checkpoint_epoch,
            "evaluation_state": "raw_model",
            "training_performed": False,
            "evaluation_scope": manifest["evaluation_scope"],
            "subset_size": int(args.subset_size),
            "elapsed_seconds": elapsed,
            "metrics": metrics,
        }
        _write_json(metrics_path, metrics_payload)
        manifest.update(
            {
                "status": "complete",
                "checkpoint_epoch": checkpoint_epoch,
                "state_load_strict": True,
                "resolved_intervention_config": config.to_dict(),
                "metrics_path": str(metrics_path),
                "predictions_path": (
                    str(predictions_path) if args.save_predictions else None
                ),
                "elapsed_seconds": elapsed,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate_intervention(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
