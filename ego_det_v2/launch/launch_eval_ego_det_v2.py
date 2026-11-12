from __future__ import annotations

import argparse
import json

import torch

from ..configs.config import EgoDetV2Config
from ..configs.project import (
    evaluation_model_config, load_settings, path_from_settings, resolve_path, resolve_root,
)
from dataclasses import replace
from pathlib import Path
from ..engine.evaluator import evaluate
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an EGO-Det V2 checkpoint")
    parser.add_argument("--config", help="YAML configuration (default: configs/egodet.yaml)")
    parser.add_argument("--ckpt", default=None, help="Detector checkpoint; defaults to YAML paths.checkpoint")
    ema_group = parser.add_mutually_exclusive_group()
    ema_group.add_argument("--use-ema", dest="use_ema", action="store_true", default=None)
    ema_group.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--project-root")
    parser.add_argument("--val-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="DataLoader batch size (not model config)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--output-json", help="COCO-format predictions JSON (NOT metrics)")
    parser.add_argument("--metrics-json", help="Evaluation metrics JSON")
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.subset_size is not None and args.subset_size < 0:
        parser.error("--subset-size must be >= 0")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_root(args.project_root)
    runtime = settings["evaluation"]
    checkpoint_path = path_from_settings(settings, "checkpoint", root, args.ckpt)
    if checkpoint_path is None:
        raise ValueError("Set paths.checkpoint in YAML or pass --ckpt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_values = checkpoint.get("config", {})
    allowed = EgoDetV2Config.__dataclass_fields__
    config = EgoDetV2Config(
        **{key: value for key, value in config_values.items() if key in allowed}
    )
    config = evaluation_model_config(config, settings)
    paths = resolve_paths(str(root))
    pretrained = path_from_settings(settings, "overlock_weights", root)
    if pretrained is not None:
        paths = replace(paths, overlock_weights=pretrained)
    device = choose_device(args.device or runtime.get("device"))
    model = EgoDetV2(config, paths=paths)
    use_ema = bool(args.use_ema if args.use_ema is not None else runtime.get("use_ema", False))
    state = checkpoint.get("ema") if use_ema else checkpoint.get("model")
    if state is None:
        raise KeyError("requested model state is absent from checkpoint")
    model.load_state_dict(state, strict=True)
    model.to(device)
    criterion = EgoDetV2Criterion(config)
    val_annotation = path_from_settings(settings, "val_annotations", root, args.val_annotations)
    images = path_from_settings(settings, "image_root", root, args.image_root)
    batch_size = args.batch_size if args.batch_size is not None else runtime.get("batch_size")
    num_workers = args.num_workers if args.num_workers is not None else runtime.get("num_workers")
    subset_size = args.subset_size if args.subset_size is not None else int(runtime.get("subset_size", 0))
    predictions = args.output_json if args.output_json is not None else runtime.get("predictions_json")
    metrics_path = args.metrics_json if args.metrics_json is not None else runtime.get("metrics_json")
    loader, dataset = make_loader(
        "val",
        paths=paths,
        config=config,
        annotation_file=val_annotation,
        image_root=images,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    metrics = evaluate(
        model,
        loader,
        matcher=criterion.matcher,
        device=device,
        dataset=dataset,
        output_json=resolve_path(predictions, root),
        subset_size=subset_size,
        show_progress=not args.no_progress,
    )
    if "coco_eval_error" in metrics:
        raise RuntimeError(f"COCO evaluation failed: {metrics['coco_eval_error']}")
    if metrics_path is not None:
        metrics_file = Path(metrics_path).expanduser()
        if not metrics_file.is_absolute():
            metrics_file = root / metrics_file
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
