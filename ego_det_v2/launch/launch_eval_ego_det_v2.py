from __future__ import annotations

import argparse
import json

import torch

from ..configs.config import EgoDetV2Config
from ..engine.evaluator import evaluate
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an EGO-Det V2 checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--project-root")
    parser.add_argument("--val-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--subset-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None, help="DataLoader batch size (not model config)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--output-json")
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config_values = checkpoint.get("config", {})
    allowed = EgoDetV2Config.__dataclass_fields__
    config = EgoDetV2Config(
        **{key: value for key, value in config_values.items() if key in allowed}
    )
    paths = resolve_paths(args.project_root)
    device = choose_device(args.device)
    model = EgoDetV2(config, paths=paths)
    state = checkpoint.get("ema") if args.use_ema else checkpoint.get("model")
    if state is None:
        raise KeyError("requested model state is absent from checkpoint")
    model.load_state_dict(state, strict=True)
    model.to(device)
    criterion = EgoDetV2Criterion(config)
    loader, dataset = make_loader(
        "val",
        paths=paths,
        config=config,
        annotation_file=args.val_annotations,
        image_root=args.image_root,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics = evaluate(
        model,
        loader,
        matcher=criterion.matcher,
        device=device,
        dataset=dataset,
        output_json=args.output_json,
        subset_size=args.subset_size,
        show_progress=not args.no_progress,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
