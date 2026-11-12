"""Run a single-image diagnostic with the checkpoint-compatible EGO-Det model.

This command is NOT the COCO benchmark postprocessor. Class labels are contiguous
indices, and the display may include repeated boxes (no NMS).
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic single-image EGO-Det inference")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_detection.jpg"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--overlock-weights", type=Path)
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.short_side <= 0 or args.max_size <= 0:
        raise ValueError("--short-side and --max-size must be positive")
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if not 0.0 <= args.score_thr <= 1.0:
        raise ValueError("--score-thr must be in [0, 1]")
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    import torch
    from PIL import Image, ImageDraw
    from torchvision.transforms import functional as TF

    from egodet import EgoDet, EgoDetConfig
    from egodet.paths import EgoDetPaths

    root = args.project_root.expanduser().resolve()
    ckpt_path = (args.checkpoint or root / "checkpoints/epoch_12.pth").expanduser().resolve()
    pretrained = (args.overlock_weights or root / "weights/overlock/overlock_t_in1k_224.pth").expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"detector checkpoint not found: {ckpt_path}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"OverLoCK pretrained weights not found: {pretrained}")

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("expected an EGO-Det checkpoint containing a 'model' state dict")
    saved_config = checkpoint.get("config") or {}
    allowed = EgoDetConfig.__dataclass_fields__
    config = EgoDetConfig(**{key: value for key, value in saved_config.items() if key in allowed})
    paths = replace(EgoDetPaths.from_root(root), overlock_weights=pretrained)
    model = EgoDet(config, paths=paths)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(device).eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    width, height = image.size
    factor = min(args.short_side / min(width, height), args.max_size / max(width, height))
    resized_w = max(1, round(width * factor))
    resized_h = max(1, round(height * factor))
    resized = TF.resize(image, [resized_h, resized_w], antialias=True)
    tensor = TF.normalize(TF.to_tensor(resized), config.pixel_mean, config.pixel_std)

    # Match the right/bottom padding semantics of the official COCO collator.
    padded_h = ((resized_h + 31) // 32) * 32
    padded_w = ((resized_w + 31) // 32) * 32
    images = tensor.new_zeros((1, 3, padded_h, padded_w))
    images[0, :, :resized_h, :resized_w] = tensor
    pad_mask = torch.ones((1, padded_h, padded_w), dtype=torch.bool)
    pad_mask[0, :resized_h, :resized_w] = False
    images, pad_mask = images.to(device), pad_mask.to(device)

    if device.type == "cuda" and config.amp:
        use_bf16 = str(config.amp_dtype).lower() == "bf16" and torch.cuda.is_bf16_supported()
        autocast = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16)
    else:
        autocast = nullcontext()

    with torch.inference_mode(), autocast:
        outputs = model(images, pad_mask)

    scores = outputs["pred_scores"][0].float().cpu()
    boxes = outputs["pred_boxes"][0].float().cpu()
    count_classes = scores.shape[-1]
    flat = scores.reshape(-1)
    values, indices = torch.topk(flat, min(flat.numel(), max(args.topk * 5, args.topk)))

    draw = ImageDraw.Draw(image)
    visible = 0
    for score, index in zip(values.tolist(), indices.tolist()):
        if score < args.score_thr or visible >= args.topk:
            break
        slot, label = divmod(index, count_classes)
        cx, cy, bw, bh = boxes[slot].tolist()
        x1 = max(0, min(width, (cx - bw / 2) * width))
        y1 = max(0, min(height, (cy - bh / 2) * height))
        x2 = max(0, min(width, (cx + bw / 2) * width))
        y2 = max(0, min(height, (cy + bh / 2) * height))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 14)), f"class_index={label} {score:.3f}", fill="red")
        visible += 1

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Input: {args.image}")
    print(f"Preprocessed size: {resized_w}x{resized_h}, padded: {padded_w}x{padded_h}")
    print(f"Displayed predictions: {visible}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
