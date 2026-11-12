from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..losses.criterion import EgoDetV2Criterion
from ..models.box_ops import box_cxcywh_to_xyxy
from ..models.ego_det_v2 import EgoDetV2


def _config(full_shape: bool, efficiency_mode: str) -> EgoDetV2Config:
    if full_shape:
        return EgoDetV2Config(
            d_model=32,
            tower_convs=1,
            sem_convs=1,
            roi_size=3,
            hdn_max_pairs=10,
            batch_size=2,
            grad_accum_steps=1,
            efficiency_mode=efficiency_mode,
        )
    return EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=12,
        evidence_per_slot=6,
        bank_min_per_level=2,
        bank_pre_topk_per_level=12,
        bank_topk_train=32,
        bank_topk_eval=32,
        hdn_max_pairs=6,
        roi_size=3,
        batch_size=2,
        grad_accum_steps=1,
        efficiency_mode=efficiency_mode,
    )


def _targets(side: int, device: torch.device) -> List[Dict]:
    normalized = torch.tensor(
        [
            [0.16, 0.20, 0.12, 0.14],
            [0.32, 0.33, 0.13, 0.14],
            [0.48, 0.46, 0.12, 0.14],
            [0.64, 0.59, 0.13, 0.14],
            [0.80, 0.72, 0.12, 0.14],
        ],
        device=device,
        dtype=torch.float32,
    )
    absolute = box_cxcywh_to_xyxy(normalized) * float(side)
    return [
        {
            "labels": torch.arange(5, device=device, dtype=torch.long),
            "boxes": normalized,
            "boxes_xyxy_abs": absolute,
            "image_id": 1,
            "orig_size": (side, side),
            "size": (side, side),
        },
        {
            "labels": torch.empty(0, device=device, dtype=torch.long),
            "boxes": torch.empty((0, 4), device=device, dtype=torch.float32),
            "boxes_xyxy_abs": torch.empty(
                (0, 4),
                device=device,
                dtype=torch.float32,
            ),
            "image_id": 2,
            "orig_size": (side, side),
            "size": (side, side),
        },
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EGO-Det V2 CUDA BF16 forward/loss/backward contract"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--full-shape",
        action="store_true",
        help="exercise B=2, 672x672, 300 slots and K_train=1000",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--efficiency-mode",
        choices=("reference", "optimized"),
        default="reference",
    )
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; run this contract in the server CUDA environment"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"CUDA BF16 is unsupported on {torch.cuda.get_device_name(device)}"
        )

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    config = _config(bool(args.full_shape), str(args.efficiency_mode))
    side = 672 if args.full_shape else 96
    model = EgoDetV2(config, fake_backbone=True).to(device).train()
    criterion = EgoDetV2Criterion(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    images = torch.randn(
        (2, 3, side, side),
        device=device,
        dtype=torch.float32,
    )
    pad_mask = torch.zeros(
        (2, side, side),
        device=device,
        dtype=torch.bool,
    )
    if not args.full_shape:
        pad_mask[1, 80:, :] = True
        pad_mask[1, :, 88:] = True
    targets = _targets(side, device)
    if not args.full_shape:
        targets[1]["size"] = (80, 88)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    ):
        outputs = model(images, pad_mask, targets, epoch=2)
        losses = criterion(outputs, targets, epoch=2)

    if outputs["pred_logits"].dtype != torch.bfloat16:
        raise AssertionError(
            "BF16 autocast did not produce BF16 slot logits: "
            f"{outputs['pred_logits'].dtype}"
        )
    checked_losses = {}
    for name, value in losses.items():
        if not name.startswith("loss_"):
            continue
        if value.ndim != 0:
            raise AssertionError(
                f"{name} must be scalar, got shape {tuple(value.shape)}"
            )
        if value.dtype != torch.float32:
            raise AssertionError(
                f"{name} must be float32, got {value.dtype}"
            )
        if not bool(torch.isfinite(value.detach()).all()):
            raise FloatingPointError(f"{name} is non-finite")
        checked_losses[name] = float(value.detach().cpu())

    losses["loss_total"].backward()
    finite_gradients = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite gradient in {name}")
        finite_gradients += 1
    if finite_gradients == 0:
        raise AssertionError("backward produced no parameter gradients")
    optimizer.step()

    result = {
        "status": "PASS",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "autocast_dtype": "bfloat16",
        "loss_dtype": "float32",
        "nonfinite": 0,
        "amp_dtype": "bfloat16",
        "fake_backbone": True,
        "full_shape": bool(args.full_shape),
        "efficiency_mode": config.efficiency_mode,
        "input_shape": list(images.shape),
        "num_slots": int(config.num_slots),
        "bank_topk_train": int(config.bank_topk_train),
        "output_dtypes": {
            "pred_logits": str(outputs["pred_logits"].dtype),
            "pred_boxes": str(outputs["pred_boxes"].dtype),
            "pred_quality": str(outputs["pred_quality"].dtype),
            "pred_exist": str(outputs["pred_exist"].dtype),
        },
        "loss_count": len(checked_losses),
        "losses": checked_losses,
        "finite_gradient_tensors": finite_gradients,
        "peak_memory_gb": (
            torch.cuda.max_memory_allocated(device) / float(1024**3)
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
