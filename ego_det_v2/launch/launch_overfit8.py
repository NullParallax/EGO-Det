from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from ..configs.config import EgoDetV2Config
from ..engine.diagnostics import (
    ParameterUpdateAuditor,
    append_jsonl,
    optimizer_lr_summary,
)
from ..engine.evaluator import evaluate
from ..engine.train_loop import (
    build_optimizer,
    build_optimizer_step_scheduler,
    optimizer_step_warmup_scale,
    run_fixed_steps,
    set_runtime_defaults,
)
from ..losses.criterion import EgoDetV2Criterion
from ..models.ego_det_v2 import EgoDetV2
from ._common import choose_device, make_loader, print_config, resolve_paths


DEFAULT_GEOMETRY_EVAL_STEPS = (0, 25, 50, 100, 200, 300)
DEFAULT_FULL_EVAL_STEPS = (0, 25, 50, 100, 250, 500)


def _step_csv(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in str(value).split(",")
                    if item.strip()
                }
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation steps must be comma-separated non-negative integers"
        ) from exc
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError("evaluation steps must be non-negative")
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EGO-Det V2 fixed eight-image root-cause diagnostic"
    )
    parser.add_argument("--project-root")
    parser.add_argument("--train-annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir", default="outputs/ego_det_v2/overfit8")
    parser.add_argument("--preset", choices=("diagnostic-fast",))
    parser.add_argument("--geometry-steps", type=int, default=300)
    parser.add_argument("--full-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--image-short-side", type=int, default=512)
    parser.add_argument("--image-max-size", type=int, default=768)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--base-lr", type=float, default=3.0e-4)
    parser.add_argument("--geometry-lr", type=float, default=3.0e-4)
    parser.add_argument("--fpn-lr", type=float, default=1.0e-4)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-5)
    parser.add_argument("--backbone-stem-lr", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--dense-assignment-mode",
        choices=("stable_fcos", "legacy_dynamic"),
        default="stable_fcos",
    )
    parser.add_argument(
        "--efficiency-mode",
        choices=("reference", "optimized"),
        default="reference",
    )
    parser.add_argument(
        "--proposal-identity-iou-threshold",
        type=float,
        default=0.5,
    )
    fallback_group = parser.add_mutually_exclusive_group()
    fallback_group.add_argument(
        "--assign-ensure-min-one-positive",
        dest="assign_ensure_min_one_positive",
        action="store_true",
    )
    fallback_group.add_argument(
        "--no-assign-ensure-min-one-positive",
        dest="assign_ensure_min_one_positive",
        action="store_false",
    )
    parser.add_argument(
        "--memory-sweep-result",
        default="outputs/ego_det_v2/memory_sweep_mod8.json",
        help="required batch4 evidence for --preset diagnostic-fast",
    )
    parser.add_argument(
        "--geometry-eval-steps",
        type=_step_csv,
        default=DEFAULT_GEOMETRY_EVAL_STEPS,
    )
    parser.add_argument(
        "--full-eval-steps",
        type=_step_csv,
        default=DEFAULT_FULL_EVAL_STEPS,
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
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--require-geometry-gate",
        dest="require_geometry_gate",
        action="store_true",
    )
    gate_group.add_argument(
        "--no-require-geometry-gate",
        dest="require_geometry_gate",
        action="store_false",
    )
    parser.set_defaults(
        overlock_checkpointing=True,
        require_geometry_gate=True,
        assign_ensure_min_one_positive=False,
    )
    parser.add_argument(
        "--geometry-gate-metric",
        default="dense_all_recall@0.5",
    )
    parser.add_argument("--geometry-gate-threshold", type=float, default=0.95)
    parser.add_argument(
        "--geometry-gate-bank-metric",
        default="bank_top300_recall@0.5",
    )
    parser.add_argument(
        "--geometry-gate-bank-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--slot-to-geometry-grad-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress bars (JSONL logging is unchanged)",
    )
    return parser


def _apply_preset(args: argparse.Namespace) -> Dict[str, object]:
    if args.preset != "diagnostic-fast":
        return {}
    expansion: Dict[str, object] = {
        "batch_size": 4,
        "grad_accum": 1,
        "image_short_side": 512,
        "image_max_size": 768,
        "warmup_steps": 25,
        "base_lr": 3.0e-4,
        "head_lr": 3.0e-4,
        "geometry_lr": 3.0e-4,
        "fpn_lr": 1.0e-4,
        "backbone_lr": 1.0e-5,
        "backbone_stem_lr": 1.0e-6,
        "weight_decay": 0.0,
        "random_crop": 0.0,
        "hflip": 0.0,
        "color_jitter": 0.0,
        "overlock_checkpointing": True,
        "periodic_eval": True,
    }
    for name in (
        "batch_size",
        "grad_accum",
        "image_short_side",
        "image_max_size",
        "warmup_steps",
        "base_lr",
        "geometry_lr",
        "fpn_lr",
        "backbone_lr",
        "backbone_stem_lr",
        "weight_decay",
        "overlock_checkpointing",
    ):
        setattr(args, name, expansion[name])
    return expansion


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.preset_expansion = _apply_preset(args)
    if args.geometry_steps < 0 or args.full_steps < 0:
        raise ValueError("--geometry-steps and --full-steps must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.image_short_side < 128:
        raise ValueError("--image-short-side must be >= 128")
    if args.image_max_size < args.image_short_side:
        raise ValueError("--image-max-size must be >= --image-short-side")
    if args.warmup_steps < 1:
        raise ValueError("--warmup-steps must be >= 1 optimizer step")
    for name in (
        "base_lr",
        "geometry_lr",
        "fpn_lr",
        "backbone_lr",
        "backbone_stem_lr",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0")
    if not 0.0 < args.slot_to_geometry_grad_scale <= 1.0:
        raise ValueError("--slot-to-geometry-grad-scale must be in (0, 1]")
    if not 0.0 <= args.geometry_gate_threshold <= 1.0:
        raise ValueError("--geometry-gate-threshold must be in [0, 1]")
    if not 0.0 <= args.geometry_gate_bank_threshold <= 1.0:
        raise ValueError("--geometry-gate-bank-threshold must be in [0, 1]")
    if not 0.0 <= args.proposal_identity_iou_threshold <= 1.0:
        raise ValueError(
            "--proposal-identity-iou-threshold must be in [0, 1]"
        )
    return args


def assert_exact_overfit_ids(train_dataset, eval_dataset) -> list[int]:
    train_ids = [int(value) for value in train_dataset.image_ids]
    eval_ids = [int(value) for value in eval_dataset.image_ids]
    exact = train_ids == eval_ids and len(train_ids) == 8
    if not exact:
        raise AssertionError(
            "Overfit8 train/eval datasets must contain the same ordered eight image IDs: "
            f"train={train_ids}, eval={eval_ids}"
        )
    return train_ids


def require_batch4_memory_preflight(path: Path | str) -> Dict[str, object]:
    result_path = Path(path)
    if not result_path.is_file():
        raise RuntimeError(
            "diagnostic-fast requests batch4@512/768, but its memory-sweep "
            f"evidence is missing: {result_path}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = next(
        (
            item
            for item in payload.get("cases", [])
            if int(item.get("batch_size", -1)) == 4
            and int(item.get("short_side", -1)) == 512
            and int(item.get("max_size", -1)) == 768
        ),
        None,
    )
    proven = (
        bool(payload.get("candidate_batch4_512_768_proven", False))
        and candidate is not None
        and candidate.get("status") == "PASS"
        and not bool(candidate.get("oom", True))
    )
    if not proven:
        raise RuntimeError(
            "diagnostic-fast batch4@512/768 is not proven OOM-free by "
            f"{result_path}; run launch_memory_sweep first"
        )
    return dict(candidate)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def build_composite_geometry_gate(
    metrics: Dict[str, object],
    *,
    required: bool,
    dense_metric: str = "dense_all_recall@0.5",
    dense_threshold: float = 0.95,
    bank_metric: str = "bank_top300_recall@0.5",
    bank_threshold: float = 0.95,
) -> Dict[str, object]:
    dense_value = float(metrics.get(dense_metric, 0.0))
    bank_value = float(metrics.get(bank_metric, 0.0))
    dense_passed = dense_value >= float(dense_threshold)
    bank_passed = bank_value >= float(bank_threshold)
    if not dense_passed:
        diagnosis = "assignment_or_box_regression"
    elif not bank_passed:
        diagnosis = "ranking_or_topk"
    else:
        diagnosis = "geometry_ready"
    return {
        "required": bool(required),
        "dense": {
            "metric": str(dense_metric),
            "threshold": float(dense_threshold),
            "actual": dense_value,
            "passed": dense_passed,
        },
        "bank": {
            "metric": str(bank_metric),
            "threshold": float(bank_threshold),
            "actual": bank_value,
            "passed": bank_passed,
        },
        "passed": dense_passed and bank_passed,
        "diagnosis": diagnosis,
    }


def main() -> None:
    args = parse_args()
    if args.preset_expansion:
        print(
            "[Overfit8 preset diagnostic-fast] "
            + json.dumps(
                args.preset_expansion,
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        preflight = require_batch4_memory_preflight(args.memory_sweep_result)
        print(
            "[Overfit8 memory preflight] "
            + json.dumps(preflight, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    config = replace(
        EgoDetV2Config(),
        output_dir=args.output_dir,
        num_workers=0,
        batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum),
        train_short_sides=(int(args.image_short_side),),
        train_max_size=int(args.image_max_size),
        val_short_side=int(args.image_short_side),
        val_max_size=int(args.image_max_size),
        random_crop_prob=0.0,
        hflip_prob=0.0,
        warmup_steps=int(args.warmup_steps),
        lr=float(args.base_lr),
        backbone_lr=float(args.backbone_lr),
        backbone_stem_lr=float(args.backbone_stem_lr),
        weight_decay=float(args.weight_decay),
        overlock_gradient_checkpointing=bool(args.overlock_checkpointing),
        dense_assignment_mode=str(args.dense_assignment_mode),
        proposal_identity_iou_threshold=float(
            args.proposal_identity_iou_threshold
        ),
        assign_ensure_min_one_positive=bool(
            args.assign_ensure_min_one_positive
        ),
        efficiency_mode=str(args.efficiency_mode),
    )
    print_config(config)
    print(
        "[Overfit8 dense assignment] "
        f"mode={config.dense_assignment_mode} "
        "proposal_identity_iou_threshold="
        f"{config.proposal_identity_iou_threshold:.6g} "
        "assign_ensure_min_one_positive="
        f"{config.assign_ensure_min_one_positive}",
        flush=True,
    )
    set_runtime_defaults(config.seed)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = resolve_paths(args.project_root)
    discovery_loader, discovery_dataset = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        training_override=False,
        shuffle=False,
        batch_size=config.batch_size,
    )
    del discovery_loader
    image_ids = [int(value) for value in discovery_dataset.image_ids[:8]]
    train_loader, train_dataset = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        image_ids=image_ids,
        training_override=False,
        shuffle=True,
        batch_size=config.batch_size,
    )
    eval_loader, eval_dataset = make_loader(
        "train",
        paths=paths,
        config=config,
        annotation_file=args.train_annotations,
        image_root=args.image_root,
        image_ids=image_ids,
        training_override=False,
        shuffle=False,
        batch_size=config.batch_size,
    )
    image_ids = assert_exact_overfit_ids(train_dataset, eval_dataset)
    print(f"[Overfit8 image_ids] {image_ids}", flush=True)
    print("[Overfit8 train/eval ids exact_match=True]", flush=True)
    print(
        "[Overfit8 augmentation] random_crop=0 hflip=0 color_jitter=0 "
        "fixed_resize=True drop_last=False",
        flush=True,
    )

    device = choose_device(args.device)
    model = EgoDetV2(config, paths=paths).to(device)
    criterion = EgoDetV2Criterion(config).to(device)
    optimizer = build_optimizer(
        model,
        config,
        base_lr=args.base_lr,
        geometry_lr=args.geometry_lr,
        fpn_lr=args.fpn_lr,
        backbone_lr=args.backbone_lr,
        backbone_stem_lr=args.backbone_stem_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_optimizer_step_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
    )
    lr_diagnostic_path = output / "optimizer_lr.jsonl"
    update_diagnostic_path = output / "parameter_update_audit.jsonl"
    competition_path = output / "geometry_gradient_competition.jsonl"
    for path in (
        lr_diagnostic_path,
        update_diagnostic_path,
        competition_path,
    ):
        path.write_text("", encoding="utf-8")
    initial_lrs: Dict[str, object] = {
        "kind": "optimizer_lr",
        "optimizer_step": 0,
        "phase": "geometry",
        **optimizer_lr_summary(optimizer),
    }
    append_jsonl(lr_diagnostic_path, initial_lrs)
    print(
        "[EgoDetV2 optimizer LR] "
        + json.dumps(initial_lrs, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    scheduler_contract = {
        "formula": "base_lr * (0.01 + 0.99 * min(1, (optimizer_step + 1) / warmup_steps))",
        "warmup_unit": "optimizer_step",
        "step_semantics": (
            "scheduler index; step 0 is the LR used by the first optimizer update"
        ),
        "scheduler_call_order": "optimizer.step() then scheduler.step()",
        "optimizer_recreated_between_geometry_and_full": False,
        "progress_lr_group": "geometry",
        "steps": {
            str(step): {
                "scale": optimizer_step_warmup_scale(step, args.warmup_steps),
                "lr_stem": args.backbone_stem_lr
                * optimizer_step_warmup_scale(step, args.warmup_steps),
                "lr_backbone": args.backbone_lr
                * optimizer_step_warmup_scale(step, args.warmup_steps),
                "lr_fpn": args.fpn_lr
                * optimizer_step_warmup_scale(step, args.warmup_steps),
                "lr_geometry": args.geometry_lr
                * optimizer_step_warmup_scale(step, args.warmup_steps),
                "lr_slot": args.base_lr
                * optimizer_step_warmup_scale(step, args.warmup_steps),
                "lr_heads": args.base_lr
                * optimizer_step_warmup_scale(step, args.warmup_steps),
            }
            for step in (0, 1, 10, 25, 50, 300, 500)
        },
    }
    _write_json(output / "scheduler_contract.json", scheduler_contract)
    print(
        "[Overfit8 scheduler contract] "
        + json.dumps(scheduler_contract, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    auditor = ParameterUpdateAuditor(
        optimizer,
        jsonl_path=update_diagnostic_path,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    geometry_curve: list[Dict[str, object]] = []
    full_curve: list[Dict[str, object]] = []

    def evaluate_geometry_point(stage_step: int) -> Dict[str, object]:
        metrics = evaluate(
            model,
            eval_loader,
            matcher=criterion.matcher,
            device=device,
            dataset=None,
            show_progress=not args.no_progress,
            include_geometry_diagnostics=True,
            geometry_only=True,
        )
        record: Dict[str, object] = {
            "optimizer_step": int(stage_step),
            **metrics,
        }
        geometry_curve.append(record)
        _write_json(output / "geometry_curve.json", geometry_curve)
        print(
            "[Overfit8 geometry curve] "
            + json.dumps(record, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return metrics

    def evaluate_full_point(stage_step: int) -> Dict[str, object]:
        metrics = evaluate(
            model,
            eval_loader,
            matcher=criterion.matcher,
            device=device,
            dataset=eval_dataset,
            show_progress=not args.no_progress,
            include_geometry_diagnostics=True,
            geometry_only=False,
        )
        record: Dict[str, object] = {
            "optimizer_step": int(stage_step),
            **metrics,
        }
        full_curve.append(record)
        _write_json(output / "full_curve.json", full_curve)
        print(
            "[Overfit8 full curve] "
            + json.dumps(record, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return metrics

    if 0 in args.geometry_eval_steps:
        evaluate_geometry_point(0)
    print(
        "[Overfit8] Stage A: geometry-only, "
        f"optimizer_steps={args.geometry_steps}",
        flush=True,
    )

    def geometry_callback(stage_step: int, global_step: int) -> None:
        del global_step
        if stage_step in args.geometry_eval_steps:
            evaluate_geometry_point(stage_step)

    geometry_train: Dict[str, object]
    if args.geometry_steps > 0:
        geometry_train = run_fixed_steps(
            model,
            criterion,
            train_loader,
            optimizer,
            config=config,
            device=device,
            steps=args.geometry_steps,
            geometry_only=True,
            show_progress=not args.no_progress,
            scheduler=scheduler,
            global_step_offset=0,
            diagnostic_jsonl=lr_diagnostic_path,
            parameter_auditor=auditor,
            optimizer_step_callback=geometry_callback,
        )
    else:
        geometry_train = {"optimizer_step": 0.0, **optimizer_lr_summary(optimizer)}

    if geometry_curve and geometry_curve[-1]["optimizer_step"] == args.geometry_steps:
        geometry_eval = {
            key: value
            for key, value in geometry_curve[-1].items()
            if key != "optimizer_step"
        }
    else:
        geometry_eval = evaluate_geometry_point(args.geometry_steps)

    gate = build_composite_geometry_gate(
        geometry_eval,
        required=bool(args.require_geometry_gate),
        dense_metric=str(args.geometry_gate_metric),
        dense_threshold=float(args.geometry_gate_threshold),
        bank_metric=str(args.geometry_gate_bank_metric),
        bank_threshold=float(args.geometry_gate_bank_threshold),
    )
    gate_passed = bool(gate["passed"])
    print(
        "[Overfit8 geometry gate] "
        + json.dumps(gate, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    full_train: Dict[str, object] = {"skipped": True}
    full_eval: Dict[str, object] = {}
    full_skipped_by_gate = (
        args.full_steps > 0 and args.require_geometry_gate and not gate_passed
    )
    if args.full_steps > 0 and not full_skipped_by_gate:
        if 0 in args.full_eval_steps:
            evaluate_full_point(0)
        print(
            "[Overfit8] Stage B: full model, "
            f"optimizer_steps={args.full_steps}",
            flush=True,
        )

        def full_callback(stage_step: int, global_step: int) -> None:
            del global_step
            if stage_step in args.full_eval_steps:
                evaluate_full_point(stage_step)

        full_train = run_fixed_steps(
            model,
            criterion,
            train_loader,
            optimizer,
            config=config,
            device=device,
            steps=args.full_steps,
            epoch=2,
            show_progress=not args.no_progress,
            scheduler=scheduler,
            global_step_offset=int(args.geometry_steps),
            diagnostic_jsonl=lr_diagnostic_path,
            parameter_auditor=auditor,
            gradient_competition_steps=(1, 25, 100),
            gradient_competition_jsonl=competition_path,
            slot_to_geometry_grad_scale=args.slot_to_geometry_grad_scale,
            optimizer_step_callback=full_callback,
        )
        if full_curve and full_curve[-1]["optimizer_step"] == args.full_steps:
            full_eval = {
                key: value
                for key, value in full_curve[-1].items()
                if key != "optimizer_step"
            }
        else:
            full_eval = evaluate_full_point(args.full_steps)
    elif full_skipped_by_gate:
        print(
            "[Overfit8] Full stage skipped: geometry gate failed. "
            "Use --no-require-geometry-gate only for an explicit diagnostic bypass.",
            flush=True,
        )

    acceptance = {
        "geometry_gate": gate_passed,
        "matched_accuracy": (
            float(full_eval.get("matched_accuracy", 0.0))
            >= config.overfit_min_matched_accuracy
            if full_eval
            else False
        ),
        "matched_mean_iou": (
            float(full_eval.get("matched_mean_iou", 0.0))
            >= config.overfit_min_matched_iou
            if full_eval
            else False
        ),
        "matched_recall_50": (
            float(full_eval.get("matched_recall_50", 0.0))
            >= config.overfit_min_matched_recall
            if full_eval
            else False
        ),
    }
    result = {
        "preset": args.preset,
        "preset_expansion": args.preset_expansion,
        "dense_assignment": {
            "mode": config.dense_assignment_mode,
            "proposal_identity_iou_threshold": (
                config.proposal_identity_iou_threshold
            ),
            "assign_ensure_min_one_positive": (
                config.assign_ensure_min_one_positive
            ),
        },
        "image_ids": image_ids,
        "train_eval_ids_exact_match": True,
        "augmentation": {
            "random_crop": 0.0,
            "hflip": 0.0,
            "color_jitter": 0.0,
            "fixed_resize": [
                int(args.image_short_side),
                int(args.image_max_size),
            ],
            "drop_last": False,
        },
        "scheduler_contract": scheduler_contract,
        "geometry_train": geometry_train,
        "geometry_eval": geometry_eval,
        "geometry_gate": gate,
        "full_skipped_by_gate": full_skipped_by_gate,
        "full_train": full_train,
        "full_eval": full_eval,
        "acceptance": acceptance,
        "coco_ap_is_contract_gate": False,
        "slot_to_geometry_grad_scale": args.slot_to_geometry_grad_scale,
    }
    _write_json(output / "overfit8_result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    required_acceptance = ["geometry_gate"] if args.require_geometry_gate else []
    if args.full_steps > 0 and not full_skipped_by_gate:
        required_acceptance.extend(
            ("matched_accuracy", "matched_mean_iou", "matched_recall_50")
        )
    if not all(acceptance[name] for name in required_acceptance):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
