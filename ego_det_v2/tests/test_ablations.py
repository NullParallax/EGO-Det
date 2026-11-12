from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from ego_det_v2.ablations.registry import (
    ABLATION_TABLES,
    ABLATIONS,
    INSTRUMENTATION_FIELDS,
    METHOD_FIELDS,
    build_variant_config,
    config_fingerprint,
    control_fingerprint,
)
from ego_det_v2.ablations.summarize import collect_results
from ego_det_v2.ablations.train import (
    _load_reference_manifest,
    _runtime_updates,
    _validate_args,
    _validate_reference_updates,
    _validate_resume,
    build_parser,
)
from ego_det_v2.configs.config import EgoDetV2Config
from ego_det_v2.engine.train_loop import _should_run_full_eval, build_optimizer
from ego_det_v2.losses.criterion import EgoDetV2Criterion
from ego_det_v2.models.box_ops import box_cxcywh_to_xyxy
from ego_det_v2.models.ego_det_v2 import EgoDetV2
from ego_det_v2.models.slot_seeding import SlotSeeder


def _compact_base() -> EgoDetV2Config:
    return EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=8,
        evidence_per_slot=4,
        bank_min_per_level=1,
        bank_pre_topk_per_level=8,
        bank_topk_train=16,
        bank_topk_eval=16,
        hdn_max_pairs=2,
        roi_size=3,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        amp=False,
        overlock_gradient_checkpointing=False,
    )


def _target(side: int) -> dict:
    boxes = torch.tensor(
        [
            [0.30, 0.35, 0.25, 0.20],
            [0.70, 0.65, 0.20, 0.25],
        ],
        dtype=torch.float32,
    )
    return {
        "labels": torch.tensor([1, 4], dtype=torch.long),
        "boxes": boxes,
        "boxes_xyxy_abs": box_cxcywh_to_xyxy(boxes) * float(side),
        "image_id": 1,
        "orig_size": (side, side),
        "size": (side, side),
    }


def _batch(side: int = 64):
    return (
        torch.randn(1, 3, side, side),
        torch.zeros(1, side, side, dtype=torch.bool),
        [_target(side)],
    )


def test_registry_contract() -> None:
    assert len(ABLATIONS) == 10
    assert set(ABLATION_TABLES) == {
        "overall_component_analysis",
        "object_slot_formation",
        "slot_conditioned_geometric_readout",
    }
    base = _compact_base()
    assert "freeze_backbone" not in METHOD_FIELDS
    assert "freeze_backbone" not in INSTRUMENTATION_FIELDS
    assert "full_eval_last_epoch_only" in INSTRUMENTATION_FIELDS
    fingerprints = {
        control_fingerprint(build_variant_config(key, base=base))
        for key in ABLATIONS
    }
    assert len(fingerprints) == 1
    reduced = replace(
        base,
        freeze_backbone=True,
        full_eval_last_epoch_only=True,
        train_short_sides=(448, 480, 512),
        train_max_size=800,
        batch_size=16,
        grad_accum_steps=1,
        epochs=5,
        lr_drop_epoch=4,
    )
    reduced_fingerprints = {
        control_fingerprint(build_variant_config(key, base=reduced))
        for key in ABLATIONS
    }
    assert len(reduced_fingerprints) == 1
    assert control_fingerprint(reduced) != control_fingerprint(
        replace(reduced, freeze_backbone=False)
    )
    direct = build_variant_config("direct_slot_regression", base=base)
    assert direct.readout_mode == "direct_regression"
    assert direct.w_ownership == 0
    assert direct.w_dustbin == 0
    assert direct.w_gain == 0
    assert sum(
        row.variant == "direct_slot_regression"
        for table in ABLATION_TABLES.values()
        for row in table.rows
    ) == 2
    try:
        build_variant_config(
            "full",
            base=base,
            extra_updates={"readout_mode": "whole_box"},
        )
    except ValueError as exc:
        assert "method fields" in str(exc)
    else:
        raise AssertionError("runtime override changed an ablation method field")

    diagnostic_only = replace(
        base,
        quick_eval_every_steps=0,
        full_eval_last_epoch_only=True,
        oracle_monitor_every_optimizer_steps=0,
        ckpt_every_steps=0,
        log_every_steps=100,
    )
    assert control_fingerprint(base) == control_fingerprint(diagnostic_only)


def test_reduced_protocol_cli_contract(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--variant",
            "full",
            "--epochs",
            "5",
            "--lr-drop-epoch",
            "4",
            "--batch-size",
            "16",
            "--grad-accum",
            "1",
            "--train-short-sides",
            "448,480,512",
            "--train-max-size",
            "800",
            "--seed",
            "3407",
            "--freeze-backbone",
            "--no-overlock-checkpointing",
            "--full-eval-last-epoch-only",
        ]
    )
    _validate_args(args)
    updates = _runtime_updates(args, _compact_base(), tmp_path / "full")
    assert updates["train_short_sides"] == (448, 480, 512)
    assert updates["train_max_size"] == 800
    assert updates["epochs"] == 5
    assert updates["lr_drop_epoch"] == 4
    assert updates["batch_size"] == 16
    assert updates["grad_accum_steps"] == 1
    assert updates["seed"] == 3407
    assert updates["freeze_backbone"] is True
    assert updates["overlock_gradient_checkpointing"] is False
    assert updates["full_eval_last_epoch_only"] is True

    invalid = build_parser().parse_args(
        ["--variant", "full", "--train-short-sides", "512,448"]
    )
    try:
        _validate_args(invalid)
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("unsorted training short sides were accepted")


def test_frozen_backbone_contract() -> None:
    config = replace(_compact_base(), freeze_backbone=True)
    model = EgoDetV2(config, fake_backbone=True).train()
    assert not model.backbone.training
    assert all(
        not parameter.requires_grad for parameter in model.backbone.parameters()
    )
    optimizer = build_optimizer(model, config, print_groups=False)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids.isdisjoint(
        id(parameter) for parameter in model.backbone.parameters()
    )

    images, pad_mask, targets = _batch()
    images.requires_grad_(True)
    losses = EgoDetV2Criterion(config)(
        model(images, pad_mask, targets, epoch=2),
        targets,
        epoch=2,
        epoch_progress=1.0,
    )
    losses["loss_total"].backward()
    assert images.grad is None
    assert all(
        parameter.grad is None for parameter in model.backbone.parameters()
    )


def test_full_eval_cadence_contract() -> None:
    regular = replace(_compact_base(), epochs=5)
    reduced = replace(regular, full_eval_last_epoch_only=True)
    assert all(_should_run_full_eval(regular, epoch) for epoch in range(1, 6))
    assert [
        _should_run_full_eval(reduced, epoch) for epoch in range(1, 6)
    ] == [False, False, False, False, True]


def test_reference_manifest_contract(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    full = replace(
        _compact_base(),
        batch_size=8,
        grad_accum_steps=2,
        efficiency_mode="optimized",
        train_short_sides=(480, 512, 544, 576, 608),
        output_dir=str(tmp_path / "full"),
    )
    manifest = {
        "variant": "full",
        "status": "imported_complete",
        "config_fingerprint": config_fingerprint(
            full,
            exclude=("output_dir",),
        ),
        "resolved_config": full.to_dict(),
    }
    path = tmp_path / "ablation_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, _ = _load_reference_manifest(path)
    assert loaded.batch_size == 8
    assert loaded.grad_accum_steps == 2
    assert loaded.efficiency_mode == "optimized"
    assert tuple(loaded.train_short_sides) == (480, 512, 544, 576, 608)
    _validate_reference_updates(
        loaded,
        {
            "output_dir": str(tmp_path / "variant"),
            "quick_eval_every_steps": 0,
            "full_eval_last_epoch_only": True,
            "oracle_monitor_every_optimizer_steps": 0,
        },
    )
    try:
        _validate_reference_updates(loaded, {"batch_size": 4})
    except RuntimeError as exc:
        assert "controlled training configuration" in str(exc)
    else:
        raise AssertionError("reference training settings were allowed to drift")


def test_top_score_seeding_contract() -> None:
    config = EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=3,
        evidence_per_slot=2,
        bank_min_per_level=0,
        bank_pre_topk_per_level=4,
        bank_topk_train=4,
        bank_topk_eval=4,
        hdn_max_pairs=1,
        roi_size=3,
        seed_selection_mode="top_score",
    )
    seeder = SlotSeeder(config)
    indices, valid = seeder.select_indices(
        {
            "valid": torch.tensor([[True, True, True, True]]),
            "score": torch.tensor([[[0.1], [0.9], [0.3], [0.7]]]),
        }
    )
    assert indices.tolist() == [[1, 3, 2]]
    assert valid.tolist() == [[True, True, True]]


def test_all_ablation_forward_routes() -> None:
    images, pad_mask, _ = _batch()
    for key in ABLATIONS:
        config = build_variant_config(key, base=_compact_base())
        model = EgoDetV2(config, fake_backbone=True).eval()
        if key == "no_semantic_completion":
            model.semantic_support.retrieve = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("regional retrieval was not bypassed")
            )
            for update in model.slot_updates:
                update.semantic_attention.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("global semantic attention was not bypassed")
                )
        if key == "no_slot_competition":
            for update in model.slot_updates:
                update.competition.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("slot competition was not bypassed")
                )
        with torch.no_grad():
            outputs = model(images, pad_mask, targets=None)
        assert outputs["pred_boxes"].shape == (1, config.num_slots, 4)
        assert torch.isfinite(outputs["pred_boxes"]).all()
        if key == "direct_slot_regression":
            assert len(model.readouts) == 0
            assert len(model.direct_box_heads) == config.slot_rounds
            assert not outputs["ownership"][-1]["cand_valid"].any()
        if key == "no_bounded_residual":
            torch.testing.assert_close(
                outputs["readout"][-1]["b"],
                outputs["readout"][-1]["b_hat_norm"],
            )
        if key == "no_dustbin_gating":
            expected = outputs["ownership"][-1]["cand_valid"].any(dim=-1)
            torch.testing.assert_close(
                outputs["ownership"][-1]["geometry_gate"],
                expected.to(outputs["pred_boxes"].dtype),
            )
        if key == "full":
            torch.testing.assert_close(
                outputs["ownership"][-1]["geometry_gate"],
                outputs["ownership"][-1]["proposal_mass"],
            )


def test_direct_regression_training_contract() -> None:
    config = build_variant_config(
        "direct_slot_regression",
        base=_compact_base(),
    )
    model = EgoDetV2(config, fake_backbone=True).train()
    criterion = EgoDetV2Criterion(config)
    images, pad_mask, targets = _batch()
    outputs = model(images, pad_mask, targets, epoch=2)
    losses = criterion(outputs, targets, epoch=2, epoch_progress=1.0)
    assert torch.isfinite(losses["loss_total"])
    for round_index in range(1, config.slot_rounds + 1):
        assert float(
            losses[f"loss_ownership_r{round_index}"].detach()
        ) == 0.0
        assert float(
            losses[f"loss_dustbin_r{round_index}"].detach()
        ) == 0.0
    assert float(losses["loss_gain"].detach()) == 0.0
    losses["loss_total"].backward()
    gradient = model.direct_box_heads[-1].regressor[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_controlled_resume_contract(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = build_variant_config("full", base=_compact_base())
    checkpoint = {
        "config": config.to_dict(),
        "epoch": 1,
    }
    epoch_checkpoint = tmp_path / "epoch_1.pth"
    torch.save(checkpoint, epoch_checkpoint)
    _validate_resume(
        epoch_checkpoint,
        config,
        allow_non_epoch_resume=False,
    )

    latest_checkpoint = tmp_path / "latest.pth"
    torch.save(checkpoint, latest_checkpoint)
    try:
        _validate_resume(
            latest_checkpoint,
            config,
            allow_non_epoch_resume=False,
        )
    except RuntimeError as exc:
        assert "epoch-boundary" in str(exc)
    else:
        raise AssertionError("latest.pth bypassed controlled resume safety")
    _validate_resume(
        latest_checkpoint,
        config,
        allow_non_epoch_resume=True,
    )

    other_variant = build_variant_config(
        "no_slot_competition",
        base=_compact_base(),
    )
    try:
        _validate_resume(
            epoch_checkpoint,
            other_variant,
            allow_non_epoch_resume=False,
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("cross-variant checkpoint resume was accepted")


def _write_synthetic_run(root: Path, key: str, index: int) -> None:
    run = root / key
    run.mkdir(parents=True)
    manifest = {
        "variant": key,
        "control_fingerprint": "shared-control",
        "data": {
            "project_root": "/project",
            "train_annotations": "/project/train.json",
            "val_annotations": "/project/val.json",
            "image_root": "/project/images",
        },
        "resolved_config": {"epochs": 12},
    }
    (run / "ablation_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    quick = {
        "kind": "quick500",
        "epoch": 12,
        "AP": 0.99,
    }
    full = {
        "epoch": 12,
        "global_optimizer_step": 1000,
        "AP": 0.30 + 0.001 * index,
        "AP50": 0.50 + 0.001 * index,
        "AP75": 0.32 + 0.001 * index,
        "AP_small": 0.15,
        "AP_medium": 0.33,
        "AP_large": 0.45,
        "matched_mean_iou": 0.70 + 0.001 * index,
        "duplicate_rate": 0.02 + 0.0001 * index,
    }
    (run / "eval_history.jsonl").write_text(
        json.dumps(quick) + "\n" + json.dumps(full) + "\n",
        encoding="utf-8",
    )


def test_ablation_summary_contract(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    for index, key in enumerate(ABLATIONS):
        _write_synthetic_run(run_root, key, index)
    output = tmp_path / "summary"
    result = collect_results(run_root, output, strict=True)
    assert not result["incomplete_variants"]
    assert len(result["tables"]["overall_component_analysis"]) == 4
    assert len(result["tables"]["object_slot_formation"]) == 4
    assert len(result["tables"]["slot_conditioned_geometric_readout"]) == 5
    expected = {
        "ablation_results.json",
        "all_variants.csv",
        "overall_component_analysis.csv",
        "object_slot_formation.csv",
        "slot_conditioned_geometric_readout.csv",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert not list(output.glob("*.tex"))
    with (output / "overall_component_analysis.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["Variant"] == "EGO-Det"
    assert float(rows[-1]["AP"]) < 100.0


def main() -> None:
    test_registry_contract()
    test_frozen_backbone_contract()
    test_full_eval_cadence_contract()
    test_top_score_seeding_contract()
    test_all_ablation_forward_routes()
    test_direct_regression_training_contract()
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        test_reduced_protocol_cli_contract(temporary_path / "cli")
        test_reference_manifest_contract(temporary_path / "reference")
        test_controlled_resume_contract(temporary_path / "resume")
        test_ablation_summary_contract(temporary_path)
    print("ego_det_v2 ablation contracts: PASS")


if __name__ == "__main__":
    main()
