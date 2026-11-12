from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List
from unittest import mock

import torch
import torch.nn.functional as F
from PIL import Image

from ..configs.config import EgoDetV2Config
from ..data.coco_dataset import CocoDetectionV2
from ..data.collate import collate_detection_batch
from ..engine.diagnostics import (
    OPTIMIZER_GROUP_NAMES,
    ParameterUpdateAuditor,
    measure_geometry_gradient_competition,
)
from ..engine.evaluator import evaluate
from ..engine.geometry_diagnostics import (
    GeometryDiagnosticsAccumulator,
    brute_force_proposal_metrics,
)
from ..engine.train_loop import (
    build_optimizer,
    build_optimizer_step_scheduler,
    optimizer_step_warmup_scale,
    run_fixed_steps,
)
from ..launch.launch_memory_sweep import memory_sweep_cases
from ..launch.launch_overfit8 import (
    assert_exact_overfit_ids,
    build_composite_geometry_gate,
    parse_args as parse_overfit_args,
    require_batch4_memory_preflight,
)
from ..losses.criterion import EgoDetV2Criterion
from ..losses.dense_losses import dense_geometry_losses
from ..losses.ownership_losses import (
    build_soft_ownership_targets,
    soft_ownership_losses,
)
from ..losses.slot_losses import (
    duplicate_margin_loss,
    existence_loss,
    readout_gain_loss,
    slot_round_losses,
)
from ..models.box_ops import (
    aligned_box_iou,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
)
from ..models.backbone import _configure_overlock_checkpointing
from ..models.dense_geometry_branch import (
    DenseGeometryBranch,
    assign_proposal_identity_from_current_iou,
    correct_assign_gt_with_current_iou,
)
from ..models.ego_det_v2 import EgoDetV2
from ..models.evidence_readout import apply_bounded_residual
from ..models.hdn import HypothesisDenoising
from ..models.heads import compose_scores
from ..models.matcher import HungarianMatcherV2
from ..models.slot_update import ZeroMLP


def _assert_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AssertionError(f"{name} contains non-finite values")


def _compact_config(full_shape: bool) -> EgoDetV2Config:
    if full_shape:
        return EgoDetV2Config(
            d_model=32,
            tower_convs=1,
            sem_convs=1,
            roi_size=3,
            hdn_max_pairs=10,
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
    )


def _synthetic_targets(side: int, gt_count: int = 5) -> List[Dict]:
    boxes0 = []
    labels0 = []
    for index in range(gt_count):
        cx = 0.12 + 0.16 * index
        cy = 0.18 + 0.13 * index
        width = 0.12 + 0.01 * (index % 2)
        height = 0.14
        boxes0.append([cx, cy, width, height])
        labels0.append(index)
    normalized = torch.tensor(boxes0, dtype=torch.float32)
    absolute = box_cxcywh_to_xyxy(normalized) * float(side)
    return [
        {
            "labels": torch.tensor(labels0, dtype=torch.long),
            "boxes": normalized,
            "boxes_xyxy_abs": absolute,
            "image_id": 1,
            "orig_size": (side, side),
            "size": (side, side),
        },
        {
            "labels": torch.empty(0, dtype=torch.long),
            "boxes": torch.empty(0, 4),
            "boxes_xyxy_abs": torch.empty(0, 4),
            "image_id": 2,
            "orig_size": (side, side),
            "size": (side, side),
        },
    ]


def test_defaults() -> None:
    config = EgoDetV2Config()
    assert config.bank_topk_train == 1000
    assert config.bank_topk_eval == 1500
    assert config.bank_min_per_level == 64
    assert config.evidence_per_slot == 48
    assert config.slot_rounds == 3
    assert config.overlock_gradient_checkpointing is True
    assert config.dense_assignment_mode == "stable_fcos"
    assert config.proposal_identity_iou_threshold == 0.5
    assert config.assign_ensure_min_one_positive is False
    for invalid in (
        {"dense_assignment_mode": "dynamic"},
        {"proposal_identity_iou_threshold": -0.01},
        {"proposal_identity_iou_threshold": 1.01},
    ):
        try:
            replace(config, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid assignment config was accepted: {invalid}")


def test_overlock_checkpoint_attribute_configuration() -> None:
    class HasCheckpoint(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_checkpoint = False

    class NoCheckpoint(torch.nn.Module):
        pass

    root = torch.nn.Module()
    root.first = HasCheckpoint()
    root.nested = torch.nn.Sequential(NoCheckpoint(), HasCheckpoint())
    untouched = root.nested[0]

    configured = _configure_overlock_checkpointing(root, True)

    assert configured == 2
    assert root.first.use_checkpoint is True
    assert root.nested[1].use_checkpoint is True
    assert not hasattr(untouched, "use_checkpoint")

    configured = _configure_overlock_checkpointing(root, False)

    assert configured == 2
    assert root.first.use_checkpoint is False
    assert root.nested[1].use_checkpoint is False
    assert not hasattr(untouched, "use_checkpoint")


def test_overfit_memory_parser() -> None:
    defaults = parse_overfit_args([])
    assert defaults.batch_size == 1
    assert defaults.grad_accum == 1
    assert defaults.image_short_side == 512
    assert defaults.image_max_size == 768
    assert defaults.overlock_checkpointing is True
    assert defaults.warmup_steps == 25
    assert defaults.base_lr == 3.0e-4
    assert defaults.geometry_lr == 3.0e-4
    assert defaults.fpn_lr == 1.0e-4
    assert defaults.backbone_lr == 1.0e-5
    assert defaults.backbone_stem_lr == 1.0e-6
    assert defaults.weight_decay == 0.0
    assert defaults.require_geometry_gate is True
    assert defaults.geometry_gate_metric == "dense_all_recall@0.5"
    assert defaults.geometry_gate_threshold == 0.95
    assert defaults.geometry_gate_bank_metric == "bank_top300_recall@0.5"
    assert defaults.geometry_gate_bank_threshold == 0.95
    assert defaults.dense_assignment_mode == "stable_fcos"
    assert defaults.proposal_identity_iou_threshold == 0.5
    assert defaults.assign_ensure_min_one_positive is False
    assert defaults.geometry_eval_steps == (0, 25, 50, 100, 200, 300)
    assert defaults.full_eval_steps == (0, 25, 50, 100, 250, 500)
    assert defaults.no_progress is False

    disabled = parse_overfit_args(["--no-overlock-checkpointing"])
    assert disabled.overlock_checkpointing is False
    hidden_progress = parse_overfit_args(["--no-progress"])
    assert hidden_progress.no_progress is True

    custom = parse_overfit_args(
        [
            "--batch-size",
            "2",
            "--grad-accum",
            "8",
            "--image-short-side",
            "448",
            "--image-max-size",
            "672",
            "--overlock-checkpointing",
        ]
    )
    assert custom.batch_size == 2
    assert custom.grad_accum == 8
    assert custom.image_short_side == 448
    assert custom.image_max_size == 672
    assert custom.overlock_checkpointing is True
    assignment = parse_overfit_args(
        [
            "--dense-assignment-mode",
            "legacy_dynamic",
            "--proposal-identity-iou-threshold",
            "0.6",
            "--assign-ensure-min-one-positive",
        ]
    )
    assert assignment.dense_assignment_mode == "legacy_dynamic"
    assert assignment.proposal_identity_iou_threshold == 0.6
    assert assignment.assign_ensure_min_one_positive is True

    preset = parse_overfit_args(["--preset", "diagnostic-fast"])
    assert preset.batch_size == 4
    assert preset.grad_accum == 1
    assert preset.image_short_side == 512
    assert preset.image_max_size == 768
    assert preset.warmup_steps == 25
    assert preset.overlock_checkpointing is True
    assert preset.preset_expansion["periodic_eval"] is True

    invalid_cases = [
        (["--batch-size", "0"], "--batch-size"),
        (["--grad-accum", "0"], "--grad-accum"),
        (["--image-short-side", "127"], "--image-short-side"),
        (["--warmup-steps", "0"], "--warmup-steps"),
        (["--base-lr", "0"], "--base-lr"),
        (
            ["--proposal-identity-iou-threshold", "1.01"],
            "--proposal-identity-iou-threshold",
        ),
        (
            ["--geometry-gate-bank-threshold", "-0.1"],
            "--geometry-gate-bank-threshold",
        ),
        (
            ["--slot-to-geometry-grad-scale", "1.1"],
            "--slot-to-geometry-grad-scale",
        ),
        (
            ["--image-short-side", "512", "--image-max-size", "511"],
            "--image-max-size",
        ),
    ]
    for arguments, expected_name in invalid_cases:
        try:
            parse_overfit_args(arguments)
        except ValueError as error:
            assert expected_name in str(error)
        else:
            raise AssertionError(
                f"invalid Overfit memory arguments were accepted: {arguments}"
            )


def test_optimizer_named_groups_and_scheduler_contract() -> None:
    config = _compact_config(False)
    model = EgoDetV2(config, fake_backbone=True)
    optimizer = build_optimizer(
        model,
        config,
        base_lr=3.0e-4,
        geometry_lr=3.0e-4,
        fpn_lr=1.0e-4,
        backbone_lr=1.0e-5,
        backbone_stem_lr=1.0e-6,
        weight_decay=0.0,
        print_groups=False,
    )
    names = {str(group.get("name", "")) for group in optimizer.param_groups}
    assert set(OPTIMIZER_GROUP_NAMES).issubset(names)
    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert set(parameter_ids) == trainable_ids
    expected_lrs = {
        "backbone_stem": 1.0e-6,
        "backbone": 1.0e-5,
        "fpn": 1.0e-4,
        "geometry_tower": 3.0e-4,
        "geometry_obj_head": 3.0e-4,
        "geometry_box_head": 3.0e-4,
        "geometry_quality_head": 3.0e-4,
        "semantic_tower": 3.0e-4,
        "slot_modules": 3.0e-4,
        "readout": 3.0e-4,
        "prediction_heads": 3.0e-4,
        "hdn": 3.0e-4,
    }
    for group in optimizer.param_groups:
        assert float(group["lr"]) == expected_lrs[str(group["name"])]

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    scheduler_optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 3.0e-4, "name": "geometry_tower"}]
    )
    scheduler = build_optimizer_step_scheduler(
        scheduler_optimizer,
        warmup_steps=25,
    )
    observed = {0: float(scheduler_optimizer.param_groups[0]["lr"])}
    for optimizer_step in range(1, 501):
        scheduler_optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.zeros_like(parameter)
        scheduler_optimizer.step()
        scheduler.step()
        if optimizer_step in {1, 10, 25, 50, 300, 500}:
            observed[optimizer_step] = float(
                scheduler_optimizer.param_groups[0]["lr"]
            )
    for optimizer_step in (0, 1, 10, 25, 50, 300, 500):
        expected = 3.0e-4 * optimizer_step_warmup_scale(
            optimizer_step,
            25,
        )
        assert abs(observed[optimizer_step] - expected) < 1.0e-12


def test_overfit_dataset_and_memory_sweep_contract() -> None:
    class DatasetIds:
        def __init__(self, image_ids):
            self.image_ids = image_ids

    expected = list(range(8))
    assert assert_exact_overfit_ids(
        DatasetIds(expected),
        DatasetIds(expected.copy()),
    ) == expected
    for invalid in (list(range(7)), list(reversed(expected))):
        try:
            assert_exact_overfit_ids(DatasetIds(expected), DatasetIds(invalid))
        except AssertionError:
            pass
        else:
            raise AssertionError("mismatched Overfit8 IDs were accepted")

    cases = memory_sweep_cases()
    assert len(cases) == 12
    assert (4, 512, 768) in cases
    assert all(short_side in {512, 608} for _, short_side, _ in cases)
    assert all(max_size in {768, 896} for _, _, max_size in cases)
    assert all(short_side != 672 for _, short_side, _ in cases)
    with tempfile.TemporaryDirectory() as temporary:
        evidence = Path(temporary) / "sweep.json"
        evidence.write_text(
            json.dumps(
                {
                    "candidate_batch4_512_768_proven": True,
                    "cases": [
                        {
                            "batch_size": 4,
                            "short_side": 512,
                            "max_size": 768,
                            "status": "PASS",
                            "oom": False,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        candidate = require_batch4_memory_preflight(evidence)
        assert candidate["status"] == "PASS"
        evidence.write_text(
            json.dumps(
                {
                    "candidate_batch4_512_768_proven": False,
                    "cases": [],
                }
            ),
            encoding="utf-8",
        )
        try:
            require_batch4_memory_preflight(evidence)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unproven batch4 preflight was accepted")


def test_brute_force_proposal_metric_contract() -> None:
    gt = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    proposals = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [40.0, 40.0, 50.0, 50.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.1])
    metrics = brute_force_proposal_metrics(
        proposals,
        gt,
        scores=scores,
        ks=(1, 3),
    )
    assert metrics["Recall@1/0.50"] == 0.5
    assert metrics["Recall@1/0.75"] == 0.5
    assert metrics["AR@1"] == 0.5
    assert metrics["Recall@3/0.50"] == 1.0
    assert metrics["AR@3"] == 1.0
    assert metrics["best_iou_mean"] == 1.0
    assert metrics["best_iou_median"] == 1.0
    assert metrics["best_iou_min"] == 1.0

    config = EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=4,
        evidence_per_slot=2,
        bank_min_per_level=2,
        bank_pre_topk_per_level=5,
        bank_topk_train=10,
        bank_topk_eval=10,
        hdn_max_pairs=2,
        roi_size=3,
    )
    branch = DenseGeometryBranch(config)
    level_scores = [
        torch.tensor([[0.99, 0.98, 0.97, 0.96, 0.95]]),
        torch.tensor([[0.40, 0.39, 0.38, 0.37, 0.36]]),
        torch.tensor([[0.30, 0.29, 0.28, 0.27, 0.26]]),
        torch.tensor([[0.20, 0.19, 0.18, 0.17, 0.16]]),
    ]
    valid = [torch.ones_like(value, dtype=torch.bool) for value in level_scores]
    indices, selected_valid, counts = branch._select_indices(
        level_scores,
        valid,
        10,
    )
    assert int(selected_valid.sum()) == 10
    assert counts.tolist() == [[4, 2, 2, 2]]
    assert indices[0].tolist() == [0, 1, 2, 3, 5, 6, 10, 11, 15, 16]


def test_evaluation_restores_model_mode() -> None:
    model = torch.nn.Linear(2, 2).train()
    with mock.patch(
        "ego_det_v2.engine.evaluator._evaluate_impl",
        side_effect=lambda current_model, data_loader, **kwargs: (
            current_model.eval(),
            {},
        )[1],
    ):
        result = evaluate(
            model,
            [],
            matcher=None,
            device=torch.device("cpu"),
        )
    assert result == {}
    assert model.training is True


def test_gradient_diagnostics_contract() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.SGD(
        [
            {
                "name": "geometry_tower",
                "params": [parameter],
                "lr": 0.1,
            }
        ]
    )
    partitions = {
        "dense": (2.0 * parameter).square().sum(),
        "slot": (-parameter).square().sum(),
        "ownership": (3.0 * parameter).square().sum(),
    }
    competition = measure_geometry_gradient_competition(
        partitions,
        optimizer,
        optimizer_step=25,
        stage_optimizer_step=25,
    )
    for name in (
        "geometry_grad_from_dense",
        "geometry_grad_from_slot",
        "geometry_grad_from_ownership",
        "slot_to_dense_grad_ratio",
        "ownership_to_dense_grad_ratio",
        "cosine_dense_vs_slot",
    ):
        assert torch.isfinite(torch.tensor(float(competition[name])))
    assert competition["geometry_grad_from_dense"] > 0.0
    assert competition["geometry_grad_from_slot"] > 0.0

    auditor = ParameterUpdateAuditor(optimizer)
    optimizer.zero_grad(set_to_none=True)
    partitions["dense"].backward()
    auditor.capture_before_step()
    optimizer.step()
    record = auditor.capture_after_step(
        optimizer_step=25,
        phase="geometry",
    )
    module = record["modules"]["geometry_tower"]
    assert module["grad_norm"] > 0.0
    assert module["parameter_norm"] > 0.0
    assert module["update_norm"] > 0.0
    assert module["relative_update_norm"] > 0.0


def test_coordinate_and_residual_contract() -> None:
    boxes = torch.tensor([[0.5, 0.4, 0.2, 0.3]], dtype=torch.float32)
    roundtrip = box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes))
    torch.testing.assert_close(roundtrip, boxes)
    raw = torch.tensor([[0.3, -0.2, 0.4, -0.5]], dtype=torch.float32)
    actual = apply_bounded_residual(boxes, raw)
    expected = torch.stack(
        (
            boxes[:, 0] + 0.5 * torch.tanh(raw[:, 0]) * boxes[:, 2],
            boxes[:, 1] + 0.5 * torch.tanh(raw[:, 1]) * boxes[:, 3],
            boxes[:, 2] * torch.exp(0.35 * torch.tanh(raw[:, 2])),
            boxes[:, 3] * torch.exp(0.35 * torch.tanh(raw[:, 3])),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_iou_corrected_assign_gt() -> None:
    gt = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    proposals = torch.tensor(
        [
            [20.0, 20.0, 30.0, 30.0],
            [0.0, 0.0, 10.0, 10.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    base = torch.tensor([0, -1, -1])
    corrected, assigned_iou = correct_assign_gt_with_current_iou(
        base,
        proposals,
        gt,
        promote_threshold=0.5,
    )
    assert corrected.tolist() == [1, 0, -1]
    torch.testing.assert_close(assigned_iou[:2], torch.ones(2))


def _assignment_test_config(
    *,
    mode: str = "stable_fcos",
    fallback: bool = False,
) -> EgoDetV2Config:
    return EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=4,
        evidence_per_slot=2,
        bank_min_per_level=1,
        bank_pre_topk_per_level=4,
        bank_topk_train=8,
        bank_topk_eval=8,
        hdn_max_pairs=2,
        roi_size=3,
        dense_assignment_mode=mode,
        assign_ensure_min_one_positive=fallback,
    )


def test_stable_dense_assignment_decouples_proposal_identity() -> None:
    config = _assignment_test_config()
    branch = DenseGeometryBranch(config)
    gt = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    targets = [{"boxes_xyxy_abs": gt}]

    # The FCOS location belongs to GT0, while its current decoded box is GT1.
    result = branch._assign_level(
        torch.tensor([[5.0, 5.0]]),
        gt[1:2][None],
        targets,
        stride=8,
        scale_range=(0.0, 64.0),
    )
    train_assign, proposal_assign, proposal_iou, target_boxes, source = result
    assert train_assign.tolist() == [[0]]
    assert proposal_assign.tolist() == [[1]]
    torch.testing.assert_close(proposal_iou, torch.ones_like(proposal_iou))
    torch.testing.assert_close(target_boxes[0, 0], gt[0])
    assert source.tolist() == [[0]]

    # A background location with a high-IoU decoded proposal remains a dense
    # negative while retaining its independent proposal identity.
    result = branch._assign_level(
        torch.tensor([[50.0, 50.0]]),
        gt[0:1][None],
        targets,
        stride=8,
        scale_range=(0.0, 64.0),
    )
    train_assign, proposal_assign, _, target_boxes, source = result
    assert train_assign.tolist() == [[-1]]
    assert proposal_assign.tolist() == [[0]]
    assert source.tolist() == [[-1]]
    torch.testing.assert_close(target_boxes, torch.zeros_like(target_boxes))
    dense_level = {
        "obj_logits": torch.zeros((1, 1, 1), requires_grad=True),
        "qual_logits": torch.zeros((1, 1, 1), requires_grad=True),
        "boxes": gt[0:1][None].clone().requires_grad_(),
        "target_boxes": target_boxes,
        "assign_gt": train_assign,
        "train_assign_gt": train_assign,
        "train_assign_source": source,
        "valid": torch.ones((1, 1), dtype=torch.bool),
    }
    losses = dense_geometry_losses({"levels": [dense_level]})
    assert float(losses["dense_positive_count"]) == 0.0
    assert float(losses["native_positive_count"]) == 0.0

    empty_identity, empty_iou = assign_proposal_identity_from_current_iou(
        gt,
        torch.empty((0, 4)),
        threshold=0.5,
    )
    assert empty_identity.tolist() == [-1, -1]
    torch.testing.assert_close(empty_iou, torch.zeros_like(empty_iou))


def test_bank_uses_proposal_identity() -> None:
    config = _assignment_test_config()
    branch = DenseGeometryBranch(config).train()
    train_assign = torch.tensor([[0, 0]])
    proposal_assign = torch.tensor([[1, -1]])
    level = {
        "boxes": torch.tensor(
            [[[0.0, 0.0, 10.0, 10.0], [30.0, 30.0, 40.0, 40.0]]]
        ),
        "desc": torch.zeros((1, 2, config.d_model)),
        "obj_logits": torch.tensor([[[5.0], [0.0]]]),
        "qual_logits": torch.tensor([[[5.0], [5.0]]]),
        "valid": torch.ones((1, 2), dtype=torch.bool),
        "assign_gt": train_assign,
        "train_assign_gt": train_assign,
        "proposal_gt_assign": proposal_assign,
        "assign_iou": torch.tensor([[0.9, 0.0]]),
        "proposal_assign_iou": torch.tensor([[0.9, 0.0]]),
    }
    bank = branch._build_bank([level])
    assert bank["selection_indices"][0, :2].tolist() == [0, 1]
    assert bank["gt_assign"][0, :2].tolist() == [1, -1]
    assert bank["gt_assign"][0, 0] != train_assign[0, 0]


def test_legacy_dynamic_assignment_reproduction() -> None:
    config = _assignment_test_config(mode="legacy_dynamic")
    branch = DenseGeometryBranch(config)
    gt = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    locations = torch.tensor([[5.0, 5.0], [25.0, 25.0], [50.0, 50.0]])
    decoded = torch.stack((gt[1], gt[1], gt[0]))[None]
    base = branch._fcos_assign(
        locations,
        gt,
        stride=8,
        scale_range=(0.0, 64.0),
    )
    expected_assign, _ = correct_assign_gt_with_current_iou(
        base,
        decoded[0],
        gt,
        promote_threshold=config.assign_iou_promote_threshold,
    )
    result = branch._assign_level(
        locations,
        decoded,
        [{"boxes_xyxy_abs": gt}],
        stride=8,
        scale_range=(0.0, 64.0),
    )
    train_assign, proposal_assign, proposal_iou, target_boxes, source = result
    torch.testing.assert_close(train_assign[0], expected_assign)
    assert train_assign.tolist() == [[1, 1, 0]]
    assert proposal_assign.tolist() == [[1, 1, 0]]
    torch.testing.assert_close(proposal_iou, torch.ones_like(proposal_iou))
    assert source.tolist() == [[2, 0, 2]]
    torch.testing.assert_close(target_boxes[0], gt[expected_assign])

    obj = torch.tensor([[[0.2], [-0.3], [0.8]]], requires_grad=True)
    qual = torch.tensor([[[0.4], [0.1], [-0.2]]], requires_grad=True)
    valid = torch.ones((1, 3), dtype=torch.bool)
    old_level = {
        "obj_logits": obj,
        "qual_logits": qual,
        "boxes": decoded.clone().requires_grad_(),
        "target_boxes": target_boxes,
        "assign_gt": expected_assign[None],
        "valid": valid,
    }
    new_level = {
        **old_level,
        "assign_gt": train_assign,
        "train_assign_gt": train_assign,
        "train_assign_source": source,
    }
    old_losses = dense_geometry_losses({"levels": [old_level]})
    new_losses = dense_geometry_losses({"levels": [new_level]})
    for name in ("loss_dense_obj", "loss_dense_giou", "loss_dense_quality"):
        torch.testing.assert_close(new_losses[name], old_losses[name])
    assert float(new_losses["dense_positive_count"]) == 3.0
    assert float(new_losses["native_positive_count"]) == 1.0
    assert float(new_losses["legacy_promoted_count"]) == 2.0

    common_bank_level = {
        "boxes": decoded,
        "desc": torch.zeros((1, 3, config.d_model)),
        "obj_logits": obj.detach(),
        "qual_logits": qual.detach(),
        "valid": valid,
        "assign_gt": train_assign,
        "train_assign_gt": train_assign,
        "proposal_gt_assign": proposal_assign,
        "assign_iou": proposal_iou,
        "proposal_assign_iou": proposal_iou,
    }
    current_indices = branch._build_bank([common_bank_level])[
        "selection_indices"
    ]
    legacy_identity_level = {
        **common_bank_level,
        "proposal_gt_assign": train_assign,
    }
    legacy_indices = branch._build_bank([legacy_identity_level])[
        "selection_indices"
    ]
    torch.testing.assert_close(current_indices, legacy_indices)


def _fallback_levels(
    config: EgoDetV2Config,
    *,
    p3_assign: torch.Tensor,
    p3_valid: torch.Tensor | None = None,
) -> list[Dict[str, torch.Tensor]]:
    levels: list[Dict[str, torch.Tensor]] = []
    for level_index, name in enumerate(config.fpn_levels):
        if level_index == 0:
            locations = torch.tensor([[5.0, 5.0], [6.0, 5.0], [20.0, 20.0]])
            assign = p3_assign.clone()
            valid = (
                torch.ones_like(assign, dtype=torch.bool)
                if p3_valid is None
                else p3_valid.clone()
            )
        else:
            locations = torch.tensor([[100.0, 100.0]])
            assign = torch.full((1, 1), -1, dtype=torch.long)
            valid = torch.ones_like(assign, dtype=torch.bool)
        source = torch.where(
            assign >= 0,
            torch.zeros_like(assign),
            torch.full_like(assign, -1),
        )
        target_boxes = torch.zeros((1, assign.shape[1], 4))
        levels.append(
            {
                "name": name,
                "locations": locations,
                "boxes": torch.zeros((1, assign.shape[1], 4)),
                "valid": valid,
                "assign_gt": assign,
                "train_assign_gt": assign,
                "train_assign_source": source,
                "target_boxes": target_boxes,
            }
        )
    return levels


def test_min_one_fallback_contract() -> None:
    gt = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    )
    targets = [{"boxes_xyxy_abs": gt}]
    enabled_config = _assignment_test_config(fallback=True)
    enabled_branch = DenseGeometryBranch(enabled_config)
    levels = _fallback_levels(
        enabled_config,
        p3_assign=torch.tensor([[1, -1, -1]]),
    )
    success, failure = enabled_branch._ensure_min_one_positive(levels, targets)
    assert (success, failure) == (1, 0)
    assert levels[0]["train_assign_gt"].tolist() == [[1, 0, -1]]
    assert levels[0]["assign_gt"].tolist() == [[1, 0, -1]]
    assert levels[0]["train_assign_source"].tolist() == [[0, 1, -1]]
    torch.testing.assert_close(levels[0]["target_boxes"][0, 1], gt[0])
    assert all(
        int((level["train_assign_source"] == 1).sum()) == (1 if index == 0 else 0)
        for index, level in enumerate(levels)
    )

    disabled_config = _assignment_test_config(fallback=False)
    disabled_levels = _fallback_levels(
        disabled_config,
        p3_assign=torch.tensor([[1, -1, -1]]),
    )
    assert DenseGeometryBranch(disabled_config)._ensure_min_one_positive(
        disabled_levels,
        targets,
    ) == (0, 0)
    assert int((disabled_levels[0]["train_assign_gt"] == 0).sum()) == 0

    legacy_config = _assignment_test_config(
        mode="legacy_dynamic",
        fallback=True,
    )
    legacy_levels = _fallback_levels(
        legacy_config,
        p3_assign=torch.tensor([[1, -1, -1]]),
    )
    assert DenseGeometryBranch(legacy_config)._ensure_min_one_positive(
        legacy_levels,
        targets,
    ) == (0, 0)

    failure_levels = _fallback_levels(
        enabled_config,
        p3_assign=torch.tensor([[1, 1, 1]]),
    )
    success, failure = enabled_branch._ensure_min_one_positive(
        failure_levels,
        targets,
    )
    assert (success, failure) == (0, 1)
    assert failure_levels[0]["train_assign_gt"].tolist() == [[1, 1, 1]]


def test_assignment_positive_feedback_regression() -> None:
    gt = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    background_count = 12
    locations = torch.cat(
        (
            torch.tensor([[5.0, 5.0], [25.0, 25.0]]),
            torch.stack(
                (
                    torch.arange(background_count).float() + 100.0,
                    torch.full((background_count,), 100.0),
                ),
                dim=-1,
            ),
        ),
        dim=0,
    )
    decoded = torch.cat(
        (
            gt[1:2],
            gt[1:2],
            gt[0:1].expand(background_count, -1),
        ),
        dim=0,
    )[None]
    targets = [{"boxes_xyxy_abs": gt}]
    stable = DenseGeometryBranch(_assignment_test_config())
    legacy = DenseGeometryBranch(
        _assignment_test_config(mode="legacy_dynamic")
    )
    stable_assign, stable_identity, _, _, _ = stable._assign_level(
        locations,
        decoded,
        targets,
        stride=8,
        scale_range=(0.0, 64.0),
    )
    legacy_assign, _, _, _, legacy_source = legacy._assign_level(
        locations,
        decoded,
        targets,
        stride=8,
        scale_range=(0.0, 64.0),
    )
    assert stable_assign[0, :2].tolist() == [0, 1]
    assert stable_identity[0, :2].tolist() == [1, 1]
    assert int((stable_assign >= 0).sum()) == 2
    assert int((legacy_assign >= 0).sum()) == 2 + background_count
    assert legacy_assign[0, 0] == 1
    assert bool((legacy_source[0, 2:] == 2).all())


def test_assignment_diagnostics_and_composite_gate() -> None:
    config = _assignment_test_config()
    predicted = torch.tensor([[[0.0, 0.0, 5.0, 5.0]]])
    target_box = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    level = {
        "name": "p3",
        "valid": torch.ones((1, 1), dtype=torch.bool),
        "assign_gt": torch.tensor([[0]]),
        "train_assign_gt": torch.tensor([[0]]),
        "train_assign_source": torch.tensor([[0]]),
        "proposal_gt_assign": torch.tensor([[0]]),
        "assign_iou": torch.tensor([[1.0]]),
        "proposal_assign_iou": torch.tensor([[1.0]]),
        "boxes": predicted,
        "target_boxes": target_box,
        "obj_logits": torch.zeros((1, 1, 1)),
        "qual_logits": torch.zeros((1, 1, 1)),
    }
    bank = {
        "boxes": predicted,
        "valid": torch.ones((1, 1), dtype=torch.bool),
    }
    targets = [
        {
            "boxes_xyxy_abs": target_box[0],
            "image_id": 7,
        }
    ]
    accumulator = GeometryDiagnosticsAccumulator(config)
    accumulator.update(
        {
            "levels": [level],
            "fallback_failure_count": torch.tensor(0.0),
        },
        bank,
        targets,
    )
    metrics = accumulator.compute()
    assert abs(float(metrics["positive_pred_iou_mean"]) - 0.25) < 1.0e-6
    assert metrics["native_assignable_gt_ratio"] == 1.0
    assert metrics["train_assignable_gt_ratio"] == 1.0
    assert metrics["native_positive_locations"] == 1
    assert metrics["fallback_positive_locations"] == 0
    assert metrics["proposal_identity_assigned_locations"] == 1
    assert metrics["proposal_identity_assigned_ratio"] == 1.0
    assert metrics["proposal_identity_gt_coverage"] == 1.0
    assert "gt_with_zero_positive_on_level" in metrics["P3"]

    dense_failure = build_composite_geometry_gate(
        {
            "dense_all_recall@0.5": 0.94,
            "bank_top300_recall@0.5": 1.0,
        },
        required=True,
    )
    assert dense_failure["passed"] is False
    assert dense_failure["diagnosis"] == "assignment_or_box_regression"
    bank_failure = build_composite_geometry_gate(
        {
            "dense_all_recall@0.5": 1.0,
            "bank_top300_recall@0.5": 0.94,
        },
        required=True,
    )
    assert bank_failure["passed"] is False
    assert bank_failure["diagnosis"] == "ranking_or_topk"
    passed = build_composite_geometry_gate(
        {
            "dense_all_recall@0.5": 0.95,
            "bank_top300_recall@0.5": 0.95,
        },
        required=True,
    )
    assert passed["passed"] is True
    assert passed["diagnosis"] == "geometry_ready"


def test_level_quota() -> None:
    config = EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=4,
        evidence_per_slot=2,
        bank_min_per_level=2,
        bank_pre_topk_per_level=5,
        bank_topk_train=10,
        bank_topk_eval=10,
        hdn_max_pairs=2,
        roi_size=3,
    )
    branch = DenseGeometryBranch(config)
    scores = [
        torch.tensor([[0.99, 0.98, 0.97, 0.96, 0.95]]),
        torch.tensor([[0.40, 0.39, 0.38, 0.37, 0.36]]),
        torch.tensor([[0.30, 0.29, 0.28, 0.27, 0.26]]),
        torch.tensor([[0.20, 0.19, 0.18, 0.17, 0.16]]),
    ]
    valid = [torch.ones_like(value, dtype=torch.bool) for value in scores]
    indices, selected_valid, counts = branch._select_indices(scores, valid, 10)
    assert not indices.requires_grad
    assert int(selected_valid.sum()) == 10
    assert bool((counts >= 2).all()), counts


def _dense_test_boxes(batch: int, points: int) -> torch.Tensor:
    offset = torch.arange(points, dtype=torch.float32) * 12.0
    single = torch.stack(
        (
            offset,
            torch.zeros_like(offset),
            offset + 10.0,
            torch.full_like(offset, 10.0),
        ),
        dim=-1,
    )
    return single[None].expand(batch, -1, -1).clone()


def test_dense_quality_boolean_indexing_mixed_positive() -> None:
    batch, points = 2, 6
    qual_logits = (
        torch.linspace(-1.0, 1.0, batch * points)
        .reshape(batch, points, 1)
        .clone()
        .requires_grad_()
    )
    assign_gt = torch.tensor(
        [[0, -1, 1, -1, -1, 2], [-1, 0, -1, 1, -1, -1]]
    )
    valid = torch.ones((batch, points), dtype=torch.bool)
    boxes = _dense_test_boxes(batch, points)
    target_boxes = boxes.clone()
    level = {
        "obj_logits": torch.zeros((batch, points, 1), requires_grad=True),
        "qual_logits": qual_logits,
        "boxes": boxes,
        "target_boxes": target_boxes,
        "assign_gt": assign_gt,
        "valid": valid,
    }

    losses = dense_geometry_losses({"levels": [level]})
    quality_loss = losses["loss_dense_quality"]
    positive = (assign_gt >= 0) & valid
    iou_target = aligned_box_iou(boxes[positive], target_boxes[positive])

    assert quality_loss.ndim == 0
    _assert_finite_tensor(quality_loss, "mixed-positive dense quality")
    assert bool(torch.isfinite(iou_target).all())
    assert bool(((iou_target >= 0.0) & (iou_target <= 1.0)).all())
    quality_loss.backward()
    assert qual_logits.grad is not None
    assert bool((qual_logits.grad.squeeze(-1)[positive].abs() > 0.0).all())
    torch.testing.assert_close(
        qual_logits.grad.squeeze(-1)[~positive],
        torch.zeros_like(qual_logits.grad.squeeze(-1)[~positive]),
    )


def test_dense_quality_matches_manual_reference() -> None:
    qual_logits = torch.tensor(
        [[[0.1], [-0.3], [1.2], [0.7]]],
        requires_grad=True,
    )
    boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
                [40.0, 0.0, 50.0, 10.0],
                [60.0, 0.0, 70.0, 10.0],
            ]
        ]
    )
    target_boxes = boxes.clone()
    target_boxes[0, 2] = torch.tensor([40.0, 0.0, 45.0, 10.0])
    assign_gt = torch.tensor([[0, -1, 1, 2]])
    valid = torch.tensor([[True, True, True, False]])
    positive = (assign_gt >= 0) & valid
    level = {
        "obj_logits": torch.zeros((1, 4, 1)),
        "qual_logits": qual_logits,
        "boxes": boxes,
        "target_boxes": target_boxes,
        "assign_gt": assign_gt,
        "valid": valid,
    }

    losses = dense_geometry_losses({"levels": [level]})
    iou_target = aligned_box_iou(
        boxes[positive],
        target_boxes[positive],
    ).detach()
    manual = F.binary_cross_entropy_with_logits(
        qual_logits.squeeze(-1)[positive],
        iou_target,
        reduction="sum",
    ) / int(positive.sum())

    assert bool(torch.isfinite(iou_target).all())
    assert bool(((iou_target >= 0.0) & (iou_target <= 1.0)).all())
    torch.testing.assert_close(
        losses["loss_dense_quality"],
        manual,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_dense_losses_no_positive() -> None:
    batch, points = 2, 5
    obj_logits = torch.randn((batch, points, 1), requires_grad=True)
    qual_logits = torch.randn((batch, points, 1), requires_grad=True)
    boxes = _dense_test_boxes(batch, points).requires_grad_()
    valid = torch.tensor(
        [[True, True, False, True, False], [False, True, True, False, True]]
    )
    level = {
        "obj_logits": obj_logits,
        "qual_logits": qual_logits,
        "boxes": boxes,
        "target_boxes": boxes.detach().clone(),
        "assign_gt": torch.full((batch, points), -1, dtype=torch.long),
        "valid": valid,
    }

    losses = dense_geometry_losses({"levels": [level]})

    for name, value in losses.items():
        _assert_finite_tensor(value, name)
        assert value.ndim == 0
    torch.testing.assert_close(losses["loss_dense_giou"], torch.tensor(0.0))
    torch.testing.assert_close(losses["loss_dense_quality"], torch.tensor(0.0))
    assert losses["loss_dense_giou"].requires_grad
    assert losses["loss_dense_quality"].requires_grad
    sum(losses.values()).backward()
    assert obj_logits.grad is not None


def test_dense_losses_all_positive() -> None:
    batch, points = 1, 4
    obj_logits = torch.randn((batch, points, 1), requires_grad=True)
    qual_logits = torch.randn((batch, points, 1), requires_grad=True)
    boxes = _dense_test_boxes(batch, points).requires_grad_()
    target_boxes = boxes.detach().clone()
    target_boxes[..., 2] -= 1.0
    level = {
        "obj_logits": obj_logits,
        "qual_logits": qual_logits,
        "boxes": boxes,
        "target_boxes": target_boxes,
        "assign_gt": torch.arange(points)[None],
        "valid": torch.ones((batch, points), dtype=torch.bool),
    }

    losses = dense_geometry_losses({"levels": [level]})

    for name, value in losses.items():
        _assert_finite_tensor(value, name)
        assert value.ndim == 0
    sum(losses.values()).backward()
    for tensor, name in (
        (obj_logits, "all-positive obj_logits"),
        (qual_logits, "all-positive qual_logits"),
        (boxes, "all-positive boxes"),
    ):
        assert tensor.grad is not None, name
        _assert_finite_tensor(tensor.grad, f"{name} gradient")


def test_dense_level_shape_validation() -> None:
    batch, points = 2, 3
    base = {
        "obj_logits": torch.zeros((batch, points, 1)),
        "qual_logits": torch.zeros((batch, points, 1)),
        "boxes": torch.zeros((batch, points, 4)),
        "target_boxes": torch.zeros((batch, points, 4)),
        "assign_gt": torch.full((batch, points), -1, dtype=torch.long),
        "valid": torch.ones((batch, points), dtype=torch.bool),
    }
    invalid_cases = [
        ("qual_logits", torch.zeros((batch, points)), "qual_logits"),
        ("qual_logits", torch.zeros((batch, points, 2)), "qual_logits"),
        ("boxes", torch.zeros((batch, points, 5)), "boxes"),
        ("target_boxes", torch.zeros((batch, points, 5)), "target_boxes"),
        ("assign_gt", torch.zeros((batch, points, 1)), "assign_gt"),
        ("valid", torch.ones((batch, points)), "valid"),
        ("qual_logits", torch.zeros((batch, points + 1, 1)), "qual_logits"),
        ("obj_logits", torch.zeros((batch, points, 2)), "obj_logits"),
    ]

    for field, invalid_value, expected_name in invalid_cases:
        level = dict(base)
        level[field] = invalid_value
        try:
            dense_geometry_losses({"levels": [level]})
        except ValueError as error:
            message = str(error)
            assert "dense level 0" in message
            assert expected_name in message
        else:
            raise AssertionError(f"invalid dense {field} shape/dtype was accepted")

    missing = dict(base)
    missing.pop("qual_logits")
    try:
        dense_geometry_losses({"levels": [missing]})
    except KeyError as error:
        assert "dense level 0" in str(error)
        assert "qual_logits" in str(error)
    else:
        raise AssertionError("dense level missing qual_logits was accepted")

    try:
        dense_geometry_losses({"levels": []})
    except ValueError as error:
        assert "non-empty sequence" in str(error)
    else:
        raise AssertionError("empty dense levels were accepted")


def test_soft_ownership_distribution() -> None:
    logits = torch.zeros((1, 2, 4))
    ownership = {
        "logits": logits,
        "cand_idx": torch.tensor([[[0, 1, 2], [0, 1, 2]]]),
        "cand_valid": torch.ones((1, 2, 3), dtype=torch.bool),
    }
    probabilities = torch.tensor([0.5, 0.8, 0.9])
    quality_logits = torch.log(probabilities / (1.0 - probabilities))[None, :, None]
    bank = {
        "boxes": torch.tensor(
            [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 5.0, 10.0], [20.0, 20.0, 30.0, 30.0]]]
        ),
        "qual": quality_logits,
        "gt_assign": torch.tensor([[0, 0, 1]]),
    }
    targets = [
        {
            "boxes_xyxy_abs": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        }
    ]
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    target, matched = build_soft_ownership_targets(
        ownership,
        bank,
        matches,
        targets,
    )
    assert target.shape == (1, 2, 4)
    assert matched.shape == (1, 2)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones((1, 2)))
    expected = torch.tensor([0.5, 0.4])
    expected = expected / expected.sum()
    torch.testing.assert_close(target[0, 0, :2], expected)
    assert float(target[0, 0, 2]) == 0.0
    assert float(target[0, 0, 3]) == 0.0
    assert bool(matched[0, 0]) and not bool(matched[0, 1])
    assert float(target[0, 1, -1]) == 1.0


def test_soft_ownership_all_unmatched() -> None:
    batch, slots, proposals = 2, 3, 4
    ownership = {
        "logits": torch.zeros((batch, slots, proposals + 1)),
        "cand_idx": torch.tensor([[[0, 1, 2, 3]] * slots] * batch),
        "cand_valid": torch.ones((batch, slots, proposals), dtype=torch.bool),
    }
    bank = {
        "boxes": torch.zeros((batch, proposals, 4)),
        "qual": torch.zeros((batch, proposals, 1)),
        "gt_assign": torch.full((batch, proposals), -1, dtype=torch.long),
    }
    matches = [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        for _ in range(batch)
    ]
    targets = [{"boxes_xyxy_abs": torch.empty(0, 4)} for _ in range(batch)]

    target, matched = build_soft_ownership_targets(ownership, bank, matches, targets)

    assert target.shape == (batch, slots, proposals + 1)
    assert matched.shape == (batch, slots)
    assert not bool(matched.any())
    torch.testing.assert_close(target[..., :-1], torch.zeros_like(target[..., :-1]))
    torch.testing.assert_close(target[..., -1], torch.ones((batch, slots)))
    torch.testing.assert_close(target.sum(dim=-1), torch.ones((batch, slots)))


def test_soft_ownership_all_matched() -> None:
    slots = proposals = 3
    ownership = {
        "logits": torch.zeros((1, slots, proposals + 1)),
        "cand_idx": torch.tensor([[[0, 1, 2]] * slots]),
        "cand_valid": torch.ones((1, slots, proposals), dtype=torch.bool),
    }
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    bank = {
        "boxes": boxes[None],
        "qual": torch.zeros((1, proposals, 1)),
        "gt_assign": torch.tensor([[0, 1, 2]]),
    }
    matches = [(torch.arange(slots), torch.arange(slots))]
    targets = [{"boxes_xyxy_abs": boxes.clone()}]

    target, matched = build_soft_ownership_targets(ownership, bank, matches, targets)

    assert bool(matched.all())
    torch.testing.assert_close(target[0, :, :-1], torch.eye(slots))
    torch.testing.assert_close(target[0, :, -1], torch.zeros(slots))
    torch.testing.assert_close(target.sum(dim=-1), torch.ones((1, slots)))
    assert bool(torch.isfinite(target).all())
    assert bool((target >= 0.0).all())


def test_soft_ownership_matched_no_valid_candidates_routes_to_dustbin() -> None:
    ownership = {
        "logits": torch.zeros((1, 1, 3)),
        "cand_idx": torch.tensor([[[0, 1]]]),
        "cand_valid": torch.zeros((1, 1, 2), dtype=torch.bool),
    }
    bank = {
        "boxes": torch.tensor(
            [[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]]
        ),
        "qual": torch.zeros((1, 2, 1)),
        "gt_assign": torch.tensor([[0, 0]]),
    }
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [{"boxes_xyxy_abs": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}]

    target, matched = build_soft_ownership_targets(ownership, bank, matches, targets)
    losses = soft_ownership_losses(ownership, bank, matches, targets)

    assert bool(matched[0, 0])
    torch.testing.assert_close(target[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(losses["ownership_accuracy"], torch.tensor(0.0))
    torch.testing.assert_close(
        losses["ownership_matched_dustbin_rate"],
        torch.tensor(1.0),
    )


def test_soft_ownership_assignment_fallback() -> None:
    probabilities = torch.tensor([0.3, 0.8, 0.9])
    ownership = {
        "logits": torch.zeros((1, 1, 4)),
        "cand_idx": torch.tensor([[[0, 1, 2]]]),
        "cand_valid": torch.ones((1, 1, 3), dtype=torch.bool),
    }
    bank = {
        "boxes": torch.tensor(
            [
                [
                    [0.0, 0.0, 5.0, 10.0],
                    [0.0, 0.0, 10.0, 10.0],
                    [20.0, 20.0, 30.0, 30.0],
                ]
            ]
        ),
        "qual": torch.log(probabilities / (1.0 - probabilities))[None, :, None],
        "gt_assign": torch.tensor([[1, 1, 1]]),
    }
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [{"boxes_xyxy_abs": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}]

    target, matched = build_soft_ownership_targets(ownership, bank, matches, targets)

    assert bool(matched[0, 0])
    torch.testing.assert_close(target[0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(target.sum(dim=-1), torch.ones((1, 1)))


def test_soft_ownership_ragged_matches_batch() -> None:
    batch, slots, proposals = 3, 3, 3
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    ownership = {
        "logits": torch.zeros((batch, slots, proposals + 1)),
        "cand_idx": torch.tensor([[[0, 1, 2]] * slots] * batch),
        "cand_valid": torch.ones((batch, slots, proposals), dtype=torch.bool),
    }
    bank = {
        "boxes": boxes[None].expand(batch, -1, -1).clone(),
        "qual": torch.zeros((batch, proposals, 1)),
        "gt_assign": torch.tensor([[0, 1, 2]] * batch),
    }
    matches = [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        (torch.tensor([1]), torch.tensor([0])),
        (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])),
    ]
    targets = [
        {"boxes_xyxy_abs": torch.empty(0, 4)},
        {"boxes_xyxy_abs": boxes[:1].clone()},
        {"boxes_xyxy_abs": boxes.clone()},
    ]

    target, matched = build_soft_ownership_targets(ownership, bank, matches, targets)

    expected_matched = torch.tensor(
        [[False, False, False], [False, True, False], [True, True, True]]
    )
    torch.testing.assert_close(matched, expected_matched)
    torch.testing.assert_close(target.sum(dim=-1), torch.ones((batch, slots)))
    torch.testing.assert_close(target[0, :, -1], torch.ones(slots))
    torch.testing.assert_close(target[1, [0, 2], -1], torch.ones(2))
    torch.testing.assert_close(target[1, 1, :-1], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(target[2, :, :-1], torch.eye(slots))
    torch.testing.assert_close(target[2, :, -1], torch.zeros(slots))


def test_soft_ownership_shape_validation() -> None:
    ownership = {
        "logits": torch.zeros((1, 1, 3)),
        "cand_idx": torch.tensor([[[0, 1]]]),
        "cand_valid": torch.ones((1, 1, 2), dtype=torch.bool),
    }
    bank = {
        "boxes": torch.zeros((1, 2, 4)),
        "qual": torch.zeros((1, 2, 1)),
        "gt_assign": torch.zeros((1, 2), dtype=torch.long),
    }
    matches = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    targets = [{"boxes_xyxy_abs": torch.empty(0, 4)}]

    bad_ownership = dict(ownership)
    bad_ownership["logits"] = torch.zeros((1, 1, 2, 1))
    try:
        build_soft_ownership_targets(bad_ownership, bank, matches, targets)
    except ValueError as error:
        assert "logits must have shape" in str(error)
    else:
        raise AssertionError("invalid ownership logits rank was accepted")

    bad_bank = dict(bank)
    bad_bank["qual"] = torch.zeros((1, 2))
    try:
        build_soft_ownership_targets(ownership, bad_bank, matches, targets)
    except ValueError as error:
        assert "bank qual must have shape" in str(error)
    else:
        raise AssertionError("invalid bank quality shape was accepted")


def test_hdn_attention_mask() -> None:
    group = torch.tensor([[0, 0, 1, 1, -1]])
    valid = torch.tensor([[True, True, True, True, False]])
    mask = HypothesisDenoising.build_attention_mask(2, group, valid)
    assert bool(mask[0, :2, 2:].all())
    # DN group 0 sees both matching columns and its own two DN columns.
    assert not bool(mask[0, 2, :2].any())
    assert not bool(mask[0, 2, 2:4].any())
    assert bool(mask[0, 2, 4:6].all())
    # Invalid DN column is blocked for every row.
    assert bool(mask[0, :, -1].all())


def test_existence_loss_bfloat16_prediction_float32_iou_target() -> None:
    pred_exist = torch.tensor(
        [[[0.25], [-0.5], [0.1]], [[-0.2], [0.3], [-0.4]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    pred_boxes = torch.tensor(
        [
            [
                [0.48, 0.50, 0.20, 0.20],
                [0.20, 0.20, 0.10, 0.10],
                [0.80, 0.80, 0.10, 0.10],
            ],
            [
                [0.25, 0.25, 0.10, 0.10],
                [0.50, 0.50, 0.20, 0.20],
                [0.75, 0.75, 0.10, 0.10],
            ],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = [
        {
            "boxes": torch.tensor(
                [[0.50, 0.50, 0.20, 0.20]],
                dtype=torch.float32,
            )
        },
        {"boxes": torch.empty((0, 4), dtype=torch.float32)},
    ]
    matches = [
        (torch.tensor([0]), torch.tensor([0])),
        (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        ),
    ]
    loss = existence_loss(
        {"pred_exist": pred_exist, "pred_boxes": pred_boxes},
        targets,
        matches,
    )
    assert loss.dtype == torch.float32
    assert loss.ndim == 0
    _assert_finite_tensor(loss, "bf16 existence loss")
    loss.backward()
    assert pred_exist.grad is not None
    _assert_finite_tensor(pred_exist.grad.float(), "bf16 existence gradient")


def test_slot_losses_bfloat16_inputs() -> None:
    logits = torch.randn(
        (1, 3, 80),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    boxes = torch.tensor(
        [
            [
                [0.45, 0.50, 0.20, 0.20],
                [0.48, 0.50, 0.20, 0.20],
                [0.80, 0.80, 0.10, 0.10],
            ]
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    quality = torch.randn(
        (1, 3, 1),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    existence = torch.randn(
        (1, 3, 1),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    score_values = torch.zeros((1, 3, 80), dtype=torch.float32)
    score_values[0, 0, 2] = 0.1
    score_values[0, 1, 2] = 0.8
    scores = score_values.to(torch.bfloat16).requires_grad_()
    prediction = {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "pred_quality": quality,
        "pred_exist": existence,
        "pred_scores": scores,
    }
    target_box = torch.tensor(
        [[0.50, 0.50, 0.20, 0.20]],
        dtype=torch.float32,
    )
    target_abs = box_cxcywh_to_xyxy(target_box) * 100.0
    targets = [
        {
            "labels": torch.tensor([2], dtype=torch.long),
            "boxes": target_box,
            "boxes_xyxy_abs": target_abs,
        }
    ]
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    bank = {
        "boxes": torch.stack(
            (
                target_abs[0],
                torch.tensor([10.0, 10.0, 20.0, 20.0]),
            )
        )[None].to(torch.bfloat16),
        "gt_assign": torch.tensor([[0, -1]], dtype=torch.long),
    }
    ownership = {
        "cand_idx": torch.tensor(
            [[[0, 1], [0, 1], [0, 1]]],
            dtype=torch.long,
        ),
        "cand_valid": torch.ones((1, 3, 2), dtype=torch.bool),
    }
    losses = slot_round_losses(
        prediction,
        targets,
        matches,
        num_classes=80,
    )
    losses["loss_exist"] = existence_loss(prediction, targets, matches)
    losses["loss_duplicate"] = duplicate_margin_loss(
        prediction,
        targets,
        matches,
        margin=0.2,
    )
    losses["loss_gain"] = readout_gain_loss(
        prediction,
        bank,
        ownership,
        targets,
        matches,
    )
    expected = {
        "loss_cls",
        "loss_l1",
        "loss_giou",
        "loss_quality",
        "loss_exist",
        "loss_duplicate",
        "loss_gain",
    }
    assert set(losses) == expected
    for name, value in losses.items():
        assert value.dtype == torch.float32, (name, value.dtype)
        assert value.ndim == 0, (name, value.shape)
        _assert_finite_tensor(value, name)
    sum(losses.values()).backward()
    for name, value in (
        ("logits", logits),
        ("boxes", boxes),
        ("quality", quality),
        ("existence", existence),
        ("scores", scores),
    ):
        assert value.grad is not None, name
        _assert_finite_tensor(value.grad.float(), f"{name} gradient")


def test_matcher_bfloat16_outputs() -> None:
    matcher = HungarianMatcherV2(_compact_config(False))
    outputs = {
        "pred_logits": torch.randn((2, 4, 80), dtype=torch.bfloat16),
        "pred_boxes": torch.tensor(
            [
                [
                    [0.50, 0.50, 0.20, 0.20],
                    [0.20, 0.20, 0.10, 0.10],
                    [0.75, 0.75, 0.15, 0.15],
                    [0.40, 0.70, 0.12, 0.18],
                ],
                [
                    [0.10, 0.10, 0.10, 0.10],
                    [0.30, 0.30, 0.10, 0.10],
                    [0.60, 0.60, 0.10, 0.10],
                    [0.90, 0.90, 0.10, 0.10],
                ],
            ],
            dtype=torch.bfloat16,
        ),
    }
    targets = [
        {
            "labels": torch.tensor([3], dtype=torch.long),
            "boxes": torch.tensor(
                [[0.50, 0.50, 0.20, 0.20]],
                dtype=torch.float32,
            ),
        },
        {
            "labels": torch.empty(0, dtype=torch.long),
            "boxes": torch.empty((0, 4), dtype=torch.float32),
        },
    ]
    matches = matcher(outputs, targets)
    assert len(matches) == 2
    assert matches[0][0].numel() == 1
    assert matches[0][1].tolist() == [0]
    assert matches[1][0].numel() == 0
    assert matches[1][1].numel() == 0


def test_dense_and_ownership_bfloat16_inputs() -> None:
    obj_logits = torch.randn(
        (1, 3, 1),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    qual_logits = torch.randn(
        (1, 3, 1),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    box_values = torch.tensor(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ],
        dtype=torch.bfloat16,
    )
    dense_boxes = box_values.clone().requires_grad_()
    dense = {
        "levels": [
            {
                "obj_logits": obj_logits,
                "qual_logits": qual_logits,
                "boxes": dense_boxes,
                "target_boxes": box_values.float(),
                "assign_gt": torch.tensor([[0, -1, 1]], dtype=torch.long),
                "valid": torch.ones((1, 3), dtype=torch.bool),
            }
        ]
    }
    dense_result = dense_geometry_losses(dense)
    for name, value in dense_result.items():
        assert value.dtype == torch.float32, (name, value.dtype)
        _assert_finite_tensor(value, name)
    sum(dense_result.values()).backward()
    assert obj_logits.grad is not None
    assert qual_logits.grad is not None
    assert dense_boxes.grad is not None

    ownership_logits = torch.randn(
        (1, 1, 3),
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    ownership = {
        "logits": ownership_logits,
        "cand_idx": torch.tensor([[[0, 1]]], dtype=torch.long),
        "cand_valid": torch.ones((1, 1, 2), dtype=torch.bool),
    }
    bank = {
        "boxes": torch.tensor(
            [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 5.0, 10.0]]],
            dtype=torch.bfloat16,
        ),
        "qual": torch.zeros((1, 2, 1), dtype=torch.bfloat16),
        "gt_assign": torch.tensor([[0, 0]], dtype=torch.long),
    }
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    targets = [
        {
            "boxes_xyxy_abs": torch.tensor(
                [[0.0, 0.0, 10.0, 10.0]],
                dtype=torch.float32,
            )
        }
    ]
    target, _ = build_soft_ownership_targets(
        ownership,
        bank,
        matches,
        targets,
    )
    assert target.dtype == torch.float32
    ownership_result = soft_ownership_losses(
        ownership,
        bank,
        matches,
        targets,
    )
    for name, value in ownership_result.items():
        assert value.dtype == torch.float32, (name, value.dtype)
        _assert_finite_tensor(value, name)
    (
        ownership_result["loss_ownership"]
        + ownership_result["loss_dustbin"]
    ).backward()
    assert ownership_logits.grad is not None


def test_fixed_step_optimizer_counter_contract() -> None:
    config = _compact_config(False)
    config.grad_accum_steps = 2
    model = EgoDetV2(config, fake_backbone=True).train()
    criterion = EgoDetV2Criterion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    side = 96
    batch = (
        torch.randn((2, 3, side, side)),
        torch.zeros((2, side, side), dtype=torch.bool),
        _synthetic_targets(side),
    )
    # PyTorch 2.1 validates the CPU autocast dtype even when enabled=False.
    # The CPU fixed-step path must therefore be a true no-op context and never
    # call torch.autocast at all.
    with mock.patch(
        "torch.autocast",
        side_effect=AssertionError("CPU fixed-step entered torch.autocast"),
    ):
        metrics = run_fixed_steps(
            model,
            criterion,
            [batch],
            optimizer,
            config=config,
            device=torch.device("cpu"),
            steps=1,
            geometry_only=True,
            show_progress=False,
        )
    assert metrics["micro_step"] == 2.0
    assert metrics["optimizer_step"] == 1.0
    assert metrics["global_optimizer_step"] == 1.0
    assert metrics["nonfinite"] == 0.0


def test_zero_initialization() -> None:
    module = ZeroMLP(16, 8)
    actual = module(torch.randn(3, 16))
    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_empty_bank_path() -> None:
    config = EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=4,
        evidence_per_slot=2,
        bank_min_per_level=1,
        bank_pre_topk_per_level=2,
        bank_topk_train=4,
        bank_topk_eval=4,
        hdn_max_pairs=2,
        roi_size=3,
    )
    model = EgoDetV2(config, fake_backbone=True).eval()
    images = torch.randn(1, 3, 64, 64)
    all_padding = torch.ones(1, 64, 64, dtype=torch.bool)
    targets = [
        {
            "labels": torch.empty(0, dtype=torch.long),
            "boxes": torch.empty(0, 4),
            "boxes_xyxy_abs": torch.empty(0, 4),
            "image_id": 1,
            "orig_size": (64, 64),
            "size": (64, 64),
        }
    ]
    with torch.no_grad():
        outputs = model(images, all_padding, targets)
    assert int(outputs["bank"]["valid"].sum()) == 0
    _assert_finite_tensor(outputs["pred_boxes"], "empty-bank pred_boxes")


def test_category_roundtrip_and_collate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image_dir = root / "val2017"
        image_dir.mkdir()
        Image.new("RGB", (40, 32), color=(120, 80, 40)).save(image_dir / "sample.jpg")
        categories = [{"id": index + 1, "name": f"c{index}"} for index in range(80)]
        annotation = {
            "images": [{"id": 1, "file_name": "sample.jpg", "width": 40, "height": 32}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 80,
                    "bbox": [2, 3, 10, 12],
                    "area": 120,
                    "iscrowd": 0,
                }
            ],
            "categories": categories,
        }
        annotation_path = root / "instances.json"
        annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
        config = EgoDetV2Config(val_short_side=32, val_max_size=64)
        dataset = CocoDetectionV2(
            annotation_path,
            root,
            split="val2017",
            config=config,
            training=False,
        )
        for category_id, contiguous in dataset.category_id_to_contiguous.items():
            assert dataset.contiguous_to_category_id[contiguous] == category_id
        image, target = dataset[0]
        images, mask, targets = collate_detection_batch([(image, target)])
        assert images.shape[-2] % 32 == 0 and images.shape[-1] % 32 == 0
        assert mask.dtype == torch.bool
        assert targets[0]["labels"].tolist() == [79]


def test_model_and_gradients(full_shape: bool) -> None:
    torch.manual_seed(3407)
    config = _compact_config(full_shape)
    side = 672 if full_shape else 96
    model = EgoDetV2(config, fake_backbone=True).train()
    images = torch.randn(2, 3, side, side)
    pad_mask = torch.zeros((2, side, side), dtype=torch.bool)
    if not full_shape:
        pad_mask[1, 80:, :] = True
        pad_mask[1, :, 88:] = True
    targets = _synthetic_targets(side)
    if not full_shape:
        targets[1]["size"] = (80, 88)
    outputs = model(images, pad_mask, targets)
    for level in outputs["dense"]["levels"]:
        for key in (
            "assign_gt",
            "train_assign_gt",
            "proposal_gt_assign",
            "assign_iou",
            "proposal_assign_iou",
            "target_boxes",
            "train_target_boxes",
            "train_assign_source",
        ):
            assert key in level
        torch.testing.assert_close(level["assign_gt"], level["train_assign_gt"])
        torch.testing.assert_close(
            level["assign_iou"],
            level["proposal_assign_iou"],
        )
        assert level["train_assign_gt"].dtype == torch.long
        assert level["proposal_gt_assign"].dtype == torch.long
        assert level["proposal_assign_iou"].dtype == torch.float32
        assert level["train_target_boxes"].dtype == torch.float32
        assert level["train_assign_source"].dtype == torch.long
    geometry_diagnostics = GeometryDiagnosticsAccumulator(config)
    geometry_diagnostics.update(outputs["dense"], outputs["bank"], targets)
    geometry_metrics = geometry_diagnostics.compute()
    for required_metric in (
        "assignable_gt_ratio",
        "dense_all_recall@0.5",
        "dense_all_recall@0.75",
        "after_per_level_pre_topk_recall@0.5",
        "after_level_quota_recall@0.5",
        "bank_top100_recall@0.5",
        "bank_top300_recall@0.5",
        "bank_top1000_recall@0.5",
        "bank_top1500_recall@0.5",
        "combined_score_vs_iou_spearman",
    ):
        assert required_metric in geometry_metrics
        assert torch.isfinite(torch.tensor(float(geometry_metrics[required_metric])))
    for level_name in ("P3", "P4", "P5", "P6"):
        assert level_name in geometry_metrics
        assert "positive_per_gt_mean" in geometry_metrics[level_name]
    assert outputs["pred_logits"].shape == (
        2,
        config.num_slots,
        config.num_classes,
    )
    assert outputs["pred_boxes"].shape == (2, config.num_slots, 4)
    assert len(outputs["aux"]) == 2
    assert len(outputs["ownership"]) == 3
    for ownership in outputs["ownership"]:
        assert "proposal_mass" in ownership
        assert bool((ownership["proposal_mass"] >= 0).all())
        assert bool((ownership["proposal_mass"] <= 1).all())
    assert bool((outputs["pred_boxes"] > 0).all())
    assert bool((outputs["pred_boxes"] < 1).all())
    assert not outputs["bank"]["selection_indices"].requires_grad
    for batch_index in range(2):
        for level_index, count in enumerate(outputs["bank"]["level_counts"][batch_index]):
            available = int(
                outputs["dense"]["levels"][level_index]["valid"][batch_index].sum()
            )
            assert int(count) >= min(config.bank_min_per_level, available)
    for readout in outputs["readout"]:
        b_hat = readout["b_hat"]
        assert bool((b_hat[..., 2] > b_hat[..., 0]).all())
        assert bool((b_hat[..., 3] > b_hat[..., 1]).all())

    criterion = EgoDetV2Criterion(config)
    losses = criterion(outputs, targets, epoch=2)
    for name, value in losses.items():
        assert value.dtype == torch.float32, (name, value.dtype)
        assert value.ndim == 0, (name, value.shape)
        _assert_finite_tensor(value, name)
    for round_index in range(1, config.slot_rounds + 1):
        diagnostic = losses[f"ownership_matched_dustbin_rate_r{round_index}"]
        assert 0.0 <= float(diagnostic) <= 1.0

    final_prediction = {
        "pred_logits": outputs["pred_logits"],
        "pred_boxes": outputs["pred_boxes"],
        "pred_quality": outputs["pred_quality"],
    }
    matches = criterion.matcher(final_prediction, targets)
    final_losses = slot_round_losses(
        final_prediction,
        targets,
        matches,
        num_classes=config.num_classes,
    )
    final_box_loss = final_losses["loss_l1"] + final_losses["loss_giou"]
    model.zero_grad(set_to_none=True)
    final_box_loss.backward(retain_graph=True)
    obj_gradient = model.dense_geometry.obj_head.weight.grad
    quality_gradient = model.dense_geometry.qual_head.weight.grad
    assert obj_gradient is not None and float(obj_gradient.abs().sum()) > 0.0
    assert quality_gradient is not None and float(quality_gradient.abs().sum()) > 0.0
    for name, module in (
        ("mlp_q", model.readouts[-1].mlp_q),
        ("mlp_k", model.readouts[-1].mlp_k),
        ("mlp_geo", model.readouts[-1].mlp_geo),
        ("mlp_side", model.readouts[-1].mlp_side),
        ("mlp_res", model.readouts[-1].mlp_res),
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(gradient is not None for gradient in gradients), name

    model.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    required = {
        "geometry_tower": model.dense_geometry.tower,
        "quality_head": model.dense_geometry.qual_head,
        "descriptor": model.dense_geometry.desc_head,
        "semantic_tower": model.semantic_support.semantic_tower,
        "side": model.readouts[-1].mlp_side,
        "residual": model.readouts[-1].mlp_res,
        "gamma_theta": model.existence_head.theta,
        "classifier": model.round_heads[-1].classifier,
    }
    for name, module in required.items():
        gradients = [parameter.grad for parameter in module.parameters()]
        assert any(gradient is not None for gradient in gradients), name
    expected_candidate_dim = (
        config.d_model + 2 + config.level_embed_dim + 6
    )
    for round_index, readout in enumerate(model.readouts, start=1):
        assert readout.candidate_dim == expected_candidate_dim
        for name, module in (
            ("ownership_q", readout.mlp_q),
            ("ownership_k", readout.mlp_k),
            ("ownership_complete_feature", readout.mlp_geo),
            ("side", readout.mlp_side),
            ("residual", readout.mlp_res),
        ):
            gradients = [parameter.grad for parameter in module.parameters()]
            assert any(gradient is not None for gradient in gradients), (
                round_index,
                name,
            )

    # Existence starts as one constant score factor and gamma starts at 1.
    class_logits = torch.randn(2, 4, config.num_classes)
    quality_logits = torch.randn(2, 4, 1)
    content = torch.randn(2, 4, config.d_model)
    boxes = torch.tensor([[[0.2, 0.2, 0.1, 0.1]] * 4] * 2)
    existence, gamma = model.existence_head(content, boxes)
    torch.testing.assert_close(gamma, torch.ones_like(gamma), atol=1.0e-6, rtol=1.0e-6)
    score = compose_scores(class_logits, quality_logits, existence, gamma)
    base = torch.sqrt(torch.sigmoid(class_logits) * torch.sigmoid(quality_logits))
    ratio = score / base.clamp(min=1.0e-8)
    torch.testing.assert_close(ratio, torch.full_like(ratio, 0.5), atol=1.0e-6, rtol=1.0e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGO-Det V2 CPU contracts")
    parser.add_argument(
        "--fake-backbone",
        action="store_true",
        help="explicitly document that no OverLoCK weight is loaded",
    )
    parser.add_argument(
        "--full-shape",
        action="store_true",
        help="also exercise B=2, 3x672x672, N=300, K=1000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tests = [
        ("defaults", test_defaults),
        (
            "overlock_checkpoint_attribute_configuration",
            test_overlock_checkpoint_attribute_configuration,
        ),
        ("overfit_memory_parser", test_overfit_memory_parser),
        (
            "optimizer_named_groups_and_scheduler",
            test_optimizer_named_groups_and_scheduler_contract,
        ),
        (
            "overfit_dataset_and_memory_sweep",
            test_overfit_dataset_and_memory_sweep_contract,
        ),
        (
            "brute_force_proposal_metric",
            test_brute_force_proposal_metric_contract,
        ),
        ("evaluation_restores_model_mode", test_evaluation_restores_model_mode),
        ("gradient_diagnostics", test_gradient_diagnostics_contract),
        ("coordinates_and_residual", test_coordinate_and_residual_contract),
        ("iou_corrected_assign_gt", test_iou_corrected_assign_gt),
        (
            "stable_dense_assignment_decouples_proposal_identity",
            test_stable_dense_assignment_decouples_proposal_identity,
        ),
        ("bank_uses_proposal_identity", test_bank_uses_proposal_identity),
        (
            "legacy_dynamic_assignment_reproduction",
            test_legacy_dynamic_assignment_reproduction,
        ),
        ("min_one_fallback", test_min_one_fallback_contract),
        (
            "assignment_positive_feedback_regression",
            test_assignment_positive_feedback_regression,
        ),
        (
            "assignment_diagnostics_and_composite_gate",
            test_assignment_diagnostics_and_composite_gate,
        ),
        ("level_quota", test_level_quota),
        (
            "dense_quality_boolean_indexing_mixed_positive",
            test_dense_quality_boolean_indexing_mixed_positive,
        ),
        (
            "dense_quality_matches_manual_reference",
            test_dense_quality_matches_manual_reference,
        ),
        ("dense_losses_no_positive", test_dense_losses_no_positive),
        ("dense_losses_all_positive", test_dense_losses_all_positive),
        ("dense_level_shape_validation", test_dense_level_shape_validation),
        ("soft_ownership_distribution", test_soft_ownership_distribution),
        ("soft_ownership_all_unmatched", test_soft_ownership_all_unmatched),
        ("soft_ownership_all_matched", test_soft_ownership_all_matched),
        (
            "soft_ownership_matched_no_valid_candidates_routes_to_dustbin",
            test_soft_ownership_matched_no_valid_candidates_routes_to_dustbin,
        ),
        ("soft_ownership_assignment_fallback", test_soft_ownership_assignment_fallback),
        ("soft_ownership_ragged_matches_batch", test_soft_ownership_ragged_matches_batch),
        ("soft_ownership_shape_validation", test_soft_ownership_shape_validation),
        ("hdn_mask", test_hdn_attention_mask),
        (
            "existence_loss_bfloat16_prediction_float32_iou_target",
            test_existence_loss_bfloat16_prediction_float32_iou_target,
        ),
        ("slot_losses_bfloat16_inputs", test_slot_losses_bfloat16_inputs),
        ("matcher_bfloat16_outputs", test_matcher_bfloat16_outputs),
        (
            "dense_and_ownership_bfloat16_inputs",
            test_dense_and_ownership_bfloat16_inputs,
        ),
        (
            "fixed_step_optimizer_counter_contract",
            test_fixed_step_optimizer_counter_contract,
        ),
        ("zero_initialization", test_zero_initialization),
        ("empty_bank", test_empty_bank_path),
        ("category_and_collate", test_category_roundtrip_and_collate),
        ("model_and_gradients", lambda: test_model_and_gradients(args.full_shape)),
    ]
    completed = []
    for name, test in tests:
        test()
        completed.append(name)
        print(f"[contracts_cpu] PASS {name}", flush=True)
    print(
        json.dumps(
            {
                "status": "PASS",
                "fake_backbone": True,
                "full_shape": bool(args.full_shape),
                "tests": completed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
