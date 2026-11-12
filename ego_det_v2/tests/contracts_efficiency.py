from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..configs.config import EgoDetV2Config
from ..engine.throughput_profiler import (
    PROFILE_STAGE_NAMES,
    ThroughputProfiler,
    benchmark_training_throughput,
    build_benchmark_comparison,
    trace_training_profile,
)
from ..engine.train_loop import (
    _optimizer_logical_name,
    build_optimizer,
    build_optimizer_step_scheduler,
)
from ..launch.launch_overfit8 import parse_args as parse_overfit_args
from ..launch.launch_profile_throughput import (
    parse_args as parse_profile_args,
    profile_output_paths,
)
from ..launch.launch_smoke300 import parse_args as parse_smoke_args
from ..launch.launch_train_ego_det_v2_e2e import (
    parse_args as parse_train_args,
)
from ..losses import criterion as criterion_module
from ..losses.criterion import EgoDetV2Criterion
from ..losses.ownership_losses import (
    build_soft_ownership_targets_optimized,
    build_soft_ownership_targets_reference,
    soft_ownership_losses,
)
from ..losses.slot_losses import (
    denoising_round_losses_optimized,
    denoising_round_losses_reference,
    duplicate_margin_loss_optimized,
    duplicate_margin_loss_reference,
    existence_loss_optimized,
    existence_loss_reference,
    flatten_matches,
    readout_gain_loss_optimized,
    readout_gain_loss_reference,
    slot_round_losses_optimized,
    slot_round_losses_reference,
)
from ..models.box_ops import box_cxcywh_to_xyxy
from ..models.ego_det_v2 import EgoDetV2
from ..models.matcher import prepare_targets_for_matching_and_losses
from ..models.slot_seeding import SlotSeeder


def _config(
    *,
    efficiency_mode: str,
    full_shape: bool = False,
) -> EgoDetV2Config:
    common = {
        "d_model": 32,
        "tower_convs": 1,
        "sem_convs": 1,
        "roi_size": 3,
        "level_embed_dim": 4,
        "batch_size": 1,
        "grad_accum_steps": 1,
        "num_workers": 0,
        "amp": False,
        "efficiency_mode": efficiency_mode,
    }
    if full_shape:
        return EgoDetV2Config(
            **common,
            hdn_max_pairs=10,
        )
    return EgoDetV2Config(
        **common,
        num_slots=12,
        evidence_per_slot=6,
        bank_min_per_level=2,
        bank_pre_topk_per_level=12,
        bank_topk_train=32,
        bank_topk_eval=32,
        hdn_max_pairs=6,
    )


def _targets(
    side: int,
    *,
    batch: int = 1,
    empty_last: bool = False,
    num_classes: int = 80,
) -> List[Dict]:
    normalized = torch.tensor(
        [
            [0.20, 0.22, 0.13, 0.15],
            [0.42, 0.45, 0.18, 0.16],
            [0.73, 0.67, 0.14, 0.19],
        ],
        dtype=torch.float32,
    )
    result = []
    for batch_index in range(batch):
        if empty_last and batch_index == batch - 1:
            boxes = normalized[:0].clone()
            labels = torch.empty(0, dtype=torch.long)
        else:
            boxes = normalized.clone()
            labels = torch.arange(
                boxes.shape[0],
                dtype=torch.long,
            ) % int(num_classes)
        result.append(
            {
                "labels": labels,
                "boxes": boxes,
                "boxes_xyxy_abs": box_cxcywh_to_xyxy(boxes) * float(side),
                "image_id": batch_index + 1,
                "orig_size": (side, side),
                "size": (side, side),
            }
        )
    return result


def _random_bank(
    *,
    batch: int,
    bank_size: int,
    valid_counts: Sequence[int],
    identical_scores: bool = False,
    identical_boxes: bool = False,
) -> Dict[str, torch.Tensor]:
    raw = torch.rand(batch, bank_size, 4)
    lower = torch.minimum(raw[..., :2], raw[..., 2:])
    upper = torch.maximum(raw[..., :2], raw[..., 2:]) + 0.01
    boxes = torch.cat((lower, upper), dim=-1)
    if identical_boxes:
        boxes[:] = torch.tensor([0.1, 0.1, 0.9, 0.9])
    score = (
        torch.zeros(batch, bank_size, 1)
        if identical_scores
        else torch.randn(batch, bank_size, 1)
    )
    valid = torch.zeros(batch, bank_size, dtype=torch.bool)
    for batch_index, count in enumerate(valid_counts):
        valid[batch_index, : int(count)] = True
    return {"boxes": boxes, "score": score, "valid": valid}


def _assert_scalar_dict_close(
    reference: Dict[str, torch.Tensor],
    optimized: Dict[str, torch.Tensor],
    *,
    atol: float = 1.0e-6,
) -> None:
    if set(reference) != set(optimized):
        raise AssertionError(
            f"result keys differ: {set(reference)} vs {set(optimized)}"
        )
    for name in reference:
        torch.testing.assert_close(
            reference[name],
            optimized[name],
            atol=atol,
            rtol=0.0,
            msg=lambda message, key=name: f"{key}: {message}",
        )


def test_config_and_cli_contract() -> None:
    config = EgoDetV2Config()
    assert config.efficiency_mode == "reference"
    assert config.efficiency_ablation_level == 6
    assert config.oracle_monitor_every_optimizer_steps == -1
    try:
        replace(config, efficiency_mode="approximate")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid efficiency_mode was accepted")

    assert (
        parse_train_args(["--efficiency-mode", "optimized"]).efficiency_mode
        == "optimized"
    )
    assert (
        parse_smoke_args(["--efficiency-mode", "optimized"]).efficiency_mode
        == "optimized"
    )
    assert (
        parse_overfit_args(["--efficiency-mode", "optimized"]).efficiency_mode
        == "optimized"
    )
    profile = parse_profile_args(
        [
            "--mode",
            "optimized",
            "--overlock-checkpointing",
            "false",
        ]
    )
    assert profile.efficiency_mode == "optimized"
    assert profile.overlock_checkpointing is False
    assert profile.optimization_level == 6
    assert profile.profile_kind == "benchmark"
    assert profile.measure_micro_steps == 40
    trace_profile = parse_profile_args(
        [
            "--profile-kind",
            "trace",
            "--measure-micro-steps",
            "4",
        ]
    )
    assert trace_profile.profile_kind == "trace"
    try:
        parse_profile_args(["--measure-micro-steps", "39"])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "benchmark accepted fewer than 40 measured micro-steps"
        )


def test_profiler_schema_contract() -> None:
    profiler = ThroughputProfiler(device=torch.device("cpu"))
    profiler.begin_optimizer_step()
    profiler.begin_micro_step()
    with profiler.stage("data_wait"):
        value = sum(range(8))
    assert value == 28
    profiler.end_optimizer_step()
    profiler.end_micro_step()
    summary = profiler.stage_summary()
    assert tuple(summary) == PROFILE_STAGE_NAMES
    assert summary["data_wait"]["wall_ms"]["count"] == 1
    assert summary["total_micro_step"]["wall_ms"]["count"] == 1
    assert summary["total_optimizer_step"]["wall_ms"]["count"] == 1
    assert summary["backbone_fpn"]["wall_ms"]["count"] == 0


def test_benchmark_trace_separation_contract() -> None:
    benchmark_source = inspect.getsource(
        benchmark_training_throughput
    )
    trace_source = inspect.getsource(trace_training_profile)
    assert "torch.profiler.profile(" not in benchmark_source
    assert "ThroughputProfiler(" not in benchmark_source
    assert benchmark_source.count("torch.cuda.synchronize(device)") == 2
    assert "torch.profiler.profile(" in trace_source
    assert "record_shapes=True" in trace_source
    assert "profile_memory=True" in trace_source
    assert "ThroughputProfiler(" in trace_source

    output_dir = Path("outputs/ego_det_v2/throughput_profile")
    reference_benchmark, reference_chrome = profile_output_paths(
        output_dir,
        efficiency_mode="reference",
        optimization_level=6,
        profile_kind="benchmark",
    )
    optimized_benchmark, optimized_chrome = profile_output_paths(
        output_dir,
        efficiency_mode="optimized",
        optimization_level=6,
        profile_kind="benchmark",
    )
    reference_trace, reference_trace_chrome = profile_output_paths(
        output_dir,
        efficiency_mode="reference",
        optimization_level=6,
        profile_kind="trace",
    )
    optimized_trace, optimized_trace_chrome = profile_output_paths(
        output_dir,
        efficiency_mode="optimized",
        optimization_level=6,
        profile_kind="trace",
    )
    assert reference_benchmark.name == "reference_benchmark.json"
    assert optimized_benchmark.name == "optimized_benchmark.json"
    assert reference_chrome is None
    assert optimized_chrome is None
    assert reference_trace.name == "reference_trace_profile.json"
    assert optimized_trace.name == "optimized_trace_profile.json"
    assert reference_trace_chrome is not None
    assert optimized_trace_chrome is not None

    reference = {
        "status": "PASS",
        "result_kind": "benchmark",
        "gate_eligible": True,
        "config": {
            "efficiency_mode": "reference",
            "batch_size": 4,
        },
        "benchmark_contract": {
            "warmup_micro_steps": 10,
            "measure_micro_steps": 40,
            "oracle_monitor_every_optimizer_steps": -1,
        },
        "data_contract": {
            "measured_sequence_sha256": "same-data",
        },
        "environment": {"device": "cuda:0"},
        "worktree": {"git_revision": "same-revision"},
        "throughput": {"images_per_second": 2.5},
        "memory": {
            "gate_eligible": True,
            "peak_allocated_bytes": 10,
        },
    }
    optimized = copy.deepcopy(reference)
    optimized["config"]["efficiency_mode"] = "optimized"
    optimized["throughput"]["images_per_second"] = 5.5
    comparison = build_benchmark_comparison(
        reference,
        optimized,
        reference_path="reference_benchmark.json",
        optimized_path="optimized_benchmark.json",
    )
    assert comparison["source_policy"] == "benchmark_json_only"
    assert comparison["comparable_contract"] is True
    assert comparison["hard_2x_gate_passed"] is True
    assert comparison["recommended_5_images_per_second_passed"] is True
    assert "profile" not in comparison

    mismatched = copy.deepcopy(optimized)
    mismatched["data_contract"]["measured_sequence_sha256"] = (
        "different-data"
    )
    mismatch_comparison = build_benchmark_comparison(
        reference,
        mismatched,
        reference_path="reference_benchmark.json",
        optimized_path="optimized_benchmark.json",
    )
    assert mismatch_comparison["comparable_contract"] is False
    assert mismatch_comparison["hard_2x_gate_passed"] is False

    trace_payload = copy.deepcopy(reference)
    trace_payload["result_kind"] = "trace_profile"
    trace_payload["gate_eligible"] = False
    try:
        build_benchmark_comparison(
            trace_payload,
            optimized,
            reference_path="reference_trace_profile.json",
            optimized_path="optimized_benchmark.json",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "throughput comparison accepted a trace-profile payload"
        )

    class TinyProfileModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense_geometry = torch.nn.Module()
            self.dense_geometry.obj_head = torch.nn.Linear(1, 1)

        def backbone_param_groups(self):
            return {"stem": (), "backbone": (), "other": ()}

        def forward(
            self,
            images,
            pad_mask,
            targets,
            *,
            epoch,
            profiler=None,
        ):
            del pad_mask, targets, epoch, profiler
            pooled = images.mean(dim=(1, 2, 3), keepdim=False)[:, None]
            value = self.dense_geometry.obj_head(pooled)
            batch = int(images.shape[0])
            return {
                "value": value,
                "bank": {
                    "boxes": images.new_zeros((batch, 1, 4)),
                    "score": images.new_zeros((batch, 1, 1)),
                    "valid": torch.zeros(
                        batch,
                        1,
                        dtype=torch.bool,
                        device=images.device,
                    ),
                },
            }

    class TinyProfileCriterion(torch.nn.Module):
        def forward(
            self,
            outputs,
            targets,
            *,
            epoch,
            epoch_progress,
            profiler=None,
        ):
            del targets, epoch, epoch_progress, profiler
            return {"loss_total": outputs["value"].square().mean()}

    class TinyProfileDataset:
        image_ids = [101]
        split = "train2017"
        annotation_file = "tiny_annotations.json"
        image_root = "tiny_images"

        def __len__(self):
            return 1

    class TinyProfileLoader:
        dataset = TinyProfileDataset()

        def __iter__(self):
            images = torch.ones(1, 3, 4, 4)
            pad_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
            targets = [
                {
                    "image_id": 101,
                    "size": (4, 4),
                    "labels": torch.zeros(1, dtype=torch.long),
                }
            ]
            return iter([(images, pad_mask, targets)])

    runtime_config = replace(
        _config(efficiency_mode="reference"),
        oracle_monitor_every_optimizer_steps=0,
        warmup_steps=1,
    )
    benchmark_result = benchmark_training_throughput(
        TinyProfileModel(),
        TinyProfileCriterion(),
        TinyProfileLoader(),
        config=runtime_config,
        device=torch.device("cpu"),
        warmup_micro_steps=1,
        measure_micro_steps=40,
    )
    assert benchmark_result["result_kind"] == "benchmark"
    assert benchmark_result["gate_eligible"] is True
    assert benchmark_result["decision_eligibility"]["checkpointing_ab"]
    assert benchmark_result["decision_eligibility"]["batch_ab"]
    assert benchmark_result["benchmark"]["measured_optimizer_steps"] == 40
    assert benchmark_result["benchmark_contract"][
        "torch_profiler_enabled"
    ] is False
    assert len(
        benchmark_result["data_contract"]["measured_batches"]
    ) == 40
    assert "throughput" in benchmark_result
    assert "stages" not in benchmark_result

    with tempfile.TemporaryDirectory() as temporary_directory:
        trace_path = Path(temporary_directory) / "tiny_trace.json"
        trace_result = trace_training_profile(
            TinyProfileModel(),
            TinyProfileCriterion(),
            TinyProfileLoader(),
            config=runtime_config,
            device=torch.device("cpu"),
            warmup_micro_steps=1,
            measure_micro_steps=1,
            trace_path=trace_path,
        )
        assert trace_path.is_file()
    assert trace_result["result_kind"] == "trace_profile"
    assert trace_result["gate_eligible"] is False
    assert trace_result["memory_margin_eligible"] is False
    assert not any(trace_result["decision_eligibility"].values())
    assert trace_result["trace_contract"]["torch_profiler_enabled"] is True
    assert "throughput" not in trace_result
    assert "stages" in trace_result


def test_slot_seeder_exact_contract(full_shape: bool) -> None:
    seed_trials = 100 if not full_shape else 12
    for seed in range(seed_trials):
        torch.manual_seed(seed)
        batch = (1, 2, 4)[seed % 3]
        bank_size = 32
        slots = 12
        counts = [
            (0, 1, slots - 1, slots, bank_size)[
                (seed + batch_index) % 5
            ]
            for batch_index in range(batch)
        ]
        config = replace(
            _config(efficiency_mode="optimized"),
            num_slots=slots,
        )
        seeder = SlotSeeder(config)
        bank = _random_bank(
            batch=batch,
            bank_size=bank_size,
            valid_counts=counts,
            identical_scores=seed % 7 == 0,
            identical_boxes=seed % 11 == 0,
        )
        reference_indices, reference_valid = (
            seeder.select_indices_reference(bank)
        )
        optimized_indices, optimized_valid = (
            seeder.select_indices_optimized(bank)
        )
        assert torch.equal(reference_indices, optimized_indices)
        assert torch.equal(reference_valid, optimized_valid)

    # Threshold boundary: two unit boxes shifted so their IoU is the exact
    # configured diversity target up to the shared FP32 box arithmetic.
    threshold = 0.70
    shift = (1.0 - threshold) / (1.0 + threshold)
    boundary_bank = {
        "boxes": torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [shift, 0.0, 1.0 + shift, 1.0],
                    [2.0, 0.0, 3.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        ),
        "score": torch.tensor([[[3.0], [2.0], [1.0]]]),
        "valid": torch.ones(1, 3, dtype=torch.bool),
    }
    boundary_config = replace(
        _config(efficiency_mode="optimized"),
        num_slots=5,
        seed_diversity_iou=threshold,
    )
    boundary_seeder = SlotSeeder(boundary_config)
    ref = boundary_seeder.select_indices_reference(boundary_bank)
    opt = boundary_seeder.select_indices_optimized(boundary_bank)
    assert torch.equal(ref[0], opt[0])
    assert torch.equal(ref[1], opt[1])
    assert ref[1].sum().item() == 3
    assert torch.equal(ref[0][0, 3:], ref[0][0, :2])

    class _ZeroSemantic:
        @staticmethod
        def retrieve(state, boxes):
            del state
            return boxes.new_zeros((*boxes.shape[:2], 32))

    reference_config = replace(
        _config(efficiency_mode="reference"),
        num_slots=5,
    )
    optimized_config = replace(
        reference_config,
        efficiency_mode="optimized",
    )
    reference_seeder = SlotSeeder(reference_config)
    optimized_seeder = SlotSeeder(optimized_config)
    optimized_seeder.load_state_dict(reference_seeder.state_dict())
    mixed_empty_bank = _random_bank(
        batch=2,
        bank_size=8,
        valid_counts=(0, 3),
    )
    mixed_empty_bank.update(
        {
            "desc": torch.randn(2, 8, 32),
            "image_sizes": torch.tensor([[96, 96], [80, 88]]),
        }
    )
    reference_forward = reference_seeder(
        mixed_empty_bank,
        _ZeroSemantic(),
        None,
    )
    optimized_forward = optimized_seeder(
        mixed_empty_bank,
        _ZeroSemantic(),
        None,
    )
    for reference_value, optimized_value in zip(
        reference_forward,
        optimized_forward,
    ):
        torch.testing.assert_close(
            reference_value,
            optimized_value,
            atol=0.0,
            rtol=0.0,
        )

    if full_shape:
        torch.manual_seed(901)
        config = replace(
            _config(efficiency_mode="optimized", full_shape=True),
            num_slots=300,
        )
        seeder = SlotSeeder(config)
        bank = _random_bank(
            batch=4,
            bank_size=1000,
            valid_counts=(0, 299, 300, 1000),
            identical_scores=True,
            identical_boxes=False,
        )
        ref = seeder.select_indices_reference(bank)
        opt = seeder.select_indices_optimized(bank)
        assert torch.equal(ref[0], opt[0])
        assert torch.equal(ref[1], opt[1])


def _ownership_inputs(
    *,
    batch: int,
    slots: int,
    candidates: int,
    bank_size: int,
    dtype: torch.dtype,
    all_invalid: bool = False,
    assignment_support: bool = True,
    zero_fallback: bool = False,
    duplicate_indices: bool = False,
    no_matches: bool = False,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    List[Tuple[torch.Tensor, torch.Tensor]],
    List[Dict],
]:
    torch.manual_seed(
        batch * 100000 + slots * 1000 + candidates * 10 + bank_size
    )
    logits = torch.randn(batch, slots, candidates + 1, dtype=dtype)
    indices = torch.randint(0, bank_size, (batch, slots, candidates))
    if duplicate_indices and candidates > 1:
        indices[..., 1:] = indices[..., :1]
    valid = torch.rand(batch, slots, candidates) > 0.2
    if all_invalid:
        valid.zero_()
    raw_boxes = torch.rand(batch, bank_size, 4)
    lower = torch.minimum(raw_boxes[..., :2], raw_boxes[..., 2:])
    upper = torch.maximum(raw_boxes[..., :2], raw_boxes[..., 2:]) + 0.05
    boxes = torch.cat((lower, upper), dim=-1) * 100.0
    quality = torch.full(
        (batch, bank_size, 1),
        -1000.0 if zero_fallback else 0.5,
        dtype=dtype,
    )
    assign = torch.full(
        (batch, bank_size),
        -1,
        dtype=torch.long,
    )
    targets = []
    matches = []
    for batch_index in range(batch):
        gt_count = 0 if batch_index == batch - 1 and batch > 1 else 3
        gt_boxes = torch.tensor(
            [
                [10.0, 10.0, 40.0, 40.0],
                [30.0, 25.0, 70.0, 75.0],
                [55.0, 50.0, 95.0, 98.0],
            ],
            dtype=torch.float32,
        )[:gt_count]
        targets.append({"boxes_xyxy_abs": gt_boxes})
        match_count = 0 if no_matches else min(gt_count, slots)
        matches.append(
            (
                torch.arange(match_count, dtype=torch.long),
                torch.arange(match_count, dtype=torch.long),
            )
        )
        if assignment_support and match_count:
            for slot_index in range(match_count):
                candidate_index = int(indices[batch_index, slot_index, 0])
                assign[batch_index, candidate_index] = slot_index
                valid[batch_index, slot_index, 0] = True
                boxes[batch_index, candidate_index] = gt_boxes[slot_index]
    ownership = {
        "logits": logits,
        "cand_idx": indices,
        "cand_valid": valid,
    }
    bank = {
        "boxes": boxes,
        "qual": quality,
        "gt_assign": assign,
    }
    return ownership, bank, matches, targets


def _compare_ownership_case(**kwargs) -> None:
    ownership, bank, matches, targets = _ownership_inputs(**kwargs)
    reference_target, reference_mask = (
        build_soft_ownership_targets_reference(
            ownership,
            bank,
            matches,
            targets,
        )
    )
    optimized_target, optimized_mask = (
        build_soft_ownership_targets_optimized(
            ownership,
            bank,
            matches,
            targets,
        )
    )
    assert torch.equal(reference_mask, optimized_mask)
    torch.testing.assert_close(
        reference_target,
        optimized_target,
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    reference_logits = ownership["logits"].detach().clone().requires_grad_(True)
    optimized_logits = ownership["logits"].detach().clone().requires_grad_(True)
    reference_ownership = dict(ownership, logits=reference_logits)
    optimized_ownership = dict(ownership, logits=optimized_logits)
    reference_losses = soft_ownership_losses(
        reference_ownership,
        bank,
        matches,
        targets,
        efficiency_mode="reference",
    )
    optimized_losses = soft_ownership_losses(
        optimized_ownership,
        bank,
        matches,
        targets,
        efficiency_mode="optimized",
    )
    _assert_scalar_dict_close(
        reference_losses,
        optimized_losses,
        atol=1.0e-6,
    )
    reference_total = (
        reference_losses["loss_ownership"]
        + reference_losses["loss_dustbin"]
    )
    optimized_total = (
        optimized_losses["loss_ownership"]
        + optimized_losses["loss_dustbin"]
    )
    reference_total.backward()
    optimized_total.backward()
    torch.testing.assert_close(
        reference_logits.grad,
        optimized_logits.grad,
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_ownership_vectorization_contract(full_shape: bool) -> None:
    cases = [
        dict(
            batch=2,
            slots=12,
            candidates=1,
            bank_size=16,
            dtype=torch.float32,
            no_matches=True,
        ),
        dict(
            batch=2,
            slots=12,
            candidates=6,
            bank_size=32,
            dtype=torch.float32,
            all_invalid=True,
        ),
        dict(
            batch=2,
            slots=12,
            candidates=6,
            bank_size=32,
            dtype=torch.float32,
            assignment_support=True,
        ),
        dict(
            batch=2,
            slots=12,
            candidates=6,
            bank_size=32,
            dtype=torch.float32,
            assignment_support=False,
        ),
        dict(
            batch=1,
            slots=12,
            candidates=6,
            bank_size=32,
            dtype=torch.float32,
            assignment_support=False,
            zero_fallback=True,
            duplicate_indices=True,
        ),
        dict(
            batch=2,
            slots=12,
            candidates=6,
            bank_size=32,
            dtype=torch.bfloat16,
        ),
    ]
    if full_shape:
        cases.append(
            dict(
                batch=2,
                slots=300,
                candidates=48,
                bank_size=1000,
                dtype=torch.bfloat16,
                duplicate_indices=True,
            )
        )
    for case in cases:
        defaults = {
            "all_invalid": False,
            "assignment_support": True,
            "zero_fallback": False,
            "duplicate_indices": False,
            "no_matches": False,
        }
        defaults.update(case)
        _compare_ownership_case(**defaults)


def _slot_loss_inputs(
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[
    Dict[str, torch.Tensor],
    List[Dict],
    List[Tuple[torch.Tensor, torch.Tensor]],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    torch.manual_seed(712)
    batch, slots, classes, bank_size, candidates = 2, 12, 5, 32, 6
    prediction = {
        "pred_logits": torch.randn(batch, slots, classes, dtype=dtype),
        "pred_boxes": torch.rand(batch, slots, 4, dtype=dtype),
        "pred_quality": torch.randn(batch, slots, 1, dtype=dtype),
        "pred_exist": torch.randn(batch, slots, 1, dtype=dtype),
        "pred_scores": torch.rand(batch, slots, classes, dtype=dtype),
    }
    targets = _targets(100, batch=batch, empty_last=True, num_classes=classes)
    matches = [
        (
            torch.tensor([0, 4, 9], dtype=torch.long),
            torch.tensor([2, 0, 1], dtype=torch.long),
        ),
        (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        ),
    ]
    raw = torch.rand(batch, bank_size, 4)
    lower = torch.minimum(raw[..., :2], raw[..., 2:])
    upper = torch.maximum(raw[..., :2], raw[..., 2:]) + 0.01
    bank = {
        "boxes": torch.cat((lower, upper), dim=-1) * 100.0,
        "gt_assign": torch.randint(-1, 3, (batch, bank_size)),
    }
    ownership = {
        "cand_idx": torch.randint(
            0,
            bank_size,
            (batch, slots, candidates),
        ),
        "cand_valid": torch.rand(batch, slots, candidates) > 0.2,
    }
    return prediction, targets, matches, bank, ownership


def test_slot_losses_vectorization_contract() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        prediction, targets, matches, bank, ownership = _slot_loss_inputs(
            dtype=dtype
        )
        flat = flatten_matches(matches, targets, torch.device("cpu"))
        assert flat["batch_idx"].shape[0] == 3
        assert flat["flat_target_boxes_norm"].shape == (3, 4)

        _assert_scalar_dict_close(
            slot_round_losses_reference(
                prediction,
                targets,
                matches,
                num_classes=5,
            ),
            slot_round_losses_optimized(
                prediction,
                targets,
                matches,
                num_classes=5,
            ),
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            existence_loss_reference(prediction, targets, matches),
            existence_loss_optimized(prediction, targets, matches),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            duplicate_margin_loss_reference(
                prediction,
                targets,
                matches,
                margin=0.2,
            ),
            duplicate_margin_loss_optimized(
                prediction,
                targets,
                matches,
                margin=0.2,
            ),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            readout_gain_loss_reference(
                prediction,
                bank,
                ownership,
                targets,
                matches,
            ),
            readout_gain_loss_optimized(
                prediction,
                bank,
                ownership,
                targets,
                matches,
            ),
            atol=1.0e-6,
            rtol=0.0,
        )

        dn_slots = 8
        meta = {
            "valid": torch.tensor(
                [
                    [True] * dn_slots,
                    [False] * dn_slots,
                ]
            ),
            "positive": torch.tensor(
                [
                    [True, False] * (dn_slots // 2),
                    [False] * dn_slots,
                ]
            ),
            "source_gt": torch.zeros(2, dn_slots, dtype=torch.long),
        }
        dn_prediction = {
            "pred_logits": torch.randn(2, dn_slots, 5, dtype=dtype),
            "pred_boxes": torch.rand(2, dn_slots, 4, dtype=dtype),
            "pred_quality": torch.randn(2, dn_slots, 1, dtype=dtype),
            "pred_exist": torch.randn(2, dn_slots, 1, dtype=dtype),
        }
        _assert_scalar_dict_close(
            denoising_round_losses_reference(
                dn_prediction,
                meta,
                targets,
                include_existence=True,
            ),
            denoising_round_losses_optimized(
                dn_prediction,
                meta,
                targets,
                include_existence=True,
            ),
            atol=1.0e-6,
        )


def _forward_pair(
    *,
    full_shape: bool,
    training: bool,
) -> Tuple[
    EgoDetV2,
    EgoDetV2,
    torch.Tensor,
    torch.Tensor,
    List[Dict],
    Dict[str, object],
    Dict[str, object],
]:
    reference_config = _config(
        efficiency_mode="reference",
        full_shape=full_shape,
    )
    optimized_config = replace(
        reference_config,
        efficiency_mode="optimized",
    )
    torch.manual_seed(3407)
    reference_model = EgoDetV2(
        reference_config,
        fake_backbone=True,
    )
    optimized_model = EgoDetV2(
        optimized_config,
        fake_backbone=True,
    )
    optimized_model.load_state_dict(reference_model.state_dict())
    if training:
        reference_model.train()
        optimized_model.train()
    else:
        reference_model.eval()
        optimized_model.eval()
    side = 128 if full_shape else 96
    images = torch.randn(1, 3, side, side)
    pad_mask = torch.zeros(1, side, side, dtype=torch.bool)
    targets = _targets(side, batch=1)
    torch.manual_seed(991)
    reference_outputs = reference_model(images, pad_mask, targets, epoch=2)
    torch.manual_seed(991)
    optimized_outputs = optimized_model(images, pad_mask, targets, epoch=2)
    return (
        reference_model,
        optimized_model,
        images,
        pad_mask,
        targets,
        reference_outputs,
        optimized_outputs,
    )


def test_forward_and_loss_equivalence(full_shape: bool) -> None:
    (
        _,
        _,
        _,
        _,
        targets,
        reference,
        optimized,
    ) = _forward_pair(full_shape=full_shape, training=False)
    for key in ("indices", "valid"):
        assert torch.equal(reference["seed"][key], optimized["seed"][key])
    for key in ("selection_indices", "valid", "gt_assign"):
        assert torch.equal(reference["bank"][key], optimized["bank"][key])
    for round_index in range(3):
        for key in ("cand_idx", "cand_valid"):
            assert torch.equal(
                reference["ownership"][round_index][key],
                optimized["ownership"][round_index][key],
            )
    for key in ("pred_boxes", "pred_logits", "pred_quality", "pred_exist"):
        torch.testing.assert_close(
            reference[key],
            optimized[key],
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    reference_criterion = EgoDetV2Criterion(
        _config(
            efficiency_mode="reference",
            full_shape=full_shape,
        )
    )
    optimized_criterion = EgoDetV2Criterion(
        _config(
            efficiency_mode="optimized",
            full_shape=full_shape,
        )
    )
    prepared_targets = prepare_targets_for_matching_and_losses(
        targets,
        device=optimized["pred_logits"].device,
    )
    prediction_pairs = list(
        zip(
            list(reference["aux"])
            + [
                {
                    "pred_logits": reference["pred_logits"],
                    "pred_boxes": reference["pred_boxes"],
                }
            ],
            list(optimized["aux"])
            + [
                {
                    "pred_logits": optimized["pred_logits"],
                    "pred_boxes": optimized["pred_boxes"],
                }
            ],
        )
    )
    for reference_prediction, optimized_prediction in prediction_pairs:
        reference_matches = reference_criterion.matcher(
            reference_prediction,
            targets,
        )
        optimized_matches = optimized_criterion.matcher(
            optimized_prediction,
            prepared_targets,
        )
        for reference_match, optimized_match in zip(
            reference_matches,
            optimized_matches,
        ):
            assert torch.equal(reference_match[0], optimized_match[0])
            assert torch.equal(reference_match[1], optimized_match[1])
    reference_losses = reference_criterion(
        reference,
        targets,
        epoch=2,
    )
    optimized_losses = optimized_criterion(
        optimized,
        targets,
        epoch=2,
    )
    for name in reference_losses:
        if not name.startswith("loss_"):
            continue
        torch.testing.assert_close(
            reference_losses[name],
            optimized_losses[name],
            atol=1.0e-6,
            rtol=0.0,
            msg=lambda message, key=name: f"{key}: {message}",
        )


def test_inactive_extra_skip_contract() -> None:
    (
        _,
        _,
        _,
        _,
        targets,
        reference_outputs,
        optimized_outputs,
    ) = _forward_pair(full_shape=False, training=False)
    reference_criterion = EgoDetV2Criterion(
        _config(efficiency_mode="reference")
    )
    optimized_criterion = EgoDetV2Criterion(
        _config(efficiency_mode="optimized")
    )
    reference_losses = reference_criterion(
        reference_outputs,
        targets,
        epoch=0,
    )
    optimized_losses = optimized_criterion(
        optimized_outputs,
        targets,
        epoch=0,
    )
    assert optimized_losses["loss_duplicate"].item() == 0.0
    assert optimized_losses["loss_gain"].item() == 0.0
    torch.testing.assert_close(
        reference_losses["loss_total"],
        optimized_losses["loss_total"],
        atol=1.0e-6,
        rtol=0.0,
    )

    calls = {"duplicate": 0, "gain": 0}
    original_duplicate = criterion_module.duplicate_margin_loss
    original_gain = criterion_module.readout_gain_loss

    def counted_duplicate(*args, **kwargs):
        calls["duplicate"] += 1
        return original_duplicate(*args, **kwargs)

    def counted_gain(*args, **kwargs):
        calls["gain"] += 1
        return original_gain(*args, **kwargs)

    criterion_module.duplicate_margin_loss = counted_duplicate
    criterion_module.readout_gain_loss = counted_gain
    try:
        optimized_criterion(optimized_outputs, targets, epoch=0)
        assert calls == {"duplicate": 0, "gain": 0}
        reference_criterion(reference_outputs, targets, epoch=0)
        assert calls == {"duplicate": 1, "gain": 1}
    finally:
        criterion_module.duplicate_margin_loss = original_duplicate
        criterion_module.readout_gain_loss = original_gain

    # Isolate the zero-weight graph contract from all other optimized losses.
    leaf_reference = torch.randn(16, requires_grad=True)
    leaf_optimized = leaf_reference.detach().clone().requires_grad_(True)
    base_reference = leaf_reference.square().sum()
    base_optimized = leaf_optimized.square().sum()
    expensive = leaf_reference.sin().sum()
    legal_zero = leaf_optimized.sum() * 0.0
    reference_total = base_reference + expensive * 0.0
    optimized_total = base_optimized + legal_zero
    reference_total.backward()
    optimized_total.backward()
    assert torch.equal(leaf_reference.grad, leaf_optimized.grad)


def _gradient_group_names(model: EgoDetV2) -> Dict[str, str]:
    backbone_groups = model.backbone_param_groups()
    stem_ids = {id(value) for value in backbone_groups["stem"]}
    backbone_ids = {id(value) for value in backbone_groups["backbone"]}
    fpn_ids = {id(value) for value in backbone_groups.get("other", [])}
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        result[name] = _optimizer_logical_name(
            name,
            id(parameter),
            stem_ids=stem_ids,
            backbone_ids=backbone_ids,
            fpn_ids=fpn_ids,
        )
    return result


def _gradient_metrics(
    reference_model: EgoDetV2,
    optimized_model: EgoDetV2,
) -> Dict[str, Dict[str, float | bool]]:
    group_names = _gradient_group_names(reference_model)
    optimized_parameters = dict(optimized_model.named_parameters())
    grouped_reference: Dict[str, list[torch.Tensor]] = {}
    grouped_optimized: Dict[str, list[torch.Tensor]] = {}
    one_sided = []
    for name, parameter in reference_model.named_parameters():
        if name not in group_names:
            continue
        optimized_parameter = optimized_parameters[name]
        if (parameter.grad is None) != (optimized_parameter.grad is None):
            one_sided.append(name)
            continue
        if parameter.grad is None:
            continue
        group = group_names[name]
        grouped_reference.setdefault(group, []).append(
            parameter.grad.detach().float().flatten()
        )
        grouped_optimized.setdefault(group, []).append(
            optimized_parameter.grad.detach().float().flatten()
        )
    if one_sided:
        raise AssertionError(f"one-sided gradients: {one_sided}")

    result = {}
    all_groups = (
        "backbone_stem",
        "backbone",
        "fpn",
        "geometry_tower",
        "geometry_obj_head",
        "geometry_box_head",
        "geometry_quality_head",
        "semantic_tower",
        "slot_modules",
        "readout",
        "prediction_heads",
        "hdn",
    )
    for group in all_groups:
        reference_parts = grouped_reference.get(group, [])
        optimized_parts = grouped_optimized.get(group, [])
        if not reference_parts and not optimized_parts:
            result[group] = {
                "reference_grad_norm": 0.0,
                "optimized_grad_norm": 0.0,
                "cosine_similarity": 1.0,
                "relative_norm_error": 0.0,
                "max_abs_diff": 0.0,
                "both_missing": True,
            }
            continue
        reference = torch.cat(reference_parts)
        optimized = torch.cat(optimized_parts)
        reference_norm = reference.norm()
        optimized_norm = optimized.norm()
        if float(reference_norm) == 0.0 and float(optimized_norm) == 0.0:
            cosine = 1.0
        else:
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    reference,
                    optimized,
                    dim=0,
                    eps=1.0e-12,
                )
            )
        relative = float(
            (optimized_norm - reference_norm).abs()
            / reference_norm.clamp(min=1.0e-12)
        )
        max_abs = float((optimized - reference).abs().max())
        if not math.isfinite(cosine + relative + max_abs):
            raise AssertionError(f"non-finite gradient metric for {group}")
        if cosine < 0.9999:
            raise AssertionError(
                f"{group} gradient cosine {cosine} < 0.9999"
            )
        if relative > 1.0e-3:
            raise AssertionError(
                f"{group} gradient relative norm error {relative} > 1e-3"
            )
        result[group] = {
            "reference_grad_norm": float(reference_norm),
            "optimized_grad_norm": float(optimized_norm),
            "cosine_similarity": cosine,
            "relative_norm_error": relative,
            "max_abs_diff": max_abs,
            "both_missing": False,
        }
    return result


def test_gradient_and_update_equivalence(
    full_shape: bool,
) -> Dict[str, object]:
    (
        reference_model,
        optimized_model,
        images,
        pad_mask,
        targets,
        reference_outputs,
        optimized_outputs,
    ) = _forward_pair(full_shape=full_shape, training=True)
    reference_config = _config(
        efficiency_mode="reference",
        full_shape=full_shape,
    )
    optimized_config = replace(
        reference_config,
        efficiency_mode="optimized",
    )
    reference_criterion = EgoDetV2Criterion(reference_config)
    optimized_criterion = EgoDetV2Criterion(optimized_config)
    reference_losses = reference_criterion(
        reference_outputs,
        targets,
        epoch=2,
    )
    optimized_losses = optimized_criterion(
        optimized_outputs,
        targets,
        epoch=2,
    )
    torch.testing.assert_close(
        reference_losses["loss_total"],
        optimized_losses["loss_total"],
        atol=1.0e-6,
        rtol=0.0,
    )
    reference_losses["loss_total"].backward()
    optimized_losses["loss_total"].backward()
    gradient_metrics = _gradient_metrics(
        reference_model,
        optimized_model,
    )

    initial = {
        name: parameter.detach().clone()
        for name, parameter in reference_model.named_parameters()
    }
    reference_optimizer = build_optimizer(
        reference_model,
        reference_config,
    )
    optimized_optimizer = build_optimizer(
        optimized_model,
        optimized_config,
    )
    reference_scheduler = build_optimizer_step_scheduler(
        reference_optimizer,
        warmup_steps=reference_config.warmup_steps,
    )
    optimized_scheduler = build_optimizer_step_scheduler(
        optimized_optimizer,
        warmup_steps=optimized_config.warmup_steps,
    )
    torch.nn.utils.clip_grad_norm_(
        reference_model.parameters(),
        reference_config.clip_max_norm,
    )
    torch.nn.utils.clip_grad_norm_(
        optimized_model.parameters(),
        optimized_config.clip_max_norm,
    )
    reference_optimizer.step()
    optimized_optimizer.step()
    reference_scheduler.step()
    optimized_scheduler.step()

    optimized_parameters = dict(optimized_model.named_parameters())
    delta_reference = []
    delta_optimized = []
    max_abs = 0.0
    for name, parameter in reference_model.named_parameters():
        optimized_parameter = optimized_parameters[name]
        ref_delta = (parameter.detach() - initial[name]).float().flatten()
        opt_delta = (
            optimized_parameter.detach() - initial[name]
        ).float().flatten()
        delta_reference.append(ref_delta)
        delta_optimized.append(opt_delta)
        if ref_delta.numel():
            max_abs = max(
                max_abs,
                float((ref_delta - opt_delta).abs().max()),
            )
    ref_all = torch.cat(delta_reference)
    opt_all = torch.cat(delta_optimized)
    relative_update_error = float(
        (ref_all - opt_all).norm() / ref_all.norm().clamp(min=1.0e-12)
    )
    update_cosine = float(
        torch.nn.functional.cosine_similarity(
            ref_all,
            opt_all,
            dim=0,
            eps=1.0e-12,
        )
    )
    if update_cosine < 0.9999:
        raise AssertionError(
            f"parameter update cosine {update_cosine} < 0.9999"
        )
    if relative_update_error > 1.0e-3:
        raise AssertionError(
            "parameter update relative error "
            f"{relative_update_error} > 1e-3"
        )
    return {
        "gradient_groups": gradient_metrics,
        "parameter_update": {
            "cosine_similarity": update_cosine,
            "relative_norm_error": relative_update_error,
            "max_abs_diff": max_abs,
        },
        "input_shape": list(images.shape),
    }


def test_cuda_bf16_equivalence(
    *,
    device: torch.device,
    full_shape: bool,
) -> Dict[str, object]:
    if device.type != "cuda":
        raise ValueError("CUDA BF16 efficiency contract requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            f"BF16 is unsupported on {torch.cuda.get_device_name(device)}"
        )
    reference_config = replace(
        _config(
            efficiency_mode="reference",
            full_shape=full_shape,
        ),
        amp=True,
        amp_dtype="bf16",
    )
    optimized_config = replace(
        reference_config,
        efficiency_mode="optimized",
    )
    torch.manual_seed(1801)
    torch.cuda.manual_seed_all(1801)
    reference_model = EgoDetV2(
        reference_config,
        fake_backbone=True,
    ).to(device).train()
    optimized_model = EgoDetV2(
        optimized_config,
        fake_backbone=True,
    ).to(device).train()
    optimized_model.load_state_dict(reference_model.state_dict())
    reference_criterion = EgoDetV2Criterion(reference_config).to(device)
    optimized_criterion = EgoDetV2Criterion(optimized_config).to(device)
    side = 128 if full_shape else 96
    images = torch.randn(1, 3, side, side, device=device)
    pad_mask = torch.zeros(
        1,
        side,
        side,
        dtype=torch.bool,
        device=device,
    )
    targets = []
    for target in _targets(side, batch=1):
        targets.append(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
        )
    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_outputs = reference_model(
            images,
            pad_mask,
            targets,
            epoch=2,
        )
        reference_losses = reference_criterion(
            reference_outputs,
            targets,
            epoch=2,
        )
    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        optimized_outputs = optimized_model(
            images,
            pad_mask,
            targets,
            epoch=2,
        )
        optimized_losses = optimized_criterion(
            optimized_outputs,
            targets,
            epoch=2,
        )
    if reference_outputs["pred_logits"].dtype != torch.bfloat16:
        raise AssertionError(
            "CUDA autocast did not produce BF16 reference logits"
        )
    if optimized_outputs["pred_logits"].dtype != torch.bfloat16:
        raise AssertionError(
            "CUDA autocast did not produce BF16 optimized logits"
        )
    for key in ("indices", "valid"):
        assert torch.equal(
            reference_outputs["seed"][key],
            optimized_outputs["seed"][key],
        )
    for round_index in range(3):
        for key in ("cand_idx", "cand_valid"):
            assert torch.equal(
                reference_outputs["ownership"][round_index][key],
                optimized_outputs["ownership"][round_index][key],
            )
    for name in reference_losses:
        if not name.startswith("loss_"):
            continue
        torch.testing.assert_close(
            reference_losses[name],
            optimized_losses[name],
            atol=5.0e-4,
            rtol=0.0,
            msg=lambda message, key=name: f"{key}: {message}",
        )
    reference_losses["loss_total"].backward()
    optimized_losses["loss_total"].backward()
    gradient_metrics = _gradient_metrics(
        reference_model,
        optimized_model,
    )
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "autocast_dtype": "bfloat16",
        "full_shape": bool(full_shape),
        "gradient_groups": gradient_metrics,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / float(1024**3)
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EGO-Det V2 reference/optimized equivalence contracts"
    )
    parser.add_argument(
        "--full-shape",
        action="store_true",
        help="exercise K=1000, N=300 and P=48",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="optionally add the CUDA BF16 reference/optimized contract",
    )
    return parser


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    tests = [
        ("config_and_cli", test_config_and_cli_contract),
        ("profiler_schema", test_profiler_schema_contract),
        (
            "benchmark_trace_separation",
            test_benchmark_trace_separation_contract,
        ),
        (
            "slot_seeder_exact",
            lambda: test_slot_seeder_exact_contract(args.full_shape),
        ),
        (
            "ownership_vectorization",
            lambda: test_ownership_vectorization_contract(args.full_shape),
        ),
        ("slot_losses_vectorization", test_slot_losses_vectorization_contract),
        (
            "forward_and_loss_equivalence",
            lambda: test_forward_and_loss_equivalence(args.full_shape),
        ),
        ("inactive_extra_skip", test_inactive_extra_skip_contract),
    ]
    passed = []
    for name, test in tests:
        test()
        passed.append(name)
        print(f"[contracts_efficiency] PASS {name}", flush=True)
    gradient_result = test_gradient_and_update_equivalence(args.full_shape)
    passed.append("gradient_and_update_equivalence")
    print(
        "[contracts_efficiency] PASS gradient_and_update_equivalence",
        flush=True,
    )
    cuda_result = None
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        torch.cuda.set_device(requested_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(requested_device)
        cuda_result = test_cuda_bf16_equivalence(
            device=requested_device,
            full_shape=bool(args.full_shape),
        )
        passed.append("cuda_bf16_equivalence")
        print(
            "[contracts_efficiency] PASS cuda_bf16_equivalence",
            flush=True,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "version": "MOD10A",
                "full_shape": bool(args.full_shape),
                "tests": passed,
                "test_count": len(passed),
                "requested_device": str(requested_device),
                "cuda_bf16": cuda_result,
                **gradient_result,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
