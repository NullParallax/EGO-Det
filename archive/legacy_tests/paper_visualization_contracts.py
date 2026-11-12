from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ego_det_v2.paper_visualization import analyze_efficiency as efficiency_module
from ego_det_v2.paper_visualization.analyze_efficiency import count_parameters
from ego_det_v2.paper_visualization.collect_diagnostics import (
    _classify_slot_states,
    _side_weights,
)
from ego_det_v2.paper_visualization.collect_attention_diagnostics import (
    AttentionCapture,
)
from ego_det_v2.paper_visualization.export_prediction_groups import (
    _plot_comparison_gallery,
    _plot_group_gallery,
    compute_image_quality,
    compute_scene_statistics,
    render_prediction_example,
    select_quality_groups,
)
from ego_det_v2.paper_visualization.io import resolve_checkpoint_state
from ego_det_v2.paper_visualization.package_collected_data import (
    package_collected_data,
)
from ego_det_v2.paper_visualization.plot_attention import (
    plot_semantic_feature_grid_attention,
    plot_slot_competition,
)
from ego_det_v2.paper_visualization.plot_dataset import (
    _central_percentile_limits,
    generate_dataset_outputs,
)
from ego_det_v2.paper_visualization.plot_embeddings import _numpy_pca
from ego_det_v2.paper_visualization.plot_mechanism import (
    plot_boundary_readout,
    plot_progressive_slot_formation,
    plot_slot_lifecycle,
)
from ego_det_v2.paper_visualization.plot_training import (
    generate_training_outputs,
)
from ego_det_v2.paper_visualization.style import parse_vector_formats


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_vector_only_format_contract() -> None:
    assert parse_vector_formats("svg,pdf") == ("svg", "pdf")
    try:
        parse_vector_formats("png")
    except ValueError as exc:
        assert "vector-only" in str(exc)
    else:
        raise AssertionError("raster format was accepted")


def test_central_percentile_axis_contract() -> None:
    first = np.concatenate(
        (np.linspace(-0.020, 0.030, 98), np.asarray([-0.8, 0.9]))
    )
    second = np.concatenate(
        (np.linspace(-0.015, 0.025, 98), np.asarray([-0.6, 0.7]))
    )
    limits, omitted = _central_percentile_limits([first, second])
    assert limits[0] < 0 < limits[1]
    assert limits[0] > -0.05
    assert limits[1] < 0.06
    assert omitted > 0


def test_prediction_quality_matching_and_selection() -> None:
    import torch

    metrics, matching = compute_image_quality(
        image_id=7,
        prediction_boxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [21.0, 21.0, 39.0, 39.0],
                [50.0, 50.0, 60.0, 60.0],
            ]
        ),
        prediction_labels=torch.tensor([0, 1, 0]),
        prediction_scores=torch.tensor([0.9, 0.8, 0.7]),
        gt_boxes=torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 40.0, 40.0],
            ]
        ),
        gt_labels=torch.tensor([0, 1]),
        match_iou=0.5,
    )
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert np.isclose(metrics["precision"], 2.0 / 3.0)
    assert np.isclose(metrics["recall"], 1.0)
    assert np.isclose(metrics["mean_matched_iou"], 0.905)
    assert np.isclose(metrics["box_quality_score"], 0.724)
    assert matching["matched_gt"] == [0, 1, -1]

    scene = compute_scene_statistics(
        gt_boxes=torch.tensor(
            [[10.0, 10.0, 110.0, 30.0], [120.0, 20.0, 150.0, 90.0]]
        ),
        gt_labels=torch.tensor([0, 1]),
        image_size=(100, 200),
        elongated_aspect_ratio=1.5,
    )
    assert scene["image_aspect_ratio"] == 2.0
    assert scene["elongated_gt_fraction"] == 1.0
    assert scene["gt_class_count"] == 2
    assert 0.0 < scene["scene_complexity_score"] <= 1.0

    records = [
        {
            **metrics,
            "image_id": image_id,
            "box_quality_score": quality,
            "mean_matched_iou": mean_iou,
            "precision": precision,
            "recall": recall,
            "image_aspect_ratio": image_aspect_ratio,
            "scene_complexity_score": scene_complexity,
        }
        for (
            image_id,
            quality,
            mean_iou,
            precision,
            recall,
            image_aspect_ratio,
            scene_complexity,
        ) in (
            (10, 0.90, 0.95, 0.95, 1.00, 1.50, 0.60),
            (11, 0.80, 0.90, 0.90, 0.95, 0.75, 0.55),
            (14, 0.75, 0.88, 0.88, 0.92, 1.40, 0.50),
            (12, 0.20, 0.40, 0.40, 0.40, 1.45, 0.90),
            (13, 0.05, 0.20, 0.20, 0.20, 0.70, 0.80),
            (15, 0.10, 0.25, 0.25, 0.25, 1.35, 0.70),
        )
    ]
    good, poor, diagnostics = select_quality_groups(
        records,
        num_per_group=2,
        good_min_quality=0.65,
        poor_max_quality=0.30,
        min_gt=1,
        max_gt=12,
        min_image_aspect_ratio=1.20,
    )
    assert [record["image_id"] for record in good] == [10, 14]
    assert [record["image_id"] for record in poor] == [12, 15]
    assert diagnostics["good_candidate_count"] == 3
    assert diagnostics["poor_candidate_count"] == 3
    assert diagnostics["landscape_selection"]["good"]["replacements"] == [
        {
            "slot": 2,
            "replaced_image_id": 11,
            "replaced_image_aspect_ratio": 0.75,
            "replacement_image_id": 14,
            "replacement_image_aspect_ratio": 1.40,
        }
    ]
    assert diagnostics["landscape_selection"]["poor"]["replacements"] == [
        {
            "slot": 2,
            "replaced_image_id": 13,
            "replaced_image_aspect_ratio": 0.70,
            "replacement_image_id": 15,
            "replacement_image_aspect_ratio": 1.35,
        }
    ]

    good_five = [
        {
            **metrics,
            "image_id": 100 + image_id,
            "box_quality_score": 0.90 - 0.01 * image_id,
            "image_aspect_ratio": 1.40,
            "scene_complexity_score": 0.60,
        }
        for image_id in range(1, 6)
    ]
    poor_seven = [
        {
            **metrics,
            "image_id": image_id,
            "box_quality_score": 0.10,
            "image_aspect_ratio": aspect_ratio,
            "scene_complexity_score": 1.0 - 0.05 * image_id,
        }
        for image_id, aspect_ratio in (
            (1, 1.50),
            (2, 0.75),
            (3, 1.45),
            (4, 1.35),
            (5, 0.80),
            (6, 1.55),
            (7, 1.60),
        )
    ]
    _, poor_five, five_diagnostics = select_quality_groups(
        good_five + poor_seven,
        num_per_group=5,
        good_min_quality=0.65,
        poor_max_quality=0.30,
        min_gt=1,
        max_gt=12,
        min_image_aspect_ratio=1.20,
    )
    assert [record["image_id"] for record in poor_five] == [1, 6, 3, 4, 7]
    assert [
        item["slot"]
        for item in five_diagnostics["landscape_selection"]["poor"][
            "replacements"
        ]
    ] == [2, 5]


def test_prediction_example_vector_overlay(tmp_path: Path) -> None:
    payload = {
        "image": Image.new("RGB", (320, 220), color=(228, 232, 237)),
        "metrics": {
            "image_id": 17,
            "box_quality_score": 0.82,
            "true_positive": 1,
            "false_positive": 1,
            "false_negative": 1,
            "precision": 0.5,
            "recall": 0.5,
            "mean_matched_iou": 0.82,
        },
        "matching": {
            "matched_gt": [0, -1],
            "matched_iou": [0.82, 0.0],
            "unmatched_gt": [1],
        },
        "prediction_boxes": [
            [25.0, 30.0, 145.0, 175.0],
            [190.0, 40.0, 275.0, 130.0],
        ],
        "prediction_scores": [0.91, 0.62],
        "prediction_names": ["person", "dog"],
        "gt_boxes": [
            [20.0, 25.0, 150.0, 180.0],
            [180.0, 120.0, 300.0, 205.0],
        ],
    }
    saved = render_prediction_example(
        payload,
        tmp_path / "prediction_example",
        role="high_quality",
        label_topk=2,
        formats=("svg",),
    )
    text = saved[0].read_text(encoding="utf-8")
    assert text.startswith("<?xml")
    assert "HIGH QUALITY" in text
    assert "person 0.91" in text
    assert "<image" in text

    low_payload = {
        **payload,
        "metrics": {
            **payload["metrics"],
            "box_quality_score": 0.12,
        },
    }
    group_paths = _plot_group_gallery(
        [payload] * 4,
        tmp_path / "high_quality_gallery",
        role="high_quality",
        label_topk=2,
        formats=("svg",),
    )
    comparison_paths = _plot_comparison_gallery(
        [payload] * 5,
        [low_payload] * 5,
        tmp_path / "prediction_quality_comparison",
        label_topk=2,
        formats=("svg",),
    )
    for path in group_paths:
        gallery_text = path.read_text(encoding="utf-8")
        assert gallery_text.startswith("<?xml")
        assert "False positive" in gallery_text
        assert "Missed GT" in gallery_text
        assert "<image" in gallery_text
    comparison_text = comparison_paths[0].read_text(encoding="utf-8")
    assert comparison_text.startswith("<?xml")
    assert comparison_text.count("<image") == 10
    assert comparison_text.count("Matched prediction") == 1
    assert comparison_text.count("False positive") == 1
    assert "Matched GT" not in comparison_text
    assert "Missed GT" not in comparison_text
    assert "person 0.91" not in comparison_text
    assert "HIGH QUALITY" not in comparison_text


def test_training_plot_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    full = [
        {
            "epoch": epoch,
            "global_optimizer_step": epoch * 100,
            "AP": 0.10 + 0.03 * epoch,
            "AP75": 0.08 + 0.035 * epoch,
            "matched_mean_iou": 0.55 + 0.04 * epoch,
            "duplicate_rate": 0.04 / epoch,
            "readout_gain": -0.05 + 0.015 * epoch,
            "proposal_mass_mean": 0.5 + 0.02 * epoch,
            "ownership_matched_dustbin_rate": 0.01,
            "matched_accuracy": 0.55 + 0.05 * epoch,
            "matched_recall_50": 0.58 + 0.05 * epoch,
        }
        for epoch in (1, 2, 3)
    ]
    _write_jsonl(
        run_dir / "eval_history.jsonl",
        [
            {**full[0], "kind": "quick500", "epoch": 0},
            *full,
        ],
    )
    train = []
    for step in range(1, 81):
        train.append(
            {
                "epoch": min(step // 25, 3),
                "global_optimizer_step": step,
                "dense": 1.0 / (1 + 0.01 * step),
                "own": 1.2 / (1 + 0.01 * step),
                "cls": 0.9 / (1 + 0.02 * step),
                "box": 1.1 / (1 + 0.02 * step),
                "quality": 0.8 / (1 + 0.01 * step),
                "exist": 0.7 / (1 + 0.01 * step),
                "duplicate": 0.0 if step < 50 else 0.3 / (1 + 0.01 * step),
                "gain": 0.0 if step < 50 else 0.2,
            }
        )
    _write_jsonl(run_dir / "train_log.jsonl", train)
    result = generate_training_outputs(
        run_dir,
        tmp_path / "output",
        warmup_steps=10,
        formats=("svg",),
    )
    assert len(result["artifacts"]) == 2
    assert "duplicate" in result["training_log_summary"]["plotted_fields"]
    for artifact in result["artifacts"]:
        assert Path(artifact).read_text(encoding="utf-8").startswith("<?xml")


def test_dataset_plot_smoke(tmp_path: Path) -> None:
    object_records = []
    for index in range(90):
        seed = 0.2 + 0.006 * (index % 60)
        final = min(seed + 0.08 - 0.001 * (index % 7), 0.98)
        pre = max(seed + 0.02, 0)
        post = min(pre + 0.04 - 0.002 * (index % 5), 0.99)
        object_records.append(
            {
                "image_id": index // 5,
                "object_id": index,
                "size": ("small", "medium", "large")[index % 3],
                "seed_iou": seed,
                "final_iou": final,
                "pre_readout_iou": pre,
                "post_readout_iou": post,
                "readout_gain": post - pre,
                "boundary_source_count": 1 + index % 4,
                "predicted_quality": min(post + 0.03, 0.99),
            }
        )
    slot_records = []
    for index in range(180):
        group = ("matched", "unmatched", "low_iou")[index % 3]
        slot_records.append(
            {
                "slot_group": group,
                "dustbin_probability": (
                    0.15 if group == "matched" else 0.75
                )
                + 0.001 * (index % 20),
                "best_local_proposal_iou": 0.1 + 0.004 * (index % 120),
            }
        )
    recall_records = []
    for index in range(60):
        recall_records.append(
            {
                "size": ("small", "medium", "large")[index % 3],
                "best_iou_k_50": 0.35 + 0.01 * (index % 20),
                "best_iou_k_100": 0.45 + 0.01 * (index % 20),
                "best_iou_k_300": 0.55 + 0.01 * (index % 20),
                "best_iou_k_1000": 0.65 + 0.01 * (index % 20),
                "best_iou_k_1500": 0.70 + 0.01 * (index % 20),
            }
        )
    error_records = []
    for error_index, error_type in enumerate(
        (
            "classification",
            "localization",
            "duplicate",
            "background_false_positive",
            "missed_detection",
        )
    ):
        for size_index, size in enumerate(("small", "medium", "large")):
            error_records.append(
                {
                    "error_type": error_type,
                    "size": size,
                    "count": 3 + error_index + size_index,
                    "evaluation_protocol_id": "synthetic-coco-v1",
                    "attribution_policy_id": "synthetic-exclusive-v1",
                }
            )
    category_records = []
    for index in range(9):
        category_records.append(
            {
                "category": f"category_{index}",
                "supercategory": f"group_{index // 3}",
                "baseline_ap": 0.25 + 0.01 * index,
                "full_ap": 0.27 + 0.009 * index,
                "evaluation_protocol_id": "synthetic-coco-v1",
                "source_type": "same_evaluator",
            }
        )
    object_path = tmp_path / "objects.jsonl"
    slot_path = tmp_path / "slots.jsonl"
    recall_path = tmp_path / "recall.jsonl"
    error_path = tmp_path / "errors.jsonl"
    category_path = tmp_path / "categories.jsonl"
    _write_jsonl(object_path, object_records)
    _write_jsonl(slot_path, slot_records)
    _write_jsonl(recall_path, recall_records)
    _write_jsonl(error_path, error_records)
    _write_jsonl(category_path, category_records)
    result = generate_dataset_outputs(
        object_path,
        tmp_path / "output",
        slot_data=slot_path,
        recall_data=recall_path,
        error_data=error_path,
        category_data=category_path,
        formats=("svg",),
    )
    assert not result["skipped"]
    assert len(result["artifacts"]) == 8
    coverage = result["generated"]["proposal_coverage"]
    assert len(coverage["iou_thresholds"]) == 10
    assert coverage["iou_thresholds"][0] == 0.5
    assert coverage["iou_thresholds"][-1] == 0.95
    assert np.all(
        np.diff(coverage["average_recall_by_scale"]["all"]) >= -1.0e-9
    )
    gain_panel = result["generated"]["boundary_source_diversity"][
        "gain_panel"
    ]
    assert gain_panel["whisker_percentiles"] == [10.0, 90.0]
    assert gain_panel["y_limits"][0] < 0 < gain_panel["y_limits"][1]
    assert gain_panel["tail_points_not_drawn"] > 0
    assert all(Path(path).exists() for path in result["artifacts"])
    assert all(
        "<image" not in Path(path).read_text(encoding="utf-8")
        for path in result["artifacts"]
    )


def test_mechanism_plot_smoke(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    scene_dir = tmp_path / "scenes"
    image_dir.mkdir()
    scene_dir.mkdir()
    image_path = image_dir / "image_1.png"
    Image.new("RGB", (160, 100), color=(235, 238, 242)).save(image_path)
    slots = 5
    gt_boxes = [[15, 15, 65, 75], [90, 20, 145, 85]]
    stage_boxes = {
        "seed": [
            [15 + index * 2, 15, 45 + index * 2, 45]
            for index in range(slots)
        ],
        "round1": [
            [14 + index * 2, 14, 55 + index * 4, 60]
            for index in range(slots)
        ],
        "round2": [
            [14 + index * 4, 14, 65 + index * 8, 75]
            for index in range(slots)
        ],
        "round3": [
            [15, 15, 65, 75],
            [90, 20, 145, 85],
            [20, 20, 60, 70],
            [70, 15, 110, 60],
            [5, 5, 25, 25],
        ],
    }
    association = {
        stage: np.clip(
            np.asarray(
                [
                    [0.25 + 0.15 * round_index, 0.05],
                    [0.05, 0.30 + 0.14 * round_index],
                    [0.20 + 0.08 * round_index, 0.10],
                    [0.08, 0.18 + 0.06 * round_index],
                    [0.02, 0.01],
                ]
            ),
            0,
            1,
        ).tolist()
        for round_index, stage in enumerate(
            ("seed", "round1", "round2", "round3")
        )
    }
    scene = {
        "schema_version": 1,
        "image_id": 1,
        "image_path": "../images/image_1.png",
        "image_size": [100, 160],
        "orig_size": [100, 160],
        "gt_boxes": gt_boxes,
        "gt_labels": [1, 2],
        "slot_boxes": stage_boxes,
        "association_iou": association,
        "slot_states": {
            "seed": ["unstable", "unstable", "unstable", "background", "background"],
            "round1": ["matched", "unstable", "duplicate", "unstable", "background"],
            "round2": ["matched", "matched", "duplicate", "unstable", "background"],
            "round3": ["matched", "matched", "duplicate", "unstable", "background"],
        },
        "matched_pairs": [
            {"slot_id": 0, "gt_id": 0, "final_iou": 0.9},
            {"slot_id": 1, "gt_id": 1, "final_iou": 0.86},
        ],
        "readout_example": {
            "selection_policy": "test-only deterministic fixture",
            "slot_id": 0,
            "gt_id": 0,
            "candidate_boxes": [
                [12 + index * 3, 12, 62 + index * 2, 72]
                for index in range(6)
            ],
            "candidate_valid": [True] * 6,
            "proposal_weights": [0.25, 0.20, 0.18, 0.12, 0.10, 0.05],
            "side_weights": [
                [0.45, 0.20, 0.10, 0.10, 0.10, 0.05],
                [0.10, 0.45, 0.15, 0.10, 0.10, 0.10],
                [0.10, 0.10, 0.45, 0.15, 0.10, 0.10],
                [0.10, 0.10, 0.10, 0.45, 0.15, 0.10],
            ],
            "pre_box": [18, 18, 58, 66],
            "evidence_box": [14, 14, 64, 73],
            "final_box": [15, 15, 65, 75],
            "gt_box": [15, 15, 65, 75],
        },
    }
    scene_path = scene_dir / "image_1.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    saved, _ = plot_progressive_slot_formation(
        scene_path,
        tmp_path / "output",
        formats=("svg",),
    )
    assert saved[0].exists()
    saved, _ = plot_boundary_readout(
        scene_path,
        tmp_path / "output",
        formats=("svg",),
    )
    assert saved[0].exists()

    lifecycle = []
    for slot_id in range(16):
        lifecycle.append(
            {
                "image_id": 1,
                "slot_id": slot_id,
                "seed_state": ("background", "unstable")[slot_id % 2],
                "round1_state": ("unstable", "duplicate")[slot_id % 2],
                "round2_state": ("matched", "duplicate")[slot_id % 2],
                "round3_state": ("matched", "background")[slot_id % 2],
            }
        )
    lifecycle_path = tmp_path / "lifecycle.jsonl"
    _write_jsonl(lifecycle_path, lifecycle)
    saved, _ = plot_slot_lifecycle(
        lifecycle_path,
        tmp_path / "output",
        formats=("svg",),
    )
    assert saved[0].exists()


def test_attention_capture_contract() -> None:
    import torch
    import torch.nn as nn

    from ego_det_v2.configs.config import EgoDetV2Config
    from ego_det_v2.models.semantic_support import SemanticSupport
    from ego_det_v2.models.slot_update import SlotUpdateRound

    config = EgoDetV2Config(
        d_model=32,
        num_slots=5,
        slot_rounds=3,
        token_levels=("p5", "p6"),
        sem_convs=1,
    )

    class DiagnosticModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.semantic_support = SemanticSupport(config)
            self.slot_updates = nn.ModuleList(
                [SlotUpdateRound(config) for _ in range(3)]
            )

    torch.manual_seed(3407)
    model = DiagnosticModel().eval()
    capture = AttentionCapture(model)
    features = {
        "p3": torch.randn(1, 32, 8, 8),
        "p4": torch.randn(1, 32, 4, 4),
        "p5": torch.randn(1, 32, 2, 2),
        "p6": torch.randn(1, 32, 1, 1),
    }
    masks = {
        name: torch.zeros(
            (1, value.shape[-2], value.shape[-1]),
            dtype=torch.bool,
        )
        for name, value in features.items()
    }
    image_sizes = torch.tensor([[64, 64]])
    semantic_state = model.semantic_support(features, masks, image_sizes)
    s = torch.randn(1, 5, 32)
    c = torch.randn(1, 5, 32)
    boxes = torch.rand(1, 5, 4)
    boxes[..., 2:] = 0.15 + 0.35 * boxes[..., 2:]
    for update in model.slot_updates:
        s, c = update(
            s,
            c,
            boxes,
            model.semantic_support,
            semantic_state,
        )
    arrays = capture.export(matching_slots=5)
    capture.close()
    assert arrays["semantic_attention_heads"].shape == (3, 8, 5, 6)
    assert arrays["competition_attention_heads"].shape == (3, 8, 5, 5)
    assert np.allclose(
        arrays["competition_attention_heads"].astype(np.float32).sum(
            axis=-1
        ),
        1.0,
        atol=2.0e-3,
    )


def test_attention_plot_smoke(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    scene_dir = tmp_path / "scenes"
    image_dir.mkdir()
    scene_dir.mkdir()
    Image.new("RGB", (160, 100), color=(225, 232, 238)).save(
        image_dir / "image_7.png"
    )
    boxes = [
        [15, 15, 65, 75],
        [90, 20, 145, 85],
        [20, 20, 60, 70],
        [70, 15, 110, 60],
        [5, 5, 25, 25],
    ]
    scene = {
        "schema_version": 1,
        "image_id": 7,
        "image_path": "../images/image_7.png",
        "image_size": [100, 160],
        "gt_boxes": boxes[:2],
        "slot_boxes": {"round3": boxes},
        "association_iou": {
            "round3": [
                [0.9, 0.1],
                [0.1, 0.85],
                [0.6, 0.1],
                [0.1, 0.4],
                [0.01, 0.02],
            ]
        },
        "matched_pairs": [
            {"slot_id": 0, "gt_id": 0, "final_iou": 0.9},
            {"slot_id": 1, "gt_id": 1, "final_iou": 0.85},
        ],
    }
    scene_path = scene_dir / "image_7.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    rng = np.random.default_rng(3407)
    semantic = rng.random((3, 2, 5, 27), dtype=np.float32)
    semantic /= semantic.sum(axis=-1, keepdims=True)
    competition = rng.random((3, 2, 5, 5), dtype=np.float32)
    competition /= competition.sum(axis=-1, keepdims=True)
    attention_path = tmp_path / "attention_image_7.npz"
    np.savez_compressed(
        attention_path,
        schema_version=np.asarray(1),
        image_id=np.asarray(7),
        checkpoint_epoch=np.asarray(5),
        checkpoint_state=np.asarray("model"),
        semantic_attention_heads=semantic.astype(np.float16),
        competition_attention_heads=competition.astype(np.float16),
        slot_boxes_pre=np.zeros((3, 5, 4), dtype=np.float32),
        token_level_names=np.asarray(["p5", "p6", "global"]),
        token_level_heights=np.asarray([4, 2, 0]),
        token_level_widths=np.asarray([5, 3, 0]),
        token_level_offsets=np.asarray([0, 20, 26]),
        token_level_lengths=np.asarray([20, 6, 1]),
        token_mask=np.zeros(27, dtype=bool),
        image_size=np.asarray([100, 160]),
    )
    semantic_saved, semantic_summary = (
        plot_semantic_feature_grid_attention(
            attention_path,
            scene_path,
            tmp_path / "semantic",
            formats=("svg",),
        )
    )
    competition_saved, competition_summary = plot_slot_competition(
        attention_path,
        scene_path,
        tmp_path / "competition",
        formats=("svg",),
    )
    assert semantic_saved[0].exists()
    assert competition_saved[0].exists()
    assert "not a pixel-level object mask" in (
        semantic_summary["interpretation_limit"]
    )
    assert competition_summary["slice_renormalized"] is False
    assert "<image" in semantic_saved[0].read_text(encoding="utf-8")


def test_numpy_pca_fallback_is_deterministic() -> None:
    rng = np.random.default_rng(3407)
    values = rng.normal(size=(40, 8)).astype(np.float32)
    first = _numpy_pca(values, components=3)
    second = _numpy_pca(values, components=3)
    assert first.shape == (40, 3)
    assert np.allclose(first, second)


def test_attention_transfer_package(tmp_path: Path) -> None:
    collected = tmp_path / "collected_data"
    attention_dir = collected / "attention"
    scene_dir = collected / "scenes"
    image_dir = collected / "images"
    for directory in (attention_dir, scene_dir, image_dir):
        directory.mkdir(parents=True)
    (attention_dir / "attention_image_7.npz").write_bytes(b"npz-fixture")
    (attention_dir / "attention_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (collected / "collection_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (collected / "slot_embeddings.pending.npz").write_bytes(b"embedding")
    (image_dir / "image_7.png").write_bytes(b"png-fixture")
    (scene_dir / "image_7.json").write_text(
        json.dumps(
            {
                "image_id": 7,
                "image_path": "../images/image_7.png",
            }
        ),
        encoding="utf-8",
    )
    result = package_collected_data(output_root=tmp_path, mode="attention")
    archive = Path(result["archive"])
    assert archive.exists()
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    assert "transfer_manifest.json" in names
    assert "collected_data/attention/attention_image_7.npz" in names
    assert "collected_data/scenes/image_7.json" in names
    assert "collected_data/images/image_7.png" in names


def test_parameter_count_helper() -> None:
    import torch

    module = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 2, bias=False),
    )
    result = count_parameters(module)
    assert result["total"] == 24
    assert result["trainable"] == 24


def test_checkpoint_state_resolution_is_fail_closed() -> None:
    use_ema, source = resolve_checkpoint_state(
        {
            "config": {"use_ema": True},
            "model": {"weight": 1},
            "ema": {"weight": 2},
        },
        None,
    )
    assert use_ema is True
    assert source == "checkpoint config.use_ema"
    use_ema, _ = resolve_checkpoint_state(
        {"model": {"weight": 1}},
        None,
    )
    assert use_ema is False
    try:
        resolve_checkpoint_state(
            {
                "model": {"weight": 1},
                "ema": {"weight": 2},
            },
            None,
        )
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous checkpoint state was accepted")


def test_efficiency_breakdown_plot_smoke(tmp_path: Path) -> None:
    parameter_groups = (
        "Backbone + FPN",
        "Dense geometry",
        "Semantic support",
        "Slot seeding",
        "Slot updates",
        "Evidence readout",
        "Prediction heads",
        "Training-only HDN",
    )
    result = {
        "parameter_breakdown": {
            name: (index + 1) * 1_000_000
            for index, name in enumerate(parameter_groups)
        },
        "full_model": {
            "flops": {"giga_flops": 150.0},
            "latency": {"latency_ms_mean": 20.0},
        },
        "geometry_path": {
            "flops": {"giga_flops": 120.0},
            "latency": {"latency_ms_mean": 15.0},
        },
    }
    saved = efficiency_module._plot_efficiency_breakdown(
        result,
        tmp_path,
        formats=("svg",),
    )
    text = saved[0].read_text(encoding="utf-8")
    assert text.startswith("<?xml")
    assert "<image" not in text


def test_efficiency_table_uses_explicit_reported_scope(
    tmp_path: Path,
) -> None:
    result = {
        "full_model": {
            "parameters": {"total_millions": 42.0},
            "flops": {"giga_flops": 123.0},
        },
        "reported_latency": {
            "scope": "predict",
            "latency_ms_mean": 25.0,
            "fps_from_mean_latency": 40.0,
        },
        "ap": {"AP": 0.321},
    }
    paths = efficiency_module._write_csv_and_tex(tmp_path, result)
    csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
    latex_text = Path(paths["latex_row"]).read_text(encoding="utf-8")
    assert "25.0,40.0" in csv_text
    assert "25.00 & 40.0 & 32.1" in latex_text


def test_torch21_dispatch_flop_fallback() -> None:
    import torch

    module = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 8 * 8, 5),
    )
    inputs = torch.randn(1, 3, 10, 10)
    original = efficiency_module.FlopCounterMode
    efficiency_module.FlopCounterMode = None
    try:
        result = efficiency_module.measure_flops(lambda: module(inputs))
    finally:
        efficiency_module.FlopCounterMode = original
    assert result["total"] == 16_384
    assert result["counter"] == "ego_det_v2.compat.TorchDispatchMode"
    assert result["operation_breakdown"]["aten.conv2d"] == 13_824
    assert result["operation_breakdown"]["aten.linear"] == 2_560


def test_cpu_only_profiler_flop_fallback() -> None:
    import torch

    module = torch.nn.Conv2d(3, 4, 3)
    inputs = torch.randn(1, 3, 10, 10)
    original_counter = efficiency_module.FlopCounterMode
    original_compat = efficiency_module._CompatibilityFlopCounterMode
    efficiency_module.FlopCounterMode = None
    efficiency_module._CompatibilityFlopCounterMode = None
    try:
        result = efficiency_module.measure_flops(lambda: module(inputs))
    finally:
        efficiency_module.FlopCounterMode = original_counter
        efficiency_module._CompatibilityFlopCounterMode = original_compat
    assert result["total"] == 13_824
    assert result["counter"] == (
        "torch.profiler.profile(CPU, with_flops=True)"
    )
    assert result["backend_attempts"][-1]["raw_event_count"] > 0


def test_torch21_dispatch_attention_flops() -> None:
    import torch

    query = torch.randn(1, 4, 10, 8)
    key = torch.randn(1, 4, 20, 8)
    value = torch.randn(1, 4, 20, 8)
    self_query = torch.randn(1, 10, 32)
    attention = torch.nn.MultiheadAttention(
        32,
        4,
        batch_first=True,
    ).eval()
    original = efficiency_module.FlopCounterMode
    efficiency_module.FlopCounterMode = None
    try:
        sdpa = efficiency_module.measure_flops(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
            )
        )
        native = efficiency_module.measure_flops(
            lambda: attention(
                self_query,
                self_query,
                self_query,
                need_weights=False,
            )
        )
    finally:
        efficiency_module.FlopCounterMode = original
    assert sdpa["counter"] == "ego_det_v2.compat.TorchDispatchMode"
    assert sdpa["total"] == 25_600
    assert native["counter"] == "ego_det_v2.compat.TorchDispatchMode"
    assert native["total"] == 94_720


def test_ap_epoch_alignment_is_fail_closed(tmp_path: Path) -> None:
    history = tmp_path / "eval_history.jsonl"
    _write_jsonl(
        history,
        [
            {"epoch": 1, "AP": 0.2},
            {"epoch": 2, "AP": 0.3},
        ],
    )
    missing = efficiency_module._resolve_ap(
        checkpoint={"epoch": 3},
        ap_override=None,
        metrics_json=None,
        eval_history=history,
    )
    assert missing["AP"] is None
    assert missing["candidate_AP"] == 0.3
    aligned = efficiency_module._resolve_ap(
        checkpoint={"epoch": 2},
        ap_override=None,
        metrics_json=None,
        eval_history=history,
    )
    assert aligned["AP"] == 0.3
    assert aligned["epoch_aligned"] is True


def test_intermediate_diagnostic_contracts() -> None:
    import torch

    association = torch.tensor(
        [
            [0.8, 0.1],
            [0.6, 0.1],
            [0.1, 0.7],
            [0.2, 0.1],
            [0.02, 0.03],
        ]
    )
    states = _classify_slot_states(
        association,
        match_iou=0.5,
        background_iou=0.1,
    )
    assert states == [
        "matched",
        "duplicate",
        "matched",
        "unstable",
        "background",
    ]
    raw = torch.zeros(1, 2, 3, 4)
    ownership = {
        "w": torch.tensor(
            [
                [
                    [0.5, 0.3, 0.1, 0.1],
                    [0.2, 0.1, 0.0, 0.7],
                ]
            ]
        ),
        "cand_valid": torch.tensor(
            [[[True, True, True], [True, True, False]]]
        ),
    }
    side = _side_weights(raw, ownership)
    assert side.shape == (1, 2, 3, 4)
    assert torch.allclose(
        side.sum(dim=2),
        torch.ones(1, 2, 4),
        atol=1.0e-6,
    )
    assert torch.all(side[0, 1, 2] == 0)


def main() -> None:
    test_vector_only_format_contract()
    test_central_percentile_axis_contract()
    test_prediction_quality_matching_and_selection()
    test_parameter_count_helper()
    test_checkpoint_state_resolution_is_fail_closed()
    test_torch21_dispatch_flop_fallback()
    test_cpu_only_profiler_flop_fallback()
    test_torch21_dispatch_attention_flops()
    test_intermediate_diagnostic_contracts()
    test_attention_capture_contract()
    test_numpy_pca_fallback_is_deterministic()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, function in (
            ("training", test_training_plot_smoke),
            ("dataset", test_dataset_plot_smoke),
            ("mechanism", test_mechanism_plot_smoke),
            ("attention", test_attention_plot_smoke),
            ("prediction_example", test_prediction_example_vector_overlay),
            ("transfer", test_attention_transfer_package),
            ("efficiency", test_efficiency_breakdown_plot_smoke),
            (
                "efficiency_table",
                test_efficiency_table_uses_explicit_reported_scope,
            ),
            ("ap_alignment", test_ap_epoch_alignment_is_fail_closed),
        ):
            directory = root / name
            directory.mkdir()
            function(directory)
    print("paper_visualization contracts: PASS")


if __name__ == "__main__":
    main()
