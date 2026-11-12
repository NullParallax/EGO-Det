from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class EgoDetV2Config:
    """Single source of truth for the formal EGO-Det V2 workflow.

    Defaults are the formal 4090/COCO settings.  Tests may construct a smaller
    instance, but launchers intentionally start from this dataclass unchanged.
    """

    # Data
    num_classes: int = 80
    train_short_sides: Tuple[int, ...] = (480, 512, 544, 576, 608, 640, 672)
    train_max_size: int = 1024
    val_short_side: int = 672
    val_max_size: int = 1024
    random_crop_prob: float = 0.5
    hflip_prob: float = 0.5
    pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Backbone/FPN
    d_model: int = 256
    fpn_levels: Tuple[str, ...] = ("p3", "p4", "p5", "p6")
    fpn_strides: Tuple[int, ...] = (8, 16, 32, 64)
    backbone_lr: float = 1.0e-5
    backbone_stem_lr: float = 1.0e-6
    overlock_gradient_checkpointing: bool = True
    freeze_backbone: bool = False

    # Class-agnostic dense geometry
    tower_convs: int = 4
    center_sampling_radius: float = 1.5
    level_scale_ranges: Tuple[Tuple[float, float], ...] = (
        (0.0, 64.0),
        (64.0, 128.0),
        (128.0, 256.0),
        (256.0, 100000.0),
    )
    bank_pre_topk_per_level: int = 400
    bank_min_per_level: int = 64
    bank_topk_train: int = 1000
    bank_topk_eval: int = 1500
    # Legacy-only dynamic dense promotion threshold, retained for A/B
    # reproduction. stable_fcos never uses current proposal IoU as a target.
    assign_iou_promote_threshold: float = 0.5
    # MOD9A separates stable dense supervision from the current proposal
    # identity used by Bank/Ownership. legacy_dynamic reproduces the MOD8
    # dense-target behavior for contracts and server A/B evidence only.
    dense_assignment_mode: str = "stable_fcos"
    proposal_identity_iou_threshold: float = 0.5
    assign_ensure_min_one_positive: bool = False

    # Pure implementation routing. ``reference`` preserves the validated
    # MOD9A eager path; ``optimized`` may only select contract-equivalent hot
    # path implementations.
    efficiency_mode: str = "reference"
    # Profiler-only cumulative ablation level. Formal optimized execution uses
    # all levels (6); reference ignores this value.
    # 1 inactive extras, 2 seeder, 3 ownership, 4 slot losses,
    # 5 target/matcher preparation, 6 hot-path host-sync removal.
    efficiency_ablation_level: int = 6

    # Semantic support
    sem_convs: int = 2
    token_levels: Tuple[str, ...] = ("p5", "p6")
    roi_size: int = 7
    # Controlled method-ablation switches.  Defaults preserve the formal
    # EGO-Det path and old checkpoints that do not contain these fields.
    semantic_global_attention: bool = True
    semantic_regional_retrieval: bool = True

    # Slots/readout
    num_slots: int = 300
    slot_rounds: int = 3
    evidence_per_slot: int = 48
    seed_diversity_iou: float = 0.7
    level_embed_dim: int = 16
    seed_selection_mode: str = "diversity"
    slot_competition_enabled: bool = True
    readout_mode: str = "boundary"
    readout_dustbin_gating: bool = True
    readout_bounded_residual: bool = True

    # Hypothesis denoising
    hdn_max_pairs: int = 100
    hdn_pos_box_noise: float = 0.4
    hdn_neg_box_noise: Tuple[float, float] = (0.4, 0.8)
    hdn_label_noise: float = 0.2

    # Existence/score
    exist_iou_gate: float = 0.5
    gamma_init: float = 1.0
    gamma_range: Tuple[float, float] = (0.25, 2.0)

    # Matching and loss weights
    set_cost_class: float = 2.0
    set_cost_bbox: float = 5.0
    set_cost_giou: float = 2.0
    w_dense_obj: float = 1.0
    w_dense_giou: float = 2.0
    w_dense_quality: float = 1.0
    w_cls: float = 2.0
    w_l1: float = 5.0
    w_giou: float = 2.0
    w_quality: float = 1.0
    w_exist: float = 1.0
    w_ownership: float = 1.0
    w_dustbin: float = 0.1
    w_dup: float = 0.25
    w_gain: float = 0.25
    dup_margin: float = 0.2
    ownership_warmup_epochs: int = 1
    dup_gain_start_epoch: int = 2

    # Optimisation
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    clip_max_norm: float = 0.1
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 12
    lr_drop_epoch: int = 11
    warmup_steps: int = 1000
    amp: bool = True
    amp_dtype: str = "bf16"
    num_workers: int = 8
    use_ema: bool = False
    ema_decay: float = 0.9998
    seed: int = 3407

    # Evaluation/logging
    quick_eval_every_steps: int = 2500
    full_eval_last_epoch_only: bool = False
    log_every_steps: int = 25
    # -1 preserves the MOD9A behavior (oracle follows log_every_steps), 0
    # disables the expensive monitor, and a positive value gives its cadence
    # in optimizer-step units without changing any training tensor.
    oracle_monitor_every_optimizer_steps: int = -1
    ckpt_every_steps: int = 2500
    eval_nms_diagnostic: bool = False
    output_dir: str = "outputs/ego_det_v2/e2e_run1"

    # Eight-image engineering contract. COCO AP is deliberately not a gate.
    overfit_min_matched_accuracy: float = 0.90
    overfit_min_matched_iou: float = 0.70
    overfit_min_matched_recall: float = 0.90

    def __post_init__(self) -> None:
        if len(self.fpn_levels) != len(self.fpn_strides):
            raise ValueError("fpn_levels and fpn_strides must have the same length")
        if len(self.level_scale_ranges) != len(self.fpn_levels):
            raise ValueError("level_scale_ranges must have one range per FPN level")
        if self.bank_min_per_level < 0:
            raise ValueError("bank_min_per_level must be non-negative")
        minimum_budget = self.bank_min_per_level * len(self.fpn_levels)
        if self.bank_topk_train < minimum_budget or self.bank_topk_eval < minimum_budget:
            raise ValueError(
                "bank budgets must fit the requested per-level minimum: "
                f"need at least {minimum_budget}"
            )
        if self.evidence_per_slot <= 0:
            raise ValueError("evidence_per_slot must be positive")
        if self.slot_rounds != 3:
            raise ValueError("EGO-Det V2 contract requires exactly three slot rounds")
        if self.seed_selection_mode not in {"diversity", "top_score"}:
            raise ValueError(
                "seed_selection_mode must be 'diversity' or 'top_score'"
            )
        if self.readout_mode not in {
            "boundary",
            "whole_box",
            "direct_regression",
        }:
            raise ValueError(
                "readout_mode must be 'boundary', 'whole_box', or "
                "'direct_regression'"
            )
        if self.dense_assignment_mode not in {"stable_fcos", "legacy_dynamic"}:
            raise ValueError(
                "dense_assignment_mode must be 'stable_fcos' or 'legacy_dynamic'"
            )
        if not 0.0 <= float(self.proposal_identity_iou_threshold) <= 1.0:
            raise ValueError(
                "proposal_identity_iou_threshold must be in [0, 1]"
            )
        if self.efficiency_mode not in {"reference", "optimized"}:
            raise ValueError(
                "efficiency_mode must be 'reference' or 'optimized'"
            )
        if not 0 <= int(self.efficiency_ablation_level) <= 6:
            raise ValueError(
                "efficiency_ablation_level must be in [0, 6]"
            )
        if int(self.oracle_monitor_every_optimizer_steps) < -1:
            raise ValueError(
                "oracle_monitor_every_optimizer_steps must be -1, 0, or positive"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
