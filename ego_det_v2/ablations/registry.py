from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ..configs.config import EgoDetV2Config


METHOD_FIELDS = frozenset(
    {
        "semantic_global_attention",
        "semantic_regional_retrieval",
        "seed_selection_mode",
        "slot_competition_enabled",
        "readout_mode",
        "readout_dustbin_gating",
        "readout_bounded_residual",
        "w_ownership",
        "w_dustbin",
        "w_gain",
    }
)

# These fields only control diagnostics, console/file I/O, or checkpoint
# cadence.  They do not alter model parameters, optimizer updates, data
# augmentation, or the formal full-val prediction path, so formal ablation
# runs may disable them without breaking the controlled-training contract.
INSTRUMENTATION_FIELDS = frozenset(
    {
        "log_every_steps",
        "quick_eval_every_steps",
        "full_eval_last_epoch_only",
        "oracle_monitor_every_optimizer_steps",
        "ckpt_every_steps",
    }
)

REQUIRED_CORE_FILES = (
    "ego_det_v2/configs/config.py",
    "ego_det_v2/models/slot_seeding.py",
    "ego_det_v2/models/slot_update.py",
    "ego_det_v2/models/heads.py",
    "ego_det_v2/models/evidence_readout.py",
    "ego_det_v2/models/ego_det_v2.py",
    "ego_det_v2/losses/criterion.py",
    "ego_det_v2/engine/train_loop.py",
)


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    description: str
    overrides: tuple[tuple[str, object], ...] = ()

    def override_dict(self) -> dict[str, object]:
        return dict(self.overrides)


@dataclass(frozen=True)
class TableRow:
    variant: str
    label: str
    components: tuple[tuple[str, bool], ...]

    def component_dict(self) -> dict[str, bool]:
        return dict(self.components)


@dataclass(frozen=True)
class AblationTableSpec:
    key: str
    filename: str
    metrics: tuple[str, ...]
    rows: tuple[TableRow, ...]


ABLATIONS: dict[str, AblationSpec] = {
    "full": AblationSpec(
        key="full",
        label="EGO-Det",
        description="Formal EGO-Det configuration used as the shared control.",
    ),
    "no_semantic_completion": AblationSpec(
        key="no_semantic_completion",
        label="w/o semantic completion",
        description=(
            "Disable both global token attention and box-conditioned regional "
            "semantic retrieval while retaining competition and readout."
        ),
        overrides=(
            ("semantic_global_attention", False),
            ("semantic_regional_retrieval", False),
        ),
    ),
    "no_slot_competition": AblationSpec(
        key="no_slot_competition",
        label="w/o slot competition",
        description=(
            "Bypass slot-to-slot multi-head competition in every formation "
            "round; all other losses and modules remain unchanged."
        ),
        overrides=(("slot_competition_enabled", False),),
    ),
    "direct_slot_regression": AblationSpec(
        key="direct_slot_regression",
        label="Direct slot regression",
        description=(
            "Replace proposal ownership and geometric readout with a "
            "lightweight iterative box head driven by the slot states."
        ),
        overrides=(
            ("readout_mode", "direct_regression"),
            ("readout_dustbin_gating", False),
            ("readout_bounded_residual", False),
            ("w_ownership", 0.0),
            ("w_dustbin", 0.0),
            ("w_gain", 0.0),
        ),
    ),
    "top_score_seeding": AblationSpec(
        key="top_score_seeding",
        label="Top-score seeding",
        description=(
            "Select valid proposal-bank entries strictly by score without "
            "IoU-based diversity suppression."
        ),
        overrides=(("seed_selection_mode", "top_score"),),
    ),
    "global_semantics_only": AblationSpec(
        key="global_semantics_only",
        label="Global semantics only",
        description=(
            "Retain P5/P6/global-token attention and disable all regional "
            "RoI semantic retrieval, including slot initialization."
        ),
        overrides=(("semantic_regional_retrieval", False),),
    ),
    "regional_semantics_only": AblationSpec(
        key="regional_semantics_only",
        label="Regional semantics only",
        description=(
            "Retain box-conditioned regional retrieval and bypass global "
            "semantic cross-attention in every slot round."
        ),
        overrides=(("semantic_global_attention", False),),
    ),
    "whole_box_aggregation": AblationSpec(
        key="whole_box_aggregation",
        label="Whole-box aggregation",
        description=(
            "Use one ownership-conditioned proposal distribution to average "
            "all four box coordinates instead of boundary-specific weights."
        ),
        overrides=(("readout_mode", "whole_box"),),
    ),
    "no_dustbin_gating": AblationSpec(
        key="no_dustbin_gating",
        label="w/o dustbin gating",
        description=(
            "Keep the learned ownership/dustbin supervision fixed, but remove "
            "dustbin mass from both evidence and geometry acceptance gates."
        ),
        overrides=(("readout_dustbin_gating", False),),
    ),
    "no_bounded_residual": AblationSpec(
        key="no_bounded_residual",
        label="w/o bounded residual",
        description=(
            "Return the ownership-conditioned aggregated box directly and "
            "bypass the final bounded residual MLP."
        ),
        overrides=(("readout_bounded_residual", False),),
    ),
}


ABLATION_TABLES: dict[str, AblationTableSpec] = {
    "overall_component_analysis": AblationTableSpec(
        key="overall_component_analysis",
        filename="overall_component_analysis.csv",
        metrics=("AP", "AP50", "AP75"),
        rows=(
            TableRow(
                "no_semantic_completion",
                "w/o semantic completion",
                (
                    ("semantic_completion", False),
                    ("slot_competition", True),
                    ("geometric_readout", True),
                ),
            ),
            TableRow(
                "no_slot_competition",
                "w/o slot competition",
                (
                    ("semantic_completion", True),
                    ("slot_competition", False),
                    ("geometric_readout", True),
                ),
            ),
            TableRow(
                "direct_slot_regression",
                "Direct slot regression",
                (
                    ("semantic_completion", True),
                    ("slot_competition", True),
                    ("geometric_readout", False),
                ),
            ),
            TableRow(
                "full",
                "EGO-Det",
                (
                    ("semantic_completion", True),
                    ("slot_competition", True),
                    ("geometric_readout", True),
                ),
            ),
        ),
    ),
    "object_slot_formation": AblationTableSpec(
        key="object_slot_formation",
        filename="object_slot_formation.csv",
        metrics=("AP", "AP75", "matched_mean_iou", "duplicate_rate_percent"),
        rows=(
            TableRow(
                "top_score_seeding",
                "Top-score seeding",
                (
                    ("diversity_aware_seeding", False),
                    ("global_semantics", True),
                    ("regional_semantics", True),
                ),
            ),
            TableRow(
                "global_semantics_only",
                "Global semantics only",
                (
                    ("diversity_aware_seeding", True),
                    ("global_semantics", True),
                    ("regional_semantics", False),
                ),
            ),
            TableRow(
                "regional_semantics_only",
                "Regional semantics only",
                (
                    ("diversity_aware_seeding", True),
                    ("global_semantics", False),
                    ("regional_semantics", True),
                ),
            ),
            TableRow(
                "full",
                "Full formation",
                (
                    ("diversity_aware_seeding", True),
                    ("global_semantics", True),
                    ("regional_semantics", True),
                ),
            ),
        ),
    ),
    "slot_conditioned_geometric_readout": AblationTableSpec(
        key="slot_conditioned_geometric_readout",
        filename="slot_conditioned_geometric_readout.csv",
        metrics=("AP", "AP75", "matched_mean_iou"),
        rows=(
            TableRow(
                "direct_slot_regression",
                "Direct slot regression",
                (
                    ("ownership", False),
                    ("boundary_aggregation", False),
                    ("dustbin", False),
                    ("residual", False),
                ),
            ),
            TableRow(
                "whole_box_aggregation",
                "Whole-box aggregation",
                (
                    ("ownership", True),
                    ("boundary_aggregation", False),
                    ("dustbin", True),
                    ("residual", True),
                ),
            ),
            TableRow(
                "no_dustbin_gating",
                "w/o dustbin gating",
                (
                    ("ownership", True),
                    ("boundary_aggregation", True),
                    ("dustbin", False),
                    ("residual", True),
                ),
            ),
            TableRow(
                "no_bounded_residual",
                "w/o bounded residual",
                (
                    ("ownership", True),
                    ("boundary_aggregation", True),
                    ("dustbin", True),
                    ("residual", False),
                ),
            ),
            TableRow(
                "full",
                "Full readout",
                (
                    ("ownership", True),
                    ("boundary_aggregation", True),
                    ("dustbin", True),
                    ("residual", True),
                ),
            ),
        ),
    ),
}


def get_ablation(key: str) -> AblationSpec:
    try:
        return ABLATIONS[str(key)]
    except KeyError as exc:
        raise KeyError(
            f"unknown ablation {key!r}; choose from {', '.join(ABLATIONS)}"
        ) from exc


def build_variant_config(
    key: str,
    *,
    base: EgoDetV2Config | None = None,
    extra_updates: Mapping[str, object] | None = None,
) -> EgoDetV2Config:
    spec = get_ablation(key)
    updates = spec.override_dict()
    if extra_updates:
        overlap = METHOD_FIELDS.intersection(extra_updates)
        if overlap:
            raise ValueError(
                "runtime overrides may not modify ablation method fields: "
                + ", ".join(sorted(overlap))
            )
        updates.update(extra_updates)
    allowed = EgoDetV2Config.__dataclass_fields__
    unknown = sorted(set(updates).difference(allowed))
    if unknown:
        raise KeyError(f"unknown EgoDetV2Config fields: {unknown}")
    return replace(base or EgoDetV2Config(), **updates)


def config_fingerprint(
    config: EgoDetV2Config,
    *,
    exclude: Sequence[str] = (),
) -> str:
    excluded = set(exclude)
    payload = {
        key: value
        for key, value in config.to_dict().items()
        if key not in excluded
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def control_fingerprint(config: EgoDetV2Config) -> str:
    return config_fingerprint(
        config,
        exclude=tuple(
            METHOD_FIELDS | INSTRUMENTATION_FIELDS | {"output_dir"}
        ),
    )


def validate_registry() -> None:
    fields = set(EgoDetV2Config.__dataclass_fields__)
    missing_method_fields = sorted(METHOD_FIELDS.difference(fields))
    if missing_method_fields:
        raise RuntimeError(
            "EGO-Det ablation package/core version mismatch: "
            "EgoDetV2Config is missing required method fields "
            f"{missing_method_fields}. The ablations package cannot be "
            "deployed by itself. Sync the complete core patch as well: "
            + ", ".join(REQUIRED_CORE_FILES)
            + ". Do not bypass this check because the requested variants "
            "would otherwise execute the full model path."
        )
    for key, spec in ABLATIONS.items():
        if key != spec.key:
            raise AssertionError(f"registry key mismatch: {key} != {spec.key}")
        unknown = set(spec.override_dict()).difference(fields)
        if unknown:
            raise AssertionError(f"{key} has unknown overrides: {unknown}")
        build_variant_config(key)
    referenced = {
        row.variant
        for table in ABLATION_TABLES.values()
        for row in table.rows
    }
    missing = referenced.difference(ABLATIONS)
    if missing:
        raise AssertionError(f"table rows reference missing variants: {missing}")


validate_registry()
