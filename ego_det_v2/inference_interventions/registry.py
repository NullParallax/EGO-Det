from __future__ import annotations

from dataclasses import dataclass

from ..ablations.registry import ABLATIONS


@dataclass(frozen=True)
class InterventionSpec:
    key: str
    label: str
    description: str
    config_variant: str
    runtime_transform: str = "none"

    def config_overrides(self) -> dict[str, object]:
        return ABLATIONS[self.config_variant].override_dict()


@dataclass(frozen=True)
class TableRow:
    variant: str
    label: str
    components: tuple[tuple[str, bool], ...]

    def component_dict(self) -> dict[str, bool]:
        return dict(self.components)


@dataclass(frozen=True)
class InterventionTable:
    key: str
    filename: str
    metrics: tuple[str, ...]
    rows: tuple[TableRow, ...]


INTERVENTIONS: dict[str, InterventionSpec] = {
    "no_semantic_completion": InterventionSpec(
        key="no_semantic_completion",
        label="w/o semantic completion",
        description=(
            "Bypass global token attention and regional semantic retrieval "
            "at inference while retaining the epoch-12 full-model weights."
        ),
        config_variant="no_semantic_completion",
    ),
    "no_slot_competition": InterventionSpec(
        key="no_slot_competition",
        label="w/o slot competition",
        description=(
            "Bypass all slot-to-slot competition rounds at inference while "
            "retaining the epoch-12 full-model weights."
        ),
        config_variant="no_slot_competition",
    ),
    "frozen_seed_geometry": InterventionSpec(
        key="frozen_seed_geometry",
        label="Frozen seed geometry",
        description=(
            "Run semantic slot updates and evidence retrieval, but force each "
            "round to retain its incoming box so final geometry remains the "
            "initial proposal seed. This is an inference-only proxy for no "
            "geometric refinement, not direct slot regression."
        ),
        config_variant="full",
        runtime_transform="freeze_readout_geometry",
    ),
    "top_score_seeding": InterventionSpec(
        key="top_score_seeding",
        label="Top-score seeding",
        description=(
            "Replace diversity-aware proposal selection with strict top-score "
            "selection at inference."
        ),
        config_variant="top_score_seeding",
    ),
    "global_semantics_only": InterventionSpec(
        key="global_semantics_only",
        label="Global semantics only",
        description=(
            "Retain global token attention and bypass regional semantic "
            "retrieval at inference."
        ),
        config_variant="global_semantics_only",
    ),
    "regional_semantics_only": InterventionSpec(
        key="regional_semantics_only",
        label="Regional semantics only",
        description=(
            "Retain regional semantic retrieval and bypass global token "
            "attention at inference."
        ),
        config_variant="regional_semantics_only",
    ),
    "whole_box_aggregation": InterventionSpec(
        key="whole_box_aggregation",
        label="Whole-box aggregation",
        description=(
            "Use one ownership-conditioned proposal distribution for all four "
            "coordinates at inference."
        ),
        config_variant="whole_box_aggregation",
    ),
    "no_dustbin_gating": InterventionSpec(
        key="no_dustbin_gating",
        label="w/o dustbin gating",
        description=(
            "Remove dustbin mass from evidence and geometry acceptance gates "
            "at inference."
        ),
        config_variant="no_dustbin_gating",
    ),
    "no_bounded_residual": InterventionSpec(
        key="no_bounded_residual",
        label="w/o bounded residual",
        description=(
            "Return the boundary-aggregated box without the final bounded "
            "residual correction at inference."
        ),
        config_variant="no_bounded_residual",
    ),
}


TABLES: dict[str, InterventionTable] = {
    "overall_component_intervention": InterventionTable(
        key="overall_component_intervention",
        filename="overall_component_intervention.csv",
        metrics=("AP", "AP50", "AP75"),
        rows=(
            TableRow(
                "no_semantic_completion",
                "w/o semantic completion",
                (
                    ("semantic_completion", False),
                    ("slot_competition", True),
                    ("geometric_refinement", True),
                ),
            ),
            TableRow(
                "no_slot_competition",
                "w/o slot competition",
                (
                    ("semantic_completion", True),
                    ("slot_competition", False),
                    ("geometric_refinement", True),
                ),
            ),
            TableRow(
                "frozen_seed_geometry",
                "Frozen seed geometry",
                (
                    ("semantic_completion", True),
                    ("slot_competition", True),
                    ("geometric_refinement", False),
                ),
            ),
            TableRow(
                "full",
                "EGO-Det (control)",
                (
                    ("semantic_completion", True),
                    ("slot_competition", True),
                    ("geometric_refinement", True),
                ),
            ),
        ),
    ),
    "formation_intervention": InterventionTable(
        key="formation_intervention",
        filename="formation_intervention.csv",
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
                "EGO-Det (control)",
                (
                    ("diversity_aware_seeding", True),
                    ("global_semantics", True),
                    ("regional_semantics", True),
                ),
            ),
        ),
    ),
    "readout_intervention": InterventionTable(
        key="readout_intervention",
        filename="readout_intervention.csv",
        metrics=("AP", "AP75", "matched_mean_iou"),
        rows=(
            TableRow(
                "frozen_seed_geometry",
                "Frozen seed geometry",
                (
                    ("ownership_evidence", True),
                    ("boundary_geometry", False),
                    ("dustbin_gating", True),
                    ("bounded_residual", False),
                ),
            ),
            TableRow(
                "whole_box_aggregation",
                "Whole-box aggregation",
                (
                    ("ownership_evidence", True),
                    ("boundary_geometry", False),
                    ("dustbin_gating", True),
                    ("bounded_residual", True),
                ),
            ),
            TableRow(
                "no_dustbin_gating",
                "w/o dustbin gating",
                (
                    ("ownership_evidence", True),
                    ("boundary_geometry", True),
                    ("dustbin_gating", False),
                    ("bounded_residual", True),
                ),
            ),
            TableRow(
                "no_bounded_residual",
                "w/o bounded residual",
                (
                    ("ownership_evidence", True),
                    ("boundary_geometry", True),
                    ("dustbin_gating", True),
                    ("bounded_residual", False),
                ),
            ),
            TableRow(
                "full",
                "EGO-Det (control)",
                (
                    ("ownership_evidence", True),
                    ("boundary_geometry", True),
                    ("dustbin_gating", True),
                    ("bounded_residual", True),
                ),
            ),
        ),
    ),
}


def get_intervention(key: str) -> InterventionSpec:
    try:
        return INTERVENTIONS[str(key)]
    except KeyError as exc:
        raise KeyError(
            f"unknown inference intervention {key!r}; choose from "
            + ", ".join(INTERVENTIONS)
        ) from exc


def validate_registry() -> None:
    if "direct_slot_regression" in INTERVENTIONS:
        raise AssertionError(
            "an untrained direct-regression head must never be exposed as an "
            "inference-only intervention"
        )
    for key, spec in INTERVENTIONS.items():
        if key != spec.key:
            raise AssertionError(f"registry key mismatch: {key} != {spec.key}")
        if spec.config_variant not in ABLATIONS:
            raise AssertionError(
                f"{key} references unknown config variant {spec.config_variant}"
            )
        if spec.config_variant == "direct_slot_regression":
            raise AssertionError(f"{key} would require untrained new parameters")
        if spec.runtime_transform not in {"none", "freeze_readout_geometry"}:
            raise AssertionError(
                f"{key} has unknown runtime transform {spec.runtime_transform}"
            )
    referenced = {
        row.variant
        for table in TABLES.values()
        for row in table.rows
        if row.variant != "full"
    }
    missing = referenced.difference(INTERVENTIONS)
    if missing:
        raise AssertionError(f"table rows reference missing interventions: {missing}")


validate_registry()
