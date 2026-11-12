"""Controlled component ablations for EGO-Det V2."""

from .registry import (
    ABLATION_TABLES,
    ABLATIONS,
    AblationSpec,
    AblationTableSpec,
    TableRow,
    build_variant_config,
    get_ablation,
)

__all__ = [
    "ABLATION_TABLES",
    "ABLATIONS",
    "AblationSpec",
    "AblationTableSpec",
    "TableRow",
    "build_variant_config",
    "get_ablation",
]
