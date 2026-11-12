"""Inference-only component interventions for the epoch-12 EGO-Det model."""

from .registry import (
    INTERVENTIONS,
    TABLES,
    InterventionSpec,
    InterventionTable,
    TableRow,
    get_intervention,
)

__all__ = [
    "INTERVENTIONS",
    "TABLES",
    "InterventionSpec",
    "InterventionTable",
    "TableRow",
    "get_intervention",
]
