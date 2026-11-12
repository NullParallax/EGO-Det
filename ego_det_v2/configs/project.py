"""User-editable YAML runtime settings; model code and checkpoint format stay unchanged.

Precedence: explicit CLI option > YAML section > EgoDetV2Config defaults.
Only the training entrypoint applies the `model` section to model construction;
evaluation and demo recover structural configuration from the checkpoint.
"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import EgoDetV2Config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "egodet.yaml"


def _mapping(value: Any, label: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a YAML mapping")
    return value


def load_settings(path: str | Path | None = None) -> dict:
    filename = Path(path).expanduser().resolve() if path is not None else DEFAULT_CONFIG
    if not filename.is_file():
        raise FileNotFoundError(f"EGO-Det configuration not found: {filename}")
    with filename.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    data = _mapping(data, "EGO-Det configuration")
    valid_sections = {"paths", "model", "training", "evaluation", "demo"}
    extras = set(data) - valid_sections
    if extras:
        raise ValueError(f"Unrecognized top-level config sections: {sorted(extras)}")
    return {name: _mapping(data.get(name), name) for name in valid_sections}


def resolve_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root).expanduser().resolve() if project_root else REPOSITORY_ROOT


def resolve_path(value: str | Path | None, root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def path_from_settings(settings: Mapping, name: str, root: Path,
                       explicit: str | Path | None = None) -> Path | None:
    raw = explicit if explicit is not None else settings["paths"].get(name)
    return resolve_path(raw, root)


def _preserve_tuples(value: Any, template: Any) -> Any:
    if isinstance(template, tuple) and isinstance(value, (tuple, list)):
        exemplar = template[0] if template else None
        return tuple(_preserve_tuples(item, exemplar) for item in value)
    return value


def model_from_settings(settings: Mapping,
                        base: EgoDetV2Config | None = None) -> EgoDetV2Config:
    baseline = base or EgoDetV2Config()
    supplied = _mapping(settings.get("model"), "model")
    permitted = {field.name for field in fields(EgoDetV2Config)}
    unknown = set(supplied) - permitted
    if unknown:
        raise ValueError(f"Unrecognized EGO-Det model parameters: {sorted(unknown)}")
    converted = {key: _preserve_tuples(value, getattr(baseline, key))
                 for key, value in supplied.items()}
    return replace(baseline, **converted)


# Only inference-only fields may override the configuration stored in the checkpoint.
# Model architecture and training parameters MUST be restored from that checkpoint.
EVAL_SAFE_OVERRIDES = frozenset({
    "val_short_side", "val_max_size", "amp", "amp_dtype", "bank_topk_eval",
})


def evaluation_model_config(saved: EgoDetV2Config, settings: Mapping) -> EgoDetV2Config:
    supplied = _mapping(settings["evaluation"].get("model_overrides"),
                        "evaluation.model_overrides")
    illegal = set(supplied) - EVAL_SAFE_OVERRIDES
    if illegal:
        raise ValueError(f"Not safe to override checkpoint architecture for evaluation: {sorted(illegal)}")
    return replace(saved, **supplied) if supplied else saved
