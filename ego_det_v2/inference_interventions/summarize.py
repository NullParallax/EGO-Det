from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ..ablations.summarize import _read_jsonl, _select_evaluation
from ..ablations.train import _load_reference_manifest, _write_json
from .evaluate import STUDY_TYPE
from .registry import INTERVENTIONS, TABLES


AP_FIELDS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric(record: Mapping[str, object] | None, name: str) -> float | None:
    if record is None:
        return None
    source = "duplicate_rate" if name == "duplicate_rate_percent" else name
    value = record.get(source)
    if not _finite(value):
        return None
    numeric = float(value)
    if name in AP_FIELDS or name == "duplicate_rate_percent":
        numeric *= 100.0
    return numeric


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    partial.replace(path)
    return path


def _control_record(
    manifest: Mapping[str, object],
    *,
    expected_epoch: int,
) -> dict[str, object]:
    history_value = manifest.get("source_eval_history")
    if history_value:
        history_path = Path(str(history_value)).expanduser().resolve()
        if history_path.exists():
            evaluation, status = _select_evaluation(
                _read_jsonl(history_path),
                expected_epoch=expected_epoch,
                allow_incomplete=False,
            )
            if evaluation is not None and status == "complete":
                return evaluation
    final_metrics = manifest.get("final_metrics")
    if isinstance(final_metrics, Mapping) and _finite(final_metrics.get("AP")):
        return dict(final_metrics)
    raise RuntimeError(
        "registered full manifest has neither a readable final eval history "
        "nor finite final_metrics"
    )


def collect_results(
    run_root: Path | str,
    output_dir: Path | str,
    reference_manifest: Path | str,
    *,
    allow_incomplete: bool = False,
    allow_subset: bool = False,
    strict: bool = False,
) -> dict[str, object]:
    run_root = Path(run_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    reference_path = Path(reference_manifest).expanduser().resolve()
    reference_config, reference = _load_reference_manifest(reference_path)
    full_record = _control_record(
        reference,
        expected_epoch=int(reference_config.epochs),
    )

    collected: dict[str, dict[str, object]] = {}
    for key, spec in INTERVENTIONS.items():
        run_dir = run_root / key
        manifest_path = run_dir / "intervention_manifest.json"
        metrics_path = run_dir / "metrics.json"
        item: dict[str, object] = {
            "variant": key,
            "label": spec.label,
            "run_dir": str(run_dir),
            "status": "missing_run",
            "metrics": None,
        }
        if not manifest_path.exists():
            collected[key] = item
            continue
        manifest = _read_json(manifest_path)
        item["manifest"] = manifest
        problems = []
        if manifest.get("study_type") != STUDY_TYPE:
            problems.append("wrong_study_type")
        if manifest.get("variant") != key:
            problems.append("variant_mismatch")
        if bool(manifest.get("training_performed", True)):
            problems.append("training_performed_must_be_false")
        if manifest.get("evaluation_state") != "raw_model":
            problems.append("not_raw_model")
        manifest_reference = Path(
            str(manifest.get("reference_manifest", ""))
        ).expanduser().resolve()
        if manifest_reference != reference_path:
            problems.append("reference_manifest_mismatch")
        subset_size = int(manifest.get("subset_size", -1))
        if subset_size != 0 and not allow_subset:
            problems.append("not_full_coco_val2017")
        if manifest.get("status") != "complete":
            problems.append(f"manifest_status_{manifest.get('status')}")
        if problems:
            item["status"] = ";".join(problems)
            collected[key] = item
            continue
        if not metrics_path.exists():
            item["status"] = "missing_metrics"
            collected[key] = item
            continue
        payload = _read_json(metrics_path)
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping) or not _finite(metrics.get("AP")):
            item["status"] = "invalid_metrics"
            collected[key] = item
            continue
        item["status"] = (
            "complete" if subset_size == 0 else "diagnostic_subset"
        )
        item["metrics"] = dict(metrics)
        item["checkpoint_epoch"] = payload.get("checkpoint_epoch")
        item["elapsed_seconds"] = payload.get("elapsed_seconds")
        collected[key] = item

    full_row: dict[str, object] = {
        "variant": "full",
        "label": "EGO-Det (control)",
        "status": "complete",
        "checkpoint_epoch": int(reference_config.epochs),
        "runtime_transform": "none",
        "config_variant": "full",
        "elapsed_seconds": None,
        "run_dir": str(reference.get("source_run", "")),
    }
    for metric in AP_FIELDS + (
        "matched_mean_iou",
        "duplicate_rate_percent",
    ):
        full_row[metric] = _metric(full_record, metric)
    full_row.update(
        {
            "delta_AP": 0.0,
            "delta_AP50": 0.0,
            "delta_AP75": 0.0,
            "delta_matched_mean_iou": 0.0,
            "delta_duplicate_rate_pp": 0.0,
        }
    )

    rows = [full_row]
    for key, spec in INTERVENTIONS.items():
        item = collected[key]
        record = item.get("metrics")
        manifest = item.get("manifest")
        row: dict[str, object] = {
            "variant": key,
            "label": spec.label,
            "status": item["status"],
            "checkpoint_epoch": item.get("checkpoint_epoch", ""),
            "runtime_transform": spec.runtime_transform,
            "config_variant": spec.config_variant,
            "elapsed_seconds": item.get("elapsed_seconds"),
            "run_dir": item["run_dir"],
            "training_performed": (
                manifest.get("training_performed")
                if isinstance(manifest, Mapping)
                else None
            ),
        }
        for metric in AP_FIELDS + (
            "matched_mean_iou",
            "duplicate_rate_percent",
        ):
            row[metric] = _metric(
                record if isinstance(record, Mapping) else None,
                metric,
            )
        for metric, delta_name in (
            ("AP", "delta_AP"),
            ("AP50", "delta_AP50"),
            ("AP75", "delta_AP75"),
            ("matched_mean_iou", "delta_matched_mean_iou"),
            ("duplicate_rate_percent", "delta_duplicate_rate_pp"),
        ):
            value = row.get(metric)
            control = full_row.get(metric)
            row[delta_name] = (
                float(value) - float(control)
                if value is not None and control is not None
                else None
            )
        rows.append(row)

    all_fields = (
        "variant",
        "label",
        "status",
        "checkpoint_epoch",
        "training_performed",
        "config_variant",
        "runtime_transform",
        *AP_FIELDS,
        "matched_mean_iou",
        "duplicate_rate_percent",
        "delta_AP",
        "delta_AP50",
        "delta_AP75",
        "delta_matched_mean_iou",
        "delta_duplicate_rate_pp",
        "elapsed_seconds",
        "run_dir",
    )
    artifacts = [
        str(_write_csv(output_dir / "all_interventions.csv", all_fields, rows))
    ]
    row_lookup = {str(row["variant"]): row for row in rows}
    table_payloads: dict[str, list[dict[str, object]]] = {}
    for table_key, table in TABLES.items():
        component_names = []
        for table_row in table.rows:
            for name, _ in table_row.components:
                if name not in component_names:
                    component_names.append(name)
        table_rows = []
        for table_row in table.rows:
            source = row_lookup[table_row.variant]
            table_rows.append(
                {
                    "variant": table_row.variant,
                    "Variant": table_row.label,
                    **table_row.component_dict(),
                    **{metric: source.get(metric) for metric in table.metrics},
                    "delta_AP": source.get("delta_AP"),
                    "status": source.get("status"),
                }
            )
        fields = (
            "variant",
            "Variant",
            *component_names,
            *table.metrics,
            "delta_AP",
            "status",
        )
        artifacts.append(
            str(_write_csv(output_dir / table.filename, fields, table_rows))
        )
        table_payloads[table_key] = table_rows

    incomplete = [
        key
        for key, item in collected.items()
        if item["status"] not in (
            "complete",
            "diagnostic_subset" if allow_subset else "complete",
        )
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "study_type": STUDY_TYPE,
        "claim_boundary": (
            "All non-control rows are inference-time interventions of one "
            "epoch-12 raw checkpoint; no variant was independently trained."
        ),
        "direct_slot_regression_available": False,
        "direct_slot_regression_replacement": "frozen_seed_geometry",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "reference_manifest": str(reference_path),
        "incomplete_variants": incomplete,
        "variants": collected,
        "tables": table_payloads,
        "artifacts": artifacts,
    }
    result_path = _write_json(output_dir / "intervention_results.json", result)
    artifacts.append(str(result_path))
    _write_json(result_path, result)
    if strict and incomplete:
        raise RuntimeError(
            "inference intervention summary is incomplete: "
            + ", ".join(incomplete)
        )
    if not allow_incomplete and incomplete:
        raise RuntimeError(
            "missing or invalid interventions: " + ", ".join(incomplete)
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate inference-only intervention metrics; results are kept "
            "separate from independently trained ablations"
        )
    )
    parser.add_argument(
        "--run-root",
        default="outputs/ego_det_v2/inference_interventions_raw_epoch12",
    )
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or str(Path(args.run_root) / "summary")
    result = collect_results(
        args.run_root,
        output_dir,
        args.reference_manifest,
        allow_incomplete=args.allow_incomplete,
        allow_subset=args.allow_subset,
        strict=args.strict,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
