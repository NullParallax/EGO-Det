from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .registry import ABLATION_TABLES, ABLATIONS


AP_FIELDS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"expected JSON object at {path}:{line_number}"
                )
            records.append(record)
    return records


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _select_evaluation(
    records: Sequence[Mapping[str, object]],
    *,
    expected_epoch: int,
    allow_incomplete: bool,
) -> tuple[dict | None, str]:
    full = [
        dict(record)
        for record in records
        if record.get("kind") != "quick500" and _finite(record.get("AP"))
    ]
    exact = [
        record
        for record in full
        if int(record.get("epoch", -1)) == int(expected_epoch)
    ]
    if exact:
        exact.sort(
            key=lambda record: (
                int(record.get("global_optimizer_step", -1)),
                int(record.get("epoch", -1)),
            )
        )
        return exact[-1], "complete"
    if not full or not allow_incomplete:
        return None, "missing_final_evaluation"
    full.sort(
        key=lambda record: (
            int(record.get("epoch", -1)),
            int(record.get("global_optimizer_step", -1)),
        )
    )
    return full[-1], "incomplete_evaluation"


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


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    partial.replace(path)
    return path


def collect_results(
    run_root: Path | str,
    output_dir: Path | str,
    *,
    allow_incomplete: bool = False,
    allow_control_mismatch: bool = False,
    strict: bool = False,
) -> dict:
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    collected: dict[str, dict] = {}
    control_contracts: dict[str, list[str]] = {}
    data_contracts: dict[str, list[str]] = {}

    for key, spec in ABLATIONS.items():
        run_dir = run_root / key
        manifest_path = run_dir / "ablation_manifest.json"
        history_path = run_dir / "eval_history.jsonl"
        item: dict[str, object] = {
            "variant": key,
            "label": spec.label,
            "run_dir": str(run_dir),
            "status": "missing_run",
            "evaluation": None,
        }
        if not manifest_path.exists():
            collected[key] = item
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("variant") != key:
            raise RuntimeError(
                f"manifest variant mismatch at {manifest_path}: "
                f"{manifest.get('variant')!r} != {key!r}"
            )
        item["manifest"] = manifest
        control = str(manifest.get("control_fingerprint", ""))
        control_contracts.setdefault(control, []).append(key)
        data_encoded = json.dumps(
            manifest.get("data", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        data_contracts.setdefault(data_encoded, []).append(key)
        if not history_path.exists():
            item["status"] = "missing_eval_history"
            collected[key] = item
            continue
        expected_epoch = int(
            manifest.get("resolved_config", {}).get("epochs", 12)
        )
        evaluation, status = _select_evaluation(
            _read_jsonl(history_path),
            expected_epoch=expected_epoch,
            allow_incomplete=allow_incomplete,
        )
        item["status"] = status
        item["evaluation"] = evaluation
        item["expected_epoch"] = expected_epoch
        collected[key] = item

    nonempty_controls = {
        key: variants for key, variants in control_contracts.items() if key
    }
    contract_mismatch = len(nonempty_controls) > 1 or len(data_contracts) > 1
    if contract_mismatch and not allow_control_mismatch:
        raise RuntimeError(
            "ablation runs do not share one controlled configuration/data "
            "contract: "
            + json.dumps(
                {
                    "control_fingerprints": nonempty_controls,
                    "data_contract_count": len(data_contracts),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    full_record = collected["full"].get("evaluation")
    all_rows = []
    for key, spec in ABLATIONS.items():
        item = collected[key]
        record = item.get("evaluation")
        row: dict[str, object] = {
            "variant": key,
            "label": spec.label,
            "status": item["status"],
            "epoch": record.get("epoch", "") if record else "",
            "global_optimizer_step": (
                record.get("global_optimizer_step", "") if record else ""
            ),
            "run_dir": item["run_dir"],
        }
        for metric in AP_FIELDS + (
            "matched_mean_iou",
            "duplicate_rate_percent",
        ):
            row[metric] = _metric(record, metric)
        row["delta_AP"] = (
            _metric(record, "AP") - _metric(full_record, "AP")
            if _metric(record, "AP") is not None
            and _metric(full_record, "AP") is not None
            else None
        )
        row["delta_AP75"] = (
            _metric(record, "AP75") - _metric(full_record, "AP75")
            if _metric(record, "AP75") is not None
            and _metric(full_record, "AP75") is not None
            else None
        )
        row["delta_matched_mean_iou"] = (
            _metric(record, "matched_mean_iou")
            - _metric(full_record, "matched_mean_iou")
            if _metric(record, "matched_mean_iou") is not None
            and _metric(full_record, "matched_mean_iou") is not None
            else None
        )
        row["delta_duplicate_rate_pp"] = (
            _metric(record, "duplicate_rate_percent")
            - _metric(full_record, "duplicate_rate_percent")
            if _metric(record, "duplicate_rate_percent") is not None
            and _metric(full_record, "duplicate_rate_percent") is not None
            else None
        )
        all_rows.append(row)

    all_fields = (
        "variant",
        "label",
        "status",
        "epoch",
        "global_optimizer_step",
        *AP_FIELDS,
        "matched_mean_iou",
        "duplicate_rate_percent",
        "delta_AP",
        "delta_AP75",
        "delta_matched_mean_iou",
        "delta_duplicate_rate_pp",
        "run_dir",
    )
    artifacts = [
        str(_write_csv(output_dir / "all_variants.csv", all_fields, all_rows))
    ]
    row_lookup = {row["variant"]: row for row in all_rows}
    table_payloads: dict[str, list[dict]] = {}
    for table_key, table in ABLATION_TABLES.items():
        component_names = []
        for table_row in table.rows:
            for name, _ in table_row.components:
                if name not in component_names:
                    component_names.append(name)
        rows = []
        for table_row in table.rows:
            source = row_lookup[table_row.variant]
            row = {
                "variant": table_row.variant,
                "Variant": table_row.label,
                **table_row.component_dict(),
                **{metric: source.get(metric) for metric in table.metrics},
                "status": source["status"],
                "epoch": source["epoch"],
            }
            rows.append(row)
        fields = (
            "variant",
            "Variant",
            *component_names,
            *table.metrics,
            "status",
            "epoch",
        )
        artifacts.append(
            str(_write_csv(output_dir / table.filename, fields, rows))
        )
        table_payloads[table_key] = rows

    incomplete = [
        key
        for key, item in collected.items()
        if item["status"] != "complete"
    ]
    result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "control_contracts": nonempty_controls,
        "data_contract_count": len(data_contracts),
        "control_mismatch_allowed": bool(allow_control_mismatch),
        "allow_incomplete": bool(allow_incomplete),
        "incomplete_variants": incomplete,
        "variants": collected,
        "tables": table_payloads,
        "artifacts": artifacts,
    }
    result_path = _write_json(output_dir / "ablation_results.json", result)
    result["artifacts"].append(str(result_path))
    _write_json(result_path, result)
    if strict and incomplete:
        raise RuntimeError(
            "ablation summary is incomplete: " + ", ".join(incomplete)
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate controlled EGO-Det ablation runs into JSON/CSV; no "
            "LaTeX files are generated"
        )
    )
    parser.add_argument(
        "--run-root",
        default="outputs/ego_det_v2/ablations",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-control-mismatch", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or str(Path(args.run_root) / "summary")
    result = collect_results(
        args.run_root,
        output_dir,
        allow_incomplete=args.allow_incomplete,
        allow_control_mismatch=args.allow_control_mismatch,
        strict=args.strict,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
