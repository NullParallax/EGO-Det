from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import torch
from torch import nn

from ego_det_v2.ablations.registry import (
    build_variant_config,
    config_fingerprint,
)
from ego_det_v2.configs.config import EgoDetV2Config
from ego_det_v2.inference_interventions.evaluate import (
    STUDY_TYPE,
    FrozenGeometryReadout,
)
from ego_det_v2.inference_interventions.registry import (
    INTERVENTIONS,
    TABLES,
)
from ego_det_v2.inference_interventions.summarize import collect_results
from ego_det_v2.models.ego_det_v2 import EgoDetV2


def _compact_base() -> EgoDetV2Config:
    return EgoDetV2Config(
        d_model=32,
        tower_convs=1,
        sem_convs=1,
        num_slots=8,
        evidence_per_slot=4,
        bank_min_per_level=1,
        bank_pre_topk_per_level=8,
        bank_topk_train=16,
        bank_topk_eval=16,
        hdn_max_pairs=2,
        roi_size=3,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        amp=False,
        overlock_gradient_checkpointing=False,
    )


class _ChangingReadout(nn.Module):
    def forward(self, content, semantic, boxes, bank):
        del semantic, bank
        return {
            "b": boxes + 0.25,
            "c": content + 1.0,
            "marker": boxes.new_tensor(7.0),
        }


def test_registry_and_state_compatibility() -> None:
    assert len(INTERVENTIONS) == 9
    assert "direct_slot_regression" not in INTERVENTIONS
    assert "frozen_seed_geometry" in INTERVENTIONS
    assert set(TABLES) == {
        "overall_component_intervention",
        "formation_intervention",
        "readout_intervention",
    }
    base = _compact_base()
    full = EgoDetV2(base, fake_backbone=True)
    state = full.state_dict()
    for spec in INTERVENTIONS.values():
        config = build_variant_config(spec.config_variant, base=base)
        candidate = EgoDetV2(config, fake_backbone=True)
        candidate.load_state_dict(state, strict=True)


def test_frozen_geometry_wrapper() -> None:
    wrapper = FrozenGeometryReadout(_ChangingReadout())
    content = torch.randn(2, 3, 8)
    semantic = torch.randn(2, 3, 8)
    boxes = torch.rand(2, 3, 4)
    result = wrapper(content, semantic, boxes, {})
    torch.testing.assert_close(result["b"], boxes)
    torch.testing.assert_close(result["c"], content + 1.0)
    assert float(result["marker"]) == 7.0


def _write_reference(root: Path) -> Path:
    config = _compact_base()
    history = root / "full_eval_history.jsonl"
    full = {
        "epoch": 12,
        "global_optimizer_step": 100,
        "AP": 0.453,
        "AP50": 0.633,
        "AP75": 0.495,
        "AP_small": 0.264,
        "AP_medium": 0.483,
        "AP_large": 0.625,
        "matched_mean_iou": 0.771,
        "duplicate_rate": 0.0088,
    }
    history.write_text(json.dumps(full) + "\n", encoding="utf-8")
    reference = root / "full_manifest.json"
    manifest = {
        "variant": "full",
        "status": "imported_complete",
        "source_run": str(root / "source"),
        "source_checkpoint": str(root / "epoch_12.pth"),
        "source_eval_history": str(history),
        "config_fingerprint": config_fingerprint(
            config,
            exclude=("output_dir",),
        ),
        "resolved_config": config.to_dict(),
        "final_metrics": full,
    }
    reference.write_text(json.dumps(manifest), encoding="utf-8")
    return reference


def _write_intervention_runs(root: Path, reference: Path) -> None:
    for index, (key, spec) in enumerate(INTERVENTIONS.items()):
        run = root / key
        run.mkdir(parents=True)
        manifest = {
            "study_type": STUDY_TYPE,
            "variant": key,
            "status": "complete",
            "training_performed": False,
            "evaluation_state": "raw_model",
            "reference_manifest": str(reference.resolve()),
            "subset_size": 0,
            "runtime_transform": spec.runtime_transform,
        }
        (run / "intervention_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        metrics = {
            "AP": 0.30 + index * 0.005,
            "AP50": 0.50 + index * 0.005,
            "AP75": 0.32 + index * 0.005,
            "AP_small": 0.15,
            "AP_medium": 0.33,
            "AP_large": 0.45,
            "matched_mean_iou": 0.70 + index * 0.001,
            "duplicate_rate": 0.02 + index * 0.0001,
        }
        payload = {
            "variant": key,
            "checkpoint_epoch": 12,
            "elapsed_seconds": 10.0 + index,
            "metrics": metrics,
        }
        (run / "metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def test_summary_contract(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = _write_reference(tmp_path)
    runs = tmp_path / "runs"
    _write_intervention_runs(runs, reference)
    output = tmp_path / "summary"
    result = collect_results(
        runs,
        output,
        reference,
        strict=True,
    )
    assert not result["incomplete_variants"]
    assert result["direct_slot_regression_available"] is False
    assert result["direct_slot_regression_replacement"] == (
        "frozen_seed_geometry"
    )
    assert len(result["tables"]["overall_component_intervention"]) == 4
    assert len(result["tables"]["formation_intervention"]) == 4
    assert len(result["tables"]["readout_intervention"]) == 5
    expected = {
        "intervention_results.json",
        "all_interventions.csv",
        "overall_component_intervention.csv",
        "formation_intervention.csv",
        "readout_intervention.csv",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert not list(output.glob("*.tex"))
    with (output / "all_interventions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["variant"] == "full"
    assert math.isclose(float(rows[0]["AP"]), 45.3)
    frozen = next(
        row for row in rows if row["variant"] == "frozen_seed_geometry"
    )
    assert frozen["training_performed"] == "False"


def main() -> None:
    test_registry_and_state_compatibility()
    test_frozen_geometry_wrapper()
    with tempfile.TemporaryDirectory() as temporary:
        test_summary_contract(Path(temporary))
    print("ego_det_v2 inference intervention contracts: PASS")


if __name__ == "__main__":
    main()
