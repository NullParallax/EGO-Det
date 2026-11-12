# Naming and directory migration plan (do not rename numerical code yet)

## External project identity

- GitHub repository: `EGO-Det` (stable paper-facing name).
- README title: `EGO-Det` (use the precise final paper title only after checking the manuscript).
- Conda environment name: `egodet` (local developer convenience; not a source package).
- Internal Python module currently: **`ego_det_v2`** (preserve until the official checkpoint and evaluation paths are verified).

## Inventory of names and recommended action

| Existing name | Meaning in the recovered paper implementation | Action now | Later when verified |
|---|---|---|---|
| `ego_det_v2/` | Complete research model, criterion, data, train/eval and tests | **Keep** | Optionally expose an `egodet/` facade, preserving the old import path |
| `ego_joint_object_slot/` | Compatibility `configs/paths.py` and `utils/import_utils.py` | **Keep** | Fold into an explicit `egodet.utils` module; leave compatibility shims |
| `stage1/` | OverLoCK backend adapter and upstream implementation | **Keep** pending license | Move to `egodet/backbones/` with copyright/license attribution and an import shim |
| `ego_det_v2/launch/launch_*.py` | Original training/evaluation entrypoints | **Keep** | Provide `tools/train.py`, `tools/eval.py`, `tools/demo.py` wrappers that preserve original APIs |
| `ego_det_v2/ablations/`, `inference_interventions/` | Formal experiments, inference-only intervention tools | **Keep** | Expose as optional experiments with clearly labelled protocols |
| `configs/paper_epoch12_recorded.json` | Historical experiment settings, not executable YAML | **Keep read-only** | Convert into validated loadable configuration only after equivalent outputs are confirmed |
| `ego_det_v2/tests/test_paper_visualization.py` | References `ego_det_v2.paper_visualization` which is not included in staging source | **Flag as unresolved** | Recover referenced package or separate/remove orphan test in a dedicated, justified commit |
| `docs/staging_file_manifest.json` | Hash list captured during initial staging | **Keep as historical snapshot** | Do not treat hash list as current integrity manifest after modifications |

## Renaming contract

1. First confirm `epoch_12.pth` strict `state_dict` load with current code.
2. Run COCO subset followed by full val2017 evaluation.
3. Add automated smoke tests for import, checkpoint key compatibility and inference outputs.
4. Move/alias Python modules using one small commit per subsystem; maintain model classes, key names, saved-state compatibility and detector outputs.
5. Change wrappers and package names only after tests remain stable; otherwise roll back the single commit.

The repository has a single paper-facing name but can retain technical legacy import names. A mass rename now creates no scientific value and increases reproduction risk.
