# EGO-Det: Evidence-Guided Object Detection

> **REPRODUCIBILITY STAGING — NOT READY FOR PUBLIC RELEASE.** This is a private, preliminary source snapshot based on the EGO-Det V2 code tree and the historical experiment logs. It has not yet been validated against the published checkpoint on COCO val2017. Third-party redistribution permissions must be reviewed before publication.

EGO-Det is an object-centric detection framework using a class-agnostic geometry proposal bank, semantic object-slot formation, slot competition, and slot-conditioned geometry readout.

## Paper-run metrics (from historical records, not a new rerun)

| Split | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO val2017, epoch 12, **raw model** | 45.269 | 63.331 | 49.482 | 26.432 | 48.313 | 62.514 |

Recorded run ID: `ego_det_v2/train_mod10b_batch8_long`. Checkpoint: `epoch_12.pth` (562,663,950 bytes according to experiment manifest; SHA256 must still be independently verified).

## Repository structure

- `ego_det_v2/`: original end-to-end V2 models, losses, datasets, training and evaluation, tests, and ablation/intervention runners.
- `stage1/`: OverLoCK backbone adapter and borrowed upstream implementation. **Upstream license audit required.**
- `ego_joint_object_slot/configs/paths.py`, `utils/import_utils.py`: compatibility helpers that the V2 code imports. The hard-coded machine root was replaced with repository-relative defaults in this staging copy.
- `configs/paper_epoch12_recorded.json`: preserved training parameters from the experiment manifest; absolute output path is sanitized and is not the original config fingerprint.

## Install and evaluation

Paper environment: PyTorch 2.1.2, CUDA 11.8; OverLoCK-T initialized with ImageNet-1K pretrained weights. Other dependencies are listed in `requirements.txt` but not fully version-locked.

Prepare your model and data in the following locations, or use explicit CLI parameters:

```text
weights/overlock/overlock_t_in1k_224.pth
data/coco/images/{train2017,val2017}/...
data/coco/raw/annotations/{instances_train2017.json,instances_val2017.json}
checkpoints/epoch_12.pth
```

```bash
python -m ego_det_v2.launch.launch_eval_ego_det_v2 \
  --project-root . \
  --ckpt checkpoints/epoch_12.pth \
  --val-annotations data/coco/raw/annotations/instances_val2017.json \
  --image-root data/coco/images \
  --output-json outputs/paper_epoch12_eval.json
```

The CLI defaults to **raw model**; do not pass `--use-ema` when reproducing the 45.269 AP run.

A sample training invocation with historical parameter overrides is in `scripts/reproduce_paper_train.sh`. See `docs/VERIFICATION_AND_RELEASE_GATES.md` before running or publishing.

## Citation and license

The paper citation, BibTeX and license are intentionally not finalized in this staging copy; check submission status, author approval and third-party licenses first.

## Staging validation status

Static compile/import and ablation/inference-intervention contracts passed. CPU contracts are partially passed (37 checks), blocked by missing `pycocotools` in the assembly environment. Full checkpoint load and COCO AP reproduction remain pending.
