# EGO-Det: Evidence-Guided Object Detection

EGO-Det is an object-centric detection framework combining class-agnostic geometry proposals, semantic object-slot formation, slot competition and slot-conditioned geometry readout.

> **Research source (private staging).** The model has successfully loaded the final non-EMA checkpoint and run single-image inference on an RTX 2080 Ti. Full COCO val2017 AP has **not** yet been independently reproduced with this checkout. Do not publish without author approval and final license/review checks.

## Recorded paper result

The results below are **historical run records**, not a new evaluation of this repository:

| Dataset / checkpoint | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO val2017, epoch 12, **non-EMA** | 45.269 | 63.331 | 49.482 | 26.432 | 48.313 | 62.514 |

The corresponding detector checkpoint is `epoch_12.pth` (original run `ego_det_v2/train_mod10b_batch8_long`). Do not commit the checkpoint or the dataset to Git.

## Installation

Tested locally using Python 3.11, PyTorch 2.1.2+cu118, NumPy 1.26.4 and CUDA-enabled GPU. One requirements file includes the matching CUDA 11.8 PyTorch wheels; it is intended for Linux x86_64 with a compatible NVIDIA driver.

```bash
conda create -n egodet python=3.11 pip -y
conda activate egodet
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

No MMCV, MMDetection or local CUDA extension compilation is required for the core model path.

## Model assets and COCO data

```text
checkpoints/epoch_12.pth
weights/overlock/overlock_t_in1k_224.pth
data/coco/images/val2017/*.jpg
data/coco/raw/annotations/instances_val2017.json
```

To prepare the expected paths from an existing COCO directory **without copying the images**, optionally use:

```bash
python tools/prepare_coco.py --source /path/to/coco2017 --destination data/coco --split val --dry-run
python tools/prepare_coco.py --source /path/to/coco2017 --destination data/coco --split val
```

This tool creates symlinks only; original COCO annotations are read directly without a conversion step. Training requires the corresponding `train2017` images and JSON as well.

## Train / Eval / Demo

All three user-facing entrypoints live under `scripts/`. Run after editable installation from the repository root.

### Train

```bash
python scripts/train.py \
  --project-root . \
  --train-annotations data/coco/raw/annotations/instances_train2017.json \
  --val-annotations data/coco/raw/annotations/instances_val2017.json \
  --image-root data/coco/images \
  --batch-size 8 --grad-accum 2 --epochs 12 \
  --lr-drop-epoch 11 --image-short-max 608 \
  --efficiency-mode optimized
```

This is the *recorded paper-style* training configuration, not a recommended setting for an 11 GB GPU. The original training implementation and model forward are unchanged by repository cleanup.

### Evaluate

```bash
python scripts/eval.py \
  --project-root . \
  --ckpt checkpoints/epoch_12.pth \
  --val-annotations data/coco/raw/annotations/instances_val2017.json \
  --image-root data/coco/images \
  --batch-size 1 --num-workers 0 --subset-size 10
```

Omit `--subset-size` for full COCO val2017. No `--use-ema` for the historical non-EMA checkpoint. `--batch-size` and `--num-workers` change **only the evaluation DataLoader**, not the model or saved checkpoint; useful for low-memory GPUs. This entrypoint keeps the original COCO postprocessing and scoring.

### Demo

```bash
python scripts/demo.py \
  --image /path/to/image.jpg \
  --output outputs/demo_detection.jpg \
  --short-side 384 --max-size 640 --score-thr 0.3
```

Demo shows contiguous category indices (0–79) and may show repeated boxes; it is **not** COCO evaluation. Demo output and local images are Git-ignored.

## Configuring batch size

The effective **training default** is `EgoDetV2Config` in `ego_det_v2/configs/config.py`: currently `batch_size=4`, `grad_accum_steps=4` and `num_workers=8`. Pass `--batch-size` / `--grad-accum` to `scripts/train.py` to override the defaults; avoid editing model source for each run.

`configs/paper_epoch12_recorded.json` is an **archived parameter record, not an active config**. Editing it does not change a run. During evaluation, the launcher reconstructs the model configuration from `checkpoint['config']`; the DataLoader batch size can instead be overridden independently by `scripts/eval.py --batch-size 1 --num-workers 0`.

Changing training batch size or gradient accumulation can change optimization and results. Do not present a resource-adjusted run as an exact paper reproduction.

## Repository layout

```text
EGO-Det/
├── egodet/              # stable import API + single-image inference implementation
├── ego_det_v2/          # full trained detector, losses, data, evaluation and ablations
│   └── backbones/overlock/  # adapted upstream OverLoCK and its license
├── configs/             # historical paper configuration record
├── scripts/             # train.py / eval.py / demo.py
├── tools/               # COCO dataset preparation
├── requirements.txt
└── pyproject.toml
```

`from egodet import EgoDet, EgoDetConfig` is the stable import API. `ego_det_v2` is the actual checkpoint-compatible implementation. The old `ego_joint_object_slot` and `stage1` *top-level packages* have been removed because they were historical compatibility paths; no Slot layers have been removed from the detector.

## Tests and external code

Model tests and engineering contracts remain in `ego_det_v2/tests/`. For an import-free quick check, run:

```bash
python -m compileall -q egodet ego_det_v2 scripts tools
python scripts/eval.py --help
python scripts/train.py --help
python scripts/demo.py --help
```

The adapted `overlock_official_nofrills.py` originates from [LMMMEng/OverLoCK](https://github.com/LMMMEng/OverLoCK) and is subject to Apache License 2.0; its text is retained in `ego_det_v2/backbones/overlock/LICENSE`. This does **not** determine the license of the EGO-Det project as a whole. The EGO-Det project-wide license and paper citation are pending author approval.
