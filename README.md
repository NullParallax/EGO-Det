# EGO-Det — Evidence-Guided Object Detection

> **Private reproducibility staging.** The current source successfully loads the final non-EMA checkpoint and completes a CUDA forward pass, but the full COCO val2017 result has **not yet been reproduced** with this repository. Do not treat this as a finalized public release.

EGO-Det is an object-centric detection framework that uses a class-agnostic geometry proposal bank, semantic object-slot formation, slot competition, and slot-conditioned geometry readout.

## Recorded paper result

The values below come from historical experiment records, **not a new evaluation** of the GitHub source.

| Dataset / checkpoint | AP | AP50 | AP75 | APs | APm | APl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COCO val2017, epoch 12, **raw weights** | 45.269 | 63.331 | 49.482 | 26.432 | 48.313 | 62.514 |

Recorded run: `ego_det_v2/train_mod10b_batch8_long`. The evaluated checkpoint is `epoch_12.pth` without EMA.

## Installation

Recommended: Python 3.11, PyTorch 2.1.2 with CUDA 11.8. The CUDA wheel should be installed explicitly **before** the project's requirements.

```bash
conda env create -f environment.yml
conda activate egodet
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The last command enables `egodet` imports from any working directory. See [environment notes](docs/ENVIRONMENT.md) for WSL and GPU-specific details. No MMCV, MMDetection, or custom CUDA compilation is required for the initial detector workflow.

## Weights and data

Store local assets under these **Git-ignored** directories:

```text
checkpoints/epoch_12.pth
weights/overlock/overlock_t_in1k_224.pth
data/coco/images/val2017/...
data/coco/raw/annotations/instances_val2017.json
```

Do not commit model weights, the COCO dataset, or private experiment data.

## Single-image demonstration

```bash
python -m egodet.infer \
  --image /absolute/path/to/photo.jpg \
  --output outputs/demo_detection.jpg \
  --short-side 384 --max-size 640 --score-thr 0.3
```

This diagnostic visualizer labels **contiguous class indices (0–79)**, not COCO category names; it does not add NMS or define a new evaluation protocol. No sample photograph is distributed. Use `--help` for options.

## COCO evaluation

```bash
python -m ego_det_v2.launch.launch_eval_ego_det_v2 \
  --project-root . \
  --ckpt checkpoints/epoch_12.pth \
  --val-annotations data/coco/raw/annotations/instances_val2017.json \
  --image-root data/coco/images \
  --output-json outputs/paper_epoch12_eval.json
```

The default is the **raw model**; do not pass `--use-ema` when reproducing the recorded epoch-12 result. The official evaluation retains its original postprocessing. A recorded-configuration training command is in `scripts/reproduce_paper_train.sh`.

## Code layout

- `egodet/`: stable public aliases, paths, and the diagnostic image-inference command.
- `ego_det_v2/`: original checkpoint-compatible detector, training, evaluation, ablations, and inference interventions.
- `stage1/`: compatible OverLoCK import path and the adapted upstream backbone implementation.
- `ego_joint_object_slot/`: deprecated path compatibility, retained for existing imports.
- `tests/`: dependency-light repository and compatibility checks; deeper contracts remain under `ego_det_v2/tests/`.
- `archive/`: historical tests whose implementation is unavailable; not part of active test discovery.
- `third_party/overlock/`: upstream attribution and license text.

See [development guide](docs/DEVELOPMENT.md), [third-party attribution](docs/THIRD_PARTY.md), and [release gates](docs/VERIFICATION_AND_RELEASE_GATES.md).

## Current verification status

A local WSL session using an RTX 2080 Ti has successfully created the `egodet` environment, strictly loaded `epoch_12.pth` into `EgoDet`, completed a finite CUDA forward pass, and produced a single-image diagnostic output. **COCO subset and full-val AP evaluation remain pending.** Historical results must not be represented as freshly reproduced metrics.

## License and citation

The EGO-Det project-wide license and paper citation are not finalized. Third-party OverLoCK source is separately attributed; check redistribution permissions and research-review restrictions before public release.
