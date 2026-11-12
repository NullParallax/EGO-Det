#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ego_det_v2.launch.launch_train_ego_det_v2_e2e \
  --project-root "$PWD" \
  --train-annotations data/coco/raw/annotations/instances_train2017.json \
  --val-annotations data/coco/raw/annotations/instances_val2017.json \
  --image-root data/coco/images \
  --batch-size 8 \
  --grad-accum 2 \
  --epochs 12 \
  --lr-drop-epoch 11 \
  --image-short-max 608 \
  --efficiency-mode optimized \
  --output-dir outputs/ego_det_v2/paper_epoch12_reproduction
