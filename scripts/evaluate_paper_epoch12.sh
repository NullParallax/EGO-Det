#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/epoch_12.pth}"
python -m ego_det_v2.launch.launch_eval_ego_det_v2 \
  --project-root "$PWD" \
  --ckpt "$CKPT" \
  --val-annotations data/coco/raw/annotations/instances_val2017.json \
  --image-root data/coco/images \
  --output-json outputs/paper_epoch12_eval.json
