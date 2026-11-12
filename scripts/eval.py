"""Evaluate EGO-Det on COCO using the original paper postprocessor.

Example: python scripts/eval.py --ckpt checkpoints/epoch_12.pth --batch-size 1
"""

from ego_det_v2.launch.launch_eval_ego_det_v2 import main


if __name__ == "__main__":
    main()
