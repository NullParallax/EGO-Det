"""Train EGO-Det using the established end-to-end training implementation.

Example: python scripts/train.py --batch-size 8 --grad-accum 2 --epochs 12
"""

from ego_det_v2.launch.launch_train_ego_det_v2_e2e import main


if __name__ == "__main__":
    main()
