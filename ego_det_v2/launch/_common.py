from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ego_joint_object_slot.configs.paths import EGOJointPaths

from ..configs.config import EgoDetV2Config
from ..data.coco_dataset import build_coco_dataset
from ..data.collate import collate_detection_batch


def resolve_paths(project_root: Optional[str]) -> EGOJointPaths:
    if project_root:
        return EGOJointPaths.from_root(Path(project_root))
    environment_root = os.environ.get("EGO_JOINT_PROJECT_ROOT")
    return EGOJointPaths.from_root(environment_root) if environment_root else EGOJointPaths()


def make_loader(
    split: str,
    *,
    paths: EGOJointPaths,
    config: EgoDetV2Config,
    annotation_file: Optional[str] = None,
    image_root: Optional[str] = None,
    image_ids: Optional[Sequence[int]] = None,
    training_override: Optional[bool] = None,
    shuffle: bool = False,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, object]:
    dataset = build_coco_dataset(
        split,
        paths=paths,
        config=config,
        annotation_file=annotation_file,
        image_root=image_root,
        image_ids=image_ids,
        training_override=training_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or config.batch_size),
        shuffle=bool(shuffle),
        num_workers=int(config.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.num_workers) > 0,
        collate_fn=collate_detection_batch,
        drop_last=False,
    )
    return loader, dataset


def print_config(config: EgoDetV2Config) -> None:
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), flush=True)
    print(
        f"[EgoDetV2 efficiency] mode={config.efficiency_mode}",
        flush=True,
    )


def choose_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
