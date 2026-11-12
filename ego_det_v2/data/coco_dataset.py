from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from ego_joint_object_slot.configs.paths import EGOJointPaths

from ..configs.config import EgoDetV2Config
from ..models.box_ops import box_xyxy_to_cxcywh


def _compute_resize(
    height: int,
    width: int,
    short_side: int,
    max_size: int,
) -> Tuple[int, int]:
    scale = float(short_side) / float(min(height, width))
    if float(max(height, width)) * scale > float(max_size):
        scale = float(max_size) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _resize(
    image: Image.Image,
    boxes: torch.Tensor,
    short_side: int,
    max_size: int,
) -> Tuple[Image.Image, torch.Tensor]:
    old_w, old_h = image.size
    new_h, new_w = _compute_resize(old_h, old_w, short_side, max_size)
    image = TF.resize(image, [new_h, new_w], antialias=True)
    if boxes.numel() == 0:
        return image, boxes
    scale = boxes.new_tensor([new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h])
    return image, boxes * scale


def _crop(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    rng: random.Random,
) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
    width, height = image.size
    crop_h_max = min(600, height)
    crop_w_max = min(600, width)
    crop_h_min = min(384, crop_h_max)
    crop_w_min = min(384, crop_w_max)
    crop_h = rng.randint(crop_h_min, crop_h_max)
    crop_w = rng.randint(crop_w_min, crop_w_max)
    top = rng.randint(0, max(0, height - crop_h))
    left = rng.randint(0, max(0, width - crop_w))
    image = TF.crop(image, top, left, crop_h, crop_w)
    if boxes.numel() == 0:
        return image, boxes, labels
    offset = boxes.new_tensor([left, top, left, top])
    boxes = boxes - offset
    boxes[:, 0::2].clamp_(0, crop_w)
    boxes[:, 1::2].clamp_(0, crop_h)
    keep = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
    return image, boxes[keep], labels[keep]


class CocoDetectionV2(Dataset):
    """Direct pycocotools reader with DETR-style geometric augmentation."""

    def __init__(
        self,
        annotation_file: Path | str,
        image_root: Path | str,
        *,
        split: str,
        config: Optional[EgoDetV2Config] = None,
        image_ids: Optional[Sequence[int]] = None,
        seed: int = 3407,
        training: Optional[bool] = None,
    ) -> None:
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:  # pragma: no cover - environment contract
            raise RuntimeError("pycocotools is required for EGO-Det V2 data loading") from exc

        self.config = config or EgoDetV2Config()
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.split = str(split)
        self.training = (
            self.split.lower().startswith("train")
            if training is None
            else bool(training)
        )
        self.seed = int(seed)
        self.coco = COCO(str(self.annotation_file))
        all_ids = sorted(int(value) for value in self.coco.getImgIds())
        requested = set(int(value) for value in image_ids) if image_ids is not None else None
        self.image_ids = [value for value in all_ids if requested is None or value in requested]

        category_ids = sorted(int(value) for value in self.coco.getCatIds())
        self.category_id_to_contiguous = {category_id: index for index, category_id in enumerate(category_ids)}
        self.contiguous_to_category_id = {
            index: category_id
            for category_id, index in self.category_id_to_contiguous.items()
        }
        if len(category_ids) != self.config.num_classes:
            raise ValueError(
                f"annotation file contains {len(category_ids)} categories, expected {self.config.num_classes}"
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def _image_path(self, file_name: str) -> Path:
        candidates = [
            self.image_root / self.split / file_name,
            self.image_root / file_name,
            self.image_root / Path(file_name).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_annotations(self, image_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        annotation_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=None)
        annotations = self.coco.loadAnns(annotation_ids)
        boxes: List[List[float]] = []
        labels: List[int] = []
        for annotation in annotations:
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            x, y, width, height = (float(value) for value in annotation["bbox"])
            if width < 1.0 or height < 1.0:
                continue
            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_contiguous:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_id_to_contiguous[category_id])
        return (
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        image_id = self.image_ids[int(index)]
        info = self.coco.loadImgs([int(image_id)])[0]
        path = self._image_path(str(info["file_name"]))
        if not path.exists():
            raise FileNotFoundError(f"COCO image not found: {path}")
        image = Image.open(path).convert("RGB")
        orig_width, orig_height = image.size
        boxes, labels = self._load_annotations(image_id)

        # Per-sample deterministic RNG still changes across epochs when workers
        # receive a different torch initial seed.
        # PyTorch seeds Python's RNG independently in each DataLoader worker.
        # Using that evolving stream keeps augmentation reproducible yet avoids
        # replaying exactly the same crop for an image every epoch.
        rng = random
        if self.training and rng.random() < self.config.hflip_prob:
            image = TF.hflip(image)
            width = image.size[0]
            if boxes.numel() > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        if self.training and rng.random() < self.config.random_crop_prob:
            first_side = rng.choice((400, 500, 600))
            image, boxes = _resize(image, boxes, first_side, self.config.train_max_size)
            image, boxes, labels = _crop(image, boxes, labels, rng)
        final_side = (
            rng.choice(self.config.train_short_sides)
            if self.training
            else int(self.config.val_short_side)
        )
        image, boxes = _resize(
            image,
            boxes,
            int(final_side),
            self.config.train_max_size if self.training else self.config.val_max_size,
        )

        width, height = image.size
        if boxes.numel() > 0:
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            keep = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
            boxes = boxes[keep]
            labels = labels[keep]

        tensor = TF.to_tensor(np.array(image, copy=True))
        tensor = TF.normalize(tensor, self.config.pixel_mean, self.config.pixel_std)
        scales = boxes.new_tensor([width, height, width, height]).clamp(min=1.0)
        boxes_norm = box_xyxy_to_cxcywh(boxes / scales) if boxes.numel() else boxes.reshape(0, 4)
        target = {
            "labels": labels,
            "boxes": boxes_norm,
            "boxes_xyxy_abs": boxes,
            "image_id": int(image_id),
            "orig_size": (int(orig_height), int(orig_width)),
            "size": (int(height), int(width)),
        }
        return tensor.contiguous(), target


def build_coco_dataset(
    split: str,
    *,
    paths: Optional[EGOJointPaths] = None,
    config: Optional[EgoDetV2Config] = None,
    annotation_file: Optional[Path | str] = None,
    image_root: Optional[Path | str] = None,
    image_ids: Optional[Iterable[int]] = None,
    training_override: Optional[bool] = None,
) -> CocoDetectionV2:
    paths = paths or EGOJointPaths()
    is_train = str(split).lower().startswith("train")
    annotation = annotation_file or (
        paths.coco_train_instances if is_train else paths.coco_val_instances
    )
    root = image_root or paths.coco_image_root
    if annotation is None or root is None:
        raise ValueError("COCO annotation_file and image_root must be supplied")
    canonical_split = "train2017" if is_train else "val2017"
    return CocoDetectionV2(
        annotation,
        root,
        split=canonical_split,
        config=config,
        image_ids=list(image_ids) if image_ids is not None else None,
        seed=(config or EgoDetV2Config()).seed,
        training=training_override,
    )
