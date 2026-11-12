from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def collate_detection_batch(
    batch: Sequence[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Right/bottom pad a detection batch and expose True-at-padding masks."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    images, targets = zip(*batch)
    max_h = _ceil_to_multiple(max(int(image.shape[-2]) for image in images), 32)
    max_w = _ceil_to_multiple(max(int(image.shape[-1]) for image in images), 32)
    output = images[0].new_zeros((len(images), 3, max_h, max_w))
    pad_mask = torch.ones((len(images), max_h, max_w), dtype=torch.bool)
    copied_targets: List[Dict] = []
    for index, (image, target) in enumerate(zip(images, targets)):
        height, width = image.shape[-2:]
        output[index, :, :height, :width] = image
        pad_mask[index, :height, :width] = False
        item = dict(target)
        item["size"] = (int(height), int(width))
        copied_targets.append(item)
    return output, pad_mask, copied_targets
