from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_PROJECT_ROOT = Path(os.environ.get("EGO_JOINT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def _coerce_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if isinstance(path, Path) else Path(path)


@dataclass(frozen=True)
class EGOJointPaths:
    """Absolute paths used by the EGO Joint Object Slot mainline.

    Defaults target the confirmed Linux project root. Use the root argument or
    EGO_JOINT_PROJECT_ROOT explicitly for local mirrors; no silent fallback is
    performed.
    """

    root: Path = DEFAULT_PROJECT_ROOT

    overlock_weights: Optional[Path] = None

    coco_train_jsonl: Optional[Path] = None
    coco_val_jsonl: Optional[Path] = None
    coco_train_instances: Optional[Path] = None
    coco_val_instances: Optional[Path] = None
    coco_image_root: Optional[Path] = None
    coco_label2id: Optional[Path] = None

    teacher_cache_dir: Optional[Path] = None

    proposal_v1_2_ckpt: Optional[Path] = None
    proposal_v2_1x_ckpt: Optional[Path] = None

    output_root: Optional[Path] = None
    mmdet_root: Optional[Path] = None

    # Legacy / optional Visual Genome v6 Stage1 route. These are not required
    # by the COCO-native EGO Joint mainline.
    stage1_vg_v6_ckpt: Optional[Path] = None
    stage1_vg_class_text_bank: Optional[Path] = None
    stage1_vg_attr_text_bank: Optional[Path] = None
    stage1_vg_ctx_text_bank: Optional[Path] = None

    @classmethod
    def from_root(cls, root: Path | str) -> "EGOJointPaths":
        """Build the canonical project paths from an explicit project root.

        Several diagnostics and training launchers use this constructor to
        avoid inheriting a stale ``EGO_JOINT_PROJECT_ROOT`` value.  Keep the
        implementation equivalent to ``EGOJointPaths(root=...)`` and resolve
        the root once so every derived asset path is absolute and consistent.
        """
        return cls(root=Path(root).expanduser().resolve())

    def __post_init__(self) -> None:
        root = _coerce_path(self.root)
        object.__setattr__(self, "root", root)

        defaults = {
            "overlock_weights": root / "weights/overlock/overlock_t_in1k_224.pth",
            "coco_train_jsonl": root / "data/coco/derived/coco_train2017_packed.jsonl",
            "coco_val_jsonl": root / "data/coco/derived/coco_val2017_packed.jsonl",
            "coco_train_instances": root / "data/coco/annotations/instances_train2017.json",
            "coco_val_instances": root / "data/coco/annotations/instances_val2017.json",
            "coco_image_root": root / "data/coco/images",
            "coco_label2id": root / "data/coco/derived/coco_label2id.json",
            "teacher_cache_dir": root / "outputs/stage2_detector_v12_5r_tgd/teacher_cache/rtmdet_x_det",
            "proposal_v1_2_ckpt": root / "outputs/stage2_detector_v12_5r_tgd/v1_2_proposal_hier_01/tgd_v1_best.pth",
            "proposal_v2_1x_ckpt": root / "outputs/stage2_detector_v12_5r_tgd/v2_1x_proposal_rtmdet_x_dense/tgd_v2_1x_proposal_best.pth",
            "output_root": root / "outputs/ego_joint_object_slot",
            "mmdet_root": root / "third_party/mmdetection-main",
        }

        for name, default in defaults.items():
            value = _coerce_path(getattr(self, name))
            object.__setattr__(self, name, value if value is not None else default)


def _append_optional_required(items: List[Tuple[str, Optional[Path]]], name: str, path: Optional[Path]) -> None:
    items.append((name, path))


def required_asset_items(
    paths: EGOJointPaths,
    *,
    include_overlock: bool = True,
    include_coco: bool = True,
    include_teacher: bool = True,
    include_proposals: bool = True,
    include_stage1_vg: bool = False,
) -> List[Tuple[str, Optional[Path]]]:
    items: List[Tuple[str, Optional[Path]]] = []
    if include_overlock:
        items.extend([
            ("overlock_weights", paths.overlock_weights),
        ])
    if include_coco:
        items.extend([
            ("coco_train_jsonl", paths.coco_train_jsonl),
            ("coco_val_jsonl", paths.coco_val_jsonl),
            ("coco_train_instances", paths.coco_train_instances),
            ("coco_val_instances", paths.coco_val_instances),
            ("coco_image_root", paths.coco_image_root),
            ("coco_label2id", paths.coco_label2id),
        ])
    if include_teacher:
        items.extend([
            ("teacher_cache_dir", paths.teacher_cache_dir),
            ("teacher_cache_meta", paths.teacher_cache_dir / "cache_meta.json"),
            ("teacher_train_index", paths.teacher_cache_dir / "train_index.json"),
            ("teacher_val_index", paths.teacher_cache_dir / "val_index.json"),
        ])
    if include_proposals:
        items.extend([
            ("proposal_v1_2_ckpt", paths.proposal_v1_2_ckpt),
            ("proposal_v2_1x_ckpt", paths.proposal_v2_1x_ckpt),
        ])
    if include_stage1_vg:
        _append_optional_required(items, "stage1_vg_v6_ckpt", paths.stage1_vg_v6_ckpt)
        _append_optional_required(items, "stage1_vg_class_text_bank", paths.stage1_vg_class_text_bank)
    return items


def collect_missing_assets(items: Iterable[Tuple[str, Optional[Path]]]) -> List[Tuple[str, Optional[Path]]]:
    return [(name, path) for name, path in items if path is None or not Path(path).exists()]


def assert_required_assets(
    paths: EGOJointPaths,
    *,
    include_overlock: bool = True,
    include_coco: bool = True,
    include_teacher: bool = True,
    include_proposals: bool = True,
    include_stage1_vg: bool = False,
) -> None:
    missing = collect_missing_assets(
        required_asset_items(
            paths,
            include_overlock=include_overlock,
            include_coco=include_coco,
            include_teacher=include_teacher,
            include_proposals=include_proposals,
            include_stage1_vg=include_stage1_vg,
        )
    )
    if missing:
        lines = ["Missing required EGO Joint assets:"]
        lines.extend(f"  - {name}: {path}" for name, path in missing)
        raise FileNotFoundError("\n".join(lines))

# Stable, human-readable alias; legacy EGOJointPaths remains the same class.
EgoDetPaths = EGOJointPaths
