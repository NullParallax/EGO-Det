"""Prepare standard COCO2017 paths using local symlinks, without copying data.

Source may contain train2017/val2017 directly or images/train2017 etc.
No annotation conversion is required by EGO-Det's native COCO reader.
The program never overwrites existing user files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def locate(source: Path, split: str, *, annotations: bool) -> Path:
    filename = f"instances_{split}.json"
    if annotations:
        candidates = [source / "annotations" / filename, source / "raw" / "annotations" / filename]
    else:
        candidates = [source / split, source / "images" / split]
    for path in candidates:
        if path.is_file() if annotations else path.is_dir():
            return path.resolve()
    label = "annotation file" if annotations else "image directory"
    raise FileNotFoundError(f"COCO {split} {label} not found; checked {', '.join(map(str, candidates))}")


def link(source: Path, dest: Path, *, dry_run: bool) -> str:
    if dest.is_symlink() or dest.exists():
        if dest.resolve() == source.resolve():
            return "EXISTS"
        raise FileExistsError(f"Refusing to replace existing destination: {dest}")
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(source, target_is_directory=source.is_dir())
    return "DRY-RUN" if dry_run else "LINKED"


def main() -> None:
    parser = argparse.ArgumentParser(description="Link original COCO 2017 images/annotations into EGO-Det data layout")
    parser.add_argument("--source", required=True, type=Path, help="Existing COCO dataset root")
    parser.add_argument("--destination", type=Path, default=Path("data/coco"))
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Check without writing any links")
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    destination = args.destination.expanduser().absolute()
    splits = ("train2017", "val2017") if args.split == "all" else (f"{args.split}2017",)
    for split in splits:
        for annotations, dest in (
            (False, destination / "images" / split),
            (True, destination / "raw" / "annotations" / f"instances_{split}.json"),
        ):
            original = locate(source, split, annotations=annotations)
            status = link(original, dest, dry_run=args.dry_run)
            print(f"[{status}] {dest} -> {original}")


if __name__ == "__main__":
    main()
