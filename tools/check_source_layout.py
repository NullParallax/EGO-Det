#!/usr/bin/env python3
"""Static local-import audit for the EGO-Det source tree.

No PyTorch or COCO installation is required. This script never imports the
project modules and does not modify the repository.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("egodet", "ego_det_v2", "ego_joint_object_slot", "stage1")
ENTRIES = {
    "public_api": "egodet",
    "model": "ego_det_v2.models.ego_det_v2",
    "train": "ego_det_v2.launch.launch_train_ego_det_v2_e2e",
    "eval": "ego_det_v2.launch.launch_eval_ego_det_v2",
    "ablations": "ego_det_v2.ablations.__main__",
    "interventions": "ego_det_v2.inference_interventions.__main__",
}


def _module_from_path(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    pieces = relative.parts
    if pieces[-1] == "__init__":
        pieces = pieces[:-1]
    return ".".join(pieces)


def _source_of_import(current: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = current if is_package else current.rpartition(".")[0]
    parts = package.split(".")
    # Relative import level 1 means the current package.
    keep = len(parts) - (node.level - 1)
    if keep < 1:
        return ""
    prefix = ".".join(parts[:keep])
    return ".".join(part for part in (prefix, node.module) if part)


def main() -> int:
    modules = {}
    for source_root in SOURCE_ROOTS:
        for path in (ROOT / source_root).rglob("*.py"):
            if "__pycache__" not in path.parts:
                modules[_module_from_path(path)] = path

    edges = defaultdict(set)
    missing = set()

    def local(name: str) -> bool:
        return any(name == stem or name.startswith(stem + ".") for stem in SOURCE_ROOTS)

    for module, path in sorted(modules.items()):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                requested = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = _source_of_import(module, is_package, node)
                requested = [parent]
                # `from . import child` can import a sibling submodule.
                requested.extend(
                    f"{parent}.{alias.name}"
                    for alias in node.names
                    if parent and f"{parent}.{alias.name}" in modules
                )
            else:
                continue
            for target in requested:
                if not target or not local(target):
                    continue
                if target in modules:
                    edges[module].add(target)
                else:
                    # Only require the imported module (not imported symbols)
                    # to exist; a missing local module is a real issue.
                    missing.add((module, getattr(node, "lineno", 0), target))

    def closure(entry: str) -> set[str]:
        visited, pending = set(), [entry]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(edges[current] - visited)
        return visited

    print(f"[INFO] Local Python modules: {len(modules)}")
    for label, entry in ENTRIES.items():
        print(f"[INFO] {label:14s} reachable modules: {len(closure(entry))}")

    train = closure(ENTRIES["train"])
    evaluation = closure(ENTRIES["eval"])
    print(f"[INFO] Train/eval union: {len(train | evaluation)}")

    for module, lineno, target in sorted(missing):
        print(f"[ERROR] {module}:{lineno} imports missing module {target}")

    if missing:
        print(f"[FAIL] {len(missing)} unresolved local module imports")
        return 1
    print("[PASS] Core local imports resolve statically (runtime not verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
