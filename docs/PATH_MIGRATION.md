# Path dependency migration (compatibility stage)

The paper-compatible V2 source previously imported `EGOJointPaths` and
`setup_project_paths()` from the historical `ego_joint_object_slot` package.
That made the formal detector depend on a differently named research project.

## Changes

- `ego_det_v2/paths.py` now maintains the **same** paths dataclass and asset-checking helpers.
- `ego_det_v2/import_utils.py` maintains the previous project-import path helper.
- Four V2 source files import directly from these V2 modules.
- The old `ego_joint_object_slot` import paths remain as compatibility re-exports.
- `egodet.paths.EgoDetPaths` provides a clear public alias.
- `EGO_JOINT_PROJECT_ROOT` remains recognized for compatibility; a future
  environment-variable rename should be a separate tested change.
- Historical `teacher_cache_dir`, `mmdet_root`, and other path fields are still
  present. They are **not** installed dependencies; pruning these fields must
  happen only after the checkpoint and scripts pass runtime verification.

This migration does not change any registered PyTorch module, parameter,
state_dict key, model forward, checkpoint format, or loss calculation.

## Validation before installing PyTorch

```bash
python -m unittest discover -s tests -p 'test_paths_compatibility.py' -v
python -m unittest discover -s tests -p 'test_public_api_static.py' -v
python tools/check_source_layout.py
python -m compileall -q egodet ego_det_v2 ego_joint_object_slot stage1 tests tools
```

## Runtime validation after installing PyTorch

```python
from egodet import EgoDet, EgoDetConfig
from egodet.paths import EgoDetPaths
from ego_joint_object_slot.configs.paths import EGOJointPaths
assert EgoDetPaths is EGOJointPaths
```

Successful static checks **do not** establish the 45.269 COCO AP result.
Strict loading of the paper checkpoint and COCO evaluation remain release gates.
