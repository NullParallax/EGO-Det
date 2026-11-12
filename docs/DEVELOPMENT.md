# Repository development and hygiene

## Local artifacts

The diagnostic image used during setup (`test.jpg`) and the generated visual result (`outputs/demo_detection.jpg`) are not project source files. They are intentionally not committed. Remove these **specific files** only when you no longer need them:

```bash
rm -f -- test.jpg outputs/demo_detection.jpg
```

`/data`, `/weights`, `/checkpoints`, and `/outputs` are ignored by Git. **Never** use `git clean -fdx` for this project: it can delete your local checkpoints, datasets, and predictions. Review `git status --short` before staging.

## Package installation

```bash
conda activate egodet
python -m pip install -e . --no-deps
```

Use `python -m egodet.infer` for the supported diagnostic image demo; `python -m ego_det_v2.launch.launch_eval_ego_det_v2` is still the research evaluation entry. An old local-only `scripts/demo_single_image.py` can be removed after you confirm the new entry produces results on your image.

## Source-only checks

These require Python 3.11 but do not import CUDA or instantiate the model:

```bash
python -m compileall -q egodet ego_det_v2 ego_joint_object_slot stage1 scripts tests tools
python tools/check_source_layout.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

The GitHub workflow runs these source-only checks. They cannot establish model accuracy; real checkpoint inference and COCO evaluation are separate stages.

## Change discipline

Keep one behavioral change per commit. Preserve the registered PyTorch `nn.Module` attribute names and the current state-dictionary structure unless a tested compatibility migration is included. Do not remove the `stage1` backbone implementation or the `ego_joint_object_slot` compatibility package merely because their names look historical. Do not add any private photographs, checkpoints, access tokens, COCO dataset files, or machine-specific environment exports to Git.
