# OverLoCK backbone adapter migration (staging)

This change relocates the **unmodified adapter class body** from
`stage1/multiscale_adapter_v6.py` to
`ego_det_v2/backbones/overlock_adapter.py`. The new module imports the same
upstream OverLoCK implementation from
`stage1/overlock_official_nofrills.py`. The original module is now a
compatibility re-export; the detector's lazy import points to the new file.

## Frozen adapter class contract

AST SHA256 for `OverLoCKMultiScaleAdapterV6` (method bodies and constants):

`94dbb962eeddab8a3a05a3efe4a93e62d548eed335183d29ba9dc398d7489645`

Before/after class AST equality was checked while preparing this patch.
The static test checks this digest so inadvertent changes are detected before
runtime checkpoint verification. Only the import statement outside the class
changed. The upstream file got an attribution-only comment.

## Compatibility boundary

- No EGO-Det `nn.Module` instance attributes were changed.
- The adapter's submodules remain named exactly as before (`self.model`, etc.).
- All model/criterion/forward math is unchanged.
- The class's Python `__module__` is now `ego_det_v2.backbones.overlock_adapter`.
  The former path remains importable, but projects depending on Python class
  identity/string serialization still need direct validation.
- **Strict `state_dict` loading of `epoch_12.pth` is not yet verified**, nor
  is the 45.269 AP re-evaluation. Do not claim checkpoint equivalence solely
  on the basis of these static checks.

## Before committing on WSL

```bash
python -m unittest discover -s tests -p 'test_backbone_layout_static.py' -v
python tools/check_source_layout.py
python -m compileall -q egodet ego_det_v2 stage1 tests tools
git diff --check
```

After PyTorch installation, verify the adapter import from both paths and
compare checkpoint key/shape compatibility with strict loading. The paper
evaluation must still be run against full COCO val2017.

For third-party licensing see `docs/THIRD_PARTY.md`.
