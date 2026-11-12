# EGO-Det public API: compatibility stage

This change adds a stable public package name, `egodet`, without replacing or relocating the original `ego_det_v2` implementation. The paper's model class and configuration class remain the source of truth.

```python
from egodet import EgoDet, EgoDetConfig

config = EgoDetConfig()
model = EgoDet(config)
```

`EgoDet` is **an exact alias** of `ego_det_v2.models.ego_det_v2.EgoDetV2` and `EgoDetConfig` is an exact alias of `ego_det_v2.configs.config.EgoDetV2Config`. No subclass or wrapper is created. The registered PyTorch submodule names, `state_dict` keys, forward computation, and serialized checkpoint format are therefore unchanged by this API addition.

## Validation before the Python environment is installed

```bash
python -m compileall -q egodet ego_det_v2
python -m unittest discover -s tests -p 'test_public_api_static.py' -v
python tools/check_source_layout.py
```

These are static checks; they cannot establish a successful checkpoint load or replicate COCO AP.

## Runtime validation after installing dependencies

```bash
python - <<'PY'
from egodet import EgoDet, EgoDetConfig
from ego_det_v2.configs.config import EgoDetV2Config
from ego_det_v2.models.ego_det_v2 import EgoDetV2

assert EgoDet is EgoDetV2
assert EgoDetConfig is EgoDetV2Config
print('Public API resolves to the original paper implementation.')
PY
```

Do not rename internal `nn.Module` attributes, alter checkpoint key mappings, or move the V2 implementation until the `epoch_12.pth` checkpoint has been strictly loaded and evaluated. The existing training and evaluation entrypoints remain authoritative for now.
