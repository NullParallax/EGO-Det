# Third-party source and attribution: OverLoCK

This repository includes a local, modified version of the OverLoCK vision
backbone. Its original authors are **Meng Lou** and **Yizhou Yu**.

- Source: https://github.com/LMMMEng/OverLoCK
- Paper: *OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels*, CVPR 2025.
- Upstream code license: Apache License, Version 2.0.
- License copy: `third_party/overlock/LICENSE` (the standard Apache-2.0 text).
- Included local source: `stage1/overlock_official_nofrills.py`.

The adapter in `ego_det_v2/backbones/overlock_adapter.py` is EGO-Det-specific
integration code that uses the upstream model. The upstream model source is
kept unchanged **within this migration except for an attribution comment**.
It had already been adapted for local compatibility before this migration.
No claim is made here that this copy is byte-identical to any particular
upstream revision: that mapping still needs a dedicated upstream diff.

The OverLoCK pretrained weights are **not redistributed** in this repository.
Any release should preserve original copyright/attribution and any required
upstream NOTICE materials; review the authoritative upstream license and the
exact source provenance before changing the repository to Public.

This document is not a license grant for EGO-Det's original code. The license
for EGO-Det itself is still pending the authors' approval.
