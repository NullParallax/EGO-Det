from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1.overlock_official_nofrills import overlock_t


class OverLoCKMultiScaleAdapterV6(nn.Module):
    """Expose OverLoCK /8, /16 and /32 features for EGO-Net v6.

    This adapter deliberately follows the official OverLoCK internal path rather than
    interpolating a single feature map into fake scales. Humans do enjoy fake
    multi-scale features, but detectors generally punish that nonsense.

    For overlock_t with 672 input, expected grids are:
        feat8:  84 x 84, channels 128
        feat16: 42 x 42, channels 256
        feat32: 21 x 21, channels 640 after the final sub feature fusion
        ctx32:  21 x 21, channels 512 high-level context
    """

    def __init__(self, pretrained: Optional[str] = None, use_gemm: bool = False, deploy: bool = False):
        super().__init__()
        self.model = overlock_t(pretrained=pretrained, num_classes=0, use_gemm=use_gemm, deploy=deploy)
        embed_dim = self.model.embed_dim
        self.out_channels = {
            "feat8": int(embed_dim[1]),
            "feat16": int(embed_dim[2]),
            "feat32": int(embed_dim[-1] + embed_dim[-1] // 4),
            "ctx32": int(embed_dim[-1]),
        }
        print(
            "[OverLoCKMultiScaleAdapterV6] "
            f"feat8={self.out_channels['feat8']}, "
            f"feat16={self.out_channels['feat16']}, "
            f"feat32={self.out_channels['feat32']}, "
            f"ctx32={self.out_channels['ctx32']}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | Tuple[int, int]]:
        m = self.model

        # /4 stem stage.
        x = m.patch_embed1(x)
        for blk in m.blocks1:
            x = blk(x)

        # /8 localization/detail feature.
        x = m.patch_embed2(x)
        for blk in m.blocks2:
            x = blk(x)
        feat8 = x

        # /16 main object distribution feature.
        x = m.patch_embed3(x)
        for blk in m.blocks3:
            x = blk(x)

        # /32 high-level semantic context.
        ctx = m.patch_embed4(x)
        for blk in m.blocks4:
            ctx = blk(ctx)
        ctx32 = ctx

        # Official overview-first/look-closely sub path.
        ctx_ori = m.high_level_proj(ctx)  # /32, low-dimensional context for dynamic blocks.
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode="bilinear", align_corners=False)

        for idx, blk in enumerate(m.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)
        feat16 = x

        x, ctx = m.patch_embedx(x, ctx)
        for _, blk in enumerate(m.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)
        feat32 = x

        return {
            "feat8": feat8,
            "feat16": feat16,
            "feat32": feat32,
            "ctx32": ctx32,
            "grid8": tuple(feat8.shape[-2:]),
            "grid16": tuple(feat16.shape[-2:]),
            "grid32": tuple(feat32.shape[-2:]),
        }
