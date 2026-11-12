from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import normalized_cxcywh_to_absolute_xyxy


def _compatible_heads(channels: int, preferred: int) -> int:
    for heads in range(min(preferred, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class BoxPositionalEncoding(nn.Module):
    """Four-coordinate sine/cosine encoding followed by a learned projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels % 8 != 0:
            raise ValueError("d_model must be divisible by 8 for BoxPE")
        self.channels = int(channels)
        self.frequencies = self.channels // 8
        self.projection = nn.Linear(self.channels, self.channels)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        frequency = torch.exp(
            torch.arange(
                self.frequencies,
                device=boxes.device,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / max(self.frequencies - 1, 1))
        ).to(boxes.dtype)
        phase = boxes[..., None] * frequency
        encoded = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return self.projection(encoded)


class ZeroMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SlotUpdateRound(nn.Module):
    """One independent competition/semantic/two-stream update round."""

    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        channels = int(config.d_model)
        self.config = config
        self.box_pe = BoxPositionalEncoding(channels)
        self.self_heads = _compatible_heads(channels, 8)
        self.competition = nn.MultiheadAttention(
            channels,
            self.self_heads,
            batch_first=True,
        )
        self.semantic_attention = nn.MultiheadAttention(
            channels,
            self.self_heads,
            batch_first=True,
        )
        self.competition_norm = nn.LayerNorm(channels)
        self.semantic_norm = nn.LayerNorm(channels)
        self.roi_residual = ZeroMLP(2 * channels, channels)
        self.c_gate = nn.Linear(2 * channels, channels)
        self.s_gate = nn.Linear(2 * channels, channels)
        self.s_to_c = nn.Linear(channels, channels)
        self.c_to_s = nn.Linear(channels, channels)
        nn.init.zeros_(self.s_to_c.weight)
        nn.init.zeros_(self.s_to_c.bias)
        nn.init.zeros_(self.c_to_s.weight)
        nn.init.zeros_(self.c_to_s.bias)

    def _attention_mask(
        self,
        mask: Optional[torch.Tensor],
        batch: int,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.ndim == 2:
            return mask
        if mask.shape[0] != batch:
            raise ValueError("batched self-attention mask has the wrong batch dimension")
        return (
            mask[:, None]
            .expand(batch, self.self_heads, mask.shape[-2], mask.shape[-1])
            .reshape(batch * self.self_heads, mask.shape[-2], mask.shape[-1])
        )

    def forward(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
        boxes: torch.Tensor,
        semantic_support,
        semantic_state: Dict[str, object],
        *,
        self_attention_mask: Optional[torch.Tensor] = None,
        slot_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(c.shape[0])
        position = self.box_pe(boxes)
        if self.config.slot_competition_enabled:
            competition_mask = self._attention_mask(
                self_attention_mask,
                batch,
            )
            query = c + position
            competed, _ = self.competition(
                query,
                query,
                c,
                attn_mask=competition_mask,
                key_padding_mask=slot_padding_mask,
                need_weights=False,
            )
            c = self.competition_norm(c + competed)

        if self.config.semantic_global_attention:
            semantic, _ = self.semantic_attention(
                s + position,
                semantic_state["tokens"],
                semantic_state["tokens"],
                key_padding_mask=semantic_state["token_mask"],
                need_weights=False,
            )
            s = self.semantic_norm(s + semantic)

        if self.config.semantic_regional_retrieval:
            boxes_abs = normalized_cxcywh_to_absolute_xyxy(
                boxes,
                semantic_state["image_sizes"],
            )
            retrieved = semantic_support.retrieve(semantic_state, boxes_abs)
            s = s + self.roi_residual(torch.cat((s, retrieved), dim=-1))

        old_c = c
        old_s = s
        combined = torch.cat((old_s, old_c), dim=-1)
        c = old_c + torch.sigmoid(self.c_gate(combined)) * self.s_to_c(old_s)
        s = old_s + torch.sigmoid(self.s_gate(combined)) * self.c_to_s(old_c)
        if slot_padding_mask is not None:
            valid = (~slot_padding_mask)[..., None].to(s.dtype)
            s = s * valid
            c = c * valid
        return s, c
