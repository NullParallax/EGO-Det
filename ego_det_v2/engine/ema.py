from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            candidate = source[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(candidate, alpha=1.0 - self.decay)
            else:
                value.copy_(candidate)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state)
