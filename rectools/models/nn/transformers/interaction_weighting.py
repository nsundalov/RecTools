import typing as tp

import torch
from torch import nn


class InteractionWeightingBase(nn.Module):
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
