import typing as tp

import torch
from torch import nn


class InteractionWeightingBase(nn.Module):
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Base class for interaction weighting layers.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of sessions. [batch_size, maxlen, n_factors]
        weights : torch.Tensor
            Weights for each item of session. [batch_size, maxlen]
        """
        raise NotImplementedError()
