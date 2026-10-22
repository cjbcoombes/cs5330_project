"""DenseNet-style CNN encoder: image -> grid of feature vectors.

Owner: Lon ("back-half").

Follows the DenseNet encoder used in WAP / BTTR: a few dense blocks with
transition (pooling) layers between them, no final classification head --
we keep the last conv feature map (C x H' x W') and feed it, flattened with
2D positional encodings added, into the transformer decoder.
"""

from __future__ import annotations

import torch
from torch import nn


class DenseNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, growth_rate: int = 24, out_channels: int = 684):
        super().__init__()
        # TODO: stem conv + dense blocks + transitions, matching BTTR's
        # encoder config (or a smaller variant for faster local iteration).
        raise NotImplementedError("TODO: implement DenseNet encoder")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) -> features: (B, C, H', W')"""
        raise NotImplementedError
