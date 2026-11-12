"""CNN encoders: image -> grid of feature vectors.

Owner: Lon ("back-half").

Two implementations, same interface (images (B, 1, H, W) -> features
(B, C, H', W')), so model.py can swap one for the other without touching
the decoder:

- SimpleCNNEncoder: MVP target -- a handful of plain conv/pool layers.
  Just enough to get a real (if weak) feature grid out, so the full
  pipeline (data -> encoder -> decoder -> loss -> decode) can be
  validated end to end on a tiny overfit subset before sinking time into
  the real architecture. See README Roadmap, Phase 1a.
- DenseNetEncoder: the architecture from the proposal, follows WAP / BTTR
  -- a few dense blocks with transition (pooling) layers between them, no
  final classification head. Build this once the MVP loop works.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNNEncoder(nn.Module):
    """MVP encoder: plain conv/pool stack, no dense blocks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(3, 2, 1),

            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, 3, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) -> features: (B, C, H', W')"""

        return self.body(images)


class DenseNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, growth_rate: int = 24, out_channels: int = 684):
        super().__init__()
        # TODO: stem conv + dense blocks + transitions, matching BTTR's
        # encoder config (or a smaller variant for faster local iteration).
        raise NotImplementedError("TODO: implement DenseNet encoder")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) -> features: (B, C, H', W')"""
        raise NotImplementedError
