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
    """(B, 1, H, W) -> (B, C, H', W')"""

    # Four stride-2 convs plus one stride-2 pool. HMERModel reads this to
    # downsample the image padding mask to the feature grid, so it has to
    # match `body` -- if you change the strides, change this.
    total_stride = 32

    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        """(B, 1, H, W) -> (B, C, H', W')"""
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


class _Bottleneck(nn.Module):
    """BN-ReLU-1x1 conv -> BN-ReLU-3x3 conv, output concatenated onto the input."""

    def __init__(self, in_channels: int, growth_rate: int, dropout: float):
        super().__init__()
        inter = 4 * growth_rate
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, growth_rate, 3, padding=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.body(x)], dim=1)


class _Transition(nn.Module):
    """BN-ReLU-1x1 conv (channel compression) -> 2x2 avg pool."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Dropout(dropout),
            nn.AvgPool2d(2, 2, ceil_mode=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DenseNetEncoder(nn.Module):
    """DenseNet-BC feature extractor, BTTR-style (3 dense blocks, 16x downsampling).

    With the defaults (growth_rate=24, 16 layers per block, reduction=0.5) the
    channel count goes 48 -> 432 -> 216 -> 600 -> 300 -> 684, so the natural
    output is 684 channels. If out_channels is anything else (e.g. d_model
    from HMERModel), a 1x1 conv projects to it.
    """

    # Stem conv /2 + stem maxpool /2, then two transition avg-pools /2.
    # Read by HMERModel to downsample the image padding mask; keep in sync.
    total_stride = 16

    def __init__(
        self,
        in_channels: int = 1,
        growth_rate: int = 24,
        out_channels: int = 684,
        num_layers: int = 16,
        reduction: float = 0.5,
        dropout: float = 0.2,
    ):
        super().__init__()

        channels = 2 * growth_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

        stages: list[nn.Module] = []
        for i in range(3):
            for _ in range(num_layers):
                stages.append(_Bottleneck(channels, growth_rate, dropout))
                channels += growth_rate
            if i < 2:
                compressed = int(channels * reduction)
                stages.append(_Transition(channels, compressed, dropout))
                channels = compressed
        self.stages = nn.Sequential(*stages)

        self.post_norm = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.proj = nn.Identity() if out_channels == channels else nn.Conv2d(channels, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) -> features: (B, C, H', W'), H' ~ H/16, W' ~ W/16"""
        x = self.stem(images)
        x = self.stages(x)
        x = self.post_norm(x)
        return self.proj(x)
