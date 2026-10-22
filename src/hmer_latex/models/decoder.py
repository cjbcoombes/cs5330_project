"""Transformer decoder: encoder feature grid -> LaTeX token sequence.

Owner: Chris ("front-half").

Follows BTTR (Zhao et al. 2021): standard transformer decoder with 2D
positional encodings added to the (flattened) encoder features, trained
bidirectionally (L2R + R2L) and decoded by fusing both directions at
inference. Coverage attention (from WAP) is explicitly something we want
to evaluate the need for, per the proposal -- keep that as an optional
module, not baked into the base decoder.
"""

from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        raise NotImplementedError("TODO: 2D sinusoidal positional encoding over (H', W') grid")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, H', W') -> same shape, with position info added."""
        raise NotImplementedError


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3):
        super().__init__()
        # TODO: token embedding + nn.TransformerDecoder + output projection.
        # Trained bidirectionally per BTTR: a direction flag/token distinguishes
        # L2R vs R2L targets within the same decoder.
        raise NotImplementedError("TODO: implement transformer decoder")

    def forward(
        self,
        memory: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """memory: (B, H'*W', C) flattened encoder features (with pos. enc.)
        target_ids: (B, T) teacher-forced token ids
        -> logits: (B, T, vocab_size)
        """
        raise NotImplementedError
