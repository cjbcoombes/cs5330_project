"""Transformer decoder: encoder feature grid -> LaTeX token sequence.

Owner: Chris ("front-half").

Follows BTTR (Zhao et al. 2021): standard transformer decoder with 2D
positional encodings added to the (flattened) encoder features, trained
bidirectionally (L2R + R2L) and decoded by fusing both directions at
inference. Coverage attention (from WAP) is explicitly something we want
to evaluate the need for, per the proposal -- keep that as an optional
module, not baked into the base decoder.

MVP target (README Roadmap, Phase 1a): instantiate this same class with
nhead=1, num_layers=1, train L2R only (single direction, no bidirectional
fusion), and decode greedy-only in model.py::generate. The class/forward
signature below should already support that -- bidirectional training and
beam search are additions on top, not interface changes.
"""

from __future__ import annotations

import torch
from torch import nn

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, seq_denom: int = 10000, max_len: int = 100):
        super().__init__()

        assert d_model % 2 == 0, "requires even embedding dimension"

        self.d_model = d_model
        self.seq_denom = seq_denom
        self.max_len = max_len

        C = d_model
        L = max_len
        fourth = C // 4

        pos = torch.arange(0, L, dtype=torch.float)
        i = torch.arange(0, fourth, dtype=torch.float)
        denom = 1 / torch.pow(self.seq_denom, 2 * i / C)

        # (C/4, L)
        self.cross_prod = pos.reshape((1, L)) * denom.reshape((fourth, 1))

    def get_pe(self, L) -> torch.Tensor:
        # Creates PE along one dimension (W or H) to be combined

        C = self.d_model // 2
        pe = torch.zeros((C, L), dtype=torch.float)
        cross = self.cross_prod[:(C//2), :L] / L
        pe[0::2, :] = torch.sin(cross)
        pe[1::2, :] = torch.cos(cross)

        return pe

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, H', W') -> same shape, with position info added."""

        B, C, H, W = features.shape
        assert self.d_model == C

        half = C // 2

        # (B, W, C, H) for adding (C, H) pe
        a = features.permute(0, 3, 1, 2)
        a[:, :, :half, :] += self.get_pe(H)
        # (B, H, C, W) for adding (C, W) pe
        b = a.permute(0, 3, 2, 1)
        b[:, :, half:, :] += self.get_pe(W)

        # (B, C, H, W) for return
        out = b.permute(0, 2, 1, 3)

        return out


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
