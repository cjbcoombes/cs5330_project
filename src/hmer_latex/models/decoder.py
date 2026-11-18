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

# Hello?
class PositionalEncoding2D(nn.Module):
    """(B, C, H, W) -> (B, C, H, W)"""

    def __init__(self, d_model: int, seq_denom: int = 10000, max_len: int = 100):
        """(B, C, H, W) -> (B, C, H, W)"""
        super().__init__()

        assert d_model % 4 == 0, "requires embedding dimension to be a multiple of 4"

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
        assert H <= self.max_len
        assert W <= self.max_len

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

class PositionalEncoding1D(nn.Module):
    """(B, C, L) -> (B, C, L)"""

    def __init__(self, d_model: int, seq_denom: int = 10000, max_len: int = 100):
        """(B, C, L) -> (B, C, L)"""
        super().__init__()

        assert d_model % 2 == 0, "requires even embedding dimension"

        self.d_model = d_model
        self.seq_denom = seq_denom
        self.max_len = max_len

        C = d_model
        L = max_len
        half = C // 2

        pos = torch.arange(0, L, dtype=torch.float)
        i = torch.arange(0, half, dtype=torch.float)
        denom = 1 / torch.pow(self.seq_denom, 2 * i / C)

        # (C/2, L)
        cross = pos.reshape((1, L)) * denom.reshape((half, 1))
        # (C, L)
        pe = torch.zeros((C, L), dtype=torch.float)
        pe[0::2, :] = torch.sin(cross)
        pe[1::2, :] = torch.cos(cross)
        self.pe = pe

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, L) -> same shape, with position info added."""

        B, C, L = features.shape
        assert self.d_model == C
        assert L <= self.max_len

        out = features + self.pe[:, :L]

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3, max_seq_len: int = 100):
        super().__init__()

        embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        posenc = PositionalEncoding1D(d_model=d_model, max_len=max_seq_len)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True) # has a lot more options too

        # fwd pass is
        #    tgt      = seq so far                  (B, T, D) taken from embedding of (B, T)
        #    memory   = output of image encoder     (B, L, C), L=H'*W'
        #    tgt_mask = hides future chars in tgt   (T, T)
        # ->
        #    attended = (B, T, D)
        trans = nn.TransformerDecoder(decoder_layer=layer, num_layers=num_layers)

        proj = nn.Linear(in_features=d_model, out_features=vocab_size)

        self.embed = embed
        self.posenc = posenc
        self.trans = trans
        self.proj = proj

        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(
        self,
        memory: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """memory: (B, H'*W', C) flattened encoder features (with pos. enc.)
        target_ids: (B, T) teacher-forced token ids
        -> logits: (B, T, vocab_size)
        """
        B, L, C = memory.shape
        _, T = target_ids.shape
        D = self.d_model
        V = self.vocab_size

        mask = nn.Transformer.generate_square_subsequent_mask(T)
        assert mask.shape == (T, T)

        embed = self.embed(target_ids)
        assert embed.shape == (B, T, D)

        pos_embed = self.posenc(embed)
        assert pos_embed.shape == (B, T, D)

        # TODO: multiply by sqrt(D)? why?

        trans = self.trans(
            tgt = pos_embed,
            tgt_mask = mask,
            memory = memory
        )
        assert trans.shape == (B, T, D)

        proj = self.proj(trans)
        assert proj.shape == (B, T, V)

        return proj
