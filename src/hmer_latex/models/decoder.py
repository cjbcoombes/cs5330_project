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
fusion), and decode greedy-only in model.py::generate.

Bidirectional training and beam search are additions on top of this
interface. Variable-size *images* were not: padding images to a common
size within a batch meant `forward` had to grow a
`memory_key_padding_mask` argument, since padded feature positions must be
kept out of cross-attention the same way padded target positions are kept
out of self-attention.
"""

from __future__ import annotations

import math

import torch
from torch import nn


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

        pos = torch.arange(0, L, dtype=torch.float)
        i = torch.arange(0, C // 4, dtype=torch.float)
        denom = 1 / torch.pow(self.seq_denom, 2 * i / C)

        # (C/4, L)
        cross_prod = pos.reshape((1, L)) * denom.reshape((C // 4, 1))
        self.register_buffer("cross_prod", cross_prod, persistent=False)

        # (L, dtype, device) -> Tensor
        self.pe_cache = {}

    def get_pe(self, L) -> torch.Tensor:
        # Creates PE along one dimension (W or H) to be combined

        cross = self.cross_prod
        dtype = cross.dtype
        device = cross.device
        key = (L, dtype, device)
        if key in self.pe_cache:
            return self.pe_cache[key]

        C = self.d_model // 2
        # Absolute positions, *not* divided by L. Images in a batch are padded
        # to a common H, W (see data/datasets.py::collate_batch), so dividing
        # by the grid extent would make a given pixel's encoding depend on
        # whichever other samples happened to land in its batch.
        pe = torch.zeros((C, L), dtype=dtype, device=device)
        cross_scaled = self.cross_prod[:(C//2), :L]
        pe[0::2, :] = torch.sin(cross_scaled)
        pe[1::2, :] = torch.cos(cross_scaled)

        self.pe_cache[key] = pe
        return pe

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, H', W') -> same shape, with position info added."""

        B, C, H, W = features.shape
        assert self.d_model == C
        assert H <= self.max_len
        assert W <= self.max_len

        half = C // 2

        # (B, W, C, H) for adding (C, H) pe
        a = features.permute(0, 3, 1, 2).clone()
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
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, L) -> same shape, with position info added."""

        B, C, L = features.shape
        assert self.d_model == C
        assert L <= self.max_len

        out = features + self.pe[:, :L]

        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_seq_len: int = 256,
    ):
        super().__init__()

        embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # Initialized at std 1/sqrt(d_model) so that the * sqrt(d_model) in
        # forward() lands back at unit scale. The two go together: PyTorch's
        # default init is std 1, and scaling *that* by sqrt(d_model) inflates
        # pre-softmax attention logits by ~d_model, which saturates attention
        # into near-one-hot at initialization.
        nn.init.normal_(embed.weight, mean=0.0, std=d_model ** -0.5)
        posenc = PositionalEncoding1D(d_model=d_model, max_len=max_seq_len)
        # dim_feedforward and dropout are passed explicitly because PyTorch's
        # defaults (2048, 0.1) are both wrong for us: 2048 is 8x d_model at
        # the MVP size, and 0.1 is far too little regularization for 12k
        # training samples.
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # fwd pass is
        #    tgt      = seq so far                  (B, T, D) taken from embedding of (B, T)
        #    memory   = output of image encoder     (B, L, C), L=H'*W'
        #    tgt_mask = hides future chars in tgt   (T, T)
        # ->
        #    attended = (B, T, D)
        trans = nn.TransformerDecoder(
            decoder_layer=layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        proj = nn.Linear(in_features=d_model, out_features=vocab_size)

        self.embed = embed
        self.posenc = posenc
        self.trans = trans
        self.proj = proj

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), True, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("mask", mask, persistent=False)
        

    def forward(
        self,
        memory: torch.Tensor,
        target_ids: torch.Tensor,
        target_ids_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """memory: (B, H'*W', C) flattened encoder features (with pos. enc.)
        target_ids: (B, T) teacher-forced token ids
        target_ids_mask: (B, T) bool, True on padded target positions
        memory_key_padding_mask: (B, H'*W') bool, True on padded image
            positions -- images in a batch are padded to a common size, so
            without this, cross-attention would attend to blank padding.
        -> logits: (B, T, vocab_size)
        """
        B, L, C = memory.shape
        _, T = target_ids.shape
        D = self.d_model
        V = self.vocab_size

        if T > self.max_seq_len:
            raise ValueError(
                f"target sequence length {T} exceeds max_seq_len "
                f"{self.max_seq_len}; the causal mask and positional encoding "
                f"are only built that large"
            )

        mask = self.mask[:T, :T]
        assert mask.shape == (T, T)

        # Scaled by sqrt(D) to undo the 1/sqrt(d_model) embedding init above,
        # which puts token identity at the same amplitude as the positional
        # encoding that is about to be added to it.
        embed = self.embed(target_ids) * math.sqrt(D)
        assert embed.shape == (B, T, D)

        pos_embed = self.posenc(embed.transpose(1, 2)).transpose(1, 2)
        assert pos_embed.shape == (B, T, D)

        trans = self.trans(
            tgt = pos_embed,
            tgt_mask = mask,
            tgt_key_padding_mask = target_ids_mask,
            memory = memory,
            memory_key_padding_mask = memory_key_padding_mask,
            tgt_is_causal=True
        )
        assert trans.shape == (B, T, D)

        proj = self.proj(trans)
        assert proj.shape == (B, T, V)

        return proj
