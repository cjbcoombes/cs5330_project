"""Full encoder-decoder HMER model, wiring together DenseNetEncoder + TransformerDecoder."""

from __future__ import annotations

import torch
from torch import nn

from hmer_latex.models.decoder import TransformerDecoder
from hmer_latex.models.encoder import DenseNetEncoder


class HMERModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        self.encoder = DenseNetEncoder(out_channels=d_model)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model)

    def forward(self, images: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)  # (B, C, H', W')
        memory = features.flatten(2).permute(0, 2, 1)  # (B, H'*W', C)
        return self.decoder(memory, target_ids)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int, sos_id: int, eos_id: int) -> torch.Tensor:
        """Greedy/beam decode at inference time (no teacher forcing)."""
        raise NotImplementedError("TODO: autoregressive decode loop, fuse L2R/R2L per BTTR")
