"""Full encoder-decoder HMER model, wiring together an encoder + TransformerDecoder.

MVP (README Roadmap, Phase 1a): SimpleCNNEncoder + TransformerDecoder(nhead=1,
num_layers=1), greedy decode, L2R only. Once that loop runs end to end on a
tiny overfit subset, swap in DenseNetEncoder and scale up nhead/num_layers;
add bidirectional (L2R/R2L) training and beam search in generate() last --
those are additions, not interface changes.
"""

from __future__ import annotations

import torch
from torch import nn

from hmer_latex.models.decoder import TransformerDecoder, PositionalEncoding2D
from hmer_latex.models.encoder import SimpleCNNEncoder


class HMERModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 1, num_decoder_layers: int = 1, max_img_len: int = 100, max_seq_len: int = 100):
        super().__init__()
        self.encoder = SimpleCNNEncoder(out_channels=d_model)
        self.posenc2d = PositionalEncoding2D(d_model=d_model, max_len=max_img_len)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len
        )

    def forward(self, images: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 1, H, W)
        target_ids: (B, T)
        -> logits: (B, T, vocab_size)
        """

        features = self.encoder(images)  # (B, C, H', W')
        pos_features = self.posenc2d(features) # (B, C, H', W')
        memory = pos_features.flatten(2).permute(0, 2, 1)  # (B, H'*W', C)
        return self.decoder(memory, target_ids)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int, sos_id: int, eos_id: int) -> torch.Tensor:
        """Greedy decode at inference time (no teacher forcing).

        MVP: greedy, L2R only. Add beam search / L2R+R2L fusion per BTTR
        once the greedy loop works end to end.
        """
        raise NotImplementedError("TODO: autoregressive greedy decode loop")
