"""Full encoder-decoder HMER model, wiring together an encoder + TransformerDecoder.

The forward path is: image -> CNN encoder -> 2D positional encoding ->
flatten to a sequence of "memory" vectors -> transformer decoder -> logits
over the vocabulary.

Images in a batch have different sizes and are padded to a common one (see
data/datasets.py::collate_batch), so a mask marking the padded pixels is
carried through the encoder's downsampling and handed to cross-attention --
otherwise the decoder attends to blank padding, and a sample's output
depends on what else happens to be in its batch.

Still MVP-era: greedy decode, L2R only. BTTR-style bidirectional training
and beam search are the next additions (README Roadmap, items 13-14).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hmer_latex.models.decoder import TransformerDecoder, PositionalEncoding2D
from hmer_latex.models.encoder import DenseNetEncoder, SimpleCNNEncoder


class HMERModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder: str = "densenet",
        d_model: int = 128,
        nhead: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_seq_len: int = 256,
        growth_rate: int = 16,
        encoder_layers: int = 8,
    ):
        super().__init__()

        if encoder == "simple":
            self.encoder: nn.Module = SimpleCNNEncoder(out_channels=d_model)
        elif encoder == "densenet":
            self.encoder = DenseNetEncoder(
                growth_rate=growth_rate,
                num_layers=encoder_layers,
                out_channels=d_model,
            )
        else:
            raise ValueError(f"unknown encoder {encoder!r}; expected 'simple' or 'densenet'")

        self.posenc2d = PositionalEncoding2D(d_model=d_model)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.max_seq_len = max_seq_len

    def memory_pad_mask(self, img_pad_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """(B, H, W) pixel mask -> (B, H'*W') feature mask, True = padding.

        Max-pools the *valid* mask, so a feature cell counts as real if any
        pixel under it is real. Exact rather than approximate because collate
        rounds the batch up to a multiple of the encoder's stride.
        """
        stride = self.encoder.total_stride
        valid = (~img_pad_mask).unsqueeze(1).to(features.dtype)
        valid = F.max_pool2d(valid, stride, stride, ceil_mode=True).squeeze(1)
        if valid.shape[-2:] != features.shape[-2:]:
            raise RuntimeError(
                f"mask grid {tuple(valid.shape[-2:])} != feature grid "
                f"{tuple(features.shape[-2:])}; encoder.total_stride ({stride}) "
                f"disagrees with the encoder's actual pooling, or the batch was "
                f"not padded to a multiple of it"
            )
        # Flattened in the same order as the features below, so memory
        # position i and mask position i refer to the same cell.
        return valid.flatten(1) == 0

    def forward(
        self,
        images: torch.Tensor,
        img_pad_mask: torch.Tensor | None,
        target_ids: torch.Tensor,
        target_ids_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        images: (B, 1, H, W)
        img_pad_mask: (B, H, W) bool, True on padded pixels (None if unpadded)
        target_ids: (B, T)
        target_ids_mask: (B, T) bool, True on padded target positions
        -> logits: (B, T, vocab_size)
        """

        features = self.encoder(images)  # (B, C, H', W')
        mem_pad_mask = (
            None if img_pad_mask is None else self.memory_pad_mask(img_pad_mask, features)
        )
        pos_features = self.posenc2d(features) # (B, C, H', W')
        memory = pos_features.flatten(2).permute(0, 2, 1)  # (B, H'*W', C)
        return self.decoder(memory, target_ids, target_ids_mask, mem_pad_mask)

    def generate_greedy(
        self,
        images: torch.Tensor,
        img_pad_mask: torch.Tensor | None,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Greedy decode at inference time (no teacher forcing).

        MVP: greedy, L2R only. Add beam search / L2R+R2L fusion per BTTR
        once the greedy loop works end to end.
        """

        B, _, H, W = images.shape
        targets = torch.full(
            (B, 1), fill_value=sos_id, device=images.device
        )

        for i in range(self.max_seq_len):
            targets_mask = torch.full(
                targets.shape, fill_value=False, dtype=torch.bool, device=images.device
            )

            step = self.forward(images, img_pad_mask, targets, targets_mask)
            step = torch.argmax(step, dim=2)
            assert step.shape == (B, i+1)
            new_toks = step[:, -1].reshape((B, 1))

            targets = torch.cat((targets, new_toks), dim=1)
            done = (targets == eos_id).any(dim=1).all()
            if done:
                break

        return targets

    def generate(
        self,
        images: torch.Tensor,
        img_pad_mask: torch.Tensor | None,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Greedy decode at inference time (no teacher forcing).

        MVP: greedy, L2R only. Add beam search / L2R+R2L fusion per BTTR
        once the greedy loop works end to end.
        """

        with torch.no_grad():
            return self.generate_greedy(images, img_pad_mask, sos_id, eos_id)
