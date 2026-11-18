"""Training entry point.

Usage:
    uv run python -m hmer_latex.train
"""

from __future__ import annotations

import torch

from hmer_latex.config import ExperimentConfig
from hmer_latex.data.vocab import Vocab
from hmer_latex.data.datasets import HMEDataset, default_transform, collate_batch

from hmer_latex.models.model import HMERModel


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(config: ExperimentConfig) -> None:
    device = pick_device(config.train.device)
    print(f"Training on device: {device}")

    vocab = Vocab.load(config.data.vocab_path)
    print(f"Loaded vocab from '{config.data.vocab_path}' with {len(vocab)} symbols")

    ds = HMEDataset(
        data_dir=config.data.tiny_dir, # temporary: use tiny_dir rather than train_dir
        vocab=vocab,
        transform=default_transform()
    )
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    print(f"Loaded data from '{config.data.tiny_dir}', batch_size={config.train.batch_size}, batches={len(dl)}")

    model = HMERModel(
        vocab_size=len(vocab),
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_decoder_layers=config.model.num_decoder_layers
    ).to(device)
    print(f"Built model with d_model={config.model.d_model}, nhead={config.model.nhead}, layers={config.model.num_decoder_layers}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    num_epochs = config.train.epochs
    num_batches = len(dl)    

    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.train.lr
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs * num_batches)


    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch_i in range(num_epochs):
        model.train()

        for batch_i, (imgs, tgt_in, tgt_out, tgt_mask) in enumerate(dl):
            imgs     : torch.Tensor = imgs.to(device)
            tgt_in   : torch.Tensor = tgt_in.to(device)
            tgt_out  : torch.Tensor = tgt_out.to(device)
            tgt_mask : torch.Tensor = tgt_mask.to(device)

            opt.zero_grad()
            logits = model(imgs, tgt_in)
            loss = ce_loss(logits, tgt_out)
            loss.backward()
            opt.step()
            lr_sched.step()

    # TODO:
    # - load Vocab from config.data.vocab_path
    # - build HMEDataset/DataLoader for train + val
    # - build HMERModel, move to device
    # - optimizer (AdamW), LR schedule
    # - training loop with teacher forcing, cross-entropy loss (ignore pad)
    # - periodic validation: ExpRate (exact match) on held-out set
    # - checkpoint to config.train.checkpoint_dir, logs to config.train.log_dir
    raise NotImplementedError("TODO: implement training loop")


def main() -> None:
    train(ExperimentConfig())


if __name__ == "__main__":
    main()
