"""Training entry point.

Usage:
    uv run python -m hmer_latex.train

Logs scalars to TensorBoard under `{train.log_dir}/{train.run_name}`:

    uv run tensorboard --logdir runs
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hmer_latex.config import ExperimentConfig
from hmer_latex.data.vocab import PAD, SOS, EOS, Vocab
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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    epoch: int,
    best_loss: float,
) -> None:
    """Write everything needed to either resume training or run inference.

    `config` is here because you need d_model/nhead/num_decoder_layers to
    *construct* the model before load_state_dict will accept the weights;
    `vocab_path` because token id 47 means nothing on its own, and a
    rebuilt vocab reorders ids and silently produces fluent nonsense.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": lr_sched.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "config": asdict(config),
            "vocab_path": config.data.vocab_path,
        },
        path,
    )


def train(config: ExperimentConfig) -> None:
    device = pick_device(config.train.device)
    print(f"Training on device: {device}")

    vocab = Vocab.load(config.data.vocab_path)
    pad_id = vocab.stoi[PAD]
    sos_id = vocab.stoi[SOS]
    eos_id = vocab.stoi[EOS]
    print(f"Loaded vocab from '{config.data.vocab_path}' with {len(vocab)} symbols")

    def collate(batch):
        return collate_batch(batch, pad_id=pad_id, sos_id=sos_id, eos_id=eos_id)

    ds = HMEDataset(
        data_dir=config.data.tiny_dir, # temporary: use tiny_dir rather than train_dir
        vocab=vocab,
        transform=default_transform()
    )
    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.data.num_workers
    )
    print(f"Loaded data from '{config.data.tiny_dir}', batch_size={config.data.batch_size}, batches={len(dl)}")

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
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_epochs * num_batches)


    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    ckpt_dir = Path(config.train.checkpoint_dir) / config.train.run_name
    writer = SummaryWriter(log_dir=Path(config.train.log_dir) / config.train.run_name)
    print(f"Logging to '{writer.log_dir}', checkpointing to '{ckpt_dir}'")

    step = 0
    best_loss = float("inf")

    for epoch_i in range(num_epochs):
        model.train()

        # Token-weighted, not a mean of batch means: T varies per batch, so
        # averaging batch losses over-weights batches of short formulas.
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_tokens = 0

        bar = tqdm(dl, desc=f"epoch {epoch_i + 1}/{num_epochs}", leave=False)
        for batch_i, (imgs, tgt_in, tgt_out, tgt_mask) in enumerate(bar):
            imgs     : torch.Tensor = imgs.to(device)
            tgt_in   : torch.Tensor = tgt_in.to(device)
            tgt_out  : torch.Tensor = tgt_out.to(device)
            tgt_mask : torch.Tensor = tgt_mask.to(device)

            opt.zero_grad()
            logits = model(imgs, tgt_in, tgt_mask)
            loss = ce_loss(logits.transpose(1, 2), tgt_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            lr_sched.step()

            real = tgt_out != pad_id
            n_tokens = int(real.sum())
            with torch.no_grad():
                n_correct = int(((logits.argmax(-1) == tgt_out) & real).sum())

            epoch_loss_sum += loss.item() * n_tokens
            epoch_correct += n_correct
            epoch_tokens += n_tokens

            batch_acc = n_correct / max(n_tokens, 1)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/token_acc", batch_acc, step)
            writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], step)
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")
            step += 1

        epoch_loss = epoch_loss_sum / max(epoch_tokens, 1)
        epoch_acc = epoch_correct / max(epoch_tokens, 1)
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch_i)
        writer.add_scalar("train/token_acc_epoch", epoch_acc, epoch_i)
        print(
            f"epoch {epoch_i + 1}/{num_epochs}  train_loss={epoch_loss:.4f}  "
            f"token_acc={epoch_acc:.4f}  lr={lr_sched.get_last_lr()[0]:.2e}"
        )

        save_checkpoint(ckpt_dir / "last.pt", model, opt, lr_sched, config, epoch_i, best_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(ckpt_dir / "best.pt", model, opt, lr_sched, config, epoch_i, best_loss)

    writer.close()
    print(f"Done. best train_loss={best_loss:.4f}, checkpoints in '{ckpt_dir}'")

    # TODO:
    # - build HMEDataset/DataLoader for *val* too (currently train-only on tiny_dir)
    # - periodic validation: ExpRate (exact match) on held-out set, and
    #   select best.pt on that rather than on train loss
    # - log val/loss, val/exprate and ref-vs-pred add_text samples


def main() -> None:
    train(ExperimentConfig())


if __name__ == "__main__":
    main()
