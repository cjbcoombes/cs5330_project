# hmer-latex

Handwritten Mathematical Expression Recognition -> LaTeX, with a focus on
extending performance from clean CROHME-style input to photographed
handwriting (HME100K). See `project_proposal.pdf` for the full writeup.

## Setup

```bash
uv sync
```

This creates `.venv/` and installs everything from `pyproject.toml`. Run
anything in the project with `uv run ...` (e.g. `uv run python -m hmer_latex.train --config configs/base.yaml`)
so it uses that environment.

## Layout

```
configs/            YAML experiment configs (one per run variant) + datasets.yaml manifest
scripts/
  download_data.py  Fetches raw datasets per configs/datasets.yaml (not run yet -- URLs are TODO)
src/hmer_latex/
  data/
    render_inkml.py renders CROHME InkML strokes -> raster images + labels
    vocab.py        LaTeX token <-> id vocabulary
    datasets.py      torch Dataset/DataLoader classes
  models/
    encoder.py       DenseNet CNN encoder (Lon)
    decoder.py       Transformer decoder, BTTR-style bidirectional training (Chris)
    model.py          combined encoder-decoder model
  config.py           dataclass config, loaded from YAML
  train.py             training loop CLI
  eval.py               evaluation / ExpRate CLI
data/
  raw/                untouched downloaded datasets (gitignored)
  processed/          rendered images + labels ready for training (gitignored)
checkpoints/          model checkpoints (gitignored)
runs/                 TensorBoard logs (gitignored)
notebooks/            exploration: InkML renders, attention visualization, result plots
```

## Status

Scaffolding only -- most modules are stubs (`NotImplementedError`) marking
out the architecture described in the proposal. Not yet runnable end to
end. Data has not been downloaded.

## Next steps

1. Find and confirm real download URLs for CROHME + HME100K, fill into
   `configs/datasets.yaml`.
2. Implement `data/render_inkml.py` against a real sample `.inkml` file.
3. Implement `DenseNetEncoder` and `TransformerDecoder`, get a tiny subset
   overfitting end to end as a sanity check.
4. Fill in the training loop (`train.py`) and eval loop (`eval.py`).
