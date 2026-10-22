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

## External tools (not vendored yet)

CROHME's own toolkit, referenced by the competition data pages, may be
useful once we're rendering/evaluating InkML:

- [crohmelib](https://gitlab.univ-nantes.fr/crohme/crohmelib) -- reference
  InkML <-> image renderer and LG/InkML converters, plus `evalInkml` for
  the official structural metrics (STRUCT, REL-1/2/3). Candidate for our
  InkML rendering pipeline instead of writing one from scratch.
- [lgEval](https://gitlab.com/dprl/lgeval) -- label-graph evaluation
  library for symbol/structure-level scoring. Useful later for structural
  error analysis beyond LaTeX string accuracy; not needed for a first pass.

Not added as submodules yet -- pull in when we get to the rendering/eval
pipeline.

## Status

Scaffolding only -- most modules are stubs (`NotImplementedError`) marking
out the architecture described in the proposal. Not yet runnable end to
end. Data has not been downloaded.

## Next steps

1. ~~Find and confirm real download URLs for CROHME + HME100K.~~ Done --
   see `configs/datasets.yaml` (CROHME 2023 from Zenodo, CROHME 2016 +
   HME100K from the figshare bundle). Not downloaded yet; run
   `scripts/download_data.py` when ready.
2. Implement `data/render_inkml.py` against a real sample `.inkml` file.
3. Implement `DenseNetEncoder` and `TransformerDecoder`, get a tiny subset
   overfitting end to end as a sanity check.
4. Fill in the training loop (`train.py`) and eval loop (`eval.py`).
