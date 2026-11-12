# hmer-latex

Handwritten Mathematical Expression Recognition -> LaTeX, with a focus on
extending performance from clean CROHME-style input to photographed
handwriting (HME100K). See `project_proposal.pdf` for the full writeup.

## Setup

```bash
uv sync
```

This creates `.venv/` and installs everything from `pyproject.toml`. Run
anything in the project with `uv run ...` (e.g. `uv run python -m hmer_latex.train`)
so it uses that environment.

## Layout

```
scripts/
  download_data.py  Fetches + extracts CROHME/HME100K raw archives
src/hmer_latex/
  data/
    render_inkml.py renders CROHME InkML strokes -> raster images + labels
    vocab.py        LaTeX token <-> id vocabulary
    datasets.py      torch Dataset/DataLoader classes
  models/
    encoder.py       DenseNet CNN encoder (Lon)
    decoder.py       Transformer decoder, BTTR-style bidirectional training (Chris)
    model.py          combined encoder-decoder model
  config.py           dataclass config (edit defaults directly for now)
  train.py             training loop CLI
  eval.py               evaluation / ExpRate CLI
data/
  raw/                downloaded datasets (gitignored; already present, see below)
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
end. Raw CROHME (2016 + 2023) and HME100K data is already downloaded and
extracted under `data/raw/` (see `scripts/download_data.py` for sources).

## Roadmap

Split: Chris implements the ML models/techniques by hand (architecture,
training loop, decoding). Claude handles everything else (data pipeline,
infra, augmentation).

**Phase 0 -- data pipeline (Claude)**
1. ~~Download CROHME + HME100K.~~ Done.
2. `data/render_inkml.py`: parse InkML traces + ground-truth LaTeX
   annotation, rasterize strokes to grayscale images, tokenize LaTeX into
   whitespace-separated symbols (CROHME truth strings aren't pre-tokenized,
   e.g. `\sum_{i\leq d}`, so this needs a small regex tokenizer).
3. Run `render_dataset` over CROHME train/val/test splits ->
   `data/processed/crohme_{train,val,test}/` + `labels.tsv`.
4. Build vocab from training labels (`Vocab.build_from_labels`), save to
   `data/processed/vocab.json`.
5. `data/datasets.py::collate_batch`: pad images to a common size, pad
   token sequences, produce batched tensors.
6. (Later) HME100K prep + synthetic photographic augmentation
   (perspective/lighting/blur) -- the "extend to photographed handwriting"
   part of the proposal, also Claude's.

**Phase 1a -- end-to-end MVP (Chris, by hand)**

An encoder or decoder alone can't be validated, so build the smallest
pair that can run the *whole* pipeline (image -> encoder -> decoder ->
loss -> greedy decode) before sinking time into the real architecture:
`SimpleCNNEncoder` (a few plain conv/pool layers) + `TransformerDecoder`
with `nhead=1, num_layers=1`, trained L2R only, decoded greedy-only. Get
this overfitting a tiny slice of CROHME end to end -- that's the signal
the rest of the pipeline (data, loss, checkpointing, decoding) actually
works, before scaling up complexity.
7. `models/encoder.py::SimpleCNNEncoder`.
8. `models/decoder.py::TransformerDecoder` (MVP params), `PositionalEncoding2D`.
9. `models/model.py::generate`: greedy decode only.
10. `train.py`: loop with loss/optimizer/checkpointing, run against the MVP model.
11. `eval.py`: ExpRate over a checkpoint.

**Phase 1b -- scale up architecture (Chris, by hand)**

Once the MVP loop runs end to end, swap in the real pieces -- these are
additions on top of the same interfaces, not rewrites.
12. `models/encoder.py::DenseNetEncoder`: swap in for `SimpleCNNEncoder`.
13. Bump decoder `nhead`/`num_layers`, add BTTR-style bidirectional
    (L2R/R2L) training.
14. `models/model.py::generate`: beam search, L2R/R2L fusion.

**Phase 2 -- iteration (joint)**
15. Full CROHME training run, baseline numbers.
16. Fine-tune/evaluate on HME100K to see clean -> photographed
    degradation (the core proposal question); iterate on augmentation
    (Claude) vs. architecture changes like coverage attention (Chris)
    depending on where the error comes from.
15. Error analysis / attention visualization notebooks.
