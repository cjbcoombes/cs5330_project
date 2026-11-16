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
  prepare_data.py   Renders CROHME splits, builds vocab, carves tiny slice
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

**Phase 0 (data pipeline) is done.** `data/` renders, tokenizes, batches and
is covered by tests. The models are still stubs (`NotImplementedError`), so
the project is not yet runnable end to end -- next up is Phase 1a.

Raw CROHME (2016 + 2023) and HME100K data is already downloaded and
extracted under `data/raw/` (see `scripts/download_data.py` for sources).

```bash
uv run python scripts/prepare_data.py   # ~12s
```

renders `data/processed/crohme_{train,val,test}/` (12024 / 1702 / 3499
expressions), builds `vocab.json` (138 tokens), and carves a 64-sample
`crohme_tiny/` slice for the Phase 1a overfit test.

Source is CROHME 2023's real ink only: `INKML/train/CROHME2023_train` +
`INKML/CROHME2019`. The 147k-file `Artificial_data` corpus is deliberately
excluded for now -- it dwarfs the real ink and is a Phase 2 question.

Notes on choices that aren't obvious from the code:

- **Canvas is 384x128, not square.** 93% of CROHME expressions are wider
  than tall (median aspect 3.13:1), so a square canvas spends ~68% of its
  pixels on whitespace. The wide canvas renders symbols 44% taller using
  25% fewer pixels. It also makes the encoder's 12x4=48-position grid
  differ from `d_model=64`, so a transposed `memory` is a shape error
  rather than a silent bug.
- **Images are pre-rendered to PNG, not rendered on demand.** Parsing is
  0.32 ms/file and rendering 0.19 ms/file, so neither is a bottleneck; the
  reason to materialize is that val/test must be frozen for metrics to be
  comparable, and the overfit test needs zero input variance. Phase 2
  photographic augmentation goes in the `transform` hook (image-space, no
  re-render); only stroke-level variation would need rendering from ink.
- **Zero-ink LaTeX spacing commands (`\ `, `\!`, `\;`) are dropped** by
  `tokenize_latex`. They put no ink on the page, so nothing in the image
  can predict them. 68 occurrences across 0.47% of train labels.

## Roadmap

Split: Chris implements the ML models/techniques by hand (architecture,
training loop, decoding). Claude handles everything else (data pipeline,
infra, augmentation).

**Phase 0 -- data pipeline (Claude)**
1. ~~Download CROHME + HME100K.~~ Done.
2. ~~`data/render_inkml.py`: parse InkML traces + ground-truth LaTeX
   annotation, rasterize strokes to grayscale images, tokenize LaTeX into
   whitespace-separated symbols.~~ Done (`vocab.tokenize_latex`).
3. ~~Run `render_dataset` over CROHME train/val/test splits.~~ Done, via
   `scripts/prepare_data.py`.
4. ~~Build vocab from training labels, save to `vocab.json`.~~ Done.
5. ~~`data/datasets.py::collate_batch`.~~ Done -- returns
   `(images, tgt_in, tgt_out, tgt_pad_mask)`. Images need no padding (fixed
   canvas); `tgt_in`/`tgt_out` are the same ids offset by one, which is the
   teacher-forcing signal. `tgt_pad_mask` is built but **not yet consumed**:
   `TransformerDecoder.forward` has no padding-mask argument, and
   `ignore_index=pad_id` in the loss is what keeps the MVP correct. Wire it
   in as `tgt_key_padding_mask` before any real training run.
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
