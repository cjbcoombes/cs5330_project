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
  config.py           dataclasses + presets, overridable from the CLI
  metrics.py           ExpRate / ExpRate@1 / @2 / symbol error rate
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

## Running it

```bash
uv run python scripts/prepare_data.py              # render data, ~15s
uv run python -m hmer_latex.train --preset densenet_small
uv run tensorboard --logdir runs
```

Training logs train/val loss, token accuracy and greedy val
ExpRate/@1/@2/SER each epoch, selects `best.pt` on **val ExpRate**, and
scores the full val and test splits with it at the end. To score a
checkpoint again afterwards, import `hmer_latex.eval` (there is no CLI for
it):

```python
from hmer_latex.data.loaders import test_loader
from hmer_latex.eval import evaluate, format_metrics, load_for_inference

model, vocab, config = load_for_inference("checkpoints/densenet_small/best.pt", device)
metrics, samples = evaluate(model, test_loader(config, vocab), vocab, device)
print(format_metrics(metrics))
```

Three presets, each overridable field by field:

| preset | for |
|---|---|
| `simple_mvp` | the Phase 1a model: SimpleCNN + a 1-layer/1-head decoder. Fast enough to debug the loop on |
| `densenet_small` | default. DenseNet (growth 16, 8 layers/block) + 3-layer decoder at `d_model=128`. Sized for MPS |
| `densenet_bttr` | BTTR's own configuration (growth 24, 16 layers/block, `d_model=256`, 8 heads). For CUDA |

```bash
uv run python -m hmer_latex.train --preset simple_mvp \
    --set data.train_dir=data/processed/crohme_tiny data.augment=false train.epochs=500
```

Any `section.field` works, so a run is fully described by its command line.

## Status

**Runnable end to end.** Data pipeline, both encoders, transformer decoder,
greedy decode, training loop with per-epoch validation, and ExpRate
evaluation are implemented and tested. Still to come: BTTR-style
bidirectional (L2R + R2L) training, beam search, coverage attention,
attention visualization, and HME100K transfer.

Raw CROHME (2016 + 2023) and HME100K data is already downloaded and
extracted under `data/raw/` (see `scripts/download_data.py` for sources).

`prepare_data.py` renders `data/processed/crohme_{train,val,test}/`
(12024 / 1702 / 3499 expressions), builds `vocab.json` (138 tokens), and
carves a 64-sample `crohme_tiny/` slice for the overfit test.

Source is CROHME 2023's real ink only: `INKML/train/CROHME2023_train` +
`INKML/CROHME2019`. The 147k-file `Artificial_data` corpus is deliberately
excluded for now -- it dwarfs the real ink and is a Phase 2 question.

Notes on choices that aren't obvious from the code:

- **One canvas per expression, padded per batch.** Each expression is
  rendered at its own size -- ink 64px tall, width following the aspect
  ratio up to a 512 cap -- and `collate_batch` pads each batch to a common
  size with a mask that keeps the padding out of attention. The earlier
  approach letterboxed everything into a fixed 384x128 canvas; that
  distorted nothing (aspect ratio was preserved) but spent ~70% of its
  pixels on blank space and scaled a 3-symbol expression and a 60-symbol
  one to wildly different ink sizes. Mean pixels per image: 14.3k against
  letterbox's 49.2k. Batches of 16 still come out ~52% padding, which
  width-bucketing the sampler would mostly recover.
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
   `(images, img_pad_mask, tgt_in, tgt_out, tgt_pad_mask)`. `tgt_in`/
   `tgt_out` are the same ids offset by one, which is the teacher-forcing
   signal. Both masks mark padding with `True` and are consumed as
   attention key-padding masks: `tgt_pad_mask` keeps padded labels out of
   self-attention, `img_pad_mask` (downsampled onto the feature grid by
   `HMERModel.encode`) keeps padded image regions out of cross-attention.
   `ignore_index=pad_id` in the loss separately keeps padded positions from
   contributing gradient -- both are needed.
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
7. ~~`models/encoder.py::SimpleCNNEncoder`.~~ Done.
8. ~~`models/decoder.py::TransformerDecoder` (MVP params), `PositionalEncoding2D`.~~ Done.
9. ~~`models/model.py::generate`: greedy decode only.~~ Done.
10. ~~`train.py`: loop with loss/optimizer/checkpointing.~~ Done, plus
    per-epoch validation and ExpRate-based checkpoint selection.
11. ~~`eval.py`: ExpRate over a checkpoint.~~ Done -- ExpRate, ExpRate@1,
    ExpRate@2 and symbol error rate, per epoch on val and once on test.

**Phase 1b -- scale up architecture (Chris, by hand)**

Once the MVP loop runs end to end, swap in the real pieces.
12. ~~`models/encoder.py::DenseNetEncoder`: swap in for
    `SimpleCNNEncoder`.~~ Done -- `model.encoder` selects between them.
13. ~~Bump decoder `nhead`/`num_layers`~~ (done, see presets); add
    BTTR-style bidirectional (L2R/R2L) training.
14. `models/model.py::generate`: beam search, L2R/R2L fusion.
15. ~~Variable-size images: pad and mask rather than fit a fixed
    canvas.~~ Done -- `img_pad_mask` threaded through the encoder's
    downsampling into cross-attention.

**Phase 2 -- iteration (joint)**
15. Full CROHME training run, baseline numbers.
16. Fine-tune/evaluate on HME100K to see clean -> photographed
    degradation (the core proposal question); iterate on augmentation
    (Claude) vs. architecture changes like coverage attention (Chris)
    depending on where the error comes from.
15. Error analysis / attention visualization notebooks.
