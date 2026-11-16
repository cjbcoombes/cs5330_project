"""LaTeX token vocabulary: string <-> integer id mapping for the decoder.

Tokens are whitespace-separated LaTeX symbols (CROHME-style ground truth,
e.g. "x ^ { 2 } + y ^ { 2 } = 1"), plus special tokens for padding, start,
end, and unknown.

`tokenize_latex` produces that whitespace-separated form from raw CROHME
annotations. It runs at render time (see render_inkml.render_dataset), so
labels.tsv is already canonical and `Vocab` only ever has to `.split()`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

# Order matters: multi-character control sequences must be tried before the
# single-character fallback, or "\\frac" tokenizes as "\\" + "f" + "r" + ...
#   \\[a-zA-Z]+  control words: \\frac, \\leq, \\theta
#   \\.          control symbols: \\{, \\}, \\!, and "\\ " (backslash-space)
#   [{}^_]       structural LaTeX
#   [0-9]|[a-zA-Z]  one symbol per character ("12" is two tokens, as CROHME scores it)
#   [^\s]        anything else: + - ( ) = | . , etc.
_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|[{}^_]|[0-9]|[a-zA-Z]|[^\s]")

# LaTeX spacing commands. These render as whitespace -- they put no ink on the
# page, so there is nothing in a handwritten image for the model to condition
# on, and predicting them is pure guesswork. 68 occurrences across 56 of the
# 12024 real-ink train labels (0.47%), so dropping them costs almost nothing
# and removes three unpredictable tokens from the vocabulary.
#
# Dropping "\ " specifically is also what keeps the space-joined labels.tsv
# format lossless: a token that *contains* a space cannot survive
# " ".join(...) followed by .split().
_ZERO_INK_TOKENS = frozenset(
    {"\\ ", "\\!", "\\,", "\\;", "\\:", "\\>", "\\quad", "\\qquad", "\\thinspace"}
)


def tokenize_latex(raw: str) -> str:
    """CROHME ground-truth string -> canonical space-joined token string.

    CROHME truth annotations are not consistently formatted: some are wrapped
    in "$...$", some are already space-separated, most are not. This maps all
    of them onto one canonical form that `Vocab` can split on whitespace.

    Three things here are load-bearing and verified against the corpus:

    - The regex runs on the string with its *internal spaces intact*. The
      "\\." alternative is what captures "\\ " (backslash-space), which
      really does occur in the data. Collapsing spaces first would turn
      "\\pi d\\ = 2\\pi r" into ["\\pid", "\\=", "2", "\\pir"] -- command
      names fused with the following letter.
    - Zero-ink spacing commands are then dropped (see _ZERO_INK_TOKENS).
    - Every "$" is dropped, not just leading/trailing ones, because a handful
      of labels carry a stray "$" mid-string.

    Idempotent: tokenizing already-tokenized output is a no-op.
    """
    tokens = _TOKEN_RE.findall(raw.replace("$", "").strip())
    return " ".join(t for t in tokens if t not in _ZERO_INK_TOKENS)


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = SPECIAL_TOKENS + [t for t in tokens if t not in SPECIAL_TOKENS]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, latex: str) -> list[int]:
        unk_id = self.stoi[UNK]
        return [self.stoi.get(tok, unk_id) for tok in latex.strip().split()]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.itos[i] for i in ids if self.itos[i] not in (PAD, SOS, EOS)]
        return " ".join(tokens)

    @classmethod
    def build_from_labels(cls, labels: list[str]) -> "Vocab":
        seen: set[str] = set()
        for label in labels:
            seen.update(label.strip().split())
        return cls(sorted(seen))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        itos = json.loads(Path(path).read_text())
        non_special = [t for t in itos if t not in SPECIAL_TOKENS]
        return cls(non_special)
