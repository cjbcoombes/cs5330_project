"""LaTeX token vocabulary: string <-> integer id mapping for the decoder.

Tokens are whitespace-separated LaTeX symbols (CROHME-style ground truth,
e.g. "x ^ { 2 } + y ^ { 2 } = 1"), plus special tokens for padding, start,
end, and unknown.
"""

from __future__ import annotations

import json
from pathlib import Path

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


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
