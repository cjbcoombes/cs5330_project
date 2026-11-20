"""Expression-level recognition metrics: ExpRate and friends.

ExpRate is the standard CROHME headline number: the fraction of expressions
predicted *exactly* right. Nothing partial counts -- one wrong token in a
50-token formula scores the same as predicting nothing.

Two things worth being precise about:

- `exprate_leq1` / `exprate_leq2` are **error-tolerance bands**: the fraction
  of expressions within an edit distance of 1 / 2 tokens of the reference.
  They are not top-k beam candidates (a different axis entirely), and by
  construction exprate <= exprate_leq1 <= exprate_leq2.
- All of this is computed over *token sequences*, which is the approximation
  the literature reports. CROHME's official tool scores label *graphs*, so a
  prediction that is structurally equivalent but written differently (e.g.
  different brace placement) counts as wrong here and right there.
"""

from __future__ import annotations

from collections.abc import Sequence

import editdistance


def expression_metrics(
    preds: Sequence[Sequence[str]], refs: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Score predicted token sequences against references.

    Both arguments are sequences of token *lists* (not joined strings): the
    metric counts whole LaTeX tokens, so "\\frac" is one edit, not five.
    """
    if len(preds) != len(refs):
        raise ValueError(f"{len(preds)} predictions vs {len(refs)} references")
    if not preds:
        return {
            "exprate": 0.0,
            "exprate_leq1": 0.0,
            "exprate_leq2": 0.0,
            "symbol_error_rate": 0.0,
            "n": 0,
        }

    distances = [editdistance.eval(p, r) for p, r in zip(preds, refs)]
    n = len(distances)
    ref_tokens = sum(len(r) for r in refs)

    return {
        "exprate": sum(d == 0 for d in distances) / n,
        "exprate_leq1": sum(d <= 1 for d in distances) / n,
        "exprate_leq2": sum(d <= 2 for d in distances) / n,
        # Total edits over total reference tokens -- token-weighted, so it is
        # not thrown off by one badly-wrong short expression.
        "symbol_error_rate": sum(distances) / max(ref_tokens, 1),
        "n": n,
    }
