"""expression_metrics: the tolerance bands and the token-weighted SER."""

from hmer_latex.metrics import expression_metrics


def toks(s: str) -> list[str]:
    return s.split()


def test_perfect_predictions_score_one():
    refs = [toks("x ^ { 2 }"), toks("a + b")]
    m = expression_metrics(refs, refs)

    assert m["exprate"] == 1.0
    assert m["exprate_leq1"] == 1.0
    assert m["symbol_error_rate"] == 0.0
    assert m["n"] == 2


def test_bands_are_monotonic_and_count_token_edits():
    """One substitution, one insertion, one hopeless case.

    Distances are 1, 2 and 5, so exact=0/3, <=1 = 1/3, <=2 = 2/3.
    """
    refs = [toks("a + b"), toks("a + b"), toks("a + b")]
    preds = [toks("a - b"), toks("a + b + c"), toks("x y z w v")]

    m = expression_metrics(preds, refs)

    assert m["exprate"] == 0.0
    assert m["exprate_leq1"] == 1 / 3
    assert m["exprate_leq2"] == 2 / 3
    assert m["exprate"] <= m["exprate_leq1"] <= m["exprate_leq2"]


def test_multichar_tokens_count_as_one_edit():
    """The whole point of scoring token lists rather than characters."""
    m = expression_metrics([toks("\\alpha")], [toks("\\beta")])

    assert m["symbol_error_rate"] == 1.0


def test_symbol_error_rate_is_token_weighted():
    """2 edits over 7 reference tokens, not the mean of per-expression rates."""
    refs = [toks("a"), toks("a + b = c d")]
    preds = [toks("b"), toks("a + b = c x")]

    m = expression_metrics(preds, refs)

    assert m["symbol_error_rate"] == 2 / 7


def test_empty_input_does_not_divide_by_zero():
    assert expression_metrics([], [])["n"] == 0
