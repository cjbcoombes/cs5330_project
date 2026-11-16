"""Tokenizer tests.

These pin down behavior verified against the real CROHME 2023 corpus. If a
change to _TOKEN_RE breaks one of these, the vocab silently changes shape
and every existing checkpoint becomes undecodable -- hence the tests.
"""

import pytest

from hmer_latex.data.vocab import Vocab, tokenize_latex


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Plain, unspaced -- the CROHME 2023 style.
        ("(d+1)\\times(d+n+1)", "( d + 1 ) \\times ( d + n + 1 )"),
        # $-wrapped -- the CROHME 2012/2019 style.
        ("$x^{2}+1$", "x ^ { 2 } + 1"),
        # Already space-separated: tokenizing is a no-op.
        ("x ^ { 2 } + 1", "x ^ { 2 } + 1"),
        # Multi-char control words must not be split into backslash + letters.
        ("\\sum_{i\\leq d}", "\\sum _ { i \\leq d }"),
        # Multi-digit numbers are one token per digit, as CROHME scores them.
        ("120", "1 2 0"),
        # Control symbols: a backslash followed by a non-letter.
        ("\\{a\\}", "\\{ a \\}"),
    ],
)
def test_tokenize_latex(raw, expected):
    assert tokenize_latex(raw) == expected


def test_internal_spaces_are_not_stripped_before_tokenizing():
    """tokenize_latex must NOT collapse internal whitespace before applying
    the regex. "\\ " (backslash-space) really occurs in CROHME labels, and
    stripping spaces first fuses command names with the following letter:
    this label would become ['\\pid', '\\=', '2', '\\pir'].

    The "\\ " itself is then dropped as zero-ink, but it has to be *matched*
    as one token first for the rest of the label to come out right.
    """
    assert tokenize_latex("\\pi d\\ = 2\\pi r").split() == [
        "\\pi", "d", "=", "2", "\\pi", "r",
    ]


@pytest.mark.parametrize("spacing", ["\\ ", "\\!", "\\;", "\\quad"])
def test_zero_ink_spacing_commands_are_dropped(spacing):
    """Spacing commands put no ink on the page, so there is nothing in a
    handwritten image to predict them from.

    Written with a trailing space, as LaTeX itself requires: "a\\quadb" is
    the single command \\quadb, not \\quad followed by b, and the tokenizer
    reads it the same way.
    """
    assert tokenize_latex(f"a{spacing} b") == "a b"


def test_no_token_contains_whitespace():
    """The space-joined labels.tsv format can only round-trip tokens that
    contain no whitespace -- Vocab.encode splits on it."""
    tokens = tokenize_latex("\\pi d\\ = 2\\pi r\\!+\\;1").split()
    assert tokens == tokenize_latex(" ".join(tokens)).split()


def test_stray_mid_string_dollar_is_dropped():
    """A handful of real labels carry a '$' in the middle, not just at the ends."""
    assert "$" not in tokenize_latex("\\left| $\\frac{a}{b} \\right|")


def test_tokenize_is_idempotent():
    once = tokenize_latex("$f=Ce^{\\frac{1}{\\theta}}$")
    assert tokenize_latex(once) == once


def test_vocab_roundtrips_tokenized_labels():
    """tokenize_latex output feeds Vocab directly -- check the seam holds."""
    labels = [tokenize_latex(s) for s in ("$x^{2}$", "a+b", "\\frac{1}{2}")]
    vocab = Vocab.build_from_labels(labels)
    for label in labels:
        assert vocab.decode(vocab.encode(label)) == label
