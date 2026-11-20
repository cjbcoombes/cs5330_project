"""Renderer tests, against synthetic InkML written to tmp_path (no real data)."""

import numpy as np
import pytest

from hmer_latex.data.render_inkml import parse_inkml, render_dataset, render_strokes

INKML = """<ink xmlns="http://www.w3.org/2003/InkML">
<traceFormat><channel name="X" type="decimal"/><channel name="Y" type="decimal"/></traceFormat>
<annotation type="truth">$x^{2}$</annotation>
<trace id="0">10 10, 20 20, 30 10</trace>
<trace id="1">40 5, 45 8</trace>
</ink>
"""


@pytest.fixture
def inkml_file(tmp_path):
    path = tmp_path / "sample.inkml"
    path.write_text(INKML)
    return path


def test_parse_inkml_strokes_and_label(inkml_file):
    expr = parse_inkml(inkml_file)
    assert len(expr.strokes) == 2
    assert expr.strokes[0].shape == (3, 2)
    assert expr.strokes[1].shape == (2, 2)
    np.testing.assert_allclose(expr.strokes[0][0], [10.0, 10.0])
    # Label comes back tokenized, with the $-wrapper gone.
    assert expr.label == "x ^ { 2 }"


def test_parse_inkml_handles_third_channel(tmp_path):
    """Some CROHME years add a time channel; x and y still come first."""
    path = tmp_path / "t.inkml"
    path.write_text(INKML.replace("10 10, 20 20, 30 10", "10 10 0, 20 20 1, 30 10 2"))
    np.testing.assert_allclose(parse_inkml(path).strokes[0][1], [20.0, 20.0])


def test_single_point_stroke_renders_ink():
    """Regression test: 5.4% of CROHME files contain a one-point trace (dots,
    periods, \\cdot). PIL's draw.line renders *zero* pixels for those."""
    image = render_strokes(
        [np.array([[0.0, 0.0], [100.0, 50.0]]), np.array([[50.0, 25.0]])],
        stroke_width=5,
    )
    plain = render_strokes([np.array([[0.0, 0.0], [100.0, 50.0]])], stroke_width=5)
    assert np.asarray(image).min() == 0
    # the dot must add ink beyond the line alone
    assert (np.asarray(image) < 128).sum() > (np.asarray(plain) < 128).sum()


def test_degenerate_stroke_renders_ink():
    """A stroke whose points are all identical is also a dot."""
    image = render_strokes([np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])], stroke_width=5)
    assert (np.asarray(image) < 128).sum() > 1


def test_render_dataset_writes_pngs_and_labels(tmp_path, inkml_file):
    out = tmp_path / "out"
    written, skipped = render_dataset([inkml_file.parent], out, show_progress=False)
    assert (written, skipped) == (1, 0)
    assert (out / "sample.png").exists()
    assert (out / "labels.tsv").read_text().strip() == "sample.png\tx ^ { 2 }"


def test_render_dataset_skips_unparseable(tmp_path, inkml_file):
    (inkml_file.parent / "broken.inkml").write_text("<ink>not closed")
    written, skipped = render_dataset([inkml_file.parent], tmp_path / "out", show_progress=False)
    assert (written, skipped) == (1, 1)


# --- one canvas per expression; padding happens at batch time ---

RENDER = dict(target_height=64, max_width=512, margin=4, min_side=16)


def stroke(x0, y0, x1, y1):
    return [np.array([[float(x0), float(y0)], [float(x1), float(y1)]])]


def test_render_returns_a_grayscale_image():
    image = render_strokes(stroke(0, 0, 100, 50), **RENDER)
    assert image.mode == "L"


def test_render_fits_the_target_height():
    """An expression whose aspect ratio fits under the width cap fills the
    height exactly. (Anything wider than max_width/target_height = 8:1 is
    limited by the width instead -- see the cap test below.)"""
    image = render_strokes(stroke(0, 0, 300, 60), **RENDER)

    assert image.height == 64
    assert image.width < 512


def test_render_preserves_aspect_ratio():
    """The whole point of one scale factor across all strokes: a 10:1
    expression stays 10:1, so superscripts stay above and fractions stay
    stacked."""
    ink_w, ink_h = 300, 60
    image = render_strokes(stroke(0, 0, ink_w, ink_h), **RENDER)

    # back out the ink extent by removing the margin on both sides
    aspect = (image.width - 8) / (image.height - 8)
    assert abs(aspect - ink_w / ink_h) < 0.1


def test_render_caps_the_width():
    """A very wide expression is scaled down to fit max_width, which makes
    it short rather than clipped."""
    image = render_strokes(stroke(0, 0, 5000, 30), **RENDER)

    assert image.width == 512
    assert image.height < 64


def test_render_survives_degenerate_extents():
    """`span` is floored at 1e-6, so an expression with no vertical extent
    (a lone minus sign) or none at all (a single dot) would otherwise scale
    to a canvas of zero height."""
    flat = render_strokes(stroke(0, 10, 200, 10), **RENDER)
    dot = render_strokes([np.array([[5.0, 5.0]])], **RENDER)
    empty = render_strokes([], **RENDER)

    for image in (flat, dot, empty):
        assert image.width >= 16 and image.height >= 16
        assert image.width <= 512 and image.height <= 64


def test_render_puts_ink_in_the_top_left():
    """No centring: padding is added on the right/bottom at batch time, so
    the ink has to start at `margin` rather than being centred."""
    image = render_strokes(stroke(0, 0, 200, 40), **RENDER)
    pixels = np.asarray(image)

    ink_rows = np.where((pixels < 128).any(axis=1))[0]
    ink_cols = np.where((pixels < 128).any(axis=0))[0]
    assert ink_rows[0] <= 4 + 2
    assert ink_cols[0] <= 4 + 2
