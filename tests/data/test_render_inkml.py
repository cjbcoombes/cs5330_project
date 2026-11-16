"""Renderer tests, against synthetic InkML written to tmp_path (no real data)."""

import numpy as np
import pytest
from PIL import Image

from hmer_latex.data.render_inkml import parse_inkml, render_dataset, render_strokes

INKML = """<ink xmlns="http://www.w3.org/2003/InkML">
<traceFormat><channel name="X" type="decimal"/><channel name="Y" type="decimal"/></traceFormat>
<annotation type="truth">$x^{2}$</annotation>
<trace id="0">10 10, 20 20, 30 10</trace>
<trace id="1">40 5, 45 8</trace>
</ink>
"""


def ink_bbox(image: Image.Image):
    """(left, top, right, bottom) of non-background pixels, or None."""
    return Image.eval(image, lambda v: 255 - v).getbbox()


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


def test_render_strokes_size_and_mode():
    image = render_strokes([np.array([[0.0, 0.0], [10.0, 10.0]])], image_size=(384, 128))
    assert image.size == (384, 128)
    assert image.mode == "L"


def test_render_strokes_respects_padding():
    """Ink must not touch the canvas edge -- strokes are drawn centered on the
    path, so a stroke flush to the border loses half its width to clipping."""
    strokes = [np.array([[0.0, 0.0], [100.0, 50.0]])]
    image = render_strokes(strokes, image_size=(384, 128), padding=10, stroke_width=1)
    left, top, right, bottom = ink_bbox(image)
    assert left >= 9 and top >= 9
    assert right <= 384 - 9 and bottom <= 128 - 9


def test_render_strokes_preserves_aspect_ratio():
    """A 10:1 stroke must stay 10:1 on canvas, letterboxed, not stretched."""
    strokes = [np.array([[0.0, 0.0], [100.0, 10.0]])]
    image = render_strokes(strokes, image_size=(384, 128), padding=10, stroke_width=1)
    left, top, right, bottom = ink_bbox(image)
    assert (right - left) / (bottom - top) == pytest.approx(10.0, rel=0.15)
    # and it should be vertically centred, not pinned to the top
    assert abs(((top + bottom) / 2) - 64) < 6


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
