"""Accessibility of the report palette and markup.

- Colour pairs that carry meaning (left vs right hemisphere; right vs the
  lesion colour, which sit near each other in charts) must stay apart under
  simulated protanopia, deuteranopia and tritanopia (Machado et al. 2009,
  severity 1.0), measured as CIE76 colour difference.
- Text and status colours must meet WCAG contrast on their backgrounds.
- The page has a skip link, labelled landmarks, keyboard-sortable tables and
  alt text on images.
"""
import re

import numpy as np
import pytest

from lqtpy.report._assets import CSS

_MACHADO = {
    "protanopia": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]],
    "deuteranopia": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                     [-0.011820, 0.042940, 0.968881]],
    "tritanopia": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                   [0.004733, 0.691367, 0.303900]],
}


def _tokens(theme: str) -> dict:
    block = CSS.split(':root[data-theme="light"]')[1] if theme == "light" else CSS.split(":root {")[1]
    block = block.split("}")[0]
    return dict(re.findall(r"--([\w-]+):(#[0-9a-fA-F]{6})", block))


def _lin(hex_):
    c = np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(lin):
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = m @ np.clip(lin, 0, 1) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _delta(a, b, sim=None):
    la, lb = _lin(a), _lin(b)
    if sim:
        m = np.array(_MACHADO[sim])
        la, lb = m @ la, m @ lb
    return float(np.linalg.norm(_lab(la) - _lab(lb)))


def _contrast(fg, bg):
    def lum(h):
        return float(np.array([0.2126, 0.7152, 0.0722]) @ _lin(h))
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("sim", [None, "protanopia", "deuteranopia", "tritanopia"])
@pytest.mark.parametrize("pair", [("left", "right"), ("right", "lesion"), ("left", "lesion")])
def test_meaningful_colours_stay_apart(theme, sim, pair):
    t = _tokens(theme)
    assert _delta(t[pair[0]], t[pair[1]], sim) >= 20, (theme, sim, pair)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_text_contrast(theme):
    t = _tokens(theme)
    for fg in ("ink", "ink-2", "ink-3"):
        assert _contrast(t[fg], t["panel"]) >= 4.5, (theme, fg)
        assert _contrast(t[fg], t["bg"]) >= 4.5, (theme, fg)
    for fg in ("ok", "warn", "err", "focus"):
        assert _contrast(t[fg], t["panel"]) >= 3.0, (theme, fg)


def test_markup_landmarks_and_keyboard(tmp_path):
    from lqtpy.report import render_single
    import lqtpy
    import synthetic as syn
    res = lqtpy.analyze_single(syn.img(syn.sphere((-40, 10, 10), 8)), write=False, plots=True,
                               viewer=False)
    page = render_single(res)
    assert 'class="skip" href="#main"' in page
    assert '<main id="main"' in page and 'aria-label="Clinical report"' in page
    assert 'setAttribute("tabindex", "0")' in page          # sortable headers are focusable
    assert re.search(r'<svg class="bfig"[^>]*role="img"[^>]*aria-label="', page)
    assert "<img " not in page.split("<script")[0] or 'alt="' in page
