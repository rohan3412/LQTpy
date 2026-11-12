"""Example pictures for quality findings ("What does this look like?").

Each picture is one slice of a made-up lesion, drawn in code (no patient
data): the lesion in orange on a grey brain outline, and the voxels the
finding is about in the slice viewer's flag colours (magenta: stray, to
remove; cyan: missing, to add; yellow: to look at). Opt-in in reports
(``ReportStyle(qc_examples=True)``, ``--qc-examples``).
"""
from __future__ import annotations

import numpy as np

__all__ = ["example_svg", "CAPTIONS", "has_example"]

N = 44                                  # pixels per side
_LESION, _REMOVE, _ADD, _INSPECT = 1, 2, 3, 4
_FILL = {_LESION: "#f5793a", _REMOVE: "#d6336c", _ADD: "#22b8cf", _INSPECT: "#facc15"}

CAPTIONS = {
    "sparse_slices": "Drawn on every few slices only: seen across slices, the lesion is striped "
                     "and the gaps between the drawn slices (cyan) are missing.",
    "partial_gaps": "Some slices are drawn only in part: a piece of the outline is missing "
                    "(cyan) next to fully drawn slices.",
    "comb": "Drawn slice by slice with uneven edges: across slices the border looks like a comb.",
    "blocky": "Drawn on a coarse grid, or enlarged from one: the border climbs in steps.",
    "holes": "Small unfilled holes inside the lesion (cyan).",
    "hollow": "Only the outline was drawn; the inside (cyan) was never filled.",
    "flat": "Drawn on one slice only: a thin sheet with no depth (yellow), seen from the side.",
    "strokes": "A thin line drawn by mistake (magenta), joined to or near the lesion.",
    "specks": "Stray single voxels or tiny bits (magenta) away from the lesion.",
    "box": "A filled rectangle (yellow): often a placeholder or a cropping box, not a drawing.",
    "edge_artifact": "A piece touching the edge of the image (magenta): usually a registration "
                     "or export artefact.",
    "outside_brain": "Part of the lesion (magenta) lies beyond the brain's edge.",
    "misregistered": "The lesion sits off the brain: the mask and the template don't line up.",
    "fragmentation": "The lesion is split into very many small pieces.",
}


def has_example(finding_id: str) -> bool:
    return finding_id in CAPTIONS


def _disc(cx, cy, rx, ry=None):
    ry = rx if ry is None else ry
    y, x = np.mgrid[:N, :N]
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _brain():
    return _disc(22, 22, 19, 20)


def _picture(fid: str) -> np.ndarray:
    img = np.zeros((N, N), np.uint8)
    blob = _disc(20, 22, 9, 8)
    if fid == "sparse_slices":
        rows = np.zeros_like(blob)
        rows[::3] = True
        img[blob & rows] = _LESION
        img[blob & ~rows] = _ADD
    elif fid == "partial_gaps":
        img[blob] = _LESION
        cut = blob.copy()
        cut[:, :20] = False
        cut[: 18] = False
        cut[24:] = False
        img[cut] = _ADD
    elif fid == "comb":
        # Each row's ends jut out or fall short by turns: ragged, comb-like edges.
        for r in range(N):
            xs = np.flatnonzero(blob[r])
            if xs.size < 5:
                img[r, xs] = _LESION
                continue
            d = 2 if (r // 2) % 2 else -2
            img[r, max(0, xs[0] - d):min(N, xs[-1] + 1 + d)] = _LESION
    elif fid == "blocky":
        small = _disc(20, 22, 9, 8)[::4, ::4]
        img[np.kron(small, np.ones((4, 4), bool))[:N, :N]] = _LESION
    elif fid == "holes":
        img[blob] = _LESION
        for cx, cy in ((17, 20), (22, 25), (23, 19)):
            img[_disc(cx, cy, 1.2)] = _ADD
    elif fid == "hollow":
        inner = _disc(20, 22, 7, 6)
        img[blob & ~inner] = _LESION
        img[inner] = _ADD
    elif fid == "flat":
        img[21:23, 10:34] = _INSPECT
    elif fid == "strokes":
        img[blob] = _LESION
        for k in range(14):
            img[14 - k // 2, 28 + k] = _REMOVE
    elif fid == "specks":
        img[blob] = _LESION
        for x, y in ((33, 10), (35, 30), (12, 36), (30, 37), (9, 12)):
            img[y, x] = _REMOVE
    elif fid == "box":
        img[14:30, 13:29] = _INSPECT
    elif fid == "edge_artifact":
        img[blob] = _LESION
        img[38:44, 30:38] = _REMOVE
    elif fid == "outside_brain":
        b2 = _disc(36, 22, 8, 7)
        img[b2 & _brain()] = _LESION
        img[b2 & ~_brain()] = _REMOVE
    elif fid == "misregistered":
        img[_disc(30, 36, 9, 7)] = _INSPECT
    elif fid == "fragmentation":
        rng = np.random.default_rng(3)
        for _ in range(22):
            x, y = rng.integers(8, 36, 2)
            img[_disc(x, y, rng.uniform(0.6, 1.6))] = _LESION
    return img


def _rects(mask: np.ndarray, colour: str, px: int) -> str:
    out = []
    for r in range(mask.shape[0]):
        row = mask[r]
        c = 0
        while c < len(row):
            if row[c]:
                s = c
                while c < len(row) and row[c]:
                    c += 1
                out.append(f'<rect x="{s * px}" y="{r * px}" width="{(c - s) * px}" height="{px}"/>')
            else:
                c += 1
    return f'<g fill="{colour}">{"".join(out)}</g>' if out else ""


def example_svg(finding_id: str, px: int = 3) -> "str | None":
    """An inline SVG for ``finding_id`` (None when there's no picture)."""
    if finding_id not in CAPTIONS:
        return None
    img = _picture(finding_id)
    size = N * px
    brain = _brain()
    edge = brain & ~np.pad(brain, 1)[2:, 1:-1] | brain & ~np.pad(brain, 1)[:-2, 1:-1] \
        | brain & ~np.pad(brain, 1)[1:-1, 2:] | brain & ~np.pad(brain, 1)[1:-1, :-2]
    body = (_rects(brain, "currentColor", px).replace("<g ", '<g opacity=".10" ', 1)
            + _rects(edge, "currentColor", px).replace("<g ", '<g opacity=".35" ', 1))
    for v, colour in _FILL.items():
        body += _rects(img == v, colour, px)
    return (f'<svg class="qcex" viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
            f'shape-rendering="crispEdges" role="img" aria-label="Example: {finding_id.replace("_", " ")}">{body}</svg>')
