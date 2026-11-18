"""Save ROAR's figure functions (which return PNG bytes) as PNG and PDF."""
import io
import os

import matplotlib.figure as mfig

from . import paths

_orig = mfig.Figure.savefig
_state = {"pdf": None}


def _patched(self, fname, *a, **kw):
    r = _orig(self, fname, *a, **kw)
    if _state["pdf"] and isinstance(fname, io.BytesIO) and kw.get("format", "png") == "png":
        kw2 = dict(kw)
        kw2["format"] = "pdf"
        kw2["metadata"] = {"CreationDate": None}
        _orig(self, _state["pdf"], **kw2)
        _state["pdf"] = None
    return r


mfig.Figure.savefig = _patched


def save_roar_png(name, fn):
    """Call a ROAR figure function and write figures/<name>.png and .pdf."""
    base = paths.out(paths.FIGURES, name)
    _state["pdf"] = base + ".pdf"
    png = fn()
    with open(base + ".png", "wb") as f:
        f.write(png)
    _state["pdf"] = None
    return base


def save(fig, name):
    base = paths.out(paths.FIGURES, name)
    _orig(fig, base + ".png", format="png", bbox_inches="tight", dpi=200)
    _orig(fig, base + ".pdf", format="pdf", bbox_inches="tight", metadata={"CreationDate": None})
    return base
