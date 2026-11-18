"""Result writing: rounded, sorted JSON so committed results diff cleanly."""
import json

from . import paths


def _round(x, nd=6):
    if isinstance(x, float):
        return None if x != x else round(x, nd)   # NaN -> null (valid JSON)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    return x


def write_result(name, obj, nd=6):
    p = paths.out(paths.RESULTS, name)
    with open(p, "w") as f:
        json.dump(_round(obj, nd), f, indent=1, default=str)
        f.write("\n")
    return p
