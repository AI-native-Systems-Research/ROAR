"""Compare a quick re-run (argv[1]) with the committed results (argv[2]); see quick_test.sh."""
import gzip
import json
import os
import sys

new, old = sys.argv[1], sys.argv[2]
IGNORE = {"runtime_seconds", "quick"}


def close(x, y):
    """Equal, with NaN == NaN and a 1e-5 tolerance for the VB GLMM's floating-point noise in p-values."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(close(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(close(a, b) for a, b in zip(x, y))
    if isinstance(x, float) and isinstance(y, float):
        return (x != x and y != y) or abs(x - y) <= 1e-5
    return x == y


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in IGNORE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


bad = 0
for f in ("corpus_provenance.json", "rq1_failed_campaigns.json", "rq2_escape.json", "rq3_length_control.json"):
    same = close(strip(json.load(open(os.path.join(new, f)))), strip(json.load(open(os.path.join(old, f)))))
    print(("PASS " if same else "FAIL ") + f + " identical to committed")
    bad += not same
a = json.load(gzip.open(os.path.join(new, "rq4_roar_splits.json.gz")))
b = {(r["cell"], r["r"]): r for r in json.load(gzip.open(os.path.join(old, "rq4_roar_splits.json.gz")))}
same = all(r == b[(r["cell"], r["r"])] for r in a)
print(("PASS " if same else "FAIL ") + f"rq4 roar: {len(a)} quick split records identical to committed splits 0-3")
bad += not same
sys.exit(1 if bad else 0)
