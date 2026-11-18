"""RQ4 (our corpus): ported pipeline vs the independent re-implementation in indep_replay.py.

For the first S splits (default 3; all 200 with FULL=1) of every tested configuration, the
independent code re-selects the configurations, re-draws the split (the same permutation
ROAR's split_calibration_evaluation draws: default_rng(r) over campaigns sorted by id),
re-fits the policy and re-scores every rule, then compares each split's gain with
results/rq4_roar_splits.json.gz. Needs ROAR_DATA.
usage: python -I tests/test_rq4_independent.py
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ICLR_RESULTS_DIR", os.path.join(HERE, "..", "results"))
sys.path.insert(0, HERE)
import indep_replay as ir  # noqa: E402

RULES = {"2x25": (None, 2), "5x10": (None, 5), "Restart after 5": (6, 1), "Restart after 10": (11, 1),
         "Restart after 20": (21, 1)}
S = 200 if os.environ.get("FULL") == "1" else int(os.environ.get("SPLITS", "3"))

trajs = ir.load()
probs, cells = ir.cells_of(trajs)
tested = sorted((k for k, v in cells.items() if len(v) >= ir.CELL_MIN and ir.sd_final(v) >= ir.HEADROOM),
                key=lambda k: (k[0], -len(cells[k])))
recs = json.load(gzip.open(os.path.join(RES, "rq4_roar_splits.json.gz")))
ported = json.load(open(os.path.join(RES, "rq4_roar.json")))
assert len(tested) == ported["n_tested"] == 12, len(tested)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from analytics.q5.paper_figures import _generator  # noqa: E402  (display-name helper only)
keymap = {(p["problem"], p["algorithm"], p["model"]): i for i, p in enumerate(ported["per_configuration"])}
by = {(r["cell"], r["r"]): r for r in recs}
worst, n = 0.0, 0
for key in tested:
    ts = sorted(cells[key], key=lambda t: t["campaign_id"])
    for t in ts:
        ir.prep(t)
    idx = {t["campaign_id"]: i for i, t in enumerate(ts)}
    ci = keymap[(key[0], key[1], _generator(key[2]))]
    for r in range(S):
        order = np.random.default_rng(r).permutation(len(ts))
        A = [ts[i] for i in order[:len(ts) // 2]]
        B = [ts[i] for i in order[len(ts) // 2:]]
        k, nn = ir.fit(A)
        rules = dict(RULES, **{"Fitted (k_stop, n)": (k, nn)})
        rec = by[(ci, r)]
        assert (rec["k_stop"], rec["n_rep"]) == (k, nn), (key, r, rec["k_stop"], rec["n_rep"], k, nn)
        for name, (kk, n_) in rules.items():
            g = np.mean([ir.sim_mean([t for t in B if t is not c], n_, kk, ir.SEED + idx[c["campaign_id"]], first=c)
                         - ir.baseline(c) for c in B])
            worst = max(worst, abs(g - rec["gain"][name]))
            n += 1
print(f"compared {n} (configuration, split, rule) gains; max |difference| = {worst:.2e}")
sys.exit(0 if worst < 1e-6 else 1)
