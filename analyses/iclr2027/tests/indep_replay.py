"""Independent re-implementation of the RQ4 replay (ported from the item-3 verifier).

Own parquet loader, running best, stagnation stop, replay simulator and (k_stop, n) grid fit
with ROAR's tie-break; no ROAR function is imported. Used by test_rq4_independent.py.
"""
import hashlib
import os
import re

import numpy as np
import pandas as pd

DATA = os.environ.get("ROAR_DATA", "")
HASHES = {
    "campaigns.parquet": "6dca8718d68c14306c5e20ec418778bd6f14595f9f9205a736baa4e742638031",
    "candidates.parquet": "1f2a0cd56f5b763582cfc4a07efd9b681fd827634cc971e4c0f7ee5e93370ae0",
    "measurements.parquet": "ca0c5d810190a8c475ba1ffcc5b44663b50e6298df5289e7fed581a9503d11d6",
}
NUM = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")
BUDGET = 50
MIN_ITERS = 50
MIN_CAMPAIGNS_PROBLEM = 20
CELL_MIN = 8
HEADROOM = 2.0
SEED = 42
TRIALS = 500


def geom_grid(lo, hi, pts):
    if hi < lo:
        return []
    if hi == lo:
        return [lo]
    v = np.geomspace(lo, hi, num=max(2, pts))
    return sorted({int(round(x)) for x in v if lo <= round(x) <= hi})


K_GRID = geom_grid(2, BUDGET - 1, 9)
N_GRID = geom_grid(1, BUDGET // 10, 6)
# fixed rules need their own stop positions too
STOP_KS = sorted(set(K_GRID) | {5, 6, 10, 11, 20, 21})

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_models(v):
    if v is None or (isinstance(v, float) and v != v):
        return "unknown"
    s = v.strip()
    if s.startswith("{") and s.endswith("}"):
        parts = [x.strip().strip('"') for x in s[1:-1].split(",") if x.strip()]
        return "+".join(parts) if parts else "unknown"
    return s


def trunc(rq):
    if not rq:
        return "unknown"
    return rq[:50] + "..." if len(rq) > 50 else rq


def load(author="Roar Authors"):
    for f, h in HASHES.items():
        if sha(f"{DATA}/{f}") != h:
            raise SystemExit(f"hash mismatch {f}")
    c = pd.read_parquet(f"{DATA}/campaigns.parquet",
                        columns=["id", "author", "algorithm_used", "models_used", "research_question"])
    c = c[c.algorithm_used.notna()]
    if author:
        c = c[c.author == author]
    cand = pd.read_parquet(f"{DATA}/candidates.parquet", columns=["id", "campaign_id", "iteration_index"])
    m = pd.read_parquet(f"{DATA}/measurements.parquet", columns=["candidate_id", "name", "value"])
    m = m[(m.name == "combined_score")]
    m = m[m.value.map(lambda s: bool(NUM.match(s)) if isinstance(s, str) else False)]
    m = m.assign(score=m.value.astype(float))
    j = m.merge(cand, left_on="candidate_id", right_on="id").merge(
        c, left_on="campaign_id", right_on="id", suffixes=("", "_c"))
    it = j.groupby(["campaign_id", "iteration_index"]).score.max().reset_index()
    meta = c.set_index("id")
    trajs = []
    for cid, g in it.groupby("campaign_id"):
        g = g.sort_values("iteration_index")
        if len(g) < MIN_ITERS:
            continue
        sc = g.score.to_numpy(dtype=float)
        best = np.maximum.accumulate(sc).tolist()
        row = meta.loc[cid]
        trajs.append({"campaign_id": str(cid), "problem": trunc(row.research_question),
                      "algorithm": row.algorithm_used, "model": parse_models(row.models_used),
                      "best_scores": best})
    return trajs


# ---------------- replay reimplementation ----------------

def prep(t):
    """Precompute stagnation stop positions s_k (1-based, uncapped) per k."""
    b = t["best_scores"]
    L = len(b)
    stops = {}
    for k in STOP_KS:
        cnt = 0
        pos = None
        for i in range(L):
            if i == 0 or b[i] > b[i - 1]:
                cnt = 1
            else:
                cnt += 1
            if cnt >= k:
                pos = i + 1
                break
        stops[k] = pos if pos is not None else L
    t["_stops"] = stops
    t["_L"] = L
    return t


def consumed(t, k, cap):
    n = min(t["_L"], cap)
    if n <= 0:
        return 0
    if k is None:
        return n
    return min(t["_stops"][k], n)


def sim_mean(pool, n, k, seed, first=None, trials=TRIALS):
    if not pool and first is None:
        return None
    rng = np.random.default_rng(seed)
    per = max(1, BUDGET // n)
    out = []
    P = len(pool)
    for _ in range(trials):
        best = -np.inf
        spent = 0
        runs = 0
        while spent < BUDGET:
            if runs == 0 and first is not None:
                t = first
            elif P:
                t = pool[rng.integers(0, P)]
            else:
                break
            c = consumed(t, k, min(per, BUDGET - spent))
            if c <= 0:
                break
            spent += c
            runs += 1
            r = t["best_scores"][c - 1]
            if r > best:
                best = r
        out.append(best)
    return round(float(np.mean(np.array(out, dtype=float))), 4)


def fit(pool, seed=SEED):
    best = None
    for k in [None, *K_GRID]:
        for n in N_GRID:
            m = sim_mean(pool, n, k, seed)
            if m is None:
                continue
            key = (-m, n, k is None, k if k is not None else 0)
            if best is None or key < best[0]:
                best = (key, k, n, m)
    if best is None:
        return None, 1
    return best[1], best[2]


def baseline(t):
    b = t["best_scores"]
    return float(b[min(BUDGET, len(b)) - 1])


def cells_of(trajs):
    byp = {}
    for t in trajs:
        byp.setdefault(t["problem"], []).append(t)
    probs = sorted(p for p, ts in byp.items()
                   if len(ts) >= MIN_CAMPAIGNS_PROBLEM and len({x["best_scores"][-1] for x in ts}) > 1)
    cells = {}
    for t in trajs:
        if t["problem"] in probs:
            cells.setdefault((t["problem"], t["algorithm"], t["model"]), []).append(t)
    return probs, cells


def sd_final(ts):
    return float(np.std(np.array([t["best_scores"][-1] for t in ts], dtype=float)))


# ---------------- workers ----------------
_S = {}

