"""RQ3 with code length controlled (Section 4.4, Figure 3, Appendix I.2 / Figures 8-12).

Context composition (Section 4.4, Figure 3):
  ROAR's compute_factor_importance (VB GLMM improvement ~ z(4 factors) + z(phase) +
  (1|campaign), 1.96-SD intervals) on ROAR's query_mutation_factors rows, per problem, with
  one extra fixed effect: z(log mean code length of the context programs). The mean is over
  the context-edge sources that have a code embedding (the set ROAR's context code
  diversity averages over), and only when all of them have the length measure. ROAR's
  FACTOR_KEYS is extended for the call and restored afterwards. Length = o200k tokens of the
  code (Knapsack, Polynomino); CUDA Attention has no code text in the mirror, so its fit
  uses the artifact size in bytes.
Early diversity (Section 4.4, Appendix I.2):
  ROAR's _compute_early_diversity_campaign_data(0.25) per campaign; the control is the
  mean token length of the same early candidates. Partial Spearman = Pearson partial
  correlation of ranks, p from t on n-3 df. Reported per problem, and per mechanism and
  per model within Knapsack and Polynomino. CUDA Attention (3 campaigns, no tokens): plain
  Spearman.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import code_length, figs, parquet_backend as pb, paths, results_io  # noqa: E402

from analytics.q3_diversity import early_diversity as ed  # noqa: E402
from analytics.q3_diversity import factor_importance as fi  # noqa: E402
from analytics.utils import display_mechanism  # noqa: E402

KN, PO, CU = "Bounded 2D Knapsack", "Polynomino Packing", "CUDA Attention kernel optimization"
SHORT = {KN: "Knapsack", PO: "Polynomino", CU: "CUDA Attention"}
MEASURE = {KN: "tokens", PO: "tokens", CU: "art_bytes"}
ORIG_KEYS = list(fi.FACTOR_KEYS)


def fit(rows, problem, extra):
    fi.FACTOR_KEYS = ORIG_KEYS + list(extra)
    try:
        res = fi.compute_factor_importance(rows, None, [problem])
    finally:
        fi.FACTOR_KEYS = ORIG_KEYS
    g = res["groups"][0]
    keys = ORIG_KEYS + list(extra)
    return {"n": g["n"], "n_improvements": g["n_improvements"], "n_non_improvements": g["n_non_improvements"],
            "coef": {k: g["correlations"][k] for k in keys}, "p": {k: g["p_values"][k] for k in keys},
            "ci_lower": {k: g["ci_lower"][k] for k in keys}, "ci_upper": {k: g["ci_upper"][k] for k in keys}}


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    r = np.corrcoef([rx, ry, rz])
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    pr = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    t = pr * np.sqrt((n - 3) / (1 - pr ** 2))
    return float(pr), float(2 * stats.t.sf(abs(t), n - 3)), n


def context_lengths(L):
    t = pb.tables()
    e = t["edges"]
    emb = set(t["candidates"].loc[t["candidates"].has_emb, "id"])
    ctx = e[(e.edge_type == "context") & e.source_candidate_id.isin(emb)][["target_candidate_id", "source_candidate_id"]]
    ctx.columns = ["child_id", "ctx_id"]
    cl = ctx.join(L, on="ctx_id")
    agg = cl.groupby("child_id").agg(n=("ctx_id", "size"), tokens=("tokens", "mean"), art_bytes=("art_bytes", "mean"),
                                     n_tokens=("tokens", "count"), n_art=("art_bytes", "count"))
    agg.loc[agg.n_tokens < agg.n, "tokens"] = np.nan
    agg.loc[agg.n_art < agg.n, "art_bytes"] = np.nan
    return agg[["tokens", "art_bytes"]]


def early(L):
    rows = ed._compute_early_diversity_campaign_data(None, 0.25)
    c = pd.DataFrame(rows)
    ec, _ = pb.early_candidates(0.25)
    ln = ec[["id", "campaign_id"]].join(L, on="id").groupby("campaign_id").tokens.mean()
    c["early_tokens"] = c.campaign_id.map(ln)
    return c


def corr_block(s):
    out = {}
    if s.early_tokens.notna().all() and len(s) > 4:
        r, p, n = partial_spearman(s.early_diversity.to_numpy(), s.best_score.to_numpy(), s.early_tokens.to_numpy())
        out = {"n": n, "rho_partial_on_tokens": r, "p": p}
    rr, pp = stats.spearmanr(s.early_diversity, s.best_score)
    out.update({"n": int(len(s)), "rho_plain": float(rr), "p_plain": float(pp)})
    return out


def main():
    t0 = time.time()
    backend = pb.install("submitted")
    L = code_length.lengths()
    raw = fi.query_mutation_factors()
    cl = context_lengths(L)
    res = {"backend": backend, "glmm": {}, "early_diversity": {}}
    keep = {}
    for p in (KN, PO, CU):
        m = MEASURE[p]
        rows = [r for r in raw if r["problem"] == p]
        rows2 = []
        for r in rows:
            v = cl[m].get(r["candidate_id"], np.nan)
            if np.isfinite(v):
                r = dict(r)
                r["log_ctx_len"] = float(np.log(v))
                rows2.append(r)
        res["glmm"][SHORT[p]] = {"length_measure": m, "rows": len(rows), "rows_with_length": len(rows2),
                                 "n_campaigns": len({r["campaign_id"] for r in rows2}),
                                 "without_length": fit(rows2, p, []),
                                 "with_context_length": fit(rows2, p, ["log_ctx_len"])}
        keep[p] = rows2
    e = early(L)
    for p in (KN, PO, CU):
        s = e[e.problem == p]
        blk = {"problem": corr_block(s)}
        if p != CU:
            blk["by_mechanism"] = {display_mechanism(k): corr_block(g) for k, g in s.groupby("algorithm") if len(g) > 4}
            blk["by_model"] = {k: corr_block(g) for k, g in s.groupby("model") if len(g) > 4}
        res["early_diversity"][SHORT[p]] = blk
    res["runtime_seconds"] = round(time.time() - t0, 1)
    results_io.write_result("rq3_length_control.json", res)

    # Figure 3 panels: ROAR's factor-importance figure for the length-adjusted fits.
    fi.FACTOR_KEYS = ORIG_KEYS + ["log_ctx_len"]
    names = dict(fi.FACTOR_NAMES)
    fi.FACTOR_NAMES["log_ctx_len"] = "Context length (log)"
    try:
        for p in (KN, PO, CU):
            fr = fi.compute_factor_importance(keep[p], None, [p])
            figs.save_roar_png(f"fig3_rq3_factor_importance_{SHORT[p].split()[0].lower()}",
                               lambda fr=fr: fi.generate_factor_importance_figure(fr))
    finally:
        fi.FACTOR_KEYS = ORIG_KEYS
        fi.FACTOR_NAMES.clear()
        fi.FACTOR_NAMES.update(names)
    e.to_csv(paths.out(paths.RESULTS, "rq3_early_campaigns.csv"),
             index=False, float_format="%.6g")
    for k, v in res["glmm"].items():
        print(k, {f: round(c, 3) for f, c in v["with_context_length"]["coef"].items()})
    print({k: v["problem"] for k, v in res["early_diversity"].items()})


if __name__ == "__main__":
    main()
