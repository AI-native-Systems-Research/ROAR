"""RQ2 (Section 4.3, Figure 2): escape rates after k non-improving iterations, and the trend.

ROAR's analytics.q2_calibration is called unchanged (observations, GLMM, bootstrap,
figure); only the query is served from the parquet mirror.
  * observations: compute_stagnation_observations(query_iteration_scores()), k = 5..50;
  * escape rate p_k: uncensored episodes; 95% CI = ROAR _bootstrap_ci with campaign
    clusters (2,000 draws, seed 42 inside ROAR);
  * trend: ROAR fit_escape_trend_model (VB GLMM, escaped ~ k + (1|run)); odds ratio per
    +10 iterations. The paper reports the three problems named in Section 4.3 (Bounded 2D
    Knapsack, CUDA Attention, Polynomino Packing). The pooling with and without Palindrome
    Hamiltonian Path is also written, because that problem changes the pooled trend.
  * Figure 2: ROAR generate_aggregate_figure for the three problems (ROAR's default
    model x mechanism cluster bootstrap for the bands, as shipped).
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import figs, parquet_backend, results_io  # noqa: E402

import analytics.q2_calibration as q2  # noqa: E402

KN, CU, PO, PAL = ("Bounded 2D Knapsack", "CUDA Attention kernel optimization",
                   "Polynomino Packing", "Palindrome Hamiltonian Path")
THREE = (KN, CU, PO)
POOLS = {"paper_three": lambda rq: rq in THREE, "all_problems": lambda rq: True,
         "all_excl_palindrome": lambda rq: rq != PAL}


def rate(obs, k):
    unc = [o for o in obs if o["bin_threshold"] == k and not o["censored"]]
    if not unc:
        return {"k": k, "n_uncensored": 0}
    r = sum(o["escaped_in_window"] for o in unc) / len(unc)
    lo, hi = q2._bootstrap_ci(unc, cluster_fn=lambda o: o["campaign_id"])
    return {"k": k, "n_uncensored": len(unc), "n_escaped": int(sum(o["escaped_in_window"] for o in unc)),
            "rate": r, "ci_campaign": [lo, hi]}


def main():
    t0 = time.time()
    backend = parquet_backend.install("submitted")
    camps = q2.query_iteration_scores()
    obs = q2.compute_stagnation_observations(camps)
    rq = {c["campaign_id"]: c["research_question"] for c in camps}
    for o in obs:
        o.setdefault("research_question", rq[o["campaign_id"]])
    res = {"backend": backend, "n_campaigns": len(camps), "n_observations": len(obs), "pooled": {}, "trend": {},
           "per_problem": {}}
    for name, f in POOLS.items():
        ob = [o for o in obs if f(o["research_question"])]
        res["pooled"][name] = [rate(ob, k) for k in q2.THRESHOLDS]
        unc = [o for o in ob if not o["censored"]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = q2.fit_escape_trend_model(unc)
        res["trend"][name] = m
    for p in THREE:
        ob = [o for o in obs if o["research_question"] == p]
        res["per_problem"][p] = [rate(ob, k) for k in (10, 20, 30, 40)]
    analysis = {"line_plot_data": q2._compute_line_plot_data(obs), "_observations": obs}
    three = [p for p in analysis["line_plot_data"]["problems"] if p in THREE]
    figs.save_roar_png("fig2_rq2_escape_rates", lambda: q2.generate_aggregate_figure(analysis, three))
    res["runtime_seconds"] = round(time.time() - t0, 1)
    results_io.write_result("rq2_escape.json", res)
    t = res["trend"]["paper_three"]
    print("paper three OR per 10:", t)


if __name__ == "__main__":
    main()
