"""RQ1 failed campaigns (Section 4.2, Appendix G).

Definition (Appendix G): a candidate fails when it does not execute; a campaign is a failed
campaign when none of its candidates executed. "Executed" per problem, from the recorded
measurements:
  * judge-server problems (Knapsack, Polynomino, Palindrome): runs_successfully = 1;
  * CUDA problems: numeric combined_score present and != -1000000 (crash sentinel);
  * Dispatcher: numeric combined_score present and build_success != False;
  * KV-cache: numeric combined_score present and no `error` measurement.
Judge-infrastructure failures are candidates whose result_summary carries the judge-server
failure message.

RQ1 is ROAR's process_variance_data(group_by="model_algorithm") (BC, 1,000-resample
bootstrap, seed 42 inside ROAR), on all campaigns and with failed campaigns excluded;
configurations with n >= 10 enter RQ1. Figures 4 and 5 are ROAR's
generate_figure_for_problem on the n >= 10 configurations after the exclusion.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import figs, parquet_backend as pb, results_io  # noqa: E402

import analytics.q1_variance as q1  # noqa: E402
from analytics.utils import normalize_model_name  # noqa: E402

KN, PO, PAL = "Bounded 2D Knapsack", "Polynomino Packing", "Palindrome Hamiltonian Path"
JUDGE = {KN, PO, PAL}
INFRA = ("message=Submission failed - judge server", "message=Judge server at ")


def candidate_table():
    t = pb.tables()
    m = t["measurements"]
    cand = t["candidates"][["id", "campaign_id"]].rename(columns={"id": "candidate_id"})
    sc = pb.numeric_scores().groupby("candidate_id").score.max()

    def first(name):
        return m[m.name == name].groupby("candidate_id").value.max()
    rs = m[m.name == "result_summary"]
    infra = rs[rs.value.str.contains(INFRA[0], regex=False) | rs.value.str.contains(INFRA[1], regex=False)]
    df = cand.merge(t["campaigns"][["id", "research_question", "algorithm_used", "models_list"]],
                    left_on="campaign_id", right_on="id").drop(columns="id")
    df["score"] = df.candidate_id.map(sc)
    df["runs_successfully"] = df.candidate_id.map(first("runs_successfully"))
    df["build_success"] = df.candidate_id.map(first("build_success"))
    df["has_error"] = df.candidate_id.isin(set(m[m.name == "error"].candidate_id))
    df["infra_fail"] = df.candidate_id.isin(set(infra.candidate_id))
    rq = df.research_question
    ex = pd.Series(False, index=df.index)
    j = rq.isin(JUDGE)
    ex[j] = df.runs_successfully[j].astype(str).isin(["1", "1.0"])
    cu = rq.str.startswith("CUDA")
    ex[cu] = df.score[cu].notna() & (df.score[cu] != -1000000)
    di = rq.str.startswith("Dispatcher")
    ex[di] = df.score[di].notna() & (df.build_success[di].astype(str) != "False")
    kv = rq.str.startswith("Optimize KV")
    ex[kv] = df.score[kv].notna() & ~df.has_error[kv]
    df["executed"] = ex
    return df


def rq1_cells(campaigns):
    res = q1.process_variance_data(campaigns, min_runs=2, group_by="model_algorithm")
    cells = [c for c in res["cells"] if c["n"] >= 10 and c["research_question"] in (KN, PO)]
    return sorted(({k: c[k] for k in ("research_question", "models", "algorithm", "n", "bc", "bc_ci_lower",
                                      "bc_ci_upper", "classification")} for c in cells),
                  key=lambda c: (c["research_question"], c["models"], c["algorithm"] or ""))


def counts(cells):
    out = {}
    for p in (PO, KN):
        cs = [c for c in cells if c["research_question"] == p]
        out[p] = {"configurations": len(cs)}
        for cl in ("confirmed_bimodal", "underpowered", "unimodal", "deterministic"):
            out[p][cl] = sum(c["classification"] == cl for c in cs)
    out["both"] = {"configurations": sum(v["configurations"] for v in out.values()),
                   "confirmed_bimodal": sum(v["confirmed_bimodal"] for v in out.values())}
    return out


def main():
    backend = pb.install("submitted")
    df = candidate_table()
    g = df.groupby("campaign_id").agg(n_candidates=("candidate_id", "size"), n_executed=("executed", "sum"),
                                      n_infra=("infra_fail", "sum"))
    failed = g[g.n_executed == 0]
    camp = pb.tables()["campaigns"].set_index("id")
    scores = q1.query_skydiscover_scores()
    cfg_n = pd.Series([(c["research_question"], ", ".join(sorted(normalize_model_name(m) for m in c["models_used"]))
                        if c["models_used"] else "unknown", c["algorithm_used"]) for c in scores]).value_counts()
    failed_rows = []
    for cid, r in failed.iterrows():
        c = camp.loc[cid]
        models = ", ".join(sorted(normalize_model_name(m) for m in c.models_list)) if c.models_list else "unknown"
        failed_rows.append({"problem": c.research_question, "model": models, "mechanism": c.algorithm_used,
                            "n_candidates": int(r.n_candidates), "n_judge_infra_failures": int(r.n_infra),
                            "configuration_n": int(cfg_n.get((c.research_question, models, c.algorithm_used), 0))})
    before = rq1_cells(scores)
    kept = [c for c in scores if c["campaign_id"] not in set(failed.index)]
    after = rq1_cells(kept)
    res = {"backend": backend, "n_campaigns": int(len(g)), "n_failed_campaigns": int(len(failed)),
           "failed_campaigns": failed_rows,
           "executed_candidates_by_problem": {p: {"candidates": int(len(s)), "executed": int(s.executed.sum()),
                                                  "judge_infra_failures": int(s.infra_fail.sum())}
                                              for p, s in df.groupby("research_question")},
           "rq1_counts_all": counts(before), "rq1_counts_excluding_failed": counts(after),
           "rq1_cells_unchanged_by_exclusion": before == after,
           "rq1_cells_excluding_failed": after}
    results_io.write_result("rq1_failed_campaigns.json", res)
    var = q1.process_variance_data(kept, min_runs=10, group_by="model_algorithm")
    figs.save_roar_png("fig4_rq1_polynomino", lambda: q1.generate_figure_for_problem(var, PO))
    figs.save_roar_png("fig5_rq1_knapsack", lambda: q1.generate_figure_for_problem(var, KN))
    print(res["n_failed_campaigns"], failed_rows, res["rq1_counts_excluding_failed"], res["rq1_cells_unchanged_by_exclusion"])


if __name__ == "__main__":
    main()
