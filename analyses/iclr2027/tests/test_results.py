"""Check the committed results against (1) the values printed in the paper and (2) the outputs
of the original, independently verified analyses (tests/reference_values.json).

Pure file comparison: runs in seconds, needs no data. A result file that is absent is
reported as SKIP (e.g. parser_validation.json needs the raw run directories).
Values that the paper prints differently from what the code produces are listed in KNOWN
and reported as KNOWN-DIFF rather than failures (see README, "Differences").
usage: python -I tests/test_results.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("ICLR_RESULTS_DIR", os.path.join(HERE, "..", "results"))
REF = json.load(open(os.path.join(HERE, "reference_values.json")))
fails = []


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def check(name, got, want, tol=0.0, known=False):
    ok = (abs(got - want) <= tol + 1e-12) if isinstance(want, (int, float)) and not isinstance(want, bool) else got == want
    tag = "PASS" if ok else ("KNOWN-DIFF" if known else "FAIL")
    print(f"{tag:10s} {name}: got {got!r}, want {want!r}" + (f" (tol {tol})" if tol else ""))
    if not ok and not known:
        fails.append(name)


# ---------------------------------------------------------------- Table 4, Section 4.1
r = load("corpus_provenance.json")
if r:
    a, b = r["table4_at_submission"], r["table4_after_submission"]
    for k, v in {"Ellis & Castro (2026)": (914, 98.5), "Independent researchers": (14, 1.5)}.items():
        check(f"T4 at submission {k}", (a[k]["campaigns"], a[k]["share_pct"]), v)
    for k, v in {"Ellis & Castro (2026)": (914, 21.1), "Independent researchers": (14, 0.3),
                 "Cheng et al. (2025b)": (144, 3.3), "Gupta et al. (2026)": (3269, 75.3)}.items():
        check(f"T4 after submission {k}", (b[k]["campaigns"], b[k]["share_pct"]), v)
    check("T4 total after submission", b["Total"], 4341)
    check("4.1 corpus campaigns / candidates", (r["submitted_campaigns"], r["submitted_candidates"]), (928, 48210))
    pal = r["per_problem_submitted"]["Palindrome"]
    check("4.1 Palindrome campaigns all final score 0", (pal["campaigns"], pal["campaigns_with_final_score_zero"]), (290, 290))

# ---------------------------------------------------------------- RQ1, App. G
r = load("rq1_failed_campaigns.json")
if r:
    check("App G failed campaigns", r["n_failed_campaigns"], 1)
    f = r["failed_campaigns"][0]
    check("App G failed campaign problem / candidates / judge failures",
          (f["problem"], f["n_candidates"], f["n_judge_infra_failures"]), ("Bounded 2D Knapsack", 30, 30))
    check("App G failed campaign's configuration has < 10 runs", f["configuration_n"] < 10, True)
    check("App G exclusion leaves RQ1 cells unchanged", r["rq1_cells_unchanged_by_exclusion"], True)
    c = r["rq1_counts_excluding_failed"]
    check("4.2 Polynomino 14 configurations, 7 confirmed, 6 inconclusive, 1 unimodal",
          tuple(c["Polynomino Packing"][k] for k in ("configurations", "confirmed_bimodal", "underpowered", "unimodal")), (14, 7, 6, 1))
    check("4.2 Knapsack 10 configurations, 5 confirmed, 3 inconclusive, 2 deterministic",
          tuple(c["Bounded 2D Knapsack"][k] for k in ("configurations", "confirmed_bimodal", "underpowered", "deterministic")), (10, 5, 3, 2))
    check("4.2 12 of 24 confirmed", (c["both"]["confirmed_bimodal"], c["both"]["configurations"]), (12, 24))
    short = {"Bounded 2D Knapsack": "Knapsack", "Polynomino Packing": "Polynomino"}
    got = {(short[x["research_question"]], x["models"], x["algorithm"]): x for x in r["rq1_cells_excluding_failed"]}
    for x in REF["rq1_v1_cells"]:
        g = got[(x["problem"], x["models"], x["algorithm"])]
        lab = f"RQ1 cell {x['problem']} {x['models']}/{x['algorithm']}"
        check(lab + " (n, BC) vs original", (g["n"], g["bc"]), (x["n"], x["bc"]))
        # ROAR's BC bootstrap depends on the order of a configuration's campaigns, which q1's SQL
        # leaves unspecified; the parquet backend sorts by campaign id, so its intervals differ.
        check(lab + " (CI, class) vs original", (g["bc_ci_lower"], g["bc_ci_upper"], g["classification"]),
              (x["bc_lo"], x["bc_hi"], x["class"]), known=r["backend"] != "postgres")

# ---------------------------------------------------------------- RQ2
r = load("rq2_escape.json")
if r:
    for pool, (o, lo, hi) in REF["rq2"]["or"].items():
        t = r["trend"][pool]
        check(f"RQ2 OR per 10 ({pool}) vs original", (t["odds_ratio_per_10k"], t["or_ci_lower"], t["or_ci_upper"]), (o, lo, hi))
    t = r["trend"]["paper_three"]
    check("4.3 OR 1.00 [0.95, 1.06]", (round(t["odds_ratio_per_10k"], 2), round(t["or_ci_lower"], 2), round(t["or_ci_upper"], 2)),
          (1.0, 0.95, 1.06))
    pp = r["per_problem"]
    kn = [x["rate"] * 100 for x in pp["Bounded 2D Knapsack"]]
    check("4.3 Knapsack 4-9% at k=10..40", (round(min(kn)), round(max(kn))), (4, 9))
    po = [x["rate"] * 100 for x in pp["Polynomino Packing"][:3]]
    check("4.3 Polynomino 13-14% (k=10..30; k=40 is 0 of 87)", (round(min(po)), round(max(po))), (13, 14), known=True)
    check("4.3 CUDA Attention 40% at k=30", round(pp["CUDA Attention kernel optimization"][2]["rate"] * 100), 40)
    rates = [x["rate"] for x in r["pooled"]["paper_three"] if "rate" in x]
    check("4.3 pooled rate below 1/6 at every k", max(rates) < 1 / 6, True)

# ---------------------------------------------------------------- RQ3
r = load("rq3_length_control.json")
if r:
    pg = r["backend"] == "postgres"
    tol = 1e-4 if pg else 0.006
    for p, ref in REF["rq3_glmm_with_context_length_postgres"].items():
        got = r["glmm"][p]["with_context_length"]["coef"]
        for k, v in ref["coef"].items():
            check(f"RQ3 GLMM {p} {k} vs original (backend {r['backend']})", got[k], v, tol)
    ed = r["early_diversity"]
    for p, ref in REF["rq3_early_partial"].items():
        check(f"RQ3 early partial rho {p} vs original", ed[p]["problem"]["rho_partial_on_tokens"], ref["rho"], 1e-3)
    check("4.4 Polynomino rho 0.16, p 0.004", (round(ed["Polynomino"]["problem"]["rho_partial_on_tokens"], 2),
                                                round(ed["Polynomino"]["problem"]["p"], 3)), (0.16, 0.004))
    check("4.4 Knapsack rho 0.04, p 0.51", (round(ed["Knapsack"]["problem"]["rho_partial_on_tokens"], 2),
                                             round(ed["Knapsack"]["problem"]["p"], 2)), (0.04, 0.51))
    check("4.4 CUDA rho -0.50, p 0.667", (ed["CUDA Attention"]["problem"]["rho_plain"],
                                           round(ed["CUDA Attention"]["problem"]["p_plain"], 3)), (-0.5, 0.667))
    po = ed["Polynomino"]
    for grp, k, rho, pv in (("by_mechanism", "Best-of-N", 0.41, None), ("by_model", "eb1-preview", 0.39, 0.001),
                            ("by_model", "Kimi-K2.5", 0.36, 0.031), ("by_mechanism", "EvoX", -0.12, 0.31),
                            ("by_model", "GPT-5.4", -0.39, 0.072)):
        x = po[grp][k]
        check(f"App I Polynomino {k} rho", round(x["rho_partial_on_tokens"], 2), rho)
        if pv is not None:
            check(f"App I Polynomino {k} p", round(x["p"], 3 if pv < 0.1 else 2), pv)
    check("App I Polynomino Best-of-N p < 0.001", po["by_mechanism"]["Best-of-N"]["p"] < 0.001, True)
    g = {p: r["glmm"][p]["with_context_length"] for p in r["glmm"]}
    printed = {("Knapsack", "score_diversity"): -0.13, ("Knapsack", "worse_ratio"): 0.13,
               ("Knapsack", "context_diversity"): 0.14, ("Knapsack", "better_ratio"): 0.02,
               ("Polynomino", "worse_ratio"): 0.17, ("Polynomino", "better_ratio"): 0.10,
               ("Polynomino", "score_diversity"): 0.10,
               ("CUDA Attention", "better_ratio"): -0.48, ("CUDA Attention", "context_diversity"): 0.35}
    known = {("Knapsack", "score_diversity"), ("Knapsack", "context_diversity"), ("CUDA Attention", "better_ratio"),
             ("CUDA Attention", "context_diversity")}
    if not pg:   # tied-iteration ordering in the running best (see parquet_backend)
        known |= {("Knapsack", "better_ratio"), ("Polynomino", "worse_ratio")}
    for (p, k), v in printed.items():
        check(f"4.4 {p} {k} printed {v:+.2f}", round(g[p]["coef"][k], 2), v, known=(p, k) in known)
    check("4.4 Knapsack better ratio p 0.72", round(g["Knapsack"]["p"]["better_ratio"], 2), 0.72, known=not pg)

# ---------------------------------------------------------------- RQ4
T1 = {"roar": {"2x25": (12, 2, 9.2), "5x10": (11, 2, 12.4), "Restart after 5": (10, 5, 12.2),
               "Restart after 10": (11, 1, 11.7), "Restart after 20": (12, 1, 9.5), "Fitted (k_stop, n)": (10, 1, 8.2)},
      "external": {"2x25": (24, 3, 2.8), "5x10": (19, 2, 1.4), "Restart after 5": (21, 12, 2.0),
                   "Restart after 10": (27, 6, 3.9), "Restart after 20": (26, 2, 3.1), "Fitted (k_stop, n)": (21, 2, 2.8)}}
for corpus, rows in T1.items():
    r = load(f"rq4_{corpus}.json")
    if not r:
        print(f"SKIP       rq4_{corpus}.json absent")
        continue
    full = r["protocol"]["n_splits"] == 200
    check(f"T1 {corpus}: tested configurations", r["n_tested"], 12 if corpus == "roar" else 27)
    for name, (pos, best, mean) in rows.items():
        t = r["table1"][name]
        check(f"T1 {corpus} {name} (positive, best, mean)", (t["positive"], t["best"], round(t["mean"], 1)),
              (pos, best, mean), known=not full or corpus == "external")
    if corpus == "roar" and full:
        for p in r["per_configuration"]:
            check(f"RQ4 fitted mean gain {p['label']} vs original", p["rules"]["Fitted (k_stop, n)"]["mean"],
                  REF["rq4_roar_fitted_v1_mean_by_configuration"][p["label"]], 1e-3)
        sd = r["median_sd_across_splits"]
        check("K.2 fitted median SD across splits 8.8", round(sd["Fitted (k_stop, n)"], 1), 8.8, 0.051)
        fx = [sd[k] for k in ("2x25", "5x10", "Restart after 5", "Restart after 10", "Restart after 20")]
        check("K.2 fixed-rule median SD 2.5-6.7", (round(min(fx), 1), round(max(fx), 1)), (2.5, 6.7), known=True)
        ex = r["excluded_configurations"]
        check("K.2 17 with enough campaigns, 5 excluded with sd <= 1.3",
              (r["n_configurations_with_8"], len(ex), max(e["sd_final"] for e in ex) <= 1.3), (17, 5, True))
        check("K.2 four excluded within a point of 100", sum(e["min_final"] >= 99 for e in ex), 4)
        check("K.2 every tested sd >= 8", min(p["sd_final"] for p in r["per_configuration"]) >= 8, True)

# ---------------------------------------------------------------- App. D
r = load("parser_validation.json")
if r:
    v = REF["parser_validation"]
    check("App D campaigns / candidates (911 / 47,168)", (r["campaigns"], r["candidates"]), (v["n"], v["candidates_compared"]))
    check("App D scores and iteration indices identical", (r["score_exact"], r["iter_exact"]),
          (v["cand_score_exact"], v["cand_iter_exact"]))
    check("App D id sets, best score, iteration count per campaign",
          (r["id_sets_identical"], r["best_equal"], r["max_iter_equal"]),
          (v["campaigns_id_sets_identical"], v["roar_best_eq_raw_best"], v["roar_maxiter_eq_raw_maxiter"]))
else:
    print("SKIP       parser_validation.json absent (needs ROAR_RAW_DIR and GAMBLE_RUNS_DB)")

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
