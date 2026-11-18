"""RQ4 (Section 4.5, Table 1, Appendix K): budget-allocation rules scored on held-out campaigns.

Protocol (Section 4.5, Appendix K.1), per tested configuration and per split r = 0..R-1:
  * A, B = ROAR split_calibration_evaluation(configuration's campaigns, seed=r):
    fitting half A = floor(N/2) campaigns, evaluation half B = the rest;
  * the fitted policy is ROAR estimate_policy(A, budget=50, seed=42) (default grid);
  * every campaign c of B (external corpus: the first 60 of B in split order) is scored
    against one uninterrupted 50-iteration campaign, baseline_score(c, 50), with
    policy_score(c, pool = B minus c, 50, k_stop, n, n_trials=500, seed=42 + index of c in
    the configuration). The same seed is used for every rule, so all rules see the same
    relaunch draws;
  * a configuration's gain in split r is the mean over its scored campaigns.
Rules: 2x25 (n=2), 5x10 (n=5), restart after j non-improving iterations for j = 5, 10, 20
(n=1, k_stop = j+1 under ROAR's stagnation counter, which is 1 on the iteration that set
the best), and the fitted (k_stop, n). With --counter-variants the rules k_stop = 5/10/20
are also scored (sensitivity for the counter convention).

Tested configurations: ROAR enumerate_cells "headroom" cells (>= 8 campaigns of >= 50
iterations, population SD of final best >= 2) on ROAR's eligible problems (our corpus),
or on all three tasks (external corpus).

usage: python -I scripts/rq4_budget_allocation.py --corpus roar|external [--splits R] [--workers W]
Writes results/rq4_<corpus>.json (summary) and results/rq4_<corpus>_splits.json.gz (per split).
"""
import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import paths, parquet_backend  # noqa: E402

from analytics.q5.evaluate import baseline_score, policy_score  # noqa: E402
from analytics.q5.recommend import (  # noqa: E402
    DEFAULT_SEED, DEFAULT_TRIALS, eligible_problems, enumerate_cells, estimate_policy,
    load_trajectories, split_calibration_evaluation,
)
from analytics.utils import display_mechanism  # noqa: E402

BUDGET = 50
RULES = [("2x25", None, 2), ("5x10", None, 5), ("Restart after 5", 6, 1),
         ("Restart after 10", 11, 1), ("Restart after 20", 21, 1)]
COUNTER_VARIANTS = [("k_stop=5", 5, 1), ("k_stop=10", 10, 1), ("k_stop=20", 20, 1)]
FITTED = "Fitted (k_stop, n)"
TABLE_ROWS = [r[0] for r in RULES] + [FITTED]

_S = {}


def _init(state):
    _S.update(state)


def _task(task):
    ci, r = task
    members = _S["members"][ci]
    index = {t["campaign_id"]: i for i, t in enumerate(members)}
    A, B = split_calibration_evaluation(members, holdout=True, seed=r)
    fit = estimate_policy(A, budget=BUDGET, seed=DEFAULT_SEED)
    rules = list(_S["rules"]) + [(FITTED, fit["k_stop"], fit["n_replicates"])]
    scored = B[:_S["max_scored"]] if _S["max_scored"] else B
    gains = {name: [] for name, _, _ in rules}
    for c in scored:
        seed = DEFAULT_SEED + index[c["campaign_id"]]
        pool = [t for t in B if t["campaign_id"] != c["campaign_id"]]
        base = baseline_score(c, BUDGET)
        for name, k, n in rules:
            s = policy_score(c, pool, BUDGET, k, n, n_trials=DEFAULT_TRIALS, seed=seed)["mean_score"]
            gains[name].append(s - base)
    return {"cell": ci, "r": r, "nA": len(A), "nB": len(B), "n_scored": len(scored),
            "k_stop": fit["k_stop"], "n_rep": fit["n_replicates"],
            "gain": {name: float(np.mean(v)) for name, v in gains.items()}}


def load(corpus):
    if corpus == "roar":
        parquet_backend.install("submitted")
        traj = load_trajectories(None, min_iterations=BUDGET)
        problems = eligible_problems(traj)
        cells = enumerate_cells(traj, problems=problems)
        return traj, problems, cells, None
    from iclr2027 import harness_gen
    meta, runs = harness_gen.load_runs()
    traj = harness_gen.trajectories(runs, min_steps=BUDGET)
    cells = enumerate_cells(traj)
    return traj, sorted({t["problem"] for t in traj}), cells, {"runs_in_corpus": int(len(meta)),
                                                               "runs_with_50_steps": len(traj)}


def label(corpus, c):
    if corpus == "roar":
        from analytics.q5.paper_figures import PROBLEMS, _generator
        return "%s: %s / %s" % (PROBLEMS.get(c["problem"], c["problem"]), display_mechanism(c["algorithm"]),
                                _generator(c["model"]))
    from iclr2027.harness_gen import METHOD_LABEL, MODEL_LABEL, TASK_LABEL
    return "%s: %s / %s" % (TASK_LABEL.get(c["problem"], c["problem"]), METHOD_LABEL.get(c["algorithm"], c["algorithm"]),
                            MODEL_LABEL.get(c["model"], c["model"]))


def display_model(corpus, m):
    """Display name without provider prefix (our corpus); the external corpus has none."""
    if corpus == "roar":
        from analytics.q5.paper_figures import _generator
        return _generator(m)
    return m


def summarize(corpus, cells, recs, rule_names):
    per = []
    for ci, c in enumerate(cells):
        rs = sorted([x for x in recs if x["cell"] == ci], key=lambda x: x["r"])
        row = {"label": label(corpus, c), "problem": c["problem"], "algorithm": c["algorithm"],
               "model": display_model(corpus, c["model"]),
               "n_campaigns": c["n_campaigns"], "sd_final": c["sd_final"], "min_final": c["min_final"],
               "max_final": c["max_final"], "nA": rs[0]["nA"], "nB": rs[0]["nB"], "n_scored": rs[0]["n_scored"],
               "rules": {}}
        for name in rule_names:
            g = np.array([x["gain"][name] for x in rs])
            row["rules"][name] = {"mean": round(float(g.mean()), 4), "sd_across_splits": round(float(g.std(ddof=1)), 4),
                                  "p2_5": round(float(np.percentile(g, 2.5)), 4),
                                  "p97_5": round(float(np.percentile(g, 97.5)), 4),
                                  "frac_splits_gt0": round(float(np.mean(g > 1e-9)), 4)}
        ks = [str((x["k_stop"], x["n_rep"])) for x in rs]
        row["fitted_policy_counts"] = {k: ks.count(k) for k in sorted(set(ks))}
        per.append(row)
    means = np.array([[p["rules"][n]["mean"] for n in TABLE_ROWS] for p in per])
    best = np.argmax(means, axis=1)          # ties: first rule in table order
    table = {}
    for j, name in enumerate(TABLE_ROWS):
        table[name] = {"positive": int(np.sum(means[:, j] > 0)), "negative": int(np.sum(means[:, j] < 0)),
                       "best": int(np.sum(best == j)), "mean": round(float(means[:, j].mean()), 2)}
    for p, b in zip(per, best):
        p["best_rule"] = TABLE_ROWS[b]
    extra = {}
    for name in rule_names:
        if name not in TABLE_ROWS:
            m = np.array([p["rules"][name]["mean"] for p in per])
            extra[name] = {"positive": int(np.sum(m > 0)), "mean": round(float(m.mean()), 2)}
    sds = {name: round(float(np.median([p["rules"][name]["sd_across_splits"] for p in per])), 2) for name in rule_names}
    return per, table, extra, sds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["roar", "external"], required=True)
    ap.add_argument("--splits", type=int, default=4 if paths.QUICK else 200)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    ap.add_argument("--max-scored", type=int, default=None,
                    help="cap on scored evaluation campaigns per split (default: none for roar, 60 for external)")
    ap.add_argument("--counter-variants", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    max_scored = a.max_scored if a.max_scored is not None else (60 if a.corpus == "external" else 0)
    t0 = time.time()
    traj, problems, all_cells, meta = load(a.corpus)
    enough = [c for c in all_cells if c["n_campaigns"] >= 8]
    cells = [c for c in enough if c["headroom"]]
    slim = [{k: t[k] for k in ("campaign_id", "problem", "algorithm", "model", "best_scores")} for t in traj]
    members = {}
    for ci, c in enumerate(cells):
        key = (c["problem"], c["algorithm"], c["model"])
        members[ci] = [t for t in slim if (t["problem"], t["algorithm"], t["model"]) == key]
    rules = RULES + (COUNTER_VARIANTS if a.counter_variants else [])
    rule_names = [r[0] for r in rules] + [FITTED]
    print(f"{a.corpus}: {len(traj)} trajectories, {len(enough)} configurations with >= 8 campaigns, "
          f"{len(cells)} tested; {a.splits} splits", flush=True)
    state = {"members": members, "rules": rules, "max_scored": max_scored}
    tasks = [(ci, r) for ci in members for r in range(a.splits)]
    # largest configurations first, for load balance
    tasks.sort(key=lambda t: -len(members[t[0]]))
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(state,)) as ex:
        recs = list(ex.map(_task, tasks, chunksize=1))
    per, table, extra, sds = summarize(a.corpus, cells, recs, rule_names)
    excluded = [{"label": label(a.corpus, c), "n_campaigns": c["n_campaigns"], "sd_final": c["sd_final"],
                 "min_final": c["min_final"], "max_final": c["max_final"]} for c in enough if not c["headroom"]]
    res = {
        "corpus": a.corpus,
        "protocol": {"budget": BUDGET, "n_trials": DEFAULT_TRIALS, "fit_seed": DEFAULT_SEED,
                     "split_seeds": f"0..{a.splits - 1}", "n_splits": a.splits,
                     "max_scored_per_split": max_scored or None,
                     "relaunch_pool": "evaluation half minus the scored campaign (not capped)",
                     "rules": {name: {"k_stop": k, "n": n} for name, k, n in rules},
                     "replay_seed": "42 + index of the scored campaign in its configuration (sorted by campaign_id)"},
        "corpus_meta": meta, "problems": problems,
        "n_configurations_with_8": len(enough), "n_tested": len(cells),
        "table1": table, "counter_variants": extra,
        "median_sd_across_splits": sds, "per_configuration": per, "excluded_configurations": excluded,
        "runtime_seconds": round(time.time() - t0, 1), "quick": paths.QUICK,
    }
    tag = a.corpus + (a.tag or "")
    with open(paths.out(paths.RESULTS, f"rq4_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=False)
    blob = json.dumps(sorted(recs, key=lambda x: (x["cell"], x["r"]))).encode()
    with gzip.GzipFile(paths.out(paths.RESULTS, f"rq4_{tag}_splits.json.gz"), "wb", mtime=0) as f:
        f.write(blob)   # mtime=0: byte-identical across runs
    print(json.dumps(table, indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
