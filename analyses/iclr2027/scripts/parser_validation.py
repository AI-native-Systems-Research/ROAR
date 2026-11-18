"""Appendix D: parsed records vs the raw run directories, read by an independent reader.

The reader shares no code with ROAR's parsers. For each campaign it opens
<ROAR_RAW_DIR>/<campaign name>/frontier_cs/problem_*/checkpoints/checkpoint_N/programs/*.json,
keeps each program from the numerically latest checkpoint that holds it, and drops
migration / island-spawn copies (which ROAR's SkyDiscover parser excludes by design).
Compared with the mirror: candidate id sets (both directions), each candidate's score
(|d| <= 1e-9 relative) and iteration index, the campaign best score and the campaign's
maximum iteration index. Evaluator-log lines ("Evaluated program <id> ...: combined_score=x",
printed to 4 decimals) are a second check where present.

Campaign set: the submitted campaigns on the three combinatorial problems whose name equals
a run directory name in the run database RUNS_DB (sqlite, table runs, column
dir_name; opened read-only) -- 911 campaigns. Without RUNS_DB, every submitted
campaign with a raw directory is used and the result is labelled accordingly.
Needs ROAR_RAW_DIR. Deterministic; no random numbers.
"""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import paths, parquet_backend as pb, results_io  # noqa: E402

LOGRE = re.compile(r"Evaluated program ([0-9a-f\-]{36}) in [0-9.]+s: combined_score=(-?[0-9.]+|nan|inf)")
PROBLEMS = ("Bounded 2D Knapsack", "Polynomino Packing", "Palindrome Hamiltonian Path")


def raw_programs(pdir):
    progs = {}
    ck = pdir / "checkpoints"
    if not ck.is_dir():
        return progs
    cks = sorted(((int(m.group(1)), c) for c in ck.iterdir()
                  for m in [re.fullmatch(r"checkpoint_(\d+)", c.name)] if m and c.is_dir()), reverse=True)
    for _, c in cks:
        d = c / "programs"
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            if f.name.startswith("._") or f.stem in progs:
                continue
            try:
                o = json.loads(f.read_text())
            except Exception:
                progs[f.stem] = {"iter": None, "score": None, "copy": False}
                continue
            md = o.get("metadata") or {}
            progs[f.stem] = {"iter": o.get("iteration_found", o.get("iteration")),
                             "score": (o.get("metrics") or {}).get("combined_score"),
                             "copy": md.get("migrated_from") is not None or md.get("seeded_to_spawned_island") is not None}
    return progs


def log_scores(pdir):
    out = {}
    for lf in sorted((pdir / "logs").glob("*.log")) if (pdir / "logs").is_dir() else []:
        with open(lf, errors="replace") as fh:
            for line in fh:
                if "Evaluated program" in line:
                    m = LOGRE.search(line)
                    if m:
                        out.setdefault(m.group(1), []).append(float(m.group(2)))
    return out


def main():
    raw = Path(paths.ROAR_RAW_DIR)
    if not paths.ROAR_RAW_DIR or not raw.is_dir():
        sys.exit("set ROAR_RAW_DIR to the raw run directories")
    t = pb.tables()
    camp = t["campaigns"]
    camp = camp[camp.research_question.isin(PROBLEMS)]
    if paths.RUNS_DB:
        con = sqlite3.connect(f"file:{paths.RUNS_DB}?mode=ro", uri=True)
        names = set(pd.read_sql("select dir_name from runs", con).dir_name)
        con.close()
        camp = camp[camp.name.isin(names)]
        selection = "name matched to a run directory in the run database"
    else:
        camp = camp[camp.name.map(lambda n: (raw / n).is_dir())]
        selection = "every campaign with a raw directory (no run database given)"
    cand = t["candidates"][t["candidates"].campaign_id.isin(set(camp.id))]
    cand = cand.merge(pd.read_parquet(os.path.join(paths.ROAR_DATA, "candidates.parquet"), columns=["id", "external_id"]), on="id")
    sc = t["measurements"]
    sc = sc[sc.name == "combined_score"].assign(score=lambda d: pd.to_numeric(d.value, errors="coerce"))
    cand = cand.merge(sc.groupby("candidate_id", as_index=False).score.max(), left_on="id", right_on="candidate_id", how="left")
    by = {cid: g for cid, g in cand.groupby("campaign_id")}
    tot = {"campaigns": 0, "candidates": 0, "score_exact": 0, "iter_exact": 0, "with_log": 0, "log_agree": 0,
           "id_sets_identical": 0, "best_equal": 0, "max_iter_equal": 0, "no_single_problem_dir": 0}
    bad = []
    for r in camp.sort_values("name").itertuples():
        pd_ = list((raw / r.name).glob("frontier_cs/problem_*"))
        if len(pd_) != 1:
            tot["no_single_problem_dir"] += 1
            continue
        progs, logs = raw_programs(pd_[0]), log_scores(pd_[0])
        c = by.get(r.id, cand.iloc[:0])
        tot["campaigns"] += 1
        tot["candidates"] += len(c)
        ok = True
        for cr in c.itertuples():
            p = progs.get(cr.external_id)
            if p is not None:
                rs = 0.0 if p["score"] is None else float(p["score"])
                se = not pd.isna(cr.score) and abs(rs - cr.score) <= 1e-9 + 1e-9 * abs(cr.score)
                tot["score_exact"] += se
                tot["iter_exact"] += p["iter"] == cr.iteration_index
                ok &= se and p["iter"] == cr.iteration_index
            ls = logs.get(cr.external_id)
            if ls:
                tot["with_log"] += 1
                tot["log_agree"] += any(abs(x - cr.score) <= 5.1e-5 + 1e-9 * abs(cr.score) for x in ls)
        noncopy = {k: v for k, v in progs.items() if not v["copy"]}
        ids_ok = set(noncopy) <= set(c.external_id) <= set(progs)
        rsc = [float(v["score"]) for v in noncopy.values() if isinstance(v["score"], (int, float))]
        best_ok = bool(rsc) and abs(max(rsc) - (c.score.max() if c.score.notna().any() else 0.0)) <= 1e-6 * (1 + max(rsc))
        it = [v["iter"] for v in noncopy.values() if isinstance(v["iter"], int)]
        it_ok = bool(it) and max(it) == c.iteration_index.max()
        tot["id_sets_identical"] += ids_ok
        tot["best_equal"] += best_ok
        tot["max_iter_equal"] += it_ok
        if not (ok and ids_ok and best_ok and it_ok):
            bad.append({"problem": r.research_question, "n_candidates": int(len(c))})
    res = {"selection": selection, **{k: int(v) for k, v in tot.items()}, "campaigns_with_any_disagreement": len(bad),
           "disagreements": bad}
    results_io.write_result("parser_validation.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}))


if __name__ == "__main__":
    main()
