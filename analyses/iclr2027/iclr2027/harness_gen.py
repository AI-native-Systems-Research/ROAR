"""Trajectories of the harness-generalization runs of Gupta et al. (2026) in ROAR's format.

Translation (as in the earlier calibration of ROAR's RQ4 procedure on these runs):
  * one iteration = one search step; its score is the maximum over the step's children;
  * a step with no scored child scores 0 (kept as a non-improving iteration);
  * scores are multiplied by 100 (0-100 scale, like the submitted corpus);
  * only runs with at least 50 steps are kept (the RQ4 budget);
  * configuration = (task, search method, model): the search method plays the role of
    ROAR's `algorithm_used`.
campaign_id is the run's parquet path relative to the corpus' data/ directory.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from . import paths

SCALE = 100.0
COLS = ["task", "model", "search_method", "step", "evolved_combined_score"]

TASK_LABEL = {"circle_packing": "Circle packing", "heilbronn_triangle": "Heilbronn triangle",
              "second_autocorr_ineq": "Second autocorrelation inequality"}
METHOD_LABEL = {"sequential": "Sequential", "uct": "UCT", "puct": "PUCT"}
MODEL_LABEL = {"Qwen2.5-3B-Instruct": "Qwen2.5-3B", "Qwen3-4B-Instruct-2507": "Qwen3-4B",
               "gpt-oss-20b": "GPT-OSS-20B", "gpt-oss-120b": "GPT-OSS-120B"}


def _one(args):
    root, rel = args
    t = pd.read_parquet(os.path.join(root, rel), columns=COLS)
    s = t.groupby("step", sort=True)["evolved_combined_score"].max().fillna(0.0) * SCALE
    first = t.iloc[0]
    return {"campaign_id": rel.split("data/", 1)[1] if rel.startswith("data/") else rel,
            "task": first.task, "model": first.model, "search_method": first.search_method,
            "steps": [int(i) for i in s.index], "scores": [float(v) for v in s.to_numpy()]}


def load_runs(root=None, workers=8):
    root = root or paths.HARNESS_GEN_DATA
    meta = pd.read_parquet(os.path.join(root, "metadata", "runs.parquet"))
    with ProcessPoolExecutor(workers) as ex:
        runs = list(ex.map(_one, [(root, p) for p in meta.parquet_path], chunksize=20))
    return meta, runs


def trajectories(runs, min_steps=50):
    from analytics.q5.recommend import running_best
    out = []
    for r in runs:
        if len(r["steps"]) < min_steps:
            continue
        its = [{"index": i, "score": v} for i, v in zip(r["steps"], r["scores"])]
        out.append({"campaign_id": r["campaign_id"], "problem": r["task"], "research_question": r["task"],
                    "algorithm": r["search_method"], "model": r["model"], "models_used": [r["model"]],
                    "best_scores": running_best(its), "n_iterations": len(its)})
    return sorted(out, key=lambda t: t["campaign_id"])
