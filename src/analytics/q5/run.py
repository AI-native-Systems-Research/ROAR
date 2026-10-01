#!/usr/bin/env python3
"""Regenerate every RQ5 artifact from the database in one command.

    PYTHONPATH=/ws/src ./.venv/bin/python -m analytics.q5.run OUTDIR --workers 6

Writes to OUTDIR:
    q5_evaluation.json   per-cell rows and the across-cell test
    q5_per_problem.json  per-problem gain, campaign- and cell-weighted
    q5_curves.json       k-sweep and n-curve diagnostics
    q5_gain_forest.png, q5_k_sweep.png, q5_n_curve.png
    run.log              settings, timings and the headline numbers

The budget is counted in iterations and a launch costs exactly the iterations
it runs; nothing else is charged. match_mean is observational: it records when
the policy drew level inside a full-budget replay, and charges nothing.

Figures are rendered from the saved dicts rather than recomputed, so they
cannot disagree with the JSON.
"""

import argparse
import json
import os
import statistics
import time

from analytics.q5.evaluate import (
    DEFAULT_REFIT_BOOTSTRAP,
    get_evaluation,
    get_per_problem,
)
from analytics.q5.figures import (
    generate_gain_forest_figure,
    generate_k_sweep_figure,
    generate_n_curve_figure,
    get_diagnostic_curves,
)
from analytics.q5.recommend import (
    BUDGET_DEFAULT,
    DEFAULT_SEED,
    DEFAULT_TRIALS,
    eligible_problems,
    enumerate_cells,
    load_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget", type=int, default=BUDGET_DEFAULT)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_REFIT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_file = open(os.path.join(args.outdir, "run.log"), "w")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    def write_json(name: str, payload) -> None:
        with open(os.path.join(args.outdir, name), "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

    started = time.time()
    log(
        f"cost model: iterations only | budget={args.budget} "
        f"n_trials={args.n_trials} n_boot={args.n_boot} seed={args.seed} "
        f"workers={args.workers}"
    )

    # min_iterations is named, not positional: it equals the budget at the
    # default, but a non-default --budget must not silently change which
    # campaigns qualify.
    trajectories = load_trajectories(None, min_iterations=args.budget)
    problems = eligible_problems(trajectories)
    cells = enumerate_cells(trajectories, problems=problems)
    evaluated = [c for c in cells if c["usable"] or c["saturated"] or c["dead"]]
    log(
        f"{len(trajectories)} trajectories, {len(problems)} problems, "
        f"{len(evaluated)} cells"
    )

    step = time.time()
    evaluation = get_evaluation(
        budget=args.budget, n_trials=args.n_trials, seed=args.seed,
        n_boot=args.n_boot, trajectories=trajectories, workers=args.workers,
    )
    write_json("q5_evaluation.json", evaluation)
    across = evaluation["across_cells"]
    log(
        f"evaluation ({(time.time() - step) / 60:.1f} min): median "
        f"{across['median_gain']:+.2f}, {across['n_positive']}/"
        f"{across['n_cells']} positive, sign test p={across['sign_test_p']}"
    )
    matched = [
        r for r in evaluation["cells"] if r["match_mean"] is not None
    ]
    if matched:
        log(
            f"  match: policy drew level with the practitioner's own final "
            f"score at a median (over {len(matched)} of "
            f"{len(evaluation['cells'])} cells) of "
            f"{statistics.median(r['match_mean'] for r in matched):.2f} of "
            f"{args.budget}, at a median replay rate of "
            f"{statistics.median(r['match_rate'] for r in matched):.3f}; "
            f"{sum(r['n_baseline_never_improved'] for r in evaluation['cells'])}"
            f" campaigns excluded as never improving on their seed"
        )

    step = time.time()
    per_problem = get_per_problem(
        budget=args.budget, n_boot=args.n_boot, n_trials=args.n_trials,
        seed=args.seed, trajectories=trajectories, workers=args.workers,
    )
    write_json("q5_per_problem.json", per_problem)
    log(f"per-problem ({(time.time() - step) / 60:.1f} min):")
    for key, row in per_problem.items():
        log(
            f"  {key}: {row['mean']:+.2f} [{row['ci_low']:+.2f}, "
            f"{row['ci_high']:+.2f}] boot_p={row['boot_p']}"
        )

    step = time.time()
    curves = get_diagnostic_curves(
        budget=args.budget, n_trials=args.n_trials, seed=args.seed,
        trajectories=trajectories,
    )
    write_json("q5_curves.json", curves)
    figures = {
        "q5_gain_forest.png": generate_gain_forest_figure(evaluation),
        "q5_k_sweep.png": generate_k_sweep_figure(curves),
        "q5_n_curve.png": generate_n_curve_figure(curves),
    }
    for name, payload in figures.items():
        with open(os.path.join(args.outdir, name), "wb") as handle:
            handle.write(payload)
    log(f"curves and figures ({(time.time() - step) / 60:.1f} min)")

    log(f"all done in {(time.time() - started) / 60:.1f} min -> {args.outdir}")
    log_file.close()


if __name__ == "__main__":
    main()
