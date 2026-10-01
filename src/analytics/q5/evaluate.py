#!/usr/bin/env python3
"""Q5 headline experiment: does ROAR's policy beat what practitioners did?

The claim under test is that ROAR, given only a research question and a budget,
returns a budget-allocation policy fit on other people's runs that beats one
full-length run.

Three properties define the comparison (see README sections 5 and 7b):

1. Cell-scoped -- evidence comes from the caller's own (problem, algorithm,
   model) cell.
2. Paired and leave-one-out -- each held-out campaign is scored against its own
   counterfactual, with the policy refit on the remaining n-1, so a
   recommendation is never scored on the campaign that produced it.
3. Uncertainty comes from resampling campaigns, not replay trials.

Module structure follows the package convention: process_* transforms,
get_* combines query + process.
"""

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

from analytics.q5.recommend import (
    BUDGET_DEFAULT,
    CELL_MIN_CAMPAIGNS,
    DEFAULT_SEED,
    DEFAULT_TRIALS,
    MATCH_TOLERANCE,
    eligible_problems,
    enumerate_cells,
    estimate_policy,
    load_trajectories,
    simulate,
)

# Replicates for the per-cell refit bootstrap, enough for stable 2.5/97.5
# percentiles.
DEFAULT_REFIT_BOOTSTRAP = 1000

# The per-problem bootstrap resamples every cell once per draw from its own
# generator, seeded at this offset plus the draw index, so draws are
# independent of the per-cell intervals' streams and of worker scheduling.
PER_PROBLEM_RESAMPLE_SEED = 10_000


# =============================================================================
# Scoring primitives
# =============================================================================

def baseline_score(trajectory: dict[str, Any], budget: int) -> float:
    """What this practitioner actually got: one run of `budget` iterations.

    See README section 4 for why one full run with no early stopping is the
    faithful comparator.
    """
    best_scores = trajectory["best_scores"]
    return float(best_scores[min(budget, len(best_scores)) - 1])


def baseline_never_improved(trajectory: dict[str, Any], budget: int) -> bool:
    """Did this run end on the score it started with?

    Such campaigns are matched at iteration 1 by construction, because the
    policy's first launch IS this run, so they are excluded from the match
    statistics rather than averaged in. See README section 4.3.
    """
    best_scores = trajectory["best_scores"]
    limit = min(budget, len(best_scores))
    return best_scores[limit - 1] - best_scores[0] <= MATCH_TOLERANCE


def policy_score(
    trajectory: dict[str, Any],
    pool: list[dict[str, Any]],
    budget: int,
    k_stop: int | None,
    n: int,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Expected score if this practitioner had followed the policy instead.

    Their own run is the first one launched; restarts are drawn from `pool`. The
    match clock is always on, timed against this practitioner's own final score,
    because it consumes no random numbers and so costs the heavy bootstrap
    callers nothing -- they read "mean_score" and ignore the rest.

    Returns:
        The whole simulate() dict. "mean_score" is None only when the pool is
        empty and no run is possible.
    """
    return simulate(
        pool,
        budget=budget,
        n=n,
        k_stop=k_stop,
        n_trials=n_trials,
        seed=seed,
        first=trajectory,
        target=baseline_score(trajectory, budget),
    )


# =============================================================================
# Leave-one-out within a cell
# =============================================================================

def process_leave_one_out(
    cell_trajectories: list[dict[str, Any]],
    budget: int = BUDGET_DEFAULT,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Score every campaign in a cell against a policy fit without it.

    Each campaign is held out in turn and the policy is refit on the remaining
    n-1; cells are too small for a 50/50 split to leave a usable calibration
    half.

    Returns:
        One record per held-out campaign: the fitted policy, the baseline, the
        policy score, their difference, and the match clock.
    """
    records = []
    for index, held_out in enumerate(cell_trajectories):
        calibration = [
            t for i, t in enumerate(cell_trajectories) if i != index
        ]
        if not calibration:
            continue

        policy = estimate_policy(
            calibration,
            budget=budget,
            seed=seed,
        )
        baseline = baseline_score(held_out, budget)
        # Common random numbers: the seed is fixed per held-out campaign, so
        # the restart draws are identical across the policies being compared
        # and the difference is not contaminated by inner dice-rolling.
        replay = policy_score(
            held_out,
            calibration,
            budget=budget,
            k_stop=policy["k_stop"],
            n=policy["n_replicates"],
            n_trials=n_trials,
            seed=seed + index,
        )
        scored = replay["mean_score"]
        if scored is None:
            continue

        match = replay["match"]
        records.append({
            "campaign_id": held_out["campaign_id"],
            "k_stop": policy["k_stop"],
            "n_replicates": policy["n_replicates"],
            "baseline": round(baseline, 4),
            "policy": round(scored, 4),
            "difference": round(scored - baseline, 4),
            # When the policy drew level with this practitioner's own final
            # score. match_mean is over the replays that got there and
            # match_rate is its denominator; the flag marks the campaigns that
            # match at iteration 1 by construction.
            "match_mean": match["mean"],
            "match_rate": match["rate"],
            "baseline_never_improved": baseline_never_improved(
                held_out, budget
            ),
        })

    return records


def process_cell_refit_bootstrap(
    members: list[dict[str, Any]],
    budget: int = BUDGET_DEFAULT,
    n_boot: int = DEFAULT_REFIT_BOOTSTRAP,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Interval that also varies the recommendation itself.

    Resamples the cell's campaigns with replacement, refits (k_stop,
    n_replicates) on the resampled corpus, and scores only the out-of-bag
    campaigns -- so no campaign is ever fit on and scored in the same draw.
    Restart stand-ins are drawn from the resampled corpus too.

    This varies which campaigns we hold, what the restart pool contains, and
    the fitted policy itself. See README section 7b for how it compares with
    holding the policy fixed.
    """
    if len(members) < 2:
        return {"ci_low": None, "ci_high": None, "boot_p": None, "n_boot": 0}

    rng = np.random.default_rng(seed)
    size = len(members)
    gains: list[float] = []

    for b in range(n_boot):
        indices = rng.integers(0, size, size=size)
        pool = [members[i] for i in indices]
        oob = [members[i] for i in sorted(set(range(size)) - set(indices.tolist()))]
        if not oob:
            continue

        fitted = estimate_policy(
            pool, budget=budget, seed=seed + b
        )
        diffs = []
        for held in oob:
            base = baseline_score(held, budget)
            policy = policy_score(
                held, pool, budget, fitted["k_stop"], fitted["n_replicates"],
                n_trials=n_trials, seed=seed + b,
            )["mean_score"]
            if base is None or policy is None:
                continue
            diffs.append(policy - base)
        if diffs:
            gains.append(float(np.mean(diffs)))

    if len(gains) < 2:
        return {"ci_low": None, "ci_high": None, "boot_p": None, "n_boot": len(gains)}

    values = np.asarray(gains, dtype=float)
    return {
        "ci_low": round(float(np.percentile(values, 2.5)), 4),
        "ci_high": round(float(np.percentile(values, 97.5)), 4),
        "boot_p": round(float(np.mean(values <= 0)), 4),
        "boot_gain": round(float(values.mean()), 4),
        "n_boot": len(values),
    }


def process_cell(
    cell: dict[str, Any],
    trajectories: list[dict[str, Any]],
    budget: int = BUDGET_DEFAULT,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_REFIT_BOOTSTRAP,
) -> dict[str, Any]:
    """Evaluate one (problem, algorithm, model) cell end to end.

    Returns:
        The per-cell summary that becomes a row of q5_evaluation.json. The
        per-campaign records behind it stay internal: the artifact is per-cell.
    """
    members = [
        t for t in trajectories
        if t["problem"] == cell["problem"]
        and t["algorithm"] == cell["algorithm"]
        and t["model"] == cell["model"]
    ]

    records = process_leave_one_out(
        members,
        budget=budget,
        n_trials=n_trials,
        seed=seed,
    )
    differences = [r["difference"] for r in records]
    interval = process_cell_refit_bootstrap(
        members, budget=budget, n_boot=n_boot, n_trials=n_trials,
        seed=seed,
    )

    # The fitted policy varies across folds; the modal value is the one a
    # caller would actually be handed.
    k_values = [r["k_stop"] for r in records]
    n_values = [r["n_replicates"] for r in records]

    classification = (
        "headroom" if cell["headroom"]
        else "saturated" if cell["saturated"]
        else "dead" if cell["dead"]
        else "usable"
    )
    # Match statistics are campaign-as-unit, the same convention as `gain`, and
    # are taken over the campaigns whose own run improved on its seed -- the
    # others match at iteration 1 by construction. A campaign that never matched
    # has no mean to contribute but still counts in the rate, so n_match_folds
    # is the denominator of match_mean and n_folds_informative that of
    # match_rate. See README section 4.3.
    informative = [r for r in records if not r["baseline_never_improved"]]
    matched = [r["match_mean"] for r in informative
               if r["match_mean"] is not None]
    rates = [r["match_rate"] for r in informative
             if r["match_rate"] is not None]

    row = {
        "problem": cell["problem"],
        "algorithm": cell["algorithm"],
        "model": cell["model"],
        "n_campaigns": cell["n_campaigns"],
        "mean_final": cell["mean_final"],
        "sd_final": cell["sd_final"],
        "classification": classification,
        "k_stop": _mode(k_values),
        "n_replicates": _mode(n_values),
        "k_stop_folds": sorted({k for k in k_values if k is not None}),
        "n_replicates_folds": sorted(set(n_values)),
        "baseline": round(float(np.mean([r["baseline"] for r in records])), 4)
        if records else None,
        "policy": round(float(np.mean([r["policy"] for r in records])), 4)
        if records else None,
        "n_improved": sum(1 for d in differences if d > 0),
        "n_folds": len(records),
        # The point estimate is leave-one-out on the real corpus; the bootstrap
        # supplies only the interval. boot_gain is a consistency check, not the
        # headline number.
        "gain": round(float(np.mean(differences)), 4) if differences else None,
        # Average iteration at which the policy first reaches the practitioner's
        # own final score. Descriptive: this is a checkpoint inside a
        # full-budget replay, not a shorter run, and it is measured against a
        # score the practitioner had not yet produced. Never quote match_mean
        # without match_rate.
        "match_mean": round(float(np.mean(matched)), 2) if matched else None,
        "match_rate": round(float(np.mean(rates)), 4) if rates else None,
        "n_match_folds": len(matched),
        "n_folds_informative": len(informative),
        "n_baseline_never_improved": len(records) - len(informative),
        **interval,
    }
    return row


def _mode(values: list[Any]) -> Any:
    """Most common value, ties broken by first appearance."""
    if not values:
        return None
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda v: (counts[v], values.index(v) * -1))


# =============================================================================
# Across-cell significance
# =============================================================================

def process_across_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Test whether the policy helps across configurations, not just on average.

    The unit is the cell, so the test asks whether the gain replicates across
    configurations rather than whether one pooled average is positive. Wilcoxon
    ranks the gains; the sign test is reported alongside because it is
    scale-free.

    Only headroom cells are tested. Cells with too little final-score spread
    cannot show a difference, and saturated cells are immovable by
    construction; the caller reports those separately as the no-harm check.
    """
    tested = [r for r in rows if r["classification"] == "headroom"]
    gains = [r["gain"] for r in tested if r["gain"] is not None]

    if len(gains) < 2:
        return {
            "n_cells": len(gains), "median_gain": None,
            "wilcoxon_statistic": None, "wilcoxon_p": None,
            "sign_test_p": None, "n_positive": None,
            "note": "too few cells with headroom to test",
        }

    values = np.asarray(gains, dtype=float)
    n_positive = int(np.sum(values > 0))

    try:
        statistic, p_value = wilcoxon(values, alternative="greater")
        statistic, p_value = float(statistic), float(p_value)
    except ValueError:
        # Raised when every difference is zero.
        statistic, p_value = None, None

    sign = binomtest(n_positive, n=len(values), p=0.5, alternative="greater")

    return {
        "n_cells": len(values),
        "median_gain": round(float(np.median(values)), 4),
        "mean_gain": round(float(values.mean()), 4),
        "min_gain": round(float(values.min()), 4),
        "max_gain": round(float(values.max()), 4),
        "n_positive": n_positive,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": round(p_value, 6) if p_value is not None else None,
        "sign_test_p": round(float(sign.pvalue), 6),
    }


# =============================================================================
# High-Level API
# =============================================================================

def get_evaluation(
    database_url: str | None = None,
    budget: int = BUDGET_DEFAULT,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_REFIT_BOOTSTRAP,
    trajectories: list[dict[str, Any]] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the headline experiment: cell-scoped, paired, leave-one-out.

    Args:
        database_url: optional override.
        budget: iteration budget per comparison.
        n_trials: replay trials per policy score.
        seed: RNG seed.
        n_boot: refit-bootstrap replicates for per-cell intervals.
        trajectories: preloaded trajectories, to avoid reloading.
        workers: processes to spread cells across. Every cell seeds its own
            generators, so the result is identical for any worker count.

    Returns:
        Per-cell rows, the saturated/dead rows kept separately, and the
        across-cell test.
    """
    if trajectories is None:
        trajectories = load_trajectories(database_url)
    problems = eligible_problems(trajectories)
    cells = enumerate_cells(trajectories, problems=problems)

    todo = [c for c in cells if c["usable"] or c["saturated"] or c["dead"]]
    state = {
        "trajectories": trajectories, "budget": budget,
        "n_trials": n_trials, "seed": seed, "n_boot": n_boot,
    }
    evaluated = _map(_evaluate_cell_worker, todo, state, workers)

    tested = [r for r in evaluated if r["classification"] == "headroom"]
    degenerate = [
        r for r in evaluated if r["classification"] in ("saturated", "dead")
    ]
    uninformative = [r for r in evaluated if r["classification"] == "usable"]

    return {
        "budget": budget,
        "n_trials": n_trials,
        "cell_min_campaigns": CELL_MIN_CAMPAIGNS,
        "n_boot": n_boot,
        "seed": seed,
        "problems": problems,
        "cells": tested,
        "cells_uninformative": uninformative,
        "cells_degenerate": degenerate,
        "across_cells": process_across_cells(evaluated),
    }


def get_per_problem(
    database_url: str | None = None,
    budget: int = BUDGET_DEFAULT,
    n_boot: int = DEFAULT_REFIT_BOOTSTRAP,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    trajectories: list[dict[str, Any]] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Per-problem gain over the headroom cells, with a refit bootstrap.

    The same estimator as the per-cell interval, one level up. Within each
    draw, every headroom cell is resampled independently, refits its own
    policy on its resample, and scores its out-of-bag campaigns; the cell
    gains are then aggregated across the problem's cells two ways:

    - campaign_weighted: weighted by cell size -- "a random practitioner on
      this problem".
    - cell_weighted: each cell counts once -- "a random configuration on this
      problem".

    Resampling stays within the observed cells, so the interval is conditional
    on these configurations.

    Returns:
        Dict keyed "<problem> | <weighting>" with mean, 2.5/97.5 percentiles,
        one-sided boot_p (fraction of draws <= 0) and the number of draws.
    """
    if trajectories is None:
        trajectories = load_trajectories(database_url)
    problems = eligible_problems(trajectories)

    members: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for cell in enumerate_cells(trajectories, problems=problems):
        if not cell["headroom"]:
            continue
        members[(cell["problem"], cell["algorithm"], cell["model"])] = [
            t for t in trajectories
            if t["problem"] == cell["problem"]
            and t["algorithm"] == cell["algorithm"]
            and t["model"] == cell["model"]
        ]

    state = {
        "members": members, "budget": budget,
        "n_trials": n_trials, "seed": seed,
    }
    draws = _map(_per_problem_draw_worker, range(n_boot), state, workers,
                 chunksize=4)

    result = {}
    for problem in problems:
        for weighting in ("campaign_weighted", "cell_weighted"):
            values = np.array(
                [d[problem][weighting] for d in draws if problem in d]
            )
            if values.size == 0:
                continue
            result[f"{problem} | {weighting}"] = {
                "mean": round(float(values.mean()), 3),
                "ci_low": round(float(np.percentile(values, 2.5)), 3),
                "ci_high": round(float(np.percentile(values, 97.5)), 3),
                "boot_p": round(float(np.mean(values <= 0)), 4),
                "n_draws": int(values.size),
            }
    return result


# =============================================================================
# Worker plumbing
# =============================================================================

# Set once per worker process by _init_worker, so the trajectories are shipped
# to each process once rather than pickled with every task.
_STATE: dict[str, Any] = {}


def _init_worker(state: dict[str, Any]) -> None:
    _STATE.update(state)


def _map(func, items, state, workers, chunksize=1):
    """Map func over items, in-process or across a pool, preserving order."""
    items = list(items)
    if workers <= 1:
        _init_worker(state)
        return [func(item) for item in items]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(state,)
    ) as pool:
        return list(pool.map(func, items, chunksize=chunksize))


def _evaluate_cell_worker(cell: dict[str, Any]) -> dict[str, Any]:
    row = process_cell(
        cell, _STATE["trajectories"], budget=_STATE["budget"],
        n_trials=_STATE["n_trials"], seed=_STATE["seed"],
        n_boot=_STATE["n_boot"],
    )
    print(
        f"  done {cell['problem'][:18]:<20}{cell['algorithm']:<11}"
        f"{str(cell['model'])[:24]:<26} gain={row['gain']:+7.2f} "
        f"[{row['ci_low']}, {row['ci_high']}] "
        f"match={row['match_mean']}/{_STATE['budget']} "
        f"@ rate {row['match_rate']}",
        file=sys.stderr, flush=True,
    )
    return row


def _per_problem_draw_worker(b: int) -> dict[str, dict[str, float]]:
    """One per-problem bootstrap draw across every headroom cell."""
    budget, seed = _STATE["budget"], _STATE["seed"]
    rng = np.random.default_rng(PER_PROBLEM_RESAMPLE_SEED + b)
    per_problem: dict[str, list[tuple[float, int]]] = defaultdict(list)

    for (problem, _, _), cell_members in _STATE["members"].items():
        size = len(cell_members)
        indices = rng.integers(0, size, size=size)
        pool = [cell_members[i] for i in indices]
        oob = [
            cell_members[i]
            for i in sorted(set(range(size)) - set(indices.tolist()))
        ]
        if not oob:
            continue
        fitted = estimate_policy(pool, budget=budget, seed=seed + b)
        diffs = []
        for held in oob:
            policy = policy_score(
                held, pool, budget, fitted["k_stop"], fitted["n_replicates"],
                n_trials=_STATE["n_trials"], seed=seed + b,
            )["mean_score"]
            if policy is not None:
                diffs.append(policy - baseline_score(held, budget))
        if diffs:
            per_problem[problem].append((float(np.mean(diffs)), size))

    out = {}
    for problem, rows in per_problem.items():
        gains = np.array([r[0] for r in rows])
        weights = np.array([r[1] for r in rows], dtype=float)
        out[problem] = {
            "campaign_weighted": float((gains * weights).sum() / weights.sum()),
            "cell_weighted": float(gains.mean()),
        }
    return out


if __name__ == "__main__":
    import json

    import os

    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    evaluation = get_evaluation()
    print(json.dumps(evaluation, indent=2, default=str))

    path = os.path.join(outdir, "q5_evaluation.json")
    with open(path, "w") as handle:
        json.dump(evaluation, handle, indent=2, default=str)
    print(f"\nWrote {path}", file=sys.stderr)
