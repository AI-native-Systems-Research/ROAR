#!/usr/bin/env python3
"""Q5: Closed-Loop Configuration Recommender.

Given a user's research question, recommends two budget-allocation parameters
derived from pooled ROAR runs:

- k_stop:       stop a run after this many stagnant iterations
- n_replicates: how many independent runs to split a fixed budget across

This module is also the shared library layer for the Q5 package: the evaluation
modules import the trajectory loader and the replay simulator from here.

Module structure:
- query_*:  Database queries
- process_*/estimate_*: Data transformation and parameter estimation
- get_*:    High-level API combining query + process

See README.md for how the two parameters are grounded in Q1 and Q2, and
RQ5_EXPERIMENT_RESULTS.md for the experiment that validates them.
"""

import os
from bisect import bisect_left
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row

from analytics.q2_calibration import query_iteration_scores

DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"

# Total budget in iterations for one comparison.
BUDGET_DEFAULT = 50

# A campaign must have at least this many iterations to be replayable at
# BUDGET_DEFAULT.
MIN_ITERATIONS = 50

# A problem needs at least this many replayable campaigns before we will
# recommend problem-specific parameters for it.
MIN_CAMPAIGNS = 20

# Smallest usable stagnation threshold; k_stop=1 is vacuous.
K_STOP_MIN = 2

# Points in each log-spaced search grid. See build_k_grid / build_n_grid.
K_GRID_POINTS = 9
N_GRID_POINTS = 6

# Shortest run we are willing to call a run, so replicate counts are capped at
# budget // MIN_RUN_ITERATIONS (n <= 5 at BUDGET_DEFAULT). It bounds the n grid
# only; it charges nothing against the budget. See README section 4.1.
MIN_RUN_ITERATIONS = 10

# A (problem, algorithm, model) cell needs this many replayable campaigns
# before a policy can be fit and evaluated within it.
CELL_MIN_CAMPAIGNS = 8

# At or below this final-score spread a cell is degenerate: no allocation
# policy can be distinguished on it. Such cells are classified, not dropped.
CELL_MIN_SD = 0.0

# Above this spread a cell has enough dynamic range for the comparison to mean
# something. Cells between CELL_MIN_SD and this are usable but uninformative.
CELL_HEADROOM_SD = 2.0

# Cosine similarity above which an unseen research question is matched to a
# known problem.
SIMILARITY_THRESHOLD = 0.75

DEFAULT_SEED = 42
DEFAULT_TRIALS = 500

# Tolerance for "the policy reached the practitioner's score", so that two runs
# scoring identically are not separated by the last bit.
MATCH_TOLERANCE = 1e-9

# Directional hints from Q3/Q4. Returned to the caller but NOT validated by the
# Q5 evaluations.
ADVISORY = {
    "Polynomino Packing": {
        "exploration_bias": "positive",
        "context_composition": "show candidates across the quality spectrum, including failures",
        "source": "Q3 early diversity (rho=0.15, p=0.011); Q4 factor importance",
    },
    "Bounded 2D Knapsack": {
        "exploration_bias": "neutral",
        "context_composition": "favour worse-scoring examples; avoid score-diverse context",
        "source": "Q3 early diversity (rho=-0.00, p=0.943); Q4 factor importance",
    },
}


# =============================================================================
# Grid Construction
# =============================================================================

def _log_spaced(low: int, high: int, points: int) -> list[int]:
    """Deduplicated integer grid, log-spaced from low to high inclusive."""
    if high < low:
        return []
    if high == low:
        return [int(low)]
    values = np.geomspace(low, high, num=max(2, points))
    return sorted({int(round(v)) for v in values if low <= round(v) <= high})


def build_k_grid(budget: int = BUDGET_DEFAULT, points: int = K_GRID_POINTS) -> list[int]:
    """Stagnation thresholds that can fire within `budget` iterations.

    Bounded by K_STOP_MIN and budget - 1; a threshold of `budget` or more can
    never be reached before the budget cap, so it would duplicate "never stop".
    """
    return _log_spaced(K_STOP_MIN, budget - 1, points)


def build_n_grid(
    budget: int = BUDGET_DEFAULT,
    points: int = N_GRID_POINTS,
    min_run: int = MIN_RUN_ITERATIONS,
) -> list[int]:
    """Replicate counts whose per-run cap is at least `min_run` iterations."""
    return _log_spaced(1, max(1, budget // max(1, min_run)), points)


def _get_database_url() -> str:
    """Get database URL from environment or use default."""
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


def _truncate_problem(rq: str | None) -> str:
    """Truncate a research question to a display label.

    Matches the labelling in q2_calibration.py so Q5 labels line up with Q2's.
    """
    if not rq:
        return "unknown"
    max_len = 50
    return rq[:max_len] + "..." if len(rq) > max_len else rq


def _model_key(models_used: Any) -> str:
    """Cell key for a campaign's model list.

    Model names are joined verbatim, never normalised or mapped to a canonical
    form, so configurations that behaved differently are never merged.
    """
    if not models_used:
        return "unknown"
    if isinstance(models_used, str):
        return models_used
    return "+".join(str(m) for m in models_used)


# =============================================================================
# Shared primitives (trajectory handling and replay)
# =============================================================================

def running_best(iterations: list[dict[str, Any]]) -> list[float]:
    """Compute the running best score at each iteration.

    Args:
        iterations: list of {index, score}, any order.

    Returns:
        List of running-best scores, ordered by iteration index.
    """
    ordered = sorted(iterations, key=lambda x: x["index"])
    best = float("-inf")
    best_scores = []
    for it in ordered:
        if it["score"] > best:
            best = it["score"]
        best_scores.append(best)
    return best_scores


def stop_index(best_scores: list[float], k_stop: int | None, cap: int) -> int:
    """Number of iterations consumed before stopping.

    Runs the same stagnation counter as Q2 (reset to 1 on improvement,
    incremented otherwise) and stops the first time it reaches k_stop, or at
    `cap` iterations, whichever comes first.

    Args:
        best_scores: running best per iteration.
        k_stop: stagnation length that triggers a stop; None disables stopping.
        cap: hard iteration cap (the per-run budget).

    Returns:
        Iterations consumed (>= 1). The score achieved is
        best_scores[returned - 1].
    """
    n = min(len(best_scores), cap)
    if n <= 0:
        return 0
    if k_stop is None:
        return n

    k = 0
    for i in range(n):
        if i == 0:
            k = 1
        elif best_scores[i] > best_scores[i - 1]:
            k = 1
        else:
            k += 1
        if k >= k_stop:
            return i + 1
    return n


def first_reach(
    best_scores: list[float], target: float, limit: int
) -> int | None:
    """Iterations consumed before the running best first reaches `target`.

    The binary search is valid only because `best_scores` is a running maximum
    and therefore sorted. Only the first `limit` entries are searched, since a
    launch that stopped after `limit` iterations never observed the rest.

    Returns:
        Iterations consumed (>= 1) at the first entry reaching `target`, or
        None if the target is not reached within `limit`. Same 1-based
        convention as stop_index: the score is best_scores[returned - 1].
    """
    index = bisect_left(best_scores, target - MATCH_TOLERANCE, 0, limit)
    return index + 1 if index < limit else None


def _match_summary(
    match_iters: list[int],
    n_trials: int,
    target: float,
) -> dict[str, Any]:
    """Summarise the match clock over a simulate() run's trials.

    match_iters holds only the trials that reached the target, so the mean is
    conditional on matching and `rate` carries the denominator -- a mean without
    its rate is the misleading number.
    """
    return {
        "target": target,
        "mean": round(float(np.mean(match_iters)), 2) if match_iters else None,
        "rate": round(len(match_iters) / n_trials, 4) if n_trials else None,
        "n_matched": len(match_iters),
        "n_trials": n_trials,
    }


def simulate(
    trajectories: list[dict[str, Any]],
    budget: int = BUDGET_DEFAULT,
    n: int = 1,
    k_stop: int | None = None,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    first: dict[str, Any] | None = None,
    target: float | None = None,
) -> dict[str, Any]:
    """Replay a budget-allocation policy over stored trajectories.

    Each trial spends the budget by launching runs of at most budget // n
    iterations, stopping each at its k_stop stagnation point, and launching
    another while budget remains. Budget freed by a stopped run is reallocated,
    so with k_stop set more than `n` runs typically fit. With k_stop=None,
    exactly `n` full-length runs are drawn.

    The budget is counted in iterations and a launch costs exactly the
    iterations it runs; nothing else is charged.

    Args:
        trajectories: campaigns from load_trajectories. May contain duplicates,
            which is how a resampled corpus is passed.
        budget: total iteration budget for the trial.
        n: replicate count setting the per-run cap (budget // n).
        k_stop: stagnation threshold for early stopping, or None.
        n_trials: number of Monte Carlo trials.
        seed: RNG seed.
        first: pin the first run of every trial to this trajectory instead of
            drawing it, which is what makes the evaluation's comparison paired.
        target: score to time the crossing of. Given one, the returned dict
            carries a "match" block; omitted, it does not. estimate_policy's
            grid sweep has no held-out campaign and so passes none, which is
            what keeps q5_curves.json free of empty match blocks.

    Returns:
        Dict with mean/median best score, the spread of outcomes across trials,
        the mean iterations spent and the mean number of runs launched. With a
        target, also a "match" key: the iteration -- counted cumulatively across
        launches -- at which the running best first reached `target`. Trials that
        never match contribute nothing, so its mean must never be read without
        `rate` beside it.

    The reported spread is how much outcomes vary for the user -- a property of
    the policy, NOT a confidence interval (see README section 7b). The match
    clock consumes no random numbers, so scores are identical to the bit whether
    a target is passed or not; q5_curves.json is a raw dump of these dicts.
    """
    if not trajectories and first is None:
        empty = {
            "n_replicates": n, "k_stop": k_stop, "budget": budget,
            "mean_score": None, "median_score": None,
            "sd_score": None, "p10_score": None, "p90_score": None,
            "mean_spent": None, "mean_runs": None, "n_trials": 0,
        }
        if target is not None:
            empty["match"] = _match_summary([], 0, target)
        return empty

    rng = np.random.default_rng(seed)
    per_run = max(1, budget // n)

    scores = []
    spends = []
    run_counts = []
    match_iters = []
    for _ in range(n_trials):
        best = float("-inf")
        spent = 0
        runs = 0
        match_at = None
        while spent < budget:
            if runs == 0 and first is not None:
                traj = first
            elif trajectories:
                traj = trajectories[rng.integers(0, len(trajectories))]
            else:
                # Pinned first run with an empty restart pool: nothing to
                # restart into, so the trial ends after that one run.
                break
            cap = min(per_run, budget - spent)
            consumed = stop_index(traj["best_scores"], k_stop, cap)
            if consumed <= 0:
                break
            # Iterations already spent when this launch starts, so the two
            # clocks below accumulate across launches rather than restarting.
            launched_at = spent
            spent += consumed
            runs += 1
            reached = traj["best_scores"][consumed - 1]
            if target is not None and match_at is None:
                hit = first_reach(traj["best_scores"], target, consumed)
                if hit is not None:
                    match_at = launched_at + hit
            best = max(best, reached)
        scores.append(best)
        spends.append(spent)
        run_counts.append(runs)
        if match_at is not None:
            match_iters.append(match_at)

    scores_arr = np.array(scores, dtype=float)

    result = {
        "n_replicates": n,
        "k_stop": k_stop,
        "budget": budget,
        "iterations_per_run": per_run,
        "mean_score": round(float(np.mean(scores_arr)), 4),
        "median_score": round(float(np.median(scores_arr)), 4),
        "sd_score": round(float(np.std(scores_arr)), 4),
        "p10_score": round(float(np.percentile(scores_arr, 10)), 4),
        "p90_score": round(float(np.percentile(scores_arr, 90)), 4),
        "mean_spent": round(float(np.mean(spends)), 2),
        "mean_runs": round(float(np.mean(run_counts)), 2),
        "n_trials": n_trials,
    }
    if target is not None:
        result["match"] = _match_summary(match_iters, n_trials, target)
    return result


# =============================================================================
# Database Queries
# =============================================================================

def load_trajectories(
    database_url: str | None = None,
    min_iterations: int = MIN_ITERATIONS,
    algorithm: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Load replayable per-iteration trajectories from the database.

    Wraps query_iteration_scores (q2_calibration), drops campaigns too short to
    replay, and precomputes the running best for each.

    Args:
        database_url: optional override.
        min_iterations: campaigns shorter than this cannot be replayed.
        algorithm: restrict to one algorithm (e.g. "best_of_n"). Optional.
        model: restrict to one model key as produced by _model_key. Optional.

    Together the two filters scope the pool to a single (problem, algorithm,
    model) cell. See README section 5.1 for why that matters.

    Returns:
        List of dicts with campaign_id, problem, algorithm, model, best_scores
        (running best per iteration), and n_iterations.
    """
    campaigns = query_iteration_scores(database_url)

    trajectories = []
    for campaign in campaigns:
        iterations = campaign["iterations"]
        if len(iterations) < min_iterations:
            continue
        campaign_algorithm = campaign.get("algorithm_used") or "unknown"
        if algorithm is not None and campaign_algorithm != algorithm:
            continue
        campaign_model = _model_key(campaign.get("models_used"))
        if model is not None and campaign_model != model:
            continue
        best_scores = running_best(iterations)
        trajectories.append({
            "campaign_id": campaign["campaign_id"],
            "problem": _truncate_problem(campaign["research_question"]),
            "research_question": campaign["research_question"],
            "algorithm": campaign_algorithm,
            "model": campaign_model,
            "models_used": campaign.get("models_used"),
            "best_scores": best_scores,
            "n_iterations": len(best_scores),
            # Retained so the campaign can be handed to Q2's episode detector.
            "iterations": iterations,
        })

    return trajectories


def enumerate_cells(
    trajectories: list[dict[str, Any]],
    problems: list[str] | None = None,
    min_campaigns: int = CELL_MIN_CAMPAIGNS,
    headroom_sd: float = CELL_HEADROOM_SD,
) -> list[dict[str, Any]]:
    """Group trajectories into (problem, algorithm, model) cells.

    A cell is one configuration, and is the unit the headline evaluation
    operates on. Cells are classified rather than filtered, so the caller
    decides which to use and can report the rest:

    - usable:     enough campaigns and non-zero final-score variance.
    - headroom:   usable, and spread wide enough for the comparison to mean
                  something (sd >= headroom_sd).
    - saturated:  zero variance at the problem's attainable ceiling.
    - dead:       zero variance below the ceiling (typically all-zero cells).

    Returns:
        List of cell dicts sorted by problem then descending campaign count.
    """
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for traj in trajectories:
        if problems is not None and traj["problem"] not in problems:
            continue
        key = (traj["problem"], traj["algorithm"], traj["model"])
        by_cell.setdefault(key, []).append(traj)

    # The ceiling is the best score anyone achieved on the problem, which is
    # what "saturated" has to be measured against.
    problem_max: dict[str, float] = {}
    for (problem, _, _), trajs in by_cell.items():
        observed = max(t["best_scores"][-1] for t in trajs)
        problem_max[problem] = max(problem_max.get(problem, float("-inf")), observed)

    cells = []
    for (problem, algorithm, model), trajs in by_cell.items():
        finals = np.array([t["best_scores"][-1] for t in trajs], dtype=float)
        sd = float(np.std(finals))
        mean = float(np.mean(finals))
        degenerate = sd <= CELL_MIN_SD
        at_ceiling = mean >= problem_max[problem] - 1e-9
        cells.append({
            "problem": problem,
            "algorithm": algorithm,
            "model": model,
            "n_campaigns": len(trajs),
            "mean_final": round(mean, 4),
            "sd_final": round(sd, 4),
            "min_final": round(float(np.min(finals)), 4),
            "max_final": round(float(np.max(finals)), 4),
            "usable": len(trajs) >= min_campaigns and not degenerate,
            "headroom": len(trajs) >= min_campaigns and sd >= headroom_sd,
            "saturated": len(trajs) >= min_campaigns and degenerate and at_ceiling,
            "dead": len(trajs) >= min_campaigns and degenerate and not at_ceiling,
        })

    return sorted(cells, key=lambda c: (c["problem"], -c["n_campaigns"]))


def eligible_problems(
    trajectories: list[dict[str, Any]],
    min_campaigns: int = MIN_CAMPAIGNS,
) -> list[str]:
    """Problems with enough replayable campaigns to support a recommendation.

    Also drops degenerate problems where every campaign scores identically,
    since no allocation policy can be distinguished on them.
    """
    by_problem: dict[str, list[dict]] = {}
    for traj in trajectories:
        by_problem.setdefault(traj["problem"], []).append(traj)

    problems = []
    for problem, trajs in by_problem.items():
        if len(trajs) < min_campaigns:
            continue
        finals = {traj["best_scores"][-1] for traj in trajs}
        if len(finals) <= 1:
            continue
        problems.append(problem)

    return sorted(problems)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denominator) if denominator else 0.0


def match_problem(
    database_url: str | None,
    research_question: str,
    known_problems: list[str],
) -> dict[str, Any]:
    """Match a user's research question to a problem in the corpus.

    Exact string match first, then nearest neighbour by text-embedding cosine
    similarity over the eligible problem labels.

    The similarity search embeds the eligible problem labels in process rather
    than querying pgvector.

    Returns:
        Dict with matched_problem (or None), match_type
        (exact / similar / none), and similarity.
    """
    label = _truncate_problem(research_question)
    if label in known_problems:
        return {"matched_problem": label, "match_type": "exact", "similarity": 1.0}

    if not research_question or not known_problems:
        return {"matched_problem": None, "match_type": "none", "similarity": None}

    try:
        from embeddings import embed_text

        query_vector = embed_text(research_question)
        scored = [
            (problem, _cosine(query_vector, embed_text(problem)))
            for problem in known_problems
        ]
    except Exception:
        # Embedding model unavailable: fall back to the pooled recommendation
        # rather than erroring.
        return {"matched_problem": None, "match_type": "none", "similarity": None}

    best_problem, best_similarity = max(scored, key=lambda pair: pair[1])
    if best_similarity >= SIMILARITY_THRESHOLD:
        return {
            "matched_problem": best_problem,
            "match_type": "similar",
            "similarity": round(best_similarity, 4),
        }

    return {
        "matched_problem": None,
        "match_type": "none",
        "similarity": round(best_similarity, 4),
    }


# =============================================================================
# Parameter Estimation
# =============================================================================

def split_calibration_evaluation(
    trajectories: list[dict[str, Any]],
    holdout: bool = True,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split trajectories 50/50 at the campaign level.

    Parameters are estimated on the calibration half and evaluated on the
    evaluation half, so the recommendation is never scored on the campaigns
    that produced it. With holdout=False both halves are the full set, which
    is the in-sample robustness check.

    Has no caller: the headline evaluation uses leave-one-out instead. Retained
    as the primitive for the half-split optimism diagnostic reported in the
    paper's RQ5 appendix.
    """
    if not holdout:
        return trajectories, trajectories

    ordered = sorted(trajectories, key=lambda t: t["campaign_id"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered))
    midpoint = len(ordered) // 2
    calibration = [ordered[i] for i in order[:midpoint]]
    evaluation = [ordered[i] for i in order[midpoint:]]
    return calibration, evaluation


def estimate_policy(
    trajectories: list[dict[str, Any]],
    budget: int = BUDGET_DEFAULT,
    k_grid: list[int] | None = None,
    n_grid: list[int] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Jointly select (k_stop, n_replicates) by replay over historical runs.

    Grid-searches both parameters against the same objective the evaluation
    reports -- best score at fixed budget, with the budget freed by stopped
    runs reallocated to fresh ones. k_stop=None (never stop early) is in the
    grid, so the recommender can decline to stop when stopping does not pay.

    Returns:
        Dict with the chosen k_stop and n_replicates, the winning cell, and the
        full grid.
    """
    if k_grid is None:
        k_grid = build_k_grid(budget)
    if n_grid is None:
        n_grid = build_n_grid(budget)

    grid = []
    for k_stop in [None, *k_grid]:
        for n in n_grid:
            point = simulate(
                trajectories, budget=budget, n=n, k_stop=k_stop, seed=seed,
            )
            grid.append(point)

    scored = [point for point in grid if point["mean_score"] is not None]
    if not scored:
        return {"k_stop": None, "n_replicates": 1, "best": None, "grid": grid}

    # Ties are common -- k_stop and n are partially redundant controls -- and
    # must break deterministically. Order of preference among equal scores:
    #   1. fewest forced replicates;
    #   2. a real stagnation rule over "never stop";
    #   3. the smallest such threshold.
    best = min(scored, key=lambda p: (
        -p["mean_score"],
        p["n_replicates"],
        p["k_stop"] is None,
        p["k_stop"] if p["k_stop"] is not None else 0,
    ))
    return {
        "k_stop": best["k_stop"],
        "n_replicates": best["n_replicates"],
        "best": best,
        "grid": grid,
        "k_grid": k_grid,
        "n_grid": n_grid,
    }


# =============================================================================
# High-Level API Functions (called from api.py)
# =============================================================================

def _consensus_policy(per_problem: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine per-problem policies into a single fallback recommendation.

    Takes the lower median of each parameter across problems. `best` is None
    because no cross-problem expected score is meaningful.
    """
    k_values = sorted(
        p["k_stop"] for p in per_problem.values() if p["k_stop"] is not None
    )
    n_values = sorted(p["n_replicates"] for p in per_problem.values())

    return {
        "k_stop": k_values[(len(k_values) - 1) // 2] if k_values else None,
        "n_replicates": n_values[(len(n_values) - 1) // 2] if n_values else 1,
        "best": None,
        "grid": [],
        "per_problem": {
            problem: {"k_stop": p["k_stop"], "n_replicates": p["n_replicates"]}
            for problem, p in per_problem.items()
        },
    }


def get_recommendation(
    database_url: str | None = None,
    research_question: str = "",
    budget: int = BUDGET_DEFAULT,
    algorithm: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Recommend budget-allocation parameters for a research question.

    Main entry point for the /analytics/recommend endpoint. Fit on all
    available evidence; the held-out evaluation lives in evaluate.py.

    Args:
        database_url: optional override.
        research_question: the user's problem statement.
        budget: total iteration budget the user intends to spend.
        algorithm: the caller's search algorithm, if known. Scopes the evidence
            to their own configuration.
        model: the caller's model, if known. Same purpose.

    With algorithm and model given, the recommendation is fit only on that
    cell. Omitting them falls back to the whole problem, which mixes
    configurations the caller may not be able to reproduce. See README
    section 5.1.

    Returns:
        Dict with the two validated parameters (k_stop, n_replicates), how the
        problem was matched, how much data supports the recommendation, and an
        advisory block of unvalidated Q3/Q4 directional hints.
    """
    all_trajectories = load_trajectories(database_url)

    # Eligibility is a property of the problem's full evidence, so it must be
    # decided before the cell filter narrows the pool -- otherwise a small cell
    # would make its own problem look ineligible.
    known_problems = eligible_problems(all_trajectories)
    trajectories = [
        t for t in all_trajectories
        if (algorithm is None or t["algorithm"] == algorithm)
        and (model is None or t["model"] == model)
    ]

    match = match_problem(database_url, research_question, known_problems)
    matched = match["matched_problem"]

    if matched is not None:
        pool = [t for t in trajectories if t["problem"] == matched]
        calibration = pool
        policy = estimate_policy(calibration, budget=budget)
        coverage = "high" if len(pool) >= CELL_MIN_CAMPAIGNS else "low"
        per_problem = None
    else:
        # No matching problem. Scores are on incomparable scales across
        # problems, so the fallback must NOT pool them: estimate a policy within
        # each known problem and take the consensus.
        per_problem = {}
        calibration = []
        for problem in known_problems:
            problem_calibration = [
                t for t in trajectories if t["problem"] == problem
            ]
            calibration.extend(problem_calibration)
            per_problem[problem] = estimate_policy(
                problem_calibration, budget=budget
            )

        policy = _consensus_policy(per_problem)
        coverage = "none"

    return {
        "research_question": research_question,
        "matched_problem": matched,
        "match_type": match["match_type"],
        "similarity": match["similarity"],
        "k_stop": policy["k_stop"],
        "n_replicates": policy["n_replicates"],
        "budget_iterations": budget,
        "iterations_per_run": max(1, budget // policy["n_replicates"]),
        "expected_score": policy["best"]["mean_score"] if policy["best"] else None,
        "expected_runs": policy["best"]["mean_runs"] if policy["best"] else None,
        "scope": {
            "algorithm": algorithm,
            "model": model,
            "scoped": algorithm is not None or model is not None,
            "note": (
                "Pooled evidence: restarts are drawn from every configuration "
                "on this problem, so the expected score is an upper bound that "
                "includes model substitution. Pass algorithm and model to fit "
                "a policy you can actually execute."
                if algorithm is None and model is None
                else "Evidence is scoped to this configuration cell."
            ),
        },
        "support": {
            "n_campaigns": len(calibration),
            "per_problem_policies": policy.get("per_problem"),
        },
        "coverage": coverage,
        "advisory": ADVISORY.get(matched, {}) if matched else {},
        "advisory_note": (
            "Advisory fields are directional hints from Q3/Q4 and are NOT "
            "validated by the Q5 replay evaluations, which cannot supply the "
            "counterfactual for a policy that changes a run mid-flight."
        ),
        "known_problems": known_problems,
    }


if __name__ == "__main__":
    import json

    print("=" * 70)
    print("Usable cells")
    print("=" * 70)
    trajectories = load_trajectories()
    problems = eligible_problems(trajectories)
    for cell in enumerate_cells(trajectories, problems=problems):
        if not (cell["usable"] or cell["saturated"] or cell["dead"]):
            continue
        tag = (
            "headroom" if cell["headroom"]
            else "saturated" if cell["saturated"]
            else "dead" if cell["dead"]
            else "usable"
        )
        print(
            f"  {tag:<10} {cell['problem'][:28]:<30}{cell['algorithm']:<12}"
            f"{cell['model'][:28]:<30} n={cell['n_campaigns']:<4}"
            f" mean={cell['mean_final']:<8} sd={cell['sd_final']}"
        )

    print()
    print("=" * 70)
    print("Recommendations: pooled vs cell-scoped")
    print("=" * 70)
    demos = [
        # Pooled: an exact match and a paraphrase that must match by embedding.
        {"research_question": "Bounded 2D Knapsack"},
        {
            "research_question":
                "Pack polyominoes into a rectangular grid to maximise covered area."
        },
        # Cell-scoped: the same question, restricted to one configuration.
        {
            "research_question": "Bounded 2D Knapsack",
            "algorithm": "best_of_n",
            "model": "gemini-2.5-flash",
        },
    ]
    for kwargs in demos:
        result = get_recommendation(**kwargs)
        result.pop("known_problems", None)
        print(json.dumps(result, indent=2, default=str))
