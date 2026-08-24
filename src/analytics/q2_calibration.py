#!/usr/bin/env python3
"""Q2: Rule-of-3 Calibration Analytics.

Tests whether the rule-of-3 heuristic (k failures → p < 3/k at 95% confidence)
is calibrated for BoN runs, where history-driven context growth violates i.i.d.

The rule-of-3 predicts: after k non-improving iterations, there's a ~63% chance
of escaping (improving) within the next k/3 iterations. If actual escape rates
are lower, the heuristic is "too optimistic" - runs are more stuck than predicted.

Key concepts:
- Stagnation episode: consecutive iterations without improvement
- Escape: achieving a new best score after a stagnation period
- Censored observation: run ended before the prediction window completed
"""

import io
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg.rows import dict_row


DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"

# Font sizes for scaling: figures displayed at \columnwidth (~3.25")
# 16pt base → ~5-6pt printed (minimum readable for single-column figures)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'escape': '#648FFF',
    'stuck': '#DC267F',
    'censored': '#808080',
    'reference': '#000000',
}

# Stagnation thresholds at which we record escape observations
THRESHOLDS = list(range(5, 55, 5))  # [5, 10, 15, ..., 50]

# Subset displayed in the aggregate dot chart (visual choice only)
AGGREGATE_DISPLAY_THRESHOLDS = [10, 20, 30, 40]



def _get_database_url() -> str:
    """Get database URL from environment or use default."""
    import os
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


def _truncate_problem(rq: str) -> str:
    """Truncate research question to a reasonable display length."""
    if not rq:
        return "unknown"
    max_len = 50
    return rq[:max_len] + "..." if len(rq) > max_len else rq


from analytics.utils import abbreviate_problem as _abbreviate_problem
from analytics.utils import normalize_model_name


# =============================================================================
# Database Queries
# =============================================================================

def query_iteration_scores(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query campaign iteration-level scores from the database.

    Fetches all candidates from supported algorithms (best_of_n, adaevolve, evox)
    with their scores, grouped by campaign for stagnation analysis.

    Returns:
        List of campaign dicts, each containing:
        - campaign_id, models_used, algorithm_used, research_question
        - iterations: list of {index, score} for each candidate
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    c.id as campaign_id,
                    c.models_used,
                    c.algorithm_used,
                    c.research_question,
                    cand.iteration_index,
                    CAST(m.value AS DOUBLE PRECISION) as score
                FROM campaigns c
                JOIN candidates cand ON cand.campaign_id = c.id
                JOIN measurements m ON m.candidate_id = cand.id
                WHERE c.algorithm_used IS NOT NULL
                  AND m.name = 'combined_score'
                  AND m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                ORDER BY c.id, cand.iteration_index
            """)
            rows = cur.fetchall()

    # Group rows by campaign, taking max score per iteration_index
    # (handles cases where multiple candidates exist at the same iteration)
    campaigns: dict[str, dict] = {}
    for row in rows:
        cid = str(row["campaign_id"])
        if cid not in campaigns:
            campaigns[cid] = {
                "campaign_id": cid,
                "models_used": row["models_used"],
                "algorithm_used": row["algorithm_used"],
                "research_question": row["research_question"],
                "iterations": {},  # dict keyed by iteration_index
            }
        iter_idx = row["iteration_index"]
        score = row["score"]
        # Keep max score per iteration
        if iter_idx not in campaigns[cid]["iterations"]:
            campaigns[cid]["iterations"][iter_idx] = score
        else:
            campaigns[cid]["iterations"][iter_idx] = max(
                campaigns[cid]["iterations"][iter_idx], score
            )

    # Convert iterations dict to sorted list
    result = []
    for campaign in campaigns.values():
        result.append({
            "campaign_id": campaign["campaign_id"],
            "models_used": campaign["models_used"],
            "algorithm_used": campaign["algorithm_used"],
            "research_question": campaign["research_question"],
            "iterations": [
                {"index": idx, "score": score}
                for idx, score in sorted(campaign["iterations"].items())
            ],
        })

    return result


# =============================================================================
# Core Analysis Functions
# =============================================================================

def compute_stagnation_observations(
    campaigns: list[dict[str, Any]],
    thresholds: list[int] = THRESHOLDS,
) -> list[dict[str, Any]]:
    """Compute stagnation observations from campaign iteration data.

    For each campaign, tracks the rolling best score and stagnation count
    (iterations since last improvement). Records one observation per
    bin-threshold crossing within each stagnation episode.

    Args:
        campaigns: List of campaign data from query_bon_iteration_scores
        bin_thresholds: List of stagnation lengths to record observations at

    Returns:
        List of observation dicts, each containing:
        - campaign_id, model, algorithm, problem, research_question
        - k: stagnation count when threshold was crossed
        - bin_threshold: which threshold triggered this observation
        - best_score: rolling best score at observation time
        - escaped_in_window: True if improved in next k/3 iterations
        - censored: True if run ended before window completed
        - iteration: iteration index of the observation
    """
    observations = []

    for campaign in campaigns:
        iters = sorted(campaign["iterations"], key=lambda x: x["index"])
        if len(iters) < 15:
            continue

        model = ", ".join(sorted(normalize_model_name(m) for m in campaign["models_used"])) if campaign["models_used"] else "unknown"
        algorithm = campaign.get("algorithm_used") or "unknown"
        rq = campaign["research_question"] or "unknown"
        problem = _truncate_problem(rq)

        # Compute rolling best score at each iteration
        best_score = float('-inf')
        best_scores = []
        for it in iters:
            if it["score"] > best_score:
                best_score = it["score"]
            best_scores.append(best_score)

        # Compute stagnation count at each iteration
        stagnation_counts = []
        k = 0
        for i in range(len(iters)):
            if i == 0:
                k = 1
            elif best_scores[i] > best_scores[i - 1]:
                k = 1  # Reset on improvement
            else:
                k += 1
            stagnation_counts.append(k)

        # Record observations at threshold crossings
        thresholds_seen_this_episode = set()

        for i in range(len(iters)):
            k_t = stagnation_counts[i]
            s_star = best_scores[i]

            # Reset tracking on new episode (improvement resets stagnation)
            if k_t == 1:
                thresholds_seen_this_episode = set()

            for threshold in thresholds:
                if k_t >= threshold and threshold not in thresholds_seen_this_episode:
                    thresholds_seen_this_episode.add(threshold)

                    # Rule-of-3 prediction window: k/3 more iterations
                    window_size = k_t // 3
                    window_end = i + window_size

                    if window_end >= len(iters):
                        # Censored: run ended before window completed
                        observations.append({
                            "campaign_id": campaign["campaign_id"],
                            "model": model,
                            "algorithm": algorithm,
                            "problem": problem,
                            "research_question": rq,
                            "k": k_t,
                            "bin_threshold": threshold,
                            "best_score": s_star,
                            "escaped_in_window": None,
                            "censored": True,
                            "iteration": i,
                        })
                    else:
                        # Check if we escaped in the prediction window
                        future_best = max(best_scores[i + 1 : window_end + 1])
                        escaped = future_best > s_star

                        observations.append({
                            "campaign_id": campaign["campaign_id"],
                            "model": model,
                            "algorithm": algorithm,
                            "problem": problem,
                            "research_question": rq,
                            "k": k_t,
                            "bin_threshold": threshold,
                            "best_score": s_star,
                            "escaped_in_window": escaped,
                            "censored": False,
                            "iteration": i,
                        })

    return observations


def analyze_calibration(
    observations: list[dict[str, Any]],
    thresholds: list[int] = THRESHOLDS,
) -> dict[str, Any]:
    """Analyze rule-of-3 calibration from stagnation observations.

    Computes escape rates and compares them to the 63% reference rate
    predicted by the rule-of-3.

    Args:
        observations: List of observations from compute_stagnation_observations
        thresholds: Stagnation thresholds to compute stats for

    Returns:
        Dict containing:
        - by_threshold: stats for each stagnation threshold
        - by_model: stats grouped by model
        - line_plot_data: data for per-problem line plots
        - overall: aggregate statistics
        - _observations: raw observations (for figure generation)
    """
    results = {
        "by_threshold": {},
        "by_model": {},
        "line_plot_data": {},
        "overall": {},
    }

    # Stats by stagnation threshold
    for threshold in thresholds:
        bin_obs = [o for o in observations if o["bin_threshold"] == threshold]
        results["by_threshold"][threshold] = _compute_bin_stats(bin_obs)

    # Stats by model
    models = set(o["model"] for o in observations)
    for model in sorted(models):
        model_obs = [o for o in observations if o["model"] == model]
        results["by_model"][model] = {}
        for threshold in thresholds:
            bin_obs = [o for o in model_obs if o["bin_threshold"] == threshold]
            results["by_model"][model][threshold] = _compute_bin_stats(bin_obs)

    # Data for line plots (escape rate vs k for each problem/model)
    results["line_plot_data"] = _compute_line_plot_data(observations)

    # Overall stats
    all_uncensored = [o for o in observations if not o["censored"]]
    escaped_count = sum(1 for o in all_uncensored if o["escaped_in_window"])
    censored_count = sum(1 for o in observations if o["censored"])

    results["overall"] = {
        "total_observations": len(observations),
        "uncensored": len(all_uncensored),
        "censored": censored_count,
        "escaped": escaped_count,
        "escape_rate": escaped_count / len(all_uncensored) if all_uncensored else None,
    }

    # Mixed-effects logistic regression for trend test
    results["escape_trend_model"] = fit_escape_trend_model(observations)

    # Include raw observations for figure generation
    results["_observations"] = observations

    return results


def _compute_line_plot_data(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute data for line plots showing escape rate vs stagnation length.

    Groups data by problem and model/algorithm for the per-problem view.

    Returns:
        Dict containing:
        - problems: list of problem names
        - models: list of model names
        - k_values: list of stagnation thresholds used
        - by_generator: nested dict of problem -> model -> list of {k, escape_rate, n}
        - by_generator_algorithm: same but keyed by "model / algorithm"
    """
    all_thresholds = THRESHOLDS
    problems = sorted(set(o["problem"] for o in observations))
    models = sorted(set(o["model"] for o in observations))
    algorithms = sorted(set(o.get("algorithm", "unknown") for o in observations))

    # Group by problem -> model
    by_generator = {}
    for problem in problems:
        by_generator[problem] = {}
        for model in models:
            by_generator[problem][model] = []
            for k in all_thresholds:
                obs = [o for o in observations
                       if o["problem"] == problem
                       and o["model"] == model
                       and o["bin_threshold"] == k
                       and not o["censored"]]
                if obs:
                    escaped = sum(1 for o in obs if o["escaped_in_window"])
                    rate = escaped / len(obs)
                    by_generator[problem][model].append({
                        "k": k,
                        "escape_rate": round(rate, 3),
                        "n": len(obs),
                    })

    # Group by problem -> model+algorithm
    by_generator_algorithm = {}
    for problem in problems:
        by_generator_algorithm[problem] = {}
        for model in models:
            for algorithm in algorithms:
                key = f"{model} / {algorithm}"
                if key not in by_generator_algorithm[problem]:
                    by_generator_algorithm[problem][key] = []
                for k in all_thresholds:
                    obs = [o for o in observations
                           if o["problem"] == problem
                           and o["model"] == model
                           and o.get("algorithm", "unknown") == algorithm
                           and o["bin_threshold"] == k
                           and not o["censored"]]
                    if obs:
                        escaped = sum(1 for o in obs if o["escaped_in_window"])
                        rate = escaped / len(obs)
                        by_generator_algorithm[problem][key].append({
                            "k": k,
                            "escape_rate": round(rate, 3),
                            "n": len(obs),
                        })

    # Remove empty entries
    for problem in list(by_generator_algorithm.keys()):
        by_generator_algorithm[problem] = {
            k: v for k, v in by_generator_algorithm[problem].items() if v
        }

    for problem in list(by_generator.keys()):
        by_generator[problem] = {
            k: v for k, v in by_generator[problem].items() if v
        }

    return {
        "problems": problems,
        "models": models,
        "k_values": all_thresholds,
        "by_generator": by_generator,
        "by_generator_algorithm": by_generator_algorithm,
    }


def _bootstrap_ci(
    observations: list[dict],
    cluster_fn=lambda o: (o["model"], o["algorithm"]),
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Bootstrap CI for escape rate, resampling at the cluster level.

    Args:
        observations: List of uncensored stagnation observations.
        cluster_fn: Callable mapping an observation to its cluster key.
            Default clusters by (model, algorithm) for aggregate figures.
            Use lambda o: o["campaign_id"] for per-run resampling when
            data is already disaggregated by model or algorithm.
        n_boot: Number of bootstrap iterations.
        alpha: Significance level (0.05 → 95% CI).
    """
    clusters: dict = {}
    for o in observations:
        clusters.setdefault(cluster_fn(o), []).append(o)

    cluster_list = list(clusters.values())
    n_clusters = len(cluster_list)

    if n_clusters < 2:
        return None, None

    rng = np.random.default_rng(42)
    boot_rates = []
    for _ in range(n_boot):
        chosen = rng.choice(n_clusters, size=n_clusters, replace=True)
        pooled = [o for idx in chosen for o in cluster_list[idx]]
        if pooled:
            escaped = sum(1 for o in pooled if o["escaped_in_window"])
            boot_rates.append(escaped / len(pooled))

    if not boot_rates:
        return None, None

    lower = float(np.percentile(boot_rates, 100 * alpha / 2))
    upper = float(np.percentile(boot_rates, 100 * (1 - alpha / 2)))
    return lower, upper


def fit_escape_trend_model(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Mixed-effects logistic regression: escaped ~ k + (1|run).

    Uses BinomialBayesMixedGLM (Bayesian GLMM with variational inference)
    as a Python-native equivalent to R's glmer. Tests whether escape
    probability decreases with stagnation length, with a random intercept
    per run to account for within-run correlation.
    """
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    uncensored = [o for o in observations if not o["censored"]]
    df = pd.DataFrame([{
        "escaped": int(o["escaped_in_window"]),
        "k": o["bin_threshold"],
        "run": o["campaign_id"],
    } for o in uncensored])

    n_obs = len(df)
    n_runs = int(df["run"].nunique())

    if n_obs < 10 or n_runs < 3:
        return {
            "error": "insufficient data",
            "n_observations": n_obs,
            "n_runs": n_runs,
        }

    model = BinomialBayesMixedGLM.from_formula(
        "escaped ~ k",
        {"run": "0 + C(run)"},
        data=df,
    )
    result = model.fit_vb()

    beta_k = float(result.fe_mean[1])
    se_k = float(result.fe_sd[1])
    ci_lower = beta_k - 1.96 * se_k
    ci_upper = beta_k + 1.96 * se_k

    or_10 = np.exp(10 * beta_k)
    or_10_lower = np.exp(10 * ci_lower)
    or_10_upper = np.exp(10 * ci_upper)

    significant = bool(or_10_upper < 1.0 or or_10_lower > 1.0)

    return {
        "odds_ratio_per_10k": round(float(or_10), 3),
        "or_ci_lower": round(float(or_10_lower), 3),
        "or_ci_upper": round(float(or_10_upper), 3),
        "log_odds_per_unit_k": round(float(beta_k), 4),
        "significant": significant,
        "direction": "decreasing" if beta_k < 0 else "increasing",
        "n_observations": n_obs,
        "n_runs": n_runs,
        "random_effect_sd": round(float(result.vcp_mean[0]), 3),
    }


def _compute_bin_stats(observations: list[dict]) -> dict[str, Any]:
    """Compute statistics for a set of observations.

    Returns:
        Dict with n_total, n_uncensored, n_censored, n_escaped,
        escape_rate, ci_lower, ci_upper, miscalibration
    """
    uncensored = [o for o in observations if not o["censored"]]
    censored = [o for o in observations if o["censored"]]

    if not uncensored:
        return {
            "n_total": len(observations),
            "n_uncensored": 0,
            "n_censored": len(censored),
            "n_escaped": 0,
            "escape_rate": None,
            "ci_lower": None,
            "ci_upper": None,
            "miscalibration": None,
        }

    escaped = sum(1 for o in uncensored if o["escaped_in_window"])
    n = len(uncensored)
    rate = escaped / n

    # 95% CI via cluster bootstrap (clusters = model × algorithm)
    ci_lower, ci_upper = _bootstrap_ci(uncensored)
    if ci_lower is None:
        ci_lower = max(0, rate - 1.96 * (rate * (1 - rate) / n) ** 0.5)
        ci_upper = min(1, rate + 1.96 * (rate * (1 - rate) / n) ** 0.5)

    # Miscalibration: deviation from 63% reference rate
    reference_rate = 0.63
    miscalibration = rate - reference_rate

    return {
        "n_total": len(observations),
        "n_uncensored": n,
        "n_censored": len(censored),
        "n_escaped": escaped,
        "escape_rate": round(rate, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "miscalibration": round(miscalibration, 3),
    }


# =============================================================================
# Figure Generation
# =============================================================================

def generate_aggregate_figure(analysis: dict[str, Any], problems_filter: list[str] | None = None) -> bytes:
    """Generate aggregate calibration bar chart across selected problems.

    Shows escape rates at k≥10, k≥20, k≥30, k≥40 for each problem,
    compared to the 63% rule-of-3 reference line.

    Args:
        analysis: Calibration analysis result from analyze_calibration
        problems_filter: List of problems to include. None means all problems.

    Returns:
        PNG image as bytes
    """
    from matplotlib.ticker import PercentFormatter

    all_problems = [p for p in analysis.get("line_plot_data", {}).get("problems", []) if p != "unknown"]

    # Filter to selected problems
    if problems_filter:
        problems = [p for p in all_problems if p in problems_filter]
    else:
        problems = all_problems

    if not problems:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Select one or more problems to view', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(problems)), 6))
    thresholds = AGGREGATE_DISPLAY_THRESHOLDS
    n_thresholds = len(thresholds)

    x_positions = np.arange(len(problems))
    dot_spacing = 0.8 / n_thresholds
    colors_k = ['#648FFF', '#785EF0', '#DC267F', '#FE6100']
    markers = ['o', 's', '^', 'D']

    all_rates = []
    all_ci_uppers = []
    for i, k in enumerate(thresholds):
        rates = []
        n_samples = []
        ci_lowers = []
        ci_uppers = []
        for problem in problems:
            obs = [o for o in analysis.get("_observations", [])
                   if o["problem"] == problem
                   and o["bin_threshold"] == k
                   and not o["censored"]]
            if obs:
                escaped = sum(1 for o in obs if o["escaped_in_window"])
                rate = escaped / len(obs)
                rates.append(rate)
                n_samples.append(len(obs))
                ci_lo, ci_hi = _bootstrap_ci(obs)
                ci_lowers.append(ci_lo if ci_lo is not None else rate)
                ci_uppers.append(ci_hi if ci_hi is not None else rate)
            else:
                rates.append(None)
                n_samples.append(0)
                ci_lowers.append(None)
                ci_uppers.append(None)

        all_rates.extend(r for r in rates if r is not None)
        all_ci_uppers.extend(u for u in ci_uppers if u is not None)
        offset = (i - (n_thresholds - 1) / 2) * dot_spacing

        for j in range(len(problems)):
            if rates[j] is None:
                continue
            x = x_positions[j] + offset
            yerr_lo = max(0, rates[j] - ci_lowers[j])
            yerr_hi = max(0, ci_uppers[j] - rates[j])
            ax.errorbar(x, rates[j], yerr=[[yerr_lo], [yerr_hi]],
                       fmt=markers[i % len(markers)], color=colors_k[i % len(colors_k)],
                       markersize=10, capsize=4, capthick=1.5, elinewidth=1.5,
                       label=f'k≥{k}' if j == 0 else None)
            ax.annotate(f'n={n_samples[j]}',
                       (x, ci_uppers[j] + 0.01),
                       ha='left', rotation=45, fontsize=14)

    # Reference line at 63%
    ax.axhline(y=0.63, color=COLORS['reference'], linestyle='--', linewidth=2,
               label='Rule-of-3 bound (63%)')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([_abbreviate_problem(p) for p in problems])
    ax.set_ylabel('Escape Rate')

    max_ci = max(all_ci_uppers) if all_ci_uppers else 0.1
    y_max = max(0.70, max_ci + 0.10)
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(problems) - 0.5 + 0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # ax.set_title('Q2: Aggregate Calibration by Problem\nReference line at 63% (i.i.d. prediction)',
    #              fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_grouped_figure(
    analysis: dict[str, Any],
    problems: list[str],
    group_by: str = "model",
) -> bytes:
    """Generate escape rate vs stagnation length figure aggregated across problems.

    Shows how escape probability changes with stagnation length (k),
    grouped by model, algorithm, or both, with data pooled across selected problems.

    For model_algorithm: creates faceted subplots (one per algorithm) with models as lines.

    Args:
        analysis: Calibration analysis result from analyze_calibration
        problems: List of problem labels to include
        group_by: How to group lines - "model", "algorithm", or "model_algorithm"

    Returns:
        PNG image as bytes
    """
    from matplotlib.ticker import PercentFormatter

    observations = analysis.get("_observations", [])
    filtered_obs = [o for o in observations if o["problem"] in problems and not o["censored"]]

    if not filtered_obs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data for selected problems', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    all_thresholds = THRESHOLDS

    if group_by == "model_algorithm":
        return _generate_faceted_figure(filtered_obs, problems, all_thresholds)
    else:
        return _generate_single_figure(filtered_obs, problems, group_by, all_thresholds)


def _generate_single_figure(
    filtered_obs: list[dict[str, Any]],
    problems: list[str],
    group_by: str,
    all_thresholds: list[int],
) -> bytes:
    """Generate single-panel figure for model or algorithm grouping."""
    from matplotlib.ticker import PercentFormatter

    if group_by == "model":
        groups = sorted(set(o["model"] for o in filtered_obs))
        get_group = lambda o: o["model"]
        group_label = "Model"
    else:  # algorithm
        groups = sorted(set(o.get("algorithm", "unknown") for o in filtered_obs))
        get_group = lambda o: o.get("algorithm", "unknown")
        group_label = "Algorithm"

    n_groups = len(groups)
    fig_width = max(10, min(16, 2.5 * max(n_groups // 3, 4)))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Dynamic color palette
    if n_groups <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_groups]
    elif n_groups <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_groups]
    else:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_groups))

    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    all_rates = []
    for i, group in enumerate(groups):
        group_obs = [o for o in filtered_obs if get_group(o) == group]
        points = _compute_escape_rates(group_obs, all_thresholds, with_ci=True)

        if points:
            k_vals = [p["k"] for p in points]
            rates = [p["rate"] for p in points]
            ci_lowers = [p["ci_lower"] for p in points]
            ci_uppers = [p["ci_upper"] for p in points]
            all_rates.extend(rates)

            color = colors[i]
            ax.plot(k_vals, rates,
                    marker=markers[i % len(markers)],
                    linestyle=line_styles[i % len(line_styles)],
                    color=color,
                    label=group,
                    markersize=6, linewidth=2, alpha=0.85)
            ax.fill_between(k_vals, ci_lowers, ci_uppers,
                            color=color, alpha=0.15)

    ax.axhline(y=0.63, color=COLORS['reference'], linestyle='--', linewidth=2,
               label='Rule-of-3 bound (63%)')

    ax.set_xlabel('Stagnation Length (k)')
    ax.set_ylabel('Escape Rate')

    max_rate = max(all_rates) if all_rates else 0.1
    y_max = max(0.70, max_rate * 1.2)
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0, 55)

    if len(problems) == 1:
        title = f'{problems[0]}: Escape Rate vs Stagnation Length (by {group_label})'
    else:
        title = f'Escape Rate vs Stagnation Length (by {group_label})\nAggregated across {len(problems)} problems'

    # ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _generate_faceted_figure(
    filtered_obs: list[dict[str, Any]],
    problems: list[str],
    all_thresholds: list[int],
) -> bytes:
    """Generate faceted figure for model_algorithm: one subplot per algorithm, models as lines."""
    from matplotlib.ticker import PercentFormatter

    # Group observations by algorithm -> model
    by_algorithm: dict[str, dict[str, list]] = {}
    all_models = set()

    for o in filtered_obs:
        model = o["model"]
        algorithm = o.get("algorithm", "unknown")
        all_models.add(model)

        if algorithm not in by_algorithm:
            by_algorithm[algorithm] = {}
        if model not in by_algorithm[algorithm]:
            by_algorithm[algorithm][model] = []
        by_algorithm[algorithm][model].append(o)

    algorithms = sorted(by_algorithm.keys())
    all_models = sorted(all_models)
    n_algs = len(algorithms)
    n_models = len(all_models)

    if n_algs == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # Create consistent style mapping for models (same across all facets)
    if n_models <= 10:
        model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(all_models)}
    elif n_models <= 20:
        model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(all_models)}
    else:
        model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(all_models)}

    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    model_styles = {m: (line_styles[i % len(line_styles)], markers[i % len(markers)])
                    for i, m in enumerate(all_models)}

    # Create faceted figure
    fig_width = min(20, max(10, 5 * n_algs))
    fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 6), squeeze=False, sharey=True)
    axes = axes[0]

    all_rates = []

    for ax_idx, algorithm in enumerate(algorithms):
        ax = axes[ax_idx]
        alg_data = by_algorithm[algorithm]

        for model in all_models:
            if model not in alg_data:
                continue

            model_obs = alg_data[model]
            points = _compute_escape_rates(model_obs, all_thresholds, with_ci=True)

            if points:
                k_vals = [p["k"] for p in points]
                rates = [p["rate"] for p in points]
                ci_lowers = [p["ci_lower"] for p in points]
                ci_uppers = [p["ci_upper"] for p in points]
                all_rates.extend(rates)

                linestyle, marker = model_styles[model]
                color = model_colors[model]
                short_name = model[:15] + "..." if len(model) > 15 else model

                ax.plot(k_vals, rates,
                        marker=marker, linestyle=linestyle, color=color,
                        label=short_name if ax_idx == 0 else None,
                        markersize=6, linewidth=2, alpha=0.85)
                ax.fill_between(k_vals, ci_lowers, ci_uppers,
                                color=color, alpha=0.15)

        # Reference line
        ax.axhline(y=0.63, color=COLORS['reference'], linestyle='--', linewidth=2,
                   label='63% bound' if ax_idx == 0 else None)

        ax.set_xlabel('Stagnation Length (k)')
        if ax_idx == 0:
            ax.set_ylabel('Escape Rate')
        ax.set_xlim(0, 55)
        # Only show subplot title when multiple algorithms
        if n_algs > 1:
            ax.set_title(algorithm, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Set shared y-axis limits
    max_rate = max(all_rates) if all_rates else 0.1
    y_max = max(0.70, max_rate * 1.2)
    for ax in axes:
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Title - include algorithm name if only one
    if n_algs == 1:
        alg_suffix = f' ({algorithms[0]})'
    else:
        alg_suffix = ''
    if len(problems) == 1:
        suptitle = f'{problems[0]}: Escape Rate by Model{alg_suffix}'
    else:
        suptitle = f'Escape Rate by Model + Algorithm (across {len(problems)} problems){alg_suffix}'
    fig.suptitle(suptitle, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Single legend below the plot to avoid overlap with titles
    handles, labels = axes[0].get_legend_handles_labels()
    n_legend_cols = min(6, max(1, n_models + 1))
    fig.legend(handles, labels, loc='upper center',
               ncol=n_legend_cols, bbox_to_anchor=(0.5, -0.02))

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _compute_escape_rates(obs: list[dict], thresholds: list[int], min_samples: int = 3, with_ci: bool = False) -> list[dict]:
    """Compute escape rates at each threshold for a set of observations."""
    points = []
    for k in thresholds:
        k_obs = [o for o in obs if o["bin_threshold"] == k]
        if len(k_obs) >= min_samples:
            escaped = sum(1 for o in k_obs if o["escaped_in_window"])
            rate = escaped / len(k_obs)
            point = {"k": k, "rate": rate, "n": len(k_obs)}
            if with_ci:
                ci_lo, ci_hi = _bootstrap_ci(
                    k_obs, cluster_fn=lambda o: o["campaign_id"]
                )
                point["ci_lower"] = ci_lo if ci_lo is not None else rate
                point["ci_upper"] = ci_hi if ci_hi is not None else rate
            points.append(point)
    return points


# =============================================================================
# High-Level API Functions (called from api.py)
# =============================================================================

def get_calibration_analysis(database_url: str | None = None) -> dict[str, Any]:
    """Get complete calibration analysis.

    Main entry point for the /analytics/calibration endpoint.

    Returns:
        Full analysis dict including by_threshold, by_model,
        line_plot_data, overall stats, and metadata.
    """
    campaigns = query_iteration_scores(database_url)
    observations = compute_stagnation_observations(campaigns)
    analysis = analyze_calibration(observations)

    analysis["metadata"] = {
        "n_campaigns": len(campaigns),
        "n_observations": len(observations),
        "thresholds": THRESHOLDS,
    }

    return analysis


def get_problems(database_url: str | None = None) -> list[str]:
    """Get list of available problems for calibration analysis.

    Used by /analytics/calibration/problems endpoint.
    """
    analysis = get_calibration_analysis(database_url)
    problems = [p for p in analysis.get("line_plot_data", {}).get("problems", []) if p != "unknown"]
    return sorted(problems)


def get_aggregate_figure(database_url: str | None = None, problems: list[str] | None = None) -> bytes:
    """Get aggregate calibration bar chart as PNG.

    Used by /analytics/calibration/aggregate/figure endpoint.

    Args:
        database_url: Optional database URL override
        problems: List of problems to include. None means all problems.
    """
    analysis = get_calibration_analysis(database_url)
    return generate_aggregate_figure(analysis, problems)


def get_grouped_figure(problems: list[str], group_by: str = "model", database_url: str | None = None) -> bytes:
    """Get calibration figure aggregated across multiple problems.

    Used by /analytics/calibration/figure endpoint.

    Args:
        problems: List of problem labels to include
        group_by: "model", "algorithm", or "model_algorithm"
        database_url: Optional database URL override
    """
    analysis = get_calibration_analysis(database_url)
    return generate_grouped_figure(analysis, problems, group_by)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import json
    analysis = get_calibration_analysis()
    print(json.dumps(analysis, indent=2, default=str))
