#!/usr/bin/env python3
"""Q1: Basin Structure / Multimodality Analytics.

Analyzes whether identical configurations produce multimodal final-score
distributions, revealing distinct attractors in the optimization landscape.

Uses the bimodality coefficient (BC) to quantify distribution shape:
    BC = (γ² + 1) / (κ + 3)
where γ is skewness and κ is excess kurtosis.
BC > 5/9 ≈ 0.555 suggests bimodality; BC ≤ 5/9 suggests unimodality.

Module structure:
- query_*: Database queries
- process_*: Data transformation/aggregation
- generate_*: Figure/export generation
"""

import io
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg.rows import dict_row

from analytics.utils import normalize_model_name

# Bimodality coefficient threshold (5/9)
BC_THRESHOLD = 5 / 9  # ≈ 0.555


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"

# Publication-quality matplotlib settings
# Font sizes for scaling: 6" figure → ~5.5" at 0.85\textwidth (92% scale)
# 14pt base → ~13pt printed (comfortable for full-width figures)
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
    'savefig.pad_inches': 0.1,
})



# ---------------------------------------------------------------------------
# Query functions - Database access only
# ---------------------------------------------------------------------------

def query_skydiscover_scores(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query SkyDiscover campaign scores with embeddings.

    Returns campaigns with their best combined_score per campaign.
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
                    MAX(CAST(m.value AS DOUBLE PRECISION)) as primary_score
                FROM campaigns c
                JOIN candidates cand ON cand.campaign_id = c.id
                JOIN measurements m ON m.candidate_id = cand.id
                WHERE c.research_question IS NOT NULL
                  AND m.name = 'combined_score'
                  AND m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                GROUP BY c.id, c.models_used, c.algorithm_used, c.research_question
            """)
            rows = cur.fetchall()

    return [_parse_campaign_row(row) for row in rows]

def _parse_campaign_row(row: dict) -> dict[str, Any]:
    """Parse a campaign row."""
    return {
        "campaign_id": str(row["campaign_id"]),
        "models_used": row["models_used"],
        "algorithm_used": row.get("algorithm_used"),
        "research_question": row["research_question"],
        "primary_score": row.get("primary_score"),
    }

def _get_database_url() -> str:
    """Get database URL from environment."""
    import os
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


# ---------------------------------------------------------------------------
# Process functions - Data transformation only
# ---------------------------------------------------------------------------

def process_variance_data(
    campaigns: list[dict[str, Any]],
    min_runs: int = 2,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Process campaign data into bimodality statistics.

    Groups campaigns by research question and configuration,
    then computes bimodality coefficient (BC) to detect multimodal distributions.

    Args:
        campaigns: List of campaign dicts with primary_score and embedding.
        min_runs: Minimum runs to include a configuration.
        group_by: Grouping mode - "model", "algorithm", or "model_algorithm".
            None means aggregate all.

    Returns:
        Dictionary with cells containing bimodality stats and summary.
    """
    if not campaigns:
        return _empty_variance_result()

    # Filter to campaigns with valid data
    valid = [c for c in campaigns if c.get("primary_score") is not None]
    if not valid:
        return _empty_variance_result()

    # Group campaigns by configuration
    grouped = _group_by_configuration(valid, group_by)

    # Calculate bimodality stats for groups with sufficient runs
    cells = _compute_variance_cells(grouped, min_runs, group_by)

    # Sort by research question then by BC descending (None sorts last)
    cells.sort(key=lambda x: (x["research_question"], -(x["bc"] if x["bc"] is not None else -1)))

    return {
        "cells": cells,
        "summary": {
            "total_cells": len(cells),
            "total_campaigns": sum(c["n"] for c in cells),
            "high_bc_cells": len([c for c in cells if c["bc_ci_lower"] is not None and c["bc_ci_lower"] > BC_THRESHOLD]),
        },
    }


def _group_by_configuration(
    campaigns: list[dict],
    group_by: str | None = None,
) -> dict[tuple, list[float]]:
    """Group campaign scores by configuration.

    Args:
        campaigns: List of campaign dicts.
        group_by: Grouping mode - "algorithm", "model", or "model_algorithm"/None.
    """
    grouped: dict[tuple, list[float]] = {}

    for c in campaigns:
        rq = c["research_question"]

        models = ", ".join(sorted(normalize_model_name(m) for m in c["models_used"])) if c["models_used"] else "unknown"
        score = c["primary_score"]

        if group_by == "algorithm":
            algorithm = c.get("algorithm_used") or "unknown"
            key = (rq, algorithm)
        elif group_by == "model":
            key = (rq, models)
        else:
            algorithm = c.get("algorithm_used") or "unknown"
            key = (rq, models, algorithm)

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(score)

    return grouped


def _bootstrap_bc_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for bimodality coefficient.

    Resamples scores with replacement and recomputes BC for each resample.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(scores)
    if n < 3:
        return 0.0, 0.0, 0.0

    point_estimate, _, _ = _compute_bimodality_coefficient(scores)

    rng = np.random.default_rng(seed=42)
    bootstrap_estimates = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample = [scores[i] for i in indices]
        bc_b, _, _ = _compute_bimodality_coefficient(sample)
        bootstrap_estimates[b] = bc_b

    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper


def _compute_bimodality_coefficient(scores: list[float]) -> tuple[float, float, float]:
    """Compute bimodality coefficient, skewness, and excess kurtosis.

    BC = (γ² + 1) / (κ + 3)
    where γ is skewness and κ is excess kurtosis.

    Args:
        scores: List of score values.

    Returns:
        Tuple of (bc, skewness, kurtosis).
    """
    n = len(scores)
    if n < 3:
        return 0.0, 0.0, 0.0

    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    stddev = variance ** 0.5

    if stddev == 0:
        return 0.0, 0.0, 0.0

    # Skewness (γ)
    skewness = sum((x - mean) ** 3 for x in scores) / (n * stddev ** 3)

    # Excess kurtosis (κ)
    kurtosis = sum((x - mean) ** 4 for x in scores) / (n * stddev ** 4) - 3

    # Bimodality coefficient
    bc = (skewness ** 2 + 1) / (kurtosis + 3)

    return bc, skewness, kurtosis


def _compute_variance_cells(
    grouped: dict[tuple, list[float]],
    min_runs: int,
    group_by: str | None = None,
) -> list[dict[str, Any]]:
    """Compute bimodality statistics for each configuration group.

    Args:
        grouped: Dict mapping configuration keys to lists of scores.
        min_runs: Minimum runs to include a configuration.
        group_by: The grouping mode used to create the keys:
            - "algorithm": key is (rq, algorithm)
            - "model": key is (rq, models)
            - "model_algorithm" or None: key is (rq, models, algorithm)
    """
    cells = []

    for key, scores in grouped.items():
        n = len(scores)
        if n < min_runs:
            continue

        # Parse key based on group_by mode
        if len(key) == 3:
            # (rq, models, algorithm)
            rq, models, algorithm = key
        elif group_by == "algorithm":
            # (rq, algorithm) - display algorithm in models field
            rq, algorithm = key
            models = algorithm
            algorithm = None
        else:
            # (rq, models)
            rq, models = key
            algorithm = None

        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / n
        stddev = variance ** 0.5
        bc, skewness, kurtosis = _compute_bimodality_coefficient(scores)
        _, bc_ci_lower, bc_ci_upper = _bootstrap_bc_ci(scores)

        if stddev == 0:
            classification = "deterministic"
        elif bc_ci_lower > BC_THRESHOLD:
            classification = "confirmed_bimodal"
        elif bc > BC_THRESHOLD and bc_ci_lower <= BC_THRESHOLD:
            classification = "underpowered"
        else:
            classification = "unimodal"

        concentration = None
        if classification in ("deterministic", "underpowered"):
            score_range = max(scores) - min(scores)
            if score_range == 0:
                ceiling_mass = 1.0
                floor_mass = 0.0
            else:
                ceiling_mass = sum(1 for s in scores if s >= max(scores) - score_range * 0.05) / n
                floor_mass = sum(1 for s in scores if s <= min(scores) + score_range * 0.05) / n
            if ceiling_mass >= 0.8:
                concentration = "near ceiling"
            elif floor_mass >= 0.8:
                concentration = "near floor"
            elif ceiling_mass + floor_mass >= 0.8:
                concentration = "binary (ceiling/floor)"

        cells.append({
            "research_question": rq,
            "problem_label": _truncate(rq, 60),
            "models": models,
            "algorithm": algorithm,
            "n": n,
            "mean": round(mean, 2),
            "stddev": round(stddev, 2),
            "bc": None if classification == "deterministic" else round(bc, 3),
            "bc_ci_lower": None if classification == "deterministic" else round(bc_ci_lower, 3),
            "bc_ci_upper": None if classification == "deterministic" else round(bc_ci_upper, 3),
            "classification": classification,
            "concentration": concentration,
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "scores": sorted(scores),
        })

    return cells


def _empty_variance_result() -> dict[str, Any]:
    """Return empty result structure."""
    return {
        "cells": [],
        "summary": {"total_cells": 0, "total_campaigns": 0, "high_bc_cells": 0},
    }


# ---------------------------------------------------------------------------
# Generate functions - Figure generation only
# ---------------------------------------------------------------------------

def _generate_empty_figure(message: str) -> bytes:
    """Generate placeholder figure for empty data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# High-level API - Combines query + process + generate
# ---------------------------------------------------------------------------

def get_variance_analysis(
    database_url: str | None = None,
    min_runs: int = 2,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Get complete variance analysis data.

    Queries database, processes data, returns structured result.

    Args:
        database_url: Database connection URL.
        min_runs: Minimum runs to include a configuration.
        group_by: Grouping mode - "model", "algorithm", or "model_algorithm".
            None means aggregate all.
    """
    skydiscover = query_skydiscover_scores(database_url)
    all_campaigns = skydiscover

    return process_variance_data(all_campaigns, min_runs, group_by)


def generate_figure_for_problem(
    variance_data: dict[str, Any],
    problem_label: str,
    models_filter: list[str] | None = None,
) -> bytes:
    """Generate violin plot figure for a single problem.

    Args:
        variance_data: Output from process_variance_data().
        problem_label: The problem label to generate figure for.
        models_filter: Optional list of model names to include. If None, all models shown.

    Returns:
        Figure as PNG bytes.
    """
    cells = variance_data.get("cells", [])
    problem_cells = [c for c in cells if c["problem_label"] == problem_label]

    # Filter to specific models if requested
    if models_filter:
        problem_cells = [c for c in problem_cells if c["models"] in models_filter]

    if not problem_cells:
        return _generate_empty_figure(f"No data for {problem_label}")

    # Sort by BC (highest bimodality first, convergent cells last)
    problem_cells = sorted(problem_cells, key=lambda c: -(c["bc"] if c["bc"] is not None else -1))

    # Calculate figure width based on number of configurations
    n_configs = len(problem_cells)
    fig_width = max(12, 1.8 * n_configs)

    fig, ax = plt.subplots(figsize=(fig_width, 7))

    positions = list(range(n_configs))
    all_scores = [cell["scores"] for cell in problem_cells]

    # Draw violin plots
    parts = ax.violinplot(
        all_scores,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.7
    )

    # Color each violin based on classification
    COLOR_MAP = {
        "confirmed_bimodal": "#DC267F",  # Magenta
        "deterministic": "#888888",          # Gray
        "underpowered": "#FE6100",        # Orange
        "unimodal": "#648FFF",            # Blue
    }
    for i, (pc, cell) in enumerate(zip(parts["bodies"], problem_cells)):
        classification = cell.get("classification", "unimodal")
        color = COLOR_MAP.get(classification, "#648FFF")

        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.6)

        # Add individual points
        scores = cell["scores"]
        jitter = np.random.uniform(-0.08, 0.08, len(scores))
        ax.scatter(i + jitter, scores, c=color, alpha=0.6, s=20, zorder=3,
                  edgecolors="white", linewidths=0.3)

        # Add median line
        median = np.median(scores)
        ax.hlines(median, i - 0.15, i + 0.15, colors="black", linewidths=2, zorder=4)

        # Add quartile lines
        q1, q3 = np.percentile(scores, [25, 75])
        ax.hlines(q1, i - 0.08, i + 0.08, colors="black", linewidths=1, zorder=4)
        ax.hlines(q3, i - 0.08, i + 0.08, colors="black", linewidths=1, zorder=4)
        ax.vlines(i, q1, q3, colors="black", linewidths=1, zorder=4)

    # Format x-axis labels (model + algorithm only)
    xlabels = []
    for cell in problem_cells:
        model = cell["models"]
        algorithm = cell.get("algorithm")
        if algorithm:
            xlabels.append(f"{model}\n{algorithm}")
        else:
            xlabels.append(model)

    # Annotate n, BC, CI at top of plot (consistent position using axes coordinates)
    for i, cell in enumerate(problem_cells):
        classification = cell.get("classification", "unimodal")
        n = cell["n"]
        if classification == "deterministic":
            ann = f"n={n}\nDeterministic"
        else:
            bc = cell["bc"]
            ci_lo = cell.get("bc_ci_lower")
            ci_hi = cell.get("bc_ci_upper")
            ann = f"n={n}\nBC={bc:.2f}"
            if ci_lo is not None:
                ann += f"\n[{ci_lo:.2f},{ci_hi:.2f}]"
        ax.text(i, 1.02, ann, ha='center', va='bottom', fontsize=18,
                linespacing=1.1, transform=ax.get_xaxis_transform())

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=16)
    ax.set_ylabel("Primary Score", fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    # ax.set_title(f"Q1: Basin Structure - {_truncate(problem_label, 60)}", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, n_configs - 0.5)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_aggregate_figure_for_problem(variance_data: dict[str, Any], problem_label: str) -> bytes:
    """Generate single aggregate violin plot for a problem (no model/algorithm breakdown).

    Args:
        variance_data: Output from process_variance_data().
        problem_label: The problem label to generate figure for.

    Returns:
        Figure as PNG bytes.
    """
    cells = variance_data.get("cells", [])
    problem_cells = [c for c in cells if c["problem_label"] == problem_label]

    if not problem_cells:
        return _generate_empty_figure(f"No data for {problem_label}")

    # Aggregate all scores for this problem
    all_scores = []
    for cell in problem_cells:
        all_scores.extend(cell["scores"])

    if not all_scores:
        return _generate_empty_figure(f"No scores for {problem_label}")

    # Calculate bimodality coefficient
    bc, _, _ = _compute_bimodality_coefficient(all_scores)

    # Determine color based on BC
    if bc > BC_THRESHOLD:
        color = "#DC267F"  # Magenta - bimodal
    else:
        color = "#648FFF"  # Blue - unimodal

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw single violin plot
    parts = ax.violinplot(
        [all_scores],
        positions=[0],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.7
    )

    # Color the violin
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.6)

    # Add individual points
    jitter = np.random.uniform(-0.08, 0.08, len(all_scores))
    ax.scatter(jitter, all_scores, c=color, alpha=0.6, s=20, zorder=3,
              edgecolors="white", linewidths=0.3)

    # Add median line
    median = np.median(all_scores)
    ax.hlines(median, -0.15, 0.15, colors="black", linewidths=2, zorder=4)

    # Add quartile lines
    q1, q3 = np.percentile(all_scores, [25, 75])
    ax.hlines(q1, -0.08, 0.08, colors="black", linewidths=1, zorder=4)
    ax.hlines(q3, -0.08, 0.08, colors="black", linewidths=1, zorder=4)
    ax.vlines(0, q1, q3, colors="black", linewidths=1, zorder=4)

    # Format x-axis
    n = len(all_scores)
    ax.set_xticks([0])
    ax.set_xticklabels([f"All configs\nn={n}, BC={bc:.2f}"])
    ax.set_ylabel("Primary Score")
    # ax.set_title(f"Q1: Basin Structure - {_truncate(problem_label, 60)}\n(Aggregate)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, 0.5)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_variance_figure_for_problem(
    problem_label: str,
    database_url: str | None = None,
    min_runs: int = 2,
    group_by: str = "model_algorithm",
    models_filter: list[str] | None = None,
) -> bytes:
    """Get variance figure for a specific problem.

    Args:
        problem_label: The problem to generate figure for.
        group_by: "aggregate", "model", "algorithm", or "model_algorithm"
        models_filter: Optional list of model names to include. If None, all models shown.
    """
    if group_by == "aggregate":
        # For aggregate, we need all data for this problem merged
        data = get_variance_analysis(database_url, min_runs, "model_algorithm")
        return generate_aggregate_figure_for_problem(data, problem_label)
    else:
        data = get_variance_analysis(database_url, min_runs, group_by)
        return generate_figure_for_problem(data, problem_label, models_filter)


def get_problem_list(
    database_url: str | None = None,
    min_runs: int = 2,
) -> list[str]:
    """Get list of unique problem labels."""
    data = get_variance_analysis(database_url, min_runs, group_by="model_algorithm")
    problems = sorted(set(c["problem_label"] for c in data.get("cells", [])))
    return problems
