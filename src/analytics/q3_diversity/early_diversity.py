"""Tab 2: Early Diversity scatter plot."""

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy import stats

from analytics.q3_diversity._common import (
    COLOR_PALETTE,
    LINESTYLE_PALETTE,
    MARKER_PALETTE,
    _generate_empty_figure,
    _get_database_url,
    _truncate_problem,
    display_mechanism,
    normalize_model_name,
)


def _compute_early_diversity_campaign_data(
    candidates: list[dict[str, Any]] | None = None,
    early_fraction: float = 0.25,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Compute early diversity for each campaign using direct code embeddings."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH campaign_stats AS (
                    SELECT
                        c.campaign_id,
                        MAX(c.iteration_index) as max_iter,
                        COUNT(*) as n_candidates
                    FROM candidates c
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND c.iteration_index IS NOT NULL
                    GROUP BY c.campaign_id
                    HAVING COUNT(*) >= 10
                ),
                early_candidates AS (
                    SELECT
                        c.id,
                        c.campaign_id,
                        c.iteration_index,
                        c.direct_code_embedding
                    FROM candidates c
                    JOIN campaign_stats cs ON c.campaign_id = cs.campaign_id
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND c.iteration_index IS NOT NULL
                      AND c.iteration_index <= (cs.max_iter * %(early_fraction)s)::int
                ),
                early_diversity AS (
                    SELECT
                        e1.campaign_id,
                        COUNT(*) as n_pairs,
                        AVG(e1.direct_code_embedding <=> e2.direct_code_embedding) as early_diversity
                    FROM early_candidates e1
                    JOIN early_candidates e2 ON e1.campaign_id = e2.campaign_id AND e1.id < e2.id
                    GROUP BY e1.campaign_id
                ),
                early_counts AS (
                    SELECT campaign_id, COUNT(*) as n_early
                    FROM early_candidates
                    GROUP BY campaign_id
                    HAVING COUNT(*) >= 2
                ),
                campaign_scores AS (
                    SELECT
                        c.campaign_id,
                        MAX(CAST(m.value AS DOUBLE PRECISION)) as best_score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    GROUP BY c.campaign_id
                )
                SELECT
                    camp.id as campaign_id,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used,
                    ed.early_diversity,
                    ec.n_early as n_early_candidates,
                    cs_stats.n_candidates as n_total_candidates,
                    cs.best_score
                FROM campaigns camp
                JOIN campaign_stats cs_stats ON cs_stats.campaign_id = camp.id
                JOIN early_diversity ed ON ed.campaign_id = camp.id
                JOIN early_counts ec ON ec.campaign_id = camp.id
                JOIN campaign_scores cs ON cs.campaign_id = camp.id
                WHERE camp.research_question IS NOT NULL
                  AND cs.best_score IS NOT NULL
                ORDER BY camp.id
            """, {"early_fraction": early_fraction})
            rows = cur.fetchall()

    campaign_data = []
    for row in rows:
        problem = _truncate_problem(row["research_question"])
        if problem is None:
            continue

        campaign_data.append({
            "campaign_id": str(row["campaign_id"]),
            "problem": problem,
            "model": normalize_model_name(row["models_used"][0]) if row["models_used"] else "unknown",
            "algorithm": row["algorithm_used"] or "unknown",
            "early_diversity": round(float(row["early_diversity"]), 4),
            "best_score": round(float(row["best_score"]), 4),
            "n_early_candidates": row["n_early_candidates"],
            "n_total_candidates": row["n_total_candidates"],
        })

    return campaign_data


def process_early_diversity_scatter(
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Process early diversity data for scatter plot with grouping."""
    campaign_data = _compute_early_diversity_campaign_data(
        early_fraction=early_fraction, database_url=database_url
    )

    if problems:
        campaign_data = [c for c in campaign_data if c.get("problem") in problems]

    points = []
    for camp in campaign_data:
        if group_by == "model":
            group = camp.get("model") or "unknown"
        elif group_by == "algorithm":
            group = camp.get("algorithm") or "unknown"
        elif group_by == "model_algorithm":
            model = camp.get("model") or "unknown"
            alg = camp.get("algorithm") or "unknown"
            group = f"{model} + {alg}"
        else:
            group = "all"

        points.append({
            "campaign_id": camp["campaign_id"],
            "problem": camp["problem"],
            "early_diversity": camp["early_diversity"],
            "best_score": camp["best_score"],
            "group": group,
        })

    if not points:
        return {
            "group_by": group_by,
            "problems": problems,
            "early_fraction": early_fraction,
            "points": [],
            "group_stats": [],
            "overall_correlation": None,
        }

    all_div = [p["early_diversity"] for p in points]
    all_score = [p["best_score"] for p in points]
    if len(points) >= 4:
        overall_rho, _ = stats.spearmanr(all_div, all_score)
        overall_corr = float(overall_rho)
    else:
        overall_corr = None

    by_group: dict[str, list[dict]] = {}
    for p in points:
        g = p["group"]
        if g not in by_group:
            by_group[g] = []
        by_group[g].append(p)

    group_stats = []
    for group_name in sorted(by_group.keys()):
        group_points = by_group[group_name]
        if len(group_points) < 3:
            continue

        divs = [p["early_diversity"] for p in group_points]
        scores = [p["best_score"] for p in group_points]
        if len(group_points) >= 4:
            rho, p_val = stats.spearmanr(divs, scores)
        else:
            rho, p_val = None, None

        group_stats.append({
            "group": group_name,
            "n": len(group_points),
            "correlation": round(rho, 4) if rho is not None and not np.isnan(rho) else None,
            "spearman_rho": round(rho, 4) if rho is not None and not np.isnan(rho) else None,
            "spearman_p": round(p_val, 4) if p_val is not None and not np.isnan(p_val) else None,
            "early_diversity_mean": round(float(np.mean(divs)), 4),
            "score_mean": round(float(np.mean(scores)), 4),
        })

    return {
        "group_by": group_by,
        "problems": problems,
        "early_fraction": early_fraction,
        "points": points,
        "group_stats": group_stats,
        "overall_correlation": round(overall_corr, 4) if overall_corr and not np.isnan(overall_corr) else None,
        "total_campaigns": len(points),
    }


def generate_early_diversity_scatter_figure(data: dict[str, Any]) -> bytes:
    """Generate scatter plot of early diversity vs score with optional grouping."""
    points = data.get("points", [])
    if not points:
        return _generate_empty_figure("No early diversity data available")

    group_by = data.get("group_by")
    group_stats = data.get("group_stats", [])
    early_fraction = data.get("early_fraction", 0.25)

    if group_by is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = [p["early_diversity"] for p in points]
        y = [p["best_score"] for p in points]

        ax.scatter(x, y, c=COLOR_PALETTE[0], alpha=0.6, s=50, edgecolors='white', linewidths=0.5)

        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p_fit(x_line), '--', color=COLOR_PALETTE[0], alpha=0.8, linewidth=2)

            rho, p_value = stats.spearmanr(x, y)
            if p_value < 0.001:
                p_str = "p < .001"
            else:
                p_str = f"p = {p_value:.3f}"
            ax.set_title(f"ρ = {rho:.2f}, {p_str}", fontsize=14)

        ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}%)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)

    elif group_by == "model_algorithm":
        algorithms = sorted(set(g["group"].rsplit(" + ", 1)[1] for g in group_stats if " + " in g["group"]))
        models = sorted(set(g["group"].rsplit(" + ", 1)[0] for g in group_stats if " + " in g["group"]))

        if not algorithms:
            return _generate_empty_figure("No model+algorithm groups found")

        n_algs = len(algorithms)
        n_models = len(models)
        fig_width = min(20, max(10, 5 * n_algs))
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 6), squeeze=False, sharey=True)
        axes = axes[0]

        if n_models <= 10:
            model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(models)}
        elif n_models <= 20:
            model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(models)}
        else:
            model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(models)}

        model_styles = {m: (LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)],
                           MARKER_PALETTE[i % len(MARKER_PALETTE)])
                       for i, m in enumerate(models)}

        for ax_idx, algorithm in enumerate(algorithms):
            ax = axes[ax_idx]

            for model in models:
                group_name = f"{model} + {algorithm}"
                gs = next((g for g in group_stats if g["group"] == group_name), None)
                if not gs:
                    continue

                group_points = [p for p in points if p["group"] == group_name]
                if not group_points:
                    continue

                x = [p["early_diversity"] for p in group_points]
                y = [p["best_score"] for p in group_points]
                color = model_colors[model]
                linestyle, marker = model_styles[model]
                short_name = model[:15] + "..." if len(model) > 15 else model

                ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                          edgecolors='white', linewidths=0.5,
                          label=short_name if ax_idx == 0 else None)

                if len(x) >= 2:
                    z = np.polyfit(x, y, 1)
                    p_fit = np.poly1d(z)
                    x_line = np.linspace(min(x), max(x), 100)
                    ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                           alpha=0.7, linewidth=1.8)

            ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}%)')
            if ax_idx == 0:
                ax.set_ylabel('Best Score')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            n_legend_cols = min(6, max(1, n_models))
            fig.legend(handles, labels, loc='upper center', fontsize=9,
                      ncol=n_legend_cols, bbox_to_anchor=(0.5, -0.02))

    else:
        groups = sorted(gs["group"] for gs in group_stats)
        n_groups = len(groups)

        fig_width = max(10, min(16, 2.5 * max(n_groups // 3, 4)))
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        if n_groups <= 10:
            colors = [plt.cm.tab10(i / 10) for i in range(n_groups)]
        elif n_groups <= 20:
            colors = [plt.cm.tab20(i / 20) for i in range(n_groups)]
        else:
            colors = [plt.cm.viridis(0.1 + 0.8 * i / n_groups) for i in range(n_groups)]

        for i, gs in enumerate(group_stats):
            group_name = gs["group"]
            group_points = [p for p in points if p["group"] == group_name]
            if not group_points:
                continue

            x = [p["early_diversity"] for p in group_points]
            y = [p["best_score"] for p in group_points]

            color = colors[i]
            marker = MARKER_PALETTE[i % len(MARKER_PALETTE)]
            linestyle = LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)]

            label = display_mechanism(group_name)
            if len(label) > 20:
                label = label[:18] + "..."
            rho = gs.get("spearman_rho")
            p_val = gs.get("spearman_p")
            if rho is not None:
                if p_val is not None and p_val < 0.001:
                    label += f" (ρ={rho:.2f}, p<.001)"
                elif p_val is not None:
                    label += f" (ρ={rho:.2f}, p={p_val:.3f})"
                else:
                    label += f" (ρ={rho:.2f})"

            ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                      edgecolors='white', linewidths=0.5, label=label)

            if len(x) >= 2:
                z = np.polyfit(x, y, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                       alpha=0.7, linewidth=1.8)

        ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}% of iterations)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_early_diversity_scatter(
    database_url: str | None = None,
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get early diversity scatter data with grouping using direct code embeddings."""
    return process_early_diversity_scatter(
        early_fraction=early_fraction, group_by=group_by, problems=problems, database_url=database_url
    )


def get_early_diversity_scatter_figure(
    database_url: str | None = None,
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get early diversity scatter figure as PNG using direct code embeddings."""
    data = get_early_diversity_scatter(database_url, early_fraction, group_by, problems)
    return generate_early_diversity_scatter_figure(data)
