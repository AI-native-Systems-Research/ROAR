"""Tab 1: Diversity vs Score scatter plot."""

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analytics.q3_diversity._common import (
    COLOR_PALETTE,
    LINESTYLE_PALETTE,
    MARKER_PALETTE,
    _generate_empty_figure,
    normalize_model_name,
    query_campaign_diversity_and_scores,
)


def process_diversity_vs_score_scatter(
    campaigns: list[dict[str, Any]],
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Process campaign data for diversity vs score scatter plot."""
    if problems:
        campaigns = [c for c in campaigns if c.get("problem") in problems]

    points = []
    for camp in campaigns:
        if group_by == "model":
            group = ", ".join(sorted(normalize_model_name(m) for m in camp["models_used"])) if camp.get("models_used") else "unknown"
        elif group_by == "algorithm":
            group = camp.get("algorithm_used") or "unknown"
        elif group_by == "model_algorithm":
            model = ", ".join(sorted(normalize_model_name(m) for m in camp["models_used"])) if camp.get("models_used") else "unknown"
            alg = camp.get("algorithm_used") or "unknown"
            group = f"{model} + {alg}"
        else:
            group = "all"

        points.append({
            "campaign_id": camp["campaign_id"],
            "problem": camp["problem"],
            "diversity": camp["diversity_score"],
            "score": camp["best_score"],
            "group": group,
        })

    if not points:
        return {
            "group_by": group_by,
            "problems": problems,
            "points": [],
            "group_stats": [],
            "overall_correlation": None,
        }

    all_div = [p["diversity"] for p in points]
    all_score = [p["score"] for p in points]
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

        divs = [p["diversity"] for p in group_points]
        scores = [p["score"] for p in group_points]
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
            "diversity_mean": round(float(np.mean(divs)), 4),
            "score_mean": round(float(np.mean(scores)), 4),
        })

    return {
        "group_by": group_by,
        "problems": problems,
        "points": points,
        "group_stats": group_stats,
        "overall_correlation": round(overall_corr, 4) if overall_corr and not np.isnan(overall_corr) else None,
        "total_campaigns": len(points),
    }


def generate_diversity_vs_score_scatter_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate scatter plot of diversity vs score with optional grouping."""
    points = data.get("points", [])
    if not points:
        return _generate_empty_figure("No diversity data available")

    group_by = data.get("group_by")
    group_stats = data.get("group_stats", [])

    if group_by is None:
        fig, ax = plt.subplots(figsize=(10, 7))
        x = [p["diversity"] for p in points]
        y = [p["score"] for p in points]

        ax.scatter(x, y, c=COLOR_PALETTE[0], alpha=0.6, s=40, edgecolors='white', linewidths=0.5)

        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p_fit(x_line), '--', color=COLOR_PALETTE[0], alpha=0.8, linewidth=2)

        ax.set_xlabel('Run Diversity (1 - avg cosine similarity of code embeddings)')
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

                x = [p["diversity"] for p in group_points]
                y = [p["score"] for p in group_points]
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

            ax.set_xlabel('Run Diversity')
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

            x = [p["diversity"] for p in group_points]
            y = [p["score"] for p in group_points]

            color = colors[i]
            marker = MARKER_PALETTE[i % len(MARKER_PALETTE)]
            linestyle = LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)]

            label = group_name
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

        ax.set_xlabel('Run Diversity (1 - avg cosine similarity of code embeddings)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_diversity_vs_score_scatter(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get diversity vs score scatter data using direct code embeddings."""
    campaigns = query_campaign_diversity_and_scores(database_url)
    return process_diversity_vs_score_scatter(campaigns, group_by, problems)


def get_diversity_vs_score_scatter_figure(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get diversity vs score scatter figure as PNG using direct code embeddings."""
    data = get_diversity_vs_score_scatter(database_url, group_by, problems)
    return generate_diversity_vs_score_scatter_figure(data)
