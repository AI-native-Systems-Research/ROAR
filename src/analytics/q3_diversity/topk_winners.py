"""Tab 3: Top-K Winners Diversity bar chart."""

import io
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from analytics.q3_diversity._common import (
    PALETTE,
    _compute_cross_diversity,
    _generate_empty_figure,
    abbreviate_problem,
    normalize_model_name,
    query_campaigns_with_best_embeddings,
)


def process_topk_winners_diversity(
    campaigns: list[dict[str, Any]],
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Compute diversity ACROSS top winners vs other winners using direct code embeddings."""
    if problems_filter:
        campaigns = [c for c in campaigns if c["problem"] in problems_filter]

    campaigns = [c for c in campaigns if c["best_score"] > 0]

    if not campaigns:
        return {"groups": [], "summary": {"total_groups": 0, "excluded_groups": 0}}

    def get_group_key(camp: dict) -> str:
        if group_by == "model":
            models = camp.get("models_used")
            return ", ".join(sorted(normalize_model_name(m) for m in models)) if models else "Unknown"
        elif group_by == "algorithm":
            return camp.get("algorithm_used") or "Unknown"
        elif group_by == "model_algorithm":
            models = camp.get("models_used")
            model = ", ".join(sorted(normalize_model_name(m) for m in models)) if models else "Unknown"
            algo = camp.get("algorithm_used") or "Unknown"
            return f"{model} + {algo}"
        else:
            return camp["problem"]

    by_group: dict[str, list[dict]] = {}
    for camp in campaigns:
        key = get_group_key(camp)
        if key not in by_group:
            by_group[key] = []
        by_group[key].append(camp)

    groups = []
    excluded_count = 0
    for group_name, group_camps in sorted(by_group.items()):
        sorted_camps = sorted(group_camps, key=lambda x: -x["best_score"])

        n_top = max(1, int(len(sorted_camps) * top_pct))
        top_runs = sorted_camps[:n_top]
        other_runs = sorted_camps[n_top:]

        if len(top_runs) < 2 or len(other_runs) < 2:
            excluded_count += 1
            continue

        top_embeddings = [c["best_embedding"] for c in top_runs]
        top_diversity = _compute_cross_diversity(top_embeddings)

        other_embeddings = [c["best_embedding"] for c in other_runs]
        other_diversity = _compute_cross_diversity(other_embeddings)

        top_best_scores = [c["best_score"] for c in top_runs]
        other_best_scores = [c["best_score"] for c in other_runs]

        groups.append({
            "group": group_name,
            "top_n": len(top_runs),
            "other_n": len(other_runs),
            "total_runs": len(sorted_camps),
            "top_winners_diversity": round(top_diversity, 4),
            "other_winners_diversity": round(other_diversity, 4),
            "diversity_diff": round(top_diversity - other_diversity, 4),
            "top_score_mean": round(float(np.mean(top_best_scores)), 4),
            "top_score_range": [round(min(top_best_scores), 4), round(max(top_best_scores), 4)],
            "other_score_mean": round(float(np.mean(other_best_scores)), 4),
        })

    return {
        "groups": groups,
        "group_by": group_by,
        "top_pct": top_pct,
        "problems_filter": problems_filter,
        "summary": {
            "total_groups": len(groups),
            "excluded_groups": excluded_count,
            "groups_where_top_more_diverse": len([g for g in groups if g["diversity_diff"] > 0.01]),
            "groups_where_top_less_diverse": len([g for g in groups if g["diversity_diff"] < -0.01]),
        },
    }


def generate_topk_winners_diversity_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate bar chart comparing diversity across top winners vs other winners."""
    groups = data.get("groups", [])
    if not groups:
        excluded = data.get("summary", {}).get("excluded_groups", 0)
        msg = "No groups with sufficient data"
        if excluded > 0:
            msg += f"\n({excluded} groups excluded: need ≥2 in both top and other buckets)"
        return _generate_empty_figure(msg)

    group_by = data.get("group_by")
    top_pct = data.get("top_pct", 0.25)
    top_pct_str = f"{int(top_pct * 100)}%"

    if group_by == "model_algorithm":
        by_algorithm: dict[str, dict[str, dict]] = {}
        for g in groups:
            group_name = g["group"]
            if " + " in group_name:
                model, algorithm = group_name.rsplit(" + ", 1)
            else:
                model, algorithm = group_name, "unknown"
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = {}
            by_algorithm[algorithm][model] = g

        algorithms = sorted(by_algorithm.keys())
        n_algs = len(algorithms)

        if n_algs == 0:
            return _generate_empty_figure("No model+algorithm groups found")

        all_models = sorted(set(
            model for alg_data in by_algorithm.values() for model in alg_data.keys()
        ))
        n_models = len(all_models)

        max_model_len = max(len(m) for m in all_models) if all_models else 10

        width_per_alg = max(4, min(8, 1.5 * max(len(by_algorithm[alg]) for alg in algorithms)))
        fig_width = min(24, max(10, width_per_alg * n_algs))
        fig_height = 6 + (max_model_len * 0.05)
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
        axes = axes[0]

        if n_models <= 10:
            model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(all_models)}
        elif n_models <= 20:
            model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(all_models)}
        else:
            model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(all_models)}

        for ax_idx, algorithm in enumerate(algorithms):
            ax = axes[ax_idx]
            alg_data = by_algorithm[algorithm]

            models_in_alg = sorted(alg_data.keys())
            n_models_in_alg = len(models_in_alg)

            if n_models_in_alg == 0:
                ax.axis("off")
                continue

            x = np.arange(n_models_in_alg)
            width = 0.35

            top_diversity = [alg_data[m]["top_winners_diversity"] for m in models_in_alg]
            other_diversity = [alg_data[m]["other_winners_diversity"] for m in models_in_alg]

            bar_colors = [model_colors[m] for m in models_in_alg]

            ax.bar(x - width/2, top_diversity, width,
                  color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5)
            ax.bar(x + width/2, other_diversity, width,
                  color=bar_colors, alpha=0.4, edgecolor='white', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(models_in_alg, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            if ax_idx == 0:
                ax.set_ylabel('Best-Candidate Code Diversity')

        legend_elements = [
            Patch(facecolor='gray', alpha=0.9, label=f'Top {top_pct_str} Runs'),
            Patch(facecolor='gray', alpha=0.4, label='Other Runs'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=9,
                  bbox_to_anchor=(0.5, 0.02))

        plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    else:
        group_names = [g["group"] for g in groups]
        max_label_len = max(len(name) for name in group_names) if group_names else 10
        fig_width = max(10, min(16, 1.5 * len(groups) + max_label_len * 0.1))
        fig_height = 7 + (max_label_len * 0.03)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        top_diversity = [g["top_winners_diversity"] for g in groups]
        other_diversity = [g["other_winners_diversity"] for g in groups]

        x = np.arange(len(group_names))
        width = 0.35

        ax.bar(x - width/2, top_diversity, width, label=f'Top {top_pct_str} Runs',
               color=PALETTE[0], alpha=0.8)
        ax.bar(x + width/2, other_diversity, width, label='Other Runs',
               color=PALETTE[1], alpha=0.8)

        ax.set_ylabel('Best-Candidate Code Diversity')
        ax.set_xticks(x)
        ax.set_xticklabels([abbreviate_problem(n) for n in group_names], rotation=0, ha='center', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')

        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_topk_winners_diversity(
    database_url: str | None = None,
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get top winners diversity analysis using direct code embeddings."""
    campaigns = query_campaigns_with_best_embeddings(database_url)
    return process_topk_winners_diversity(campaigns, top_pct, group_by, problems)


def get_topk_winners_diversity_figure(
    database_url: str | None = None,
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get top winners diversity figure as PNG using direct code embeddings."""
    data = get_topk_winners_diversity(database_url, top_pct, group_by, problems)
    return generate_topk_winners_diversity_figure(data)
