"""Tab 5: Decile Diversity line plot."""

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from analytics.q3_diversity._common import (
    MARKER_PALETTE,
    PALETTE,
    _compute_cross_diversity,
    abbreviate_problem,
    normalize_model_name,
    query_campaigns_with_best_embeddings,
)


def _bootstrap_diversity_ci(
    embeddings: list[list[float]],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for average pairwise cosine distance.

    Resamples runs (embeddings) with replacement and recomputes diversity
    for each resample. Returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0.0

    emb_array = np.array(embeddings)
    point_estimate = _compute_cross_diversity(embeddings)

    rng = np.random.default_rng(seed=42)
    bootstrap_estimates = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        if len(set(indices)) < 2:
            bootstrap_estimates[b] = 0.0
            continue
        sample = emb_array[indices]
        sim_matrix = cosine_similarity(sample)
        np.fill_diagonal(sim_matrix, 0)
        k = len(sample)
        avg_sim = sim_matrix.sum() / (k * (k - 1))
        bootstrap_estimates[b] = 1 - avg_sim

    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    return point_estimate, lower, upper


def process_decile_diversity(
    campaigns: list[dict[str, Any]],
    problems_filter: list[str] | None = None,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Compute within-decile code diversity across score deciles."""
    if problems_filter:
        campaigns = [c for c in campaigns if c["problem"] in problems_filter]

    campaigns = [c for c in campaigns if c["best_score"] > 0]

    if not campaigns:
        return {"groups": [], "summary": {"total_groups": 0}}

    def get_group_key(camp: dict) -> str:
        if group_by == "model":
            models = camp.get("models_used")
            return ", ".join(sorted(normalize_model_name(m) for m in models)) if models else "Unknown"
        elif group_by == "algorithm":
            return camp.get("algorithm_used") or "Unknown"
        else:
            return camp["problem"]

    by_group: dict[str, list[dict]] = {}
    for camp in campaigns:
        key = get_group_key(camp)
        if key not in by_group:
            by_group[key] = []
        by_group[key].append(camp)

    groups = []
    for group_name, group_camps in sorted(by_group.items()):
        sorted_camps = sorted(group_camps, key=lambda x: x["best_score"])
        n = len(sorted_camps)

        if n < 2:
            continue

        n_deciles = min(10, n)
        deciles = []
        for d in range(n_deciles):
            start_idx = int(round(d * n / n_deciles))
            end_idx = int(round((d + 1) * n / n_deciles))
            bin_camps = sorted_camps[start_idx:end_idx]

            if len(bin_camps) < 2:
                continue

            embeddings = [c["best_embedding"] for c in bin_camps]
            diversity, ci_lower, ci_upper = _bootstrap_diversity_ci(embeddings)
            scores = [c["best_score"] for c in bin_camps]

            deciles.append({
                "decile": d + 1,
                "n_runs": len(bin_camps),
                "diversity": round(float(diversity), 4),
                "diversity_ci_lower": round(float(ci_lower), 4),
                "diversity_ci_upper": round(float(ci_upper), 4),
                "score_range": [round(min(scores), 4), round(max(scores), 4)],
                "score_mean": round(float(np.mean(scores)), 4),
            })

        if deciles:
            x = [d["decile"] for d in deciles]
            y = [d["diversity"] for d in deciles]
            rho, p_val = spearmanr(x, y)
            groups.append({
                "group": group_name,
                "total_runs": n,
                "deciles": deciles,
                "spearman_rho": round(float(rho), 3),
                "spearman_p": round(float(p_val), 4),
            })

    return {
        "groups": groups,
        "group_by": group_by,
        "problems_filter": problems_filter,
        "summary": {
            "total_groups": len(groups),
        },
    }


def generate_decile_diversity_figure(data: dict[str, Any]) -> bytes:
    """Generate line plot of within-decile diversity across score deciles."""
    groups = data.get("groups", [])
    if not groups:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No groups with sufficient data", ha='center', va='center')
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    group_by = data.get("group_by")
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, group in enumerate(groups):
        color = PALETTE[i % len(PALETTE)]
        marker = MARKER_PALETTE[i % len(MARKER_PALETTE)]

        deciles = group["deciles"]
        x = [d["decile"] for d in deciles]
        y = [d["diversity"] for d in deciles]

        base_label = abbreviate_problem(group["group"]) if not group_by else group["group"]
        rho = group.get("spearman_rho")
        p = group.get("spearman_p")
        if rho is not None and p is not None:
            label = f"{base_label} (ρ={rho:.2f}, p={p:.3f})"
        else:
            label = base_label

        ax.plot(x, y, color=color, marker=marker, markersize=8,
                linewidth=2, label=label)

        if deciles and "diversity_ci_lower" in deciles[0]:
            y_lower = [d["diversity_ci_lower"] for d in deciles]
            y_upper = [d["diversity_ci_upper"] for d in deciles]
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

    ax.set_xlabel("Score Decile")
    ax.set_ylabel("Within-Decile Code Diversity")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(i) for i in range(1, 11)])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def get_decile_diversity_figure(
    database_url: str | None = None,
    problems: list[str] | None = None,
    group_by: str | None = None,
) -> bytes:
    """Get decile diversity figure as PNG."""
    campaigns = query_campaigns_with_best_embeddings(database_url)
    data = process_decile_diversity(campaigns, problems, group_by)
    return generate_decile_diversity_figure(data)
