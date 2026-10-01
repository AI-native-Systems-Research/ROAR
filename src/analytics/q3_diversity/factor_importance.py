"""Tab 4: Factor Importance bar chart and heatmap."""

import io
from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy import stats

from analytics.q3_diversity._common import (
    COLORS,
    PALETTE,
    _generate_empty_figure,
    _get_database_url,
    _truncate_problem,
    display_mechanism,
    normalize_model_name,
)

FACTOR_NAMES = {
    "context_diversity": "Code Diversity",
    "score_diversity": "Score Diversity",
    "better_ratio": "Better Ratio",
    "worse_ratio": "Worse Ratio",
}

FACTOR_KEYS = ["score_diversity", "context_diversity", "better_ratio", "worse_ratio"]


def query_mutation_factors(
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Query mutation data with all factors needed for factor analysis."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH parent_edges AS (
                    SELECT target_candidate_id as child_id, source_candidate_id as parent_id
                    FROM candidate_edges
                    WHERE edge_type = 'parent'
                ),
                context_edges AS (
                    SELECT target_candidate_id as child_id, source_candidate_id as context_id
                    FROM candidate_edges
                    WHERE edge_type = 'context'
                ),
                context_stats AS (
                    SELECT
                        ce.child_id,
                        COUNT(ctx_m.value) as context_size,
                        MAX(CAST(ctx_m.value AS DOUBLE PRECISION)) as max_context_score,
                        VAR_POP(CAST(ctx_m.value AS DOUBLE PRECISION)) as context_score_variance,
                        SUM(CASE WHEN CAST(ctx_m.value AS DOUBLE PRECISION) > CAST(parent_m.value AS DOUBLE PRECISION) THEN 1 ELSE 0 END) as better_count,
                        SUM(CASE WHEN CAST(ctx_m.value AS DOUBLE PRECISION) < CAST(parent_m.value AS DOUBLE PRECISION) THEN 1 ELSE 0 END) as worse_count
                    FROM context_edges ce
                    JOIN parent_edges pe ON ce.child_id = pe.child_id
                    JOIN measurements ctx_m ON ctx_m.candidate_id = ce.context_id AND ctx_m.name = 'combined_score'
                    JOIN measurements parent_m ON parent_m.candidate_id = pe.parent_id AND parent_m.name = 'combined_score'
                    WHERE ctx_m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                      AND parent_m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                    GROUP BY ce.child_id
                ),
                running_best AS (
                    SELECT
                        c.id as candidate_id,
                        c.campaign_id,
                        MAX(CAST(m.value AS DOUBLE PRECISION)) OVER (
                            PARTITION BY c.campaign_id
                            ORDER BY c.iteration_index
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ) as best_before
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE c.iteration_index IS NOT NULL
                      AND m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                )
                SELECT
                    c.id as candidate_id,
                    c.campaign_id,
                    c.iteration_index,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used,
                    child_m.value as child_score,
                    parent_m.value as parent_score,
                    COALESCE(cs.context_size, 0) as context_size,
                    COALESCE(cs.max_context_score, CAST(parent_m.value AS DOUBLE PRECISION)) as max_context_score,
                    COALESCE(cs.context_score_variance, 0) as context_score_variance,
                    COALESCE(cs.better_count, 0) as better_count,
                    COALESCE(cs.worse_count, 0) as worse_count,
                    COALESCE(c.context_code_diversity, 0) as context_diversity,
                    rb.best_before
                FROM candidates c
                JOIN campaigns camp ON c.campaign_id = camp.id
                JOIN parent_edges pe ON pe.child_id = c.id
                JOIN measurements child_m ON child_m.candidate_id = c.id AND child_m.name = 'combined_score'
                JOIN measurements parent_m ON parent_m.candidate_id = pe.parent_id AND parent_m.name = 'combined_score'
                LEFT JOIN context_stats cs ON cs.child_id = c.id
                LEFT JOIN running_best rb ON rb.candidate_id = c.id
                WHERE c.direct_code_embedding IS NOT NULL
                  AND camp.research_question IS NOT NULL
                  AND cs.context_size > 0
                  AND child_m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                  AND parent_m.value ~ '^-?[0-9]+(\\.[0-9]+)?$'
                ORDER BY c.campaign_id, c.iteration_index
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        problem = _truncate_problem(row["research_question"])
        if problem is None:
            continue

        child_score = float(row["child_score"])
        parent_score = float(row["parent_score"])
        max_context_score = float(row["max_context_score"])
        context_size = row["context_size"]
        context_score_variance = float(row["context_score_variance"])
        better_count = row["better_count"]
        worse_count = row["worse_count"]
        best_before = float(row["best_before"]) if row["best_before"] is not None else None

        baseline = max(parent_score, max_context_score)
        score_delta = child_score - baseline

        if best_before is not None:
            score_delta_global = child_score - best_before
        else:
            score_delta_global = child_score if child_score > 0 else 0.0

        better_ratio = better_count / context_size if context_size > 0 else 0
        worse_ratio = worse_count / context_size if context_size > 0 else 0
        score_diversity = context_score_variance

        results.append({
            "candidate_id": str(row["candidate_id"]),
            "campaign_id": str(row["campaign_id"]),
            "problem": problem,
            "model": normalize_model_name(row["models_used"][0]) if row["models_used"] else "unknown",
            "algorithm": row["algorithm_used"] or "unknown",
            "iteration_index": row["iteration_index"],
            "parent_score": round(parent_score, 4),
            "child_score": round(child_score, 4),
            "max_context_score": round(max_context_score, 4),
            "best_score_so_far": round(best_before, 4) if best_before is not None else None,
            "score_delta": round(score_delta, 4),
            "score_delta_parent_only": round(child_score - parent_score, 4),
            "score_delta_global": round(score_delta_global, 4),
            "context_size": context_size,
            "context_diversity": round(float(row["context_diversity"]), 4),
            "score_diversity": round(score_diversity, 4),
            "better_count": better_count,
            "worse_count": worse_count,
            "better_ratio": round(better_ratio, 4),
            "worse_ratio": round(worse_ratio, 4),
        })

    return results


def query_mutation_factors_problems(
    database_url: str | None = None,
) -> list[str]:
    """Query distinct problems that have mutation factor data."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT camp.research_question
                FROM candidates c
                JOIN campaigns camp ON c.campaign_id = camp.id
                WHERE c.direct_code_embedding IS NOT NULL
                ORDER BY camp.research_question
            """)
            rows = cur.fetchall()

    return [_truncate_problem(r["research_question"]) for r in rows if r["research_question"]]


def get_mutation_factors_problems(
    database_url: str | None = None,
) -> list[str]:
    """Get list of problems that have mutation factor data."""
    return query_mutation_factors_problems(database_url)


def welch_ttest_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> dict[str, float]:
    """Compute Welch's t-test and Cohen's d between two independent groups."""
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    n1, n2 = len(group_a), len(group_b)
    mean_a, mean_b = float(np.mean(group_a)), float(np.mean(group_b))
    s1 = float(np.std(group_a, ddof=1))
    s2 = float(np.std(group_b, ddof=1))

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    mean_diff = mean_a - mean_b
    se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se_diff > 0:
        df_num = (s1**2 / n1 + s2**2 / n2) ** 2
        df_den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else min(n1, n2) - 1
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
    else:
        ci_lower = mean_diff
        ci_upper = mean_diff

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": float(p_value),
        "cohens_d": round(cohens_d, 4),
        "mean_diff": round(mean_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
    }


# Minimum data for a stable random-intercept GLMM fit within a group.
_MIN_PER_CLASS = 5     # improving and non-improving mutations each
_MIN_CAMPAIGNS = 3     # distinct campaigns (random-intercept levels)


def _standardize(series: "Any") -> tuple["Any", float]:
    """Z-score a series; returns (zscored, sd). sd==0 -> all-zero column."""
    sd = float(series.std(ddof=0))
    if sd > 0:
        return (series - float(series.mean())) / sd, sd
    return series * 0.0, 0.0


def _fit_group_glmm(gdf: "Any") -> dict[str, Any] | None:
    """Fit ``improvement ~ z(factors) + z(phase) + (1|campaign)`` for one group.

    A random-intercept Bayesian logistic GLMM (variational inference), matching the
    approach in ``q2_calibration.fit_escape_trend_model``. Reports each factor's
    log-odds change per 1 SD, holding campaign phase fixed and accounting for
    within-campaign clustering. Factors with no variance in the subgroup are
    reported as null effects rather than entered into the design (avoids singular
    fits). Returns None if the model cannot be fit.
    """
    from scipy import stats as sstats
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = gdf.copy()
    active = []
    for col in FACTOR_KEYS + ["phase"]:
        z, sd = _standardize(d[col])
        d["z_" + col] = z
        if sd > 0:
            active.append("z_" + col)

    if not [c for c in active if c != "z_phase"]:
        return None  # no factor varies in this subgroup

    rhs = " + ".join(active)
    try:
        result = BinomialBayesMixedGLM.from_formula(
            f"improvement ~ {rhs}",
            {"campaign": "0 + C(campaign_id)"},
            data=d,
        ).fit_vb()
    except Exception:
        return None

    names = list(result.model.exog_names)

    def coef_at(zname: str) -> dict[str, Any] | None:
        if zname not in names:
            return None
        i = names.index(zname)
        mean, sd = float(result.fe_mean[i]), float(result.fe_sd[i])
        zval = mean / sd if sd > 0 else 0.0
        p = float(2 * (1 - sstats.norm.cdf(abs(zval))))
        return {
            "coef": round(mean, 4),
            "p": p,
            "ci_lower": round(mean - 1.96 * sd, 4),
            "ci_upper": round(mean + 1.96 * sd, 4),
        }

    factors: dict[str, Any] = {}
    for factor in FACTOR_KEYS:
        c = coef_at("z_" + factor)
        # no variance in subgroup -> null effect (coef 0, p None so it renders neutral)
        factors[factor] = c or {"coef": 0.0, "p": None, "ci_lower": 0.0, "ci_upper": 0.0}

    out: dict[str, Any] = {"factors": factors}
    phase = coef_at("z_phase")
    if phase is not None:
        out["phase"] = {"coef": phase["coef"], "p": phase["p"]}
    out["random_effect_sd"] = round(float(result.vcp_mean[0]), 3)
    return out


def compute_factor_importance(
    data: list[dict[str, Any]],
    group_by: str | None = None,
    problems_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Factor importance via random-intercept logistic GLMM.

    For each group, fits ``improvement ~ z(factors) + z(phase) + (1|campaign)`` and
    reports each factor's log-odds change per 1 SD, with within-campaign clustering
    and iteration-phase controlled.
    """
    import pandas as pd

    if problems_filter:
        data = [r for r in data if r["problem"] in problems_filter]

    if not data:
        return {"groups": [], "summary": {"total_mutations": 0}}

    df = pd.DataFrame(data)
    df["improvement"] = (df["score_delta_global"] > 0).astype(int)

    # phase = normalized position within a campaign (0 = start, 1 = end); needs iteration
    df = df[df["iteration_index"].notna()].copy()
    if df.empty:
        return {"groups": [], "summary": {"total_mutations": 0}}
    df["iteration_index"] = df["iteration_index"].astype(float)
    max_iter = df.groupby("campaign_id")["iteration_index"].transform("max")
    df["phase"] = np.where(max_iter > 0, df["iteration_index"] / max_iter, 0.0)

    df = df.dropna(subset=FACTOR_KEYS).copy()
    if df.empty or df["improvement"].nunique() < 2:
        return {"groups": [], "summary": {"total_mutations": int(len(df))}}

    if group_by == "model":
        df["_group"] = df["model"].fillna("unknown")
    elif group_by == "algorithm":
        df["_group"] = df["algorithm"].fillna("unknown")
    elif group_by == "model_algorithm":
        df["_group"] = df["model"].fillna("unknown") + " + " + df["algorithm"].fillna("unknown")
    else:
        df["_group"] = "all"

    groups = []
    for group_name in sorted(df["_group"].unique()):
        gdf = df[df["_group"] == group_name]
        n_imp = int((gdf["improvement"] == 1).sum())
        n_non = int((gdf["improvement"] == 0).sum())
        if n_imp < _MIN_PER_CLASS or n_non < _MIN_PER_CLASS:
            continue
        if gdf["campaign_id"].nunique() < _MIN_CAMPAIGNS:
            continue

        fit = _fit_group_glmm(gdf)
        if fit is None:
            continue

        factors = fit["factors"]
        correlations = {f: factors[f]["coef"] for f in FACTOR_KEYS}
        groups.append({
            "group": group_name,
            "n": n_imp + n_non,
            "n_improvements": n_imp,
            "n_non_improvements": n_non,
            # key name kept as "correlations"/"cohens_d" for figure compatibility;
            # values are now GLMM log-odds per SD, not Cohen's d.
            "correlations": correlations,
            "cohens_d": correlations,
            "p_values": {f: factors[f]["p"] for f in FACTOR_KEYS},
            "ci_lower": {f: factors[f]["ci_lower"] for f in FACTOR_KEYS},
            "ci_upper": {f: factors[f]["ci_upper"] for f in FACTOR_KEYS},
            "phase_coef": fit.get("phase", {}).get("coef"),
            "random_effect_sd": fit.get("random_effect_sd"),
            "improvement_rate": round(n_imp / (n_imp + n_non), 4),
        })

    return {
        "group_by": group_by,
        "problems_filter": problems_filter,
        "groups": groups,
        "estimator": "glmm_logit_random_intercept",
        "summary": {
            "total_groups": len(groups),
            "total_mutations": sum(g["n"] for g in groups),
            "total_improvements": sum(g["n_improvements"] for g in groups),
        },
    }


def _generate_factor_heatmap(
    groups: list[dict[str, Any]],
    group_by: str,
) -> bytes:
    """Generate heatmap for factor importance by model or model+algorithm."""
    # Symmetric color scale from the data (GLMM log-odds can exceed the old +/-1
    # Cohen's d range). Floor at 1.0 so near-null panels don't over-saturate.
    _vals = [abs(v) for g in groups for v in g.get("correlations", {}).values()
             if v is not None and not np.isnan(v)]
    vabs = max([*_vals, 1.0])

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
            by_algorithm[algorithm][model] = {
                "correlations": g.get("correlations", {}),
                "p_values": g.get("p_values", {}),
                "n": g.get("n", 0),
                "n_improvements": g.get("n_improvements", 0),
            }

        algorithms = sorted(by_algorithm.keys())
        algorithms = [alg for alg in algorithms if by_algorithm[alg]]
        n_algs = len(algorithms)

        if n_algs == 0:
            return _generate_empty_figure("No data for heatmap")

        all_models = sorted(set(
            m for alg_data in by_algorithm.values() for m in alg_data.keys()
            if alg_data[m].get("correlations")
        ))

        if not all_models:
            return _generate_empty_figure("No data for heatmap")

        n_models = len(all_models)
        fig, axes = plt.subplots(
            1, n_algs, figsize=(6 * n_algs, max(6, n_models * 0.65 + 2)),
            squeeze=False)
        axes = axes[0]

        cmap = plt.cm.RdYlGn
        vmin, vmax = -vabs, vabs

        im = None
        for idx, algorithm in enumerate(algorithms):
            ax = axes[idx]
            alg_data = by_algorithm[algorithm]
            models_present = [m for m in all_models if m in alg_data and alg_data[m].get("correlations")]

            matrix = np.full((len(models_present), len(FACTOR_KEYS)), np.nan)
            p_matrix = np.full((len(models_present), len(FACTOR_KEYS)), np.nan)
            n_values = []

            for row_idx, model in enumerate(models_present):
                corrs = alg_data[model]["correlations"]
                p_vals = alg_data[model].get("p_values", {})
                n_values.append(alg_data[model].get("n", 0))
                for col_idx, factor in enumerate(FACTOR_KEYS):
                    if factor in corrs:
                        matrix[row_idx, col_idx] = corrs[factor]
                    if p_vals.get(factor) is not None:
                        p_matrix[row_idx, col_idx] = p_vals[factor]

            im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

            ax.set_xticks(np.arange(len(FACTOR_KEYS)))
            ax.set_xticklabels(
                [FACTOR_NAMES[f] for f in FACTOR_KEYS],
                rotation=30, ha="right", fontsize=13)
            ax.set_yticks(np.arange(len(models_present)))
            y_labels = [f"{m} (n={n_values[i]})" for i, m in enumerate(models_present)]
            ax.set_yticklabels(y_labels, fontsize=12)

            for i in range(len(models_present)):
                for j in range(len(FACTOR_KEYS)):
                    val = matrix[i, j]
                    if np.isnan(val):
                        continue
                    p_val = p_matrix[i, j] if not np.isnan(p_matrix[i, j]) else None
                    is_sig = p_val is not None and p_val < 0.05
                    text_color = "white" if abs(val) > 0.6 * vabs else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=12, color=text_color)
                    if not is_sig:
                        ax.add_patch(Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch='///', alpha=0.4,
                            edgecolor='gray', linewidth=0.5))

            ax.set_title(display_mechanism(algorithm), fontweight="bold", fontsize=16)

        plt.tight_layout()

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                               fraction=0.015, pad=0.06)
            cbar.set_label("Log-odds per SD (GLMM)", fontsize=14)
            cbar.ax.tick_params(labelsize=12)

    else:
        models = [g["group"] for g in groups]

        matrix = np.full((len(FACTOR_KEYS), len(models)), np.nan)
        p_matrix = np.full((len(FACTOR_KEYS), len(models)), np.nan)
        for col_idx, g in enumerate(groups):
            corrs = g.get("correlations", {})
            p_vals = g.get("p_values", {})
            for row_idx, factor in enumerate(FACTOR_KEYS):
                if factor in corrs:
                    matrix[row_idx, col_idx] = corrs[factor]
                if p_vals.get(factor) is not None:
                    p_matrix[row_idx, col_idx] = p_vals[factor]

        fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.5), 6))

        cmap = plt.cm.RdYlGn
        vmin, vmax = -vabs, vabs

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        x_labels = [f"{m}\n(n={groups[i].get('n', '?')})" for i, m in enumerate(models)]
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)
        ax.set_yticks(np.arange(len(FACTOR_KEYS)))
        ax.set_yticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], fontsize=16)

        for i in range(len(FACTOR_KEYS)):
            for j in range(len(models)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                p_val = p_matrix[i, j] if not np.isnan(p_matrix[i, j]) else None
                is_sig = p_val is not None and p_val < 0.05
                text_color = "white" if abs(val) > 0.6 * vabs else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=14, color=text_color)
                if not is_sig:
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch='///', alpha=0.4,
                        edgecolor='gray', linewidth=0.5))

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
        cbar.set_label("Log-odds per SD (GLMM)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_factor_importance_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate visualization for factor importance."""
    groups = data.get("groups", [])
    if not groups:
        return _generate_empty_figure("No mutation factor data available")

    group_by = data.get("group_by")
    n_groups = len(groups)

    if group_by is None or n_groups == 1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        group = groups[0]
        correlations = group.get("correlations", {})
        p_values = group.get("p_values", {})

        factors = []
        values = []
        colors = []
        p_vals_list = []

        for factor in FACTOR_KEYS:
            if factor in correlations:
                factors.append(FACTOR_NAMES[factor])
                val = correlations[factor]
                values.append(val)
                colors.append(COLORS["high"] if val > 0 else COLORS["low"])
                p_vals_list.append(p_values.get(factor))

        if not factors:
            return _generate_empty_figure("Insufficient data to compute correlations")

        x_pos = np.arange(len(factors))
        ax.bar(x_pos, values, color=colors, alpha=0.8, width=0.6)

        y_range = max(abs(v) for v in values) if values else 0.5
        y_pad = y_range * 0.05
        for i, (val, p_val) in enumerate(zip(values, p_vals_list)):
            if p_val is None:
                continue
            if p_val < 0.001:
                p_str = "p<.001"
            elif p_val < 0.01:
                p_str = f"p={p_val:.3f}"
            else:
                p_str = f"p={p_val:.2f}"
            if val >= 0:
                y_pos = val + y_pad
                va = 'bottom'
            else:
                y_pos = val - y_pad
                va = 'top'
            ax.text(i, y_pos, p_str, ha='center', va=va,
                    fontsize=13, color='0.2', fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(factors, rotation=30, ha='right')
        ax.set_ylabel("Log-odds per SD")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        ax.set_ylim(-1.35, 1.35)

        n_imp = group.get("n_improvements", 0)
        n_non = group.get("n_non_improvements", 0)
        ax.text(0.02, 0.98, f"$|C^{{\\uparrow}}|={n_imp}$, $|C^{{\\downarrow}}|={n_non}$",
                transform=ax.transAxes, fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    elif group_by in ("model", "model_algorithm"):
        return _generate_factor_heatmap(groups, group_by)

    else:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(FACTOR_KEYS))
        width = 0.8 / n_groups

        for i, group in enumerate(groups):
            correlations = group.get("correlations", {})
            values = [correlations.get(f, 0) for f in FACTOR_KEYS]
            offset = (i - n_groups / 2 + 0.5) * width
            label = display_mechanism(group["group"])
            if len(label) > 20:
                label = label[:18] + "..."
            n_imp = group.get('n_improvements', 0)
            n_non = group.get('n_non_improvements', 0)
            ax.bar(x + offset, values, width, label=f"{label} ($|C^\\uparrow|$={n_imp}, $|C^\\downarrow|$={n_non})",
                   color=PALETTE[i % len(PALETTE)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], rotation=15, ha="right")
        ax.set_ylabel("Log-odds per SD")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


@lru_cache(maxsize=8)
def _cached_mutation_factors(database_url: str | None) -> list[dict[str, Any]]:
    """Cache the raw mutation-factor query (static research dataset per process)."""
    return query_mutation_factors(database_url)


@lru_cache(maxsize=64)
def _cached_factor_importance(
    database_url: str | None,
    group_by: str | None,
    problems_key: tuple[str, ...] | None,
) -> dict[str, Any]:
    """Cache computed GLMM importance keyed on (db, group_by, problems).

    The GLMM fits are the expensive step (seconds), so memoizing here lets the
    FastAPI endpoint warm once and serve subsequent requests instantly. Cleared on
    process restart. Call ``_cached_factor_importance.cache_clear()`` after a data
    refresh.
    """
    problems = list(problems_key) if problems_key else None
    raw_data = _cached_mutation_factors(database_url)
    return compute_factor_importance(raw_data, group_by, problems)


def get_factor_importance_figure(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get factor importance bar chart as PNG (cached GLMM estimates)."""
    problems_key = tuple(problems) if problems else None
    importance_data = _cached_factor_importance(database_url, group_by, problems_key)
    return generate_factor_importance_figure(importance_data)
