#!/usr/bin/env python3
"""Q5 figures: one inferential, two diagnostic.

The forest plot carries the claim -- it is the visual form of the per-cell
table in evaluate.py. The two sweeps are diagnostics: they show that the
selected policy sits on a smooth landscape rather than a spike, which is
reassurance about the search, not evidence about the policy. They are labelled
as such so they are not read on their own.

Module structure follows the package convention: generate_* renders a figure
from an already-computed result, get_* combines query + process + render.
"""

import io
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytics.q5.evaluate import (
    DEFAULT_TRIALS,
    baseline_score,
    get_evaluation,
)
from analytics.utils import display_mechanism, normalize_model_name
from analytics.q5.recommend import (
    BUDGET_DEFAULT,
    DEFAULT_SEED,
    build_k_grid,
    build_n_grid,
    eligible_problems,
    enumerate_cells,
    load_trajectories,
    simulate,
)

# Matches the styling of the Q1/Q2 figures. Duplicated rather than imported so
# that no existing analysis module is modified.
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Colorblind-safe palette, as used by the Q2 figures.
COLORS = {
    'gain': '#648FFF',
    'loss': '#DC267F',
    'degenerate': '#808080',
    'reference': '#000000',
    'policy': '#785EF0',
    'baseline': '#FE6100',
}


def _render(fig) -> bytes:
    """Render a figure to PNG bytes and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _cell_label(row: dict[str, Any]) -> str:
    """Compact axis label for a cell."""
    model = row["model"]
    if len(model) > 22:
        model = model[:21] + "…"
    return f"{display_mechanism(row['algorithm'])} / {normalize_model_name(model)}"


# =============================================================================
# Figure 1 (inferential): per-cell gain with intervals
# =============================================================================

def generate_gain_forest_figure(data: dict[str, Any]) -> bytes:
    """Per-cell gain over one full run, with refit-bootstrap intervals.

    One row per configuration cell, grouped by problem. The zero line is the
    practitioner's actual outcome, so a bar crossing it is a cell where the
    evidence does not establish an improvement. Saturated cells are drawn in
    grey at zero: nothing can improve them, and the point is that the policy
    does not damage them either.
    """
    rows = list(data["cells"]) + list(data.get("cells_degenerate", []))
    # Problem first so each problem's rows form one contiguous block, then
    # degenerate cells last within the block, then by gain.
    ordered = sorted(
        rows,
        key=lambda r: (
            r["problem"],
            r["classification"] in ("saturated", "dead"),
            r["gain"] if r["gain"] is not None else 0.0,
        ),
    )

    if not ordered:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No evaluable cells", ha="center", va="center")
        ax.axis("off")
        return _render(fig)

    height = max(4.0, 0.55 * len(ordered) + 2.2)
    fig, ax = plt.subplots(figsize=(11, height))

    positions = np.arange(len(ordered))
    for position, row in zip(positions, ordered):
        gain = row["gain"] if row["gain"] is not None else 0.0
        low = row["ci_low"] if row["ci_low"] is not None else gain
        high = row["ci_high"] if row["ci_high"] is not None else gain

        if row["classification"] in ("saturated", "dead"):
            color = COLORS["degenerate"]
        elif low > 0:
            color = COLORS["gain"]
        else:
            color = COLORS["loss"]

        ax.plot([low, high], [position, position], color=color, linewidth=2.5,
                solid_capstyle="butt", zorder=2)
        ax.plot([gain], [position], marker="o", markersize=9, color=color,
                zorder=3)

    ax.axvline(0, color=COLORS["reference"], linestyle="--", linewidth=1.5,
               zorder=1)

    # Colour encodes whether the interval clears zero, so the legend has to say
    # so -- otherwise the pink reads as "got worse" rather than "not
    # established".
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLORS["reference"], linestyle="--",
               linewidth=1.5, label="One full run (current practice)"),
        Line2D([0], [0], color=COLORS["gain"], marker="o", linewidth=2.5,
               label="Interval excludes zero"),
        Line2D([0], [0], color=COLORS["loss"], marker="o", linewidth=2.5,
               label="Interval includes zero"),
        Line2D([0], [0], color=COLORS["degenerate"], marker="o", linewidth=2.5,
               label="No headroom (policy does no harm)"),
    ]

    labels = []
    for row in ordered:
        suffix = ""
        if row["classification"] == "saturated":
            suffix = "  [saturated]"
        elif row["classification"] == "dead":
            suffix = "  [degenerate]"
        labels.append(f"{_cell_label(row)}  (n={row['n_campaigns']}){suffix}")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, len(ordered) - 0.2)

    # Problem boundaries, so the two problems read as separate blocks.
    problems = [row["problem"] for row in ordered]
    for index in range(1, len(problems)):
        if problems[index] != problems[index - 1]:
            ax.axhline(index - 0.5, color="#CCCCCC", linewidth=1, zorder=0)

    # Problem name once per contiguous block, just outside the axes.
    for problem in dict.fromkeys(problems):
        indices = [i for i, p in enumerate(problems) if p == problem]
        ax.text(
            1.015, float(np.mean(indices)), problem,
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=13, rotation=270,
            color="#444444",
        )

    across = data.get("across_cells", {})
    subtitle = ""
    if across.get("median_gain") is not None:
        subtitle = (
            f"median gain {across['median_gain']:+.2f} across "
            f"{across['n_cells']} cells; "
            f"{across['n_positive']}/{across['n_cells']} positive; "
            f"Wilcoxon $p$={across['wilcoxon_p']:.4g}, "
            f"sign test $p$={across['sign_test_p']:.4g}"
        )

    ax.set_xlabel("Gain over one full run (combined_score)")
    ax.set_title(
        f"ROAR policy vs. current practice, per configuration cell\n{subtitle}",
        fontsize=15,
    )
    # Below the axes: any in-axes corner collides with one of the intervals.
    ax.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.06 - 6.0 / (len(ordered) * 12)),
        ncol=2, framealpha=0.95, fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _render(fig)


# =============================================================================
# Figure 2 (diagnostic): the k sweep
# =============================================================================

def generate_k_sweep_figure(data: dict[str, Any]) -> bytes:
    """Score against stagnation threshold, one line per problem.

    Drawn at n=1, where k_stop is the only control on run length. At the
    recommended n the per-run cap already forces short runs, so every k scores
    identically and the sweep would be flat -- the two knobs are partially
    redundant (README section 4.3).
    """
    curves = data["k_sweep"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    palette = [COLORS["gain"], COLORS["policy"], COLORS["loss"]]
    for index, (problem, curve) in enumerate(curves.items()):
        ks = [point["k_stop"] for point in curve]
        scores = [point["mean_score"] for point in curve]
        color = palette[index % len(palette)]
        ax.plot(ks, scores, marker="o", linewidth=2, color=color, label=problem)

        baseline = data["baselines"].get(problem)
        if baseline is not None:
            ax.axhline(baseline, color=color, linestyle=":", linewidth=1.5,
                       alpha=0.8)

    ax.set_xscale("log")
    # The grid is log-spaced, so label its actual values as plain integers;
    # matplotlib's default log ticks collide at this density.
    all_ks = sorted({point["k_stop"] for curve in curves.values()
                     for point in curve})
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])
    ax.minorticks_off()
    ax.set_xlabel(r"Stagnation threshold $k_{stop}$ (log scale)")
    ax.set_ylabel("Best score at fixed budget")
    ax.set_title(
        "Diagnostic: payoff against stopping threshold ($n$=1)\n"
        "dotted lines mark one full run; mechanisms react to stagnation at $k$=10--20",
        fontsize=14,
    )
    # The frameworks' own stagnation thresholds, which is what the
    # recommendation is arguing against.
    ax.axvspan(10, 20, color="#CCCCCC", alpha=0.35, zorder=0)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _render(fig)


# =============================================================================
# Figure 3 (diagnostic): the n curve
# =============================================================================

def generate_n_curve_figure(data: dict[str, Any]) -> bytes:
    """Score against replicate count at equal spend, with and without stopping.

    The interaction is the point: early stopping already generates replicates,
    so once k_stop is applied the curve in n flattens.
    """
    curves = data["n_curve"]
    fig, axes = plt.subplots(
        1, max(1, len(curves)), figsize=(6 * max(1, len(curves)), 5),
        squeeze=False,
    )

    for index, (problem, series) in enumerate(curves.items()):
        ax = axes[0][index]
        ns = [point["n_replicates"] for point in series["without_k"]]

        ax.plot(
            ns, [p["mean_score"] for p in series["without_k"]],
            marker="o", linewidth=2, color=COLORS["baseline"],
            label=r"no stopping ($k$=None)",
        )
        ax.plot(
            ns, [p["mean_score"] for p in series["with_k"]],
            marker="s", linewidth=2, color=COLORS["policy"],
            label=rf"with $k$={series['k_stop']}",
        )

        baseline = data["baselines"].get(problem)
        if baseline is not None:
            ax.axhline(baseline, color=COLORS["reference"], linestyle="--",
                       linewidth=1.5, label="one full run")

        ax.set_xlabel(r"Replicates $n$ at equal spend")
        if index == 0:
            ax.set_ylabel("Best score at fixed budget")
        ax.set_title(problem, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Diagnostic: splitting a fixed budget across replicates", fontsize=16
    )
    fig.tight_layout()
    return _render(fig)


# =============================================================================
# High-Level API
# =============================================================================

def get_diagnostic_curves(
    database_url: str | None = None,
    budget: int = BUDGET_DEFAULT,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    trajectories: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sweep k and n within each problem's largest headroom cell.

    Scoped to a cell rather than the pooled problem, for the same reason the
    headline experiment is: a pooled sweep measures configuration mixing as
    much as budget policy.
    """
    if trajectories is None:
        trajectories = load_trajectories(database_url)
    problems = eligible_problems(trajectories)
    cells = enumerate_cells(trajectories, problems=problems)

    k_sweep: dict[str, list[dict[str, Any]]] = {}
    n_curve: dict[str, dict[str, Any]] = {}
    baselines: dict[str, float] = {}

    for problem in problems:
        candidates = [
            c for c in cells if c["problem"] == problem and c["headroom"]
        ]
        if not candidates:
            continue
        cell = max(candidates, key=lambda c: c["n_campaigns"])
        members = [
            t for t in trajectories
            if t["problem"] == cell["problem"]
            and t["algorithm"] == cell["algorithm"]
            and t["model"] == cell["model"]
        ]
        label = f"{problem}\n{display_mechanism(cell['algorithm'])} / {normalize_model_name(cell['model'])}"

        baselines[label] = round(
            float(np.mean([baseline_score(t, budget) for t in members])), 4
        )

        k_sweep[label] = [
            simulate(
                members, budget=budget, n=1, k_stop=k,
                n_trials=n_trials, seed=seed,
            )
            for k in build_k_grid(budget)
        ]

        best_k = max(
            k_sweep[label], key=lambda p: p["mean_score"] or float("-inf")
        )["k_stop"]
        n_curve[label] = {
            "k_stop": best_k,
            "without_k": [
                simulate(
                    members, budget=budget, n=n, k_stop=None,
                    n_trials=n_trials, seed=seed,
                )
                for n in build_n_grid(budget)
            ],
            "with_k": [
                simulate(
                    members, budget=budget, n=n, k_stop=best_k,
                    n_trials=n_trials, seed=seed,
                )
                for n in build_n_grid(budget)
            ],
        }

    return {
        "budget": budget,
        "baselines": baselines,
        "k_sweep": k_sweep,
        "n_curve": n_curve,
    }


def get_gain_forest_figure(
    database_url: str | None = None,
    budget: int = BUDGET_DEFAULT,
) -> bytes:
    """Primary figure: per-cell gain with intervals."""
    data = get_evaluation(database_url, budget=budget)
    return generate_gain_forest_figure(data)


def get_k_sweep_figure(
    database_url: str | None = None,
    budget: int = BUDGET_DEFAULT,
) -> bytes:
    """Diagnostic figure: payoff against stopping threshold."""
    return generate_k_sweep_figure(
        get_diagnostic_curves(database_url, budget=budget)
    )


def get_n_curve_figure(
    database_url: str | None = None,
    budget: int = BUDGET_DEFAULT,
) -> bytes:
    """Diagnostic figure: budget splitting against replicate count."""
    return generate_n_curve_figure(
        get_diagnostic_curves(database_url, budget=budget)
    )


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(outdir, exist_ok=True)

    trajectories = load_trajectories()

    evaluation = get_evaluation(trajectories=trajectories)
    curves = get_diagnostic_curves(trajectories=trajectories)

    outputs = {
        "q5_gain_forest.png": generate_gain_forest_figure(evaluation),
        "q5_k_sweep.png": generate_k_sweep_figure(curves),
        "q5_n_curve.png": generate_n_curve_figure(curves),
    }
    for name, payload in outputs.items():
        path = os.path.join(outdir, name)
        with open(path, "wb") as handle:
            handle.write(payload)
        print(f"Wrote {path} ({len(payload)} bytes)")
