#!/usr/bin/env python3
"""Q5 figures as they appear in the paper.

figures.py renders the analysis-facing plates. This module renders the same
numbers under the paper's own vocabulary -- "configuration" rather than "cell",
"campaign" rather than "run" -- with three presentation differences:

  1. Only the tested configurations are plotted; the rest are reported in the
     appendix text.
  2. Each row carries the fitted policy (k_stop, n).
  3. The subtitle quotes the sign test only, matching the paper's reporting
     convention of 95% intervals plus the sign test.

Reads q5_evaluation.json written by evaluate.py, so it never touches the
database and cannot disagree with the recorded run.

Usage:
    python -m analytics.q5.paper_figures <results_dir> <paper_figures_dir>
"""

import json
import os
import re
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analytics.q5.figures import COLORS  # same colorblind-safe palette
from analytics.utils import MECHANISM_DISPLAY_NAMES as MECHANISMS, normalize_model_name

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

# Display names only; the stored values are untouched (MECHANISMS is shared in analytics.utils).
PROBLEMS = {"Bounded 2D Knapsack": "Knapsack",
            "Polynomino Packing": "Polynomino"}


def _generator(model: str) -> str:
    """Paper display name: the shared model map, then drop any trailing release date."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", normalize_model_name(model))


def _write(fig, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def gain_forest(data: dict[str, Any], path: str) -> None:
    """Per-configuration gain with 95% intervals, grouped by problem."""
    rows = list(data["cells"])

    # Index 0 is the bottom row: problems in alphabetical order from the
    # bottom, best gain on top within each problem.
    problems = sorted({r["problem"] for r in rows})
    ordered = sorted(rows, key=lambda r: (problems.index(r["problem"]),
                                          r["gain"]))

    height = 0.5 * len(ordered) + 2.6
    fig, ax = plt.subplots(figsize=(12, height))

    for position, row in enumerate(ordered):
        color = COLORS["gain"] if row["ci_low"] > 0 else COLORS["loss"]
        ax.plot([row["ci_low"], row["ci_high"]], [position, position],
                color=color, linewidth=2.5, solid_capstyle="butt", zorder=2)
        ax.plot([row["gain"]], [position], marker="o", markersize=9,
                color=color, zorder=3)

    ax.axvline(0, color=COLORS["reference"], linestyle="--", linewidth=1.5,
               zorder=1)

    labels = [f"{MECHANISMS.get(r['algorithm'], r['algorithm'])} / "
              f"{_generator(r['model'])}  ({r['n_campaigns']} campaigns)"
              for r in ordered]
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(ordered) - 0.3)

    # Fitted policy as a right-hand column, then one rotated name per problem.
    axis = ax.get_yaxis_transform()
    ax.text(1.02, len(ordered) - 0.3, r"$(k_{\mathrm{stop}}, n)$",
            transform=axis, va="bottom", ha="left", fontsize=14)
    for position, row in enumerate(ordered):
        ax.text(1.02, position, f"({row['k_stop']}, {row['n_replicates']})",
                transform=axis, va="center", ha="left", fontsize=14)

    blocks = [r["problem"] for r in ordered]
    for index in range(1, len(blocks)):
        if blocks[index] != blocks[index - 1]:
            ax.axhline(index - 0.5, color="#BBBBBB", linewidth=1, zorder=0)
    for problem in dict.fromkeys(blocks):
        indices = [i for i, value in enumerate(blocks) if value == problem]
        ax.text(1.19, float(np.mean(indices)), PROBLEMS.get(problem, problem),
                transform=axis, va="center", ha="left", fontsize=14,
                rotation=270, color="#444444")

    across = data["across_cells"]
    ax.set_title(
        f"Median gain {across['median_gain']:+.1f} points; "
        f"{across['n_positive']}/{across['n_cells']} configurations positive "
        f"(sign test $p$={across['sign_test_p']:.4f})",
        fontsize=15, pad=30)
    ax.set_xlabel("Gain in best score over current practice (points)")

    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["reference"], linestyle="--",
                   linewidth=1.5,
                   label="Current practice (one campaign, full budget)"),
            Line2D([0], [0], color=COLORS["gain"], marker="o", linewidth=2.5,
                   label="Interval excludes zero"),
            Line2D([0], [0], color=COLORS["loss"], marker="o", linewidth=2.5,
                   label="Interval includes zero"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.11),
        ncol=3, framealpha=0.95, fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _write(fig, path)


if __name__ == "__main__":
    results = sys.argv[1] if len(sys.argv) > 1 else "results"
    outdir = sys.argv[2] if len(sys.argv) > 2 else results
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(results, "q5_evaluation.json")) as handle:
        evaluation = json.load(handle)
    gain_forest(evaluation, os.path.join(outdir, "rq5_gain_forest.png"))
