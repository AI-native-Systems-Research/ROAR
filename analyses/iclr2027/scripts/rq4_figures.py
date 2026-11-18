"""Figures 16-18: per-configuration gain of each rule over one 50-iteration campaign.

Reads results/rq4_roar.json and results/rq4_external.json (no recomputation). Marker = mean
gain over the random half-splits, band = 2.5-97.5% range across splits; row label gives the
number of campaigns and the sizes of the fitting and evaluation halves. Bands that extend
beyond the axis are clipped and their end is printed. ROAR's q5 paper rcParams and palette.
Rows are grouped by problem and ordered by the fitted policy's mean gain.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from iclr2027 import figs, paths  # noqa: E402

import analytics.q5.paper_figures as pf  # noqa: E402,F401  (rcParams)
from analytics.q5.figures import COLORS  # noqa: E402
from iclr2027.harness_gen import METHOD_LABEL, MODEL_LABEL, TASK_LABEL  # noqa: E402

RULES = ["2x25", "5x10", "Restart after 5", "Restart after 10", "Restart after 20", "Fitted (k_stop, n)"]
LEGEND = ["2x25", "5x10", "Restart after 5", "Restart after 10", "Restart after 20", "Fitted policy"]
PALETTE = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000", COLORS["reference"]]
MARKERS = ["o", "s", "^", "D", "v", "*"]


def row_label(corpus, p):
    if corpus == "roar":
        from analytics.utils import display_mechanism, normalize_model_name
        lab = "%s: %s / %s" % (pf.PROBLEMS.get(p["problem"], p["problem"]), display_mechanism(p["algorithm"]),
                               pf._generator(p["model"]))
    else:
        lab = "%s / %s" % (METHOD_LABEL.get(p["algorithm"], p["algorithm"]), MODEL_LABEL.get(p["model"], p["model"]))
    return "%s (%d: %d/%d)" % (lab, p["n_campaigns"], p["nA"], p["nB"])


def forest(corpus, rows, title, xlim, name):
    rows = sorted(rows, key=lambda p: (p["problem"], p["rules"]["Fitted (k_stop, n)"]["mean"]), reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12, 0.62 * n + 1.8))
    step = 0.13
    for i, p in enumerate(rows):
        y0 = n - 1 - i
        for j, r in enumerate(RULES):
            s = p["rules"][r]
            y = y0 + (2.5 - j) * step
            lo, hi = s["p2_5"], s["p97_5"]
            ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [y, y], color=PALETTE[j], lw=1.6, alpha=0.8)
            ax.plot(s["mean"], y, MARKERS[j], color=PALETTE[j], ms=6 if MARKERS[j] != "*" else 9)
            if lo < xlim[0]:
                ax.text(xlim[0], y, "%d" % round(lo), fontsize=9, ha="left", va="bottom")
            if hi > xlim[1]:
                ax.text(xlim[1], y, "%d" % round(hi), fontsize=9, ha="right", va="bottom")
        if i < n - 1 and rows[i + 1]["problem"] != p["problem"]:
            ax.axhline(y0 - 0.5, color="#BBBBBB", lw=0.8)
    ax.axvline(0, color=COLORS["reference"], lw=0.8, ls="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels([row_label(corpus, p) for p in rows][::-1], fontsize=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Gain over one 50-iteration campaign (points)")
    if title:
        ax.set_title(title)
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6) for c, m in zip(PALETTE, MARKERS)]
    fig.legend(handles, LEGEND, loc="lower center", ncol=6, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.12)
    figs.save(fig, name)
    plt.close(fig)


def main():
    for corpus in ("roar", "external"):
        p = os.path.join(paths.RESULTS, f"rq4_{corpus}.json")
        if not os.path.exists(p):
            print("skip", corpus, "(no results)")
            continue
        d = json.load(open(p))
        rows = d["per_configuration"]
        if corpus == "roar":
            forest(corpus, rows, None, (-45, 45), "fig16_rq4_gain_by_configuration")
        else:
            a = [r for r in rows if r["problem"] in ("circle_packing", "second_autocorr_ineq")]
            b = [r for r in rows if r["problem"] == "heilbronn_triangle"]
            forest(corpus, a, "%s and %s" % (TASK_LABEL["circle_packing"], TASK_LABEL["second_autocorr_ineq"]),
                   (-25, 42), "fig17_rq4_external_circle_autocorr")
            forest(corpus, b, TASK_LABEL["heilbronn_triangle"], (-27, 22), "fig18_rq4_external_heilbronn")
    print("figures written")


if __name__ == "__main__":
    main()
