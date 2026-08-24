"""CLI entry point for generating Q3 diversity figures."""

import json
import sys


def main():
    from analytics.q3_diversity import (
        get_diversity_vs_score_scatter,
        get_diversity_vs_score_scatter_figure,
        get_early_diversity_scatter_figure,
        get_topk_winners_diversity_figure,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-figures":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        print(f"Generating figures to {output_dir}...")

        fig_data = get_diversity_vs_score_scatter_figure()
        with open(f"{output_dir}/q3_code_diversity_vs_score.png", "wb") as f:
            f.write(fig_data)
        print("  Generated: q3_code_diversity_vs_score.png")

        fig_data = get_early_diversity_scatter_figure()
        with open(f"{output_dir}/q3_code_early_diversity.png", "wb") as f:
            f.write(fig_data)
        print("  Generated: q3_code_early_diversity.png")

        fig_data = get_topk_winners_diversity_figure()
        with open(f"{output_dir}/q3_code_topk_diversity.png", "wb") as f:
            f.write(fig_data)
        print("  Generated: q3_code_topk_diversity.png")
    else:
        result = get_diversity_vs_score_scatter()
        print(json.dumps(result, indent=2, default=str))
