"""Q5: Closed-Loop Configuration (ROAR as an ADRS configurator)."""

from analytics.q5.recommend import (
    BUDGET_DEFAULT,
    CELL_MIN_CAMPAIGNS,
    MIN_RUN_ITERATIONS,
    build_k_grid,
    build_n_grid,
    enumerate_cells,
    get_recommendation,
    load_trajectories,
    simulate,
)
from analytics.q5.evaluate import (
    get_evaluation,
    get_per_problem,
)
from analytics.q5.figures import (
    get_diagnostic_curves,
    get_gain_forest_figure,
    get_k_sweep_figure,
    get_n_curve_figure,
)

__all__ = [
    # Recommender / shared primitives
    "BUDGET_DEFAULT",
    "CELL_MIN_CAMPAIGNS",
    "MIN_RUN_ITERATIONS",
    "build_k_grid",
    "build_n_grid",
    "enumerate_cells",
    "get_recommendation",
    "load_trajectories",
    "simulate",
    # Headline experiment
    "get_evaluation",
    "get_per_problem",
    # Figures
    "get_diagnostic_curves",
    "get_gain_forest_figure",
    "get_k_sweep_figure",
    "get_n_curve_figure",
]
