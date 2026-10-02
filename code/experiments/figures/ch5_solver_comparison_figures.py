"""Figures and tables for chapter 5's cross-solver comparison.

Run with::

    python -m experiments.figures.ch5_solver_comparison_figures
"""

import settings
from experiments.figures.aggregate import (
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import (
    complexity_bars,
    learning_bands,
    trajectory_overlay,
    box_metric,
)
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.store import load_experiment

EXPERIMENT = "ch5_solver_comparison"

ORDER = tuple(settings.CH5_COMPARISON_SOLVERS)


def main():
    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = with_suboptimality(tidy(records), by=("seed",))

    ax = box_metric(
        rows, "solver", "training_subopt_final", order=ORDER, log="symlog"
    )
    save(ax.figure, "ch5_solvers_training_subopt", EXPERIMENT)

    ax = learning_bands(records, "solver", x="env_steps", order=ORDER)
    save(ax.figure, "ch5_solvers_curves_steps", EXPERIMENT)

    axes = trajectory_overlay(records, "solver", dt=0.01, order=ORDER)
    save(axes[0].figure, "ch5_solvers_trajectories", EXPERIMENT)

    ax = box_metric(rows, "solver", "subopt", order=ORDER, log="symlog")
    save(ax.figure, "ch5_solvers_subopt", EXPERIMENT)

    ax = complexity_bars(rows, "solver", order=ORDER)
    save(ax.figure, "ch5_solvers_complexity", EXPERIMENT)

    write_table(
        metric_table(
            rows, "solver",
            metrics=("J_final", "training_subopt_final", "subopt",
                     "policy_complexity", "rank_final",
                     "iterations", "total_env_steps", "best_env_steps", "total_time"),
            column_order=ORDER,
            column_header="Solver",
            caption=(
                f"Chapter 5 solvers under \\emph{{{settings.CH5_PIXEL_INFLATION}}} inflation, "
                "without noise."
            ),
            label="tab:ch5-solvers",
        ),
        "ch5_solvers",
        EXPERIMENT,
    )

    print("wrote figures to", figure_dir(EXPERIMENT))
    print("wrote tables  to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
