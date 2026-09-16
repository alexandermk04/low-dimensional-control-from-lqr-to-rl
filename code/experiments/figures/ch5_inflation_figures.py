"""Figures and tables for chapter 5's inflation-size experiment.

Run with::

    python -m experiments.figures.ch5_inflation_figures
"""

import settings
from experiments.figures.aggregate import latest_per_experiment, tidy
from experiments.figures.panels import (
    complexity_bars,
    learning_bands,
    trajectory_overlay,
    box_metric,
)
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.store import MissingExperimentError, load_experiment

EXPERIMENTS = ("ch5_inflation_sizes",)

SIZE_ORDER = tuple(settings.CH5_INFLATION_SIZES)


def render(experiment: str):
    """Draw one inflation sweep, one file set per solver in it."""
    records = latest_per_experiment(load_experiment(experiment))

    solvers = sorted({r.solver for r in records})

    for solver in solvers:
        subset = [r for r in records if r.solver == solver]
        rows = tidy(subset)

        short = solver.replace("_", "-")

        # Plot 1
        ax = box_metric(rows, "size", "training_subopt_final", order=SIZE_ORDER)
        ax.set_xlabel("Inflation size")
        save(ax.figure, f"ch5_inflation_subopt_{solver}", experiment)

        # Plot 2
        ax = learning_bands(subset, "size", x="env_steps", order=SIZE_ORDER)
        save(ax.figure, f"ch5_inflation_curves_{solver}", experiment)

        # Plot 3
        axes = trajectory_overlay(subset, "size", dt=0.01, order=SIZE_ORDER)
        save(axes[0].figure, f"ch5_inflation_trajectories_{solver}", experiment)

        # Plot 6
        ax = box_metric(rows, "size", "test_cost_mean", order=SIZE_ORDER)
        ax.set_xlabel("Inflation size")
        save(ax.figure, f"ch5_inflation_testcost_{solver}", experiment)

        # Plot 7
        ax = complexity_bars(rows, "size", order=SIZE_ORDER)
        ax.set_xlabel("Inflation size")
        save(ax.figure, f"ch5_inflation_complexity_{solver}", experiment)

        # Table 2
        write_table(
            metric_table(
                rows, "size",
                metrics=("J_final", "training_subopt_final", "test_cost_mean",
                         "policy_complexity", "rank_final"),
                column_order=SIZE_ORDER,
                column_header="Inflation",
                caption=f"{solver} across inflation sizes.",
                label=f"tab:ch5-inflation-{short}",
            ),
            f"ch5_inflation_{solver}",
            experiment,
        )


def main():
    for experiment in EXPERIMENTS:
        try:
            render(experiment)
        except MissingExperimentError as error:
            print(f"skipping {experiment}: {error}")
            continue

        print("wrote figures to", figure_dir(experiment))
        print("wrote tables  to", table_dir(experiment))


if __name__ == "__main__":
    main()
