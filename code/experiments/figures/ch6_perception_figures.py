"""Figures and tables for chapter 6's architecture comparison.

Run with::

    python -m experiments.figures.ch6_perception_figures
"""

import settings
from experiments.figures.aggregate import (
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import (
    box_metric,
    complexity_bars,
    learning_bands,
    trajectory_overlay,
)
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.store import load_experiment

EXPERIMENT = "ch6_perception"

ORDER = tuple(settings.CH6_SOLVERS)

# Baseline scorer
REFERENCE = "oracle_coarse"


def main():
    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = with_suboptimality(tidy(records), reference=REFERENCE, by=("seed",))

    ax = box_metric(
        rows, "solver", "training_subopt_final", order=ORDER, log="symlog"
    )
    save(ax.figure, "ch6_perception_training_subopt", EXPERIMENT)

    ax = learning_bands(records, "solver", x="env_steps", order=ORDER)
    save(ax.figure, "ch6_perception_curves_steps", EXPERIMENT)

    axes = trajectory_overlay(records, "solver", dt=0.01, order=ORDER)
    save(axes[0].figure, "ch6_perception_trajectories", EXPERIMENT)

    ax = box_metric(rows, "solver", "subopt", order=ORDER, log="symlog")
    save(ax.figure, "ch6_perception_subopt", EXPERIMENT)

    ax = complexity_bars(rows, "solver", order=ORDER)
    save(ax.figure, "ch6_perception_complexity", EXPERIMENT)

    for metric in ("r2_position", "r2_velocity"):
        ax = box_metric(rows, "solver", metric, order=ORDER, log=False)
        save(ax.figure, f"ch6_perception_{metric}", EXPERIMENT)

    for metric in ("off_screen_rate", "null_waste_mean"):
        ax = box_metric(rows, "solver", metric, order=ORDER, log=False)
        save(ax.figure, f"ch6_perception_{metric}", EXPERIMENT)

    write_table(
        metric_table(
            rows, "solver",
            metrics=("J_final", "training_subopt_final", "subopt",
                     "r2_position", "r2_velocity", "latent_dim",
                     "off_screen_rate", "null_waste_mean",
                     "policy_complexity", "iterations", "total_env_steps", "best_env_steps", "total_time"),
            column_order=ORDER,
        ),
        "ch6_perception",
        EXPERIMENT,
    )


if __name__ == "__main__":
    main()
