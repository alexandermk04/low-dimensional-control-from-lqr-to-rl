"""Figures and table for the appendix sample-size sweep.

Run with::

    python -m experiments.figures.appendix_sample_size_figures
"""

import settings
from experiments.figures.aggregate import latest_per_experiment, tidy
from experiments.figures.panels import learning_bands, box_metric
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.store import load_experiment

EXPERIMENT = "appendix_sample_size"

CELL_LABEL = r"Rollout length $l$ / trajectories $m$"


def main():
    records = latest_per_experiment(load_experiment(EXPERIMENT))

    for solver in settings.CH4_SAMPLE_SOLVERS:
        for size in settings.INFLATION_SIZES:
            subset = [
                r for r in records
                if r.solver == solver and r.condition.get("size") == size
            ]
            if not subset:
                print(f"  no runs for {solver} at {size}, skipping")
                continue

            cell_order = settings.SAMPLE_GRID

            rows = tidy(subset)

            stem = f"appendix_sample_size_{solver}_{size}"

            # Plot 1: true cost per sample-size combination.
            ax = box_metric(rows, ("l", "m"), "training_subopt_final", order=cell_order)
            ax.set_xlabel(CELL_LABEL)
            save(ax.figure, f"{stem}_subopt", EXPERIMENT)

            # Plot 6: accumulated cost under test protocol.
            ax = box_metric(rows, ("l", "m"), "test_cost_mean", order=cell_order)
            ax.set_xlabel(CELL_LABEL)
            save(ax.figure, f"{stem}_testcost", EXPERIMENT)

            # Plot 2: the training process against env steps
            ax = learning_bands(subset, ("l", "m"), x="env_steps", order=cell_order)
            save(ax.figure, f"{stem}_curves", EXPERIMENT)

            # Table 1.
            write_table(
                metric_table(
                    rows, ("l", "m"),
                    metrics=("J_final", "training_subopt_final", "test_cost_mean"),
                    column_order=cell_order,
                    column_header=r"$(l, m)$",
                    caption=(
                        f"{solver} at the {size} inflation: cost after "
                        "training, suboptimality and accumulated test cost for "
                        "each sample-size combination."
                    ),
                    label=(
                        "tab:appendix-sample-size-"
                        f"{solver.replace('_', '-')}-{size}"
                    ),
                ),
                stem,
                EXPERIMENT,
            )

    print("wrote figures to", figure_dir(EXPERIMENT))
    print("wrote tables  to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
