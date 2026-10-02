"""Figures and the combined table for the appendix sample-size sweep.

Run with::

    python -m experiments.figures.appendix_sample_size_figures
"""

import settings
from experiments.figures.aggregate import label_for, latest_per_experiment, tidy
from experiments.figures.panels import learning_bands, box_metric
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import metric_table, usable, write_table
from experiments.store import load_experiment

EXPERIMENT = "appendix_sample_size"

CELL_LABEL = r"Rollout length $l$ / trajectories $m$"


def main():
    records = latest_per_experiment(load_experiment(EXPERIMENT))

    for solver in settings.CH4_SAMPLE_SOLVERS:
        combined = {}

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
            ax = box_metric(
                rows, ("l", "m"), "training_subopt_final",
                order=cell_order, log="symlog",
            )
            ax.set_xlabel(CELL_LABEL)
            save(ax.figure, f"{stem}_subopt", EXPERIMENT)

            # Plot 2: the training process against env steps
            ax = learning_bands(subset, ("l", "m"), x="env_steps", order=cell_order)
            save(ax.figure, f"{stem}_curves", EXPERIMENT)

            for row in filter(usable, rows):
                merged = combined.setdefault(
                    (row["l"], row["m"], row["seed"]),
                    {"l": row["l"], "m": row["m"], "J_final": row["J_final"]},
                )
                merged[size] = row.get("training_subopt_final")

        if not combined:
            continue

        write_table(
            metric_table(
                list(combined.values()), ("l", "m"),
                metrics=[size for size in settings.INFLATION_SIZES
                         if any(size in row for row in combined.values())],
                column_order=settings.SAMPLE_GRID,
                column_header=r"$(l, m)$",
                caption=(
                    f"{label_for(solver)}: training suboptimality $(J-J^*)/J^*$ per inflation "
                    "size (rows) and sample-size combination (columns)."
                ),
                label=f"tab:appendix-sample-size-{solver.replace('_', '-')}",
                census=False,
            ),
            f"appendix_sample_size_{solver}",
            EXPERIMENT,
        )

    print("wrote figures to", figure_dir(EXPERIMENT))
    print("wrote tables  to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
