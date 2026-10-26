"""Tables, learning curves and the per-size choice of the model-based S-PI's nuclear-norm weight.

Run with::

    python -m experiments.figures.appendix_spi_lambda_figures
"""

import matplotlib.pyplot as plt

import settings
from experiments.appendix_spi_lambda import EXPERIMENT, SIZES, SOLVERS
from experiments.figures.aggregate import label_for, latest_per_experiment, tidy
from experiments.figures.panels import learning_bands
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.sweeps import median, choose, print_choice, sweep_tables
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("training_subopt_final", "rank_final", "iterations")

# Rank of the optimal gain, which S-PI should recover
TARGET_RANK = 1


def render():
    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = tidy(records)

    sweep_tables(rows, EXPERIMENT, SOLVERS, SIZES, "lambda",
                 settings.SPI_LAMBDAS, METRICS, r"$\lambda$")

    for solver in SOLVERS:
        for size in SIZES:
            cell = [
                r for r in records
                if r.solver == solver and r.condition.get("size") == size
            ]
            if not cell:
                continue

            ax = learning_bands(
                cell, "lambda", x="iteration",
                order=[(lambda_,) for lambda_ in settings.SPI_LAMBDAS],
            )
            ax.set_title(f"{label_for(solver)}, {size} inflation")
            ax.set_ylim(bottom=0)
            save(ax.figure, f"{EXPERIMENT}_{solver}_{size}", EXPERIMENT)
            plt.close(ax.figure)

    # Lowest suboptimality among the weights that recover the rank on every seed
    best = choose(
        rows, SOLVERS, SIZES, "lambda",
        admissible=lambda solver, bucket: all(
            r.get("rank_final") == TARGET_RANK for r in bucket
        ),
        score=lambda solver, bucket: median(bucket, "training_subopt_final"),
    )
    print_choice(best, "lambda_")


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return


    print("wrote figures to", figure_dir(EXPERIMENT))
    print("wrote tables to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
