"""Tables and the per-size choice of the model-based S-PI's nuclear-norm weight.

Run with::

    python -m experiments.figures.appendix_spi_lambda_figures
"""

import settings
from experiments.appendix_spi_lambda import EXPERIMENT, SIZES, SOLVERS
from experiments.figures.aggregate import latest_per_experiment, tidy
from experiments.figures.style import table_dir
from experiments.figures.sweeps import median, choose, print_choice, sweep_tables
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("J_final", "training_subopt_final", "rank_final", "test_cost_mean",
           "iterations", "spectral_radius_final")

# Rank of the optimal gain, which S-PI should recover
TARGET_RANK = 1


def render():
    rows = tidy(latest_per_experiment(load_experiment(EXPERIMENT)))

    sweep_tables(rows, EXPERIMENT, SOLVERS, SIZES, "lambda",
                 settings.SPI_LAMBDAS, METRICS, r"$\lambda$")

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

    print("wrote tables to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
