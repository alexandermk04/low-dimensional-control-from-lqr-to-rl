"""Tables for the appendix on the model-free S-PI's non-convergence.

Run with::

    python -m experiments.figures.appendix_spi_mf_figures
"""

import settings
from experiments.appendix_spi_mf import EXPERIMENT, SIZES
from experiments.figures.aggregate import latest_per_experiment, tidy
from experiments.figures.style import table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("J_final", "training_subopt_final", "rank_final",
           "spectral_radius_final", "iterations", "total_env_steps", "best_env_steps")


def _slug(value: float) -> str:
    """A filename-safe stem for one hyperparameter value."""
    return f"{value:g}".replace("-", "m").replace("+", "").replace(".", "p")


def render():
    """One table per (inflation size, lambda)."""
    rows = tidy(latest_per_experiment(load_experiment(EXPERIMENT)))

    for size in SIZES:
        at_size = [r for r in rows if r.get("size") == size]
        if not at_size:
            continue

        for lambda_ in settings.SPI_MF_LAMBDAS:
            cell = [r for r in at_size if r.get("lambda") == lambda_]
            if not cell:
                continue

            write_table(
                metric_table(
                    cell, "stepsize",
                    metrics=METRICS,
                    column_order=settings.SPI_MF_STEPSIZES,
                    column_header="Step size",
                    caption=(
                        f"Model-free S-PI at lambda={lambda_:g}, {size} inflation, over "
                        f"{len(settings.SEEDS)} seeds."
                    ),
                    label=f"tab:appendix-spimf-{size}-{_slug(lambda_)}",
                ),
                f"appendix_spi_mf_{size}_{_slug(lambda_)}",
                EXPERIMENT,
            )


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return

    print("wrote tables to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
