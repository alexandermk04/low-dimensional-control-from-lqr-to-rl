"""Tables and the per-size choice of the model-free S-PI's nuclear-norm weight and step size.

Run with::

    python -m experiments.figures.appendix_spi_mf_figures
"""

import settings
from experiments.appendix_spi_mf import EXPERIMENT, SIZES
from experiments.figures.aggregate import group, label_for, latest_per_experiment, tidy
from experiments.figures.style import table_dir
from experiments.figures.sweeps import median
from experiments.figures.tables import _math, metric_table, write_table
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("J_final", "training_subopt_final", "rank_final",
           "spectral_radius_final", "iterations", "total_env_steps", "best_env_steps")

SUBOPT_TOLERANCE = 3.0


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

        for lambda_ in settings.SPI_MF_LAMBDAS[size]:
            cell = [r for r in at_size if r.get("lambda") == lambda_]
            if not cell:
                continue

            write_table(
                metric_table(
                    cell, "stepsize",
                    metrics=METRICS,
                    column_order=settings.SPI_MF_STEPSIZES[size],
                    column_header="Step size",
                    caption=(
                        f"{label_for('spi_mf')} with $\\lambda = {_math(lambda_, '{:g}')}$ under "
                        f"\\emph{{{size}}} inflation, over the seeds."
                    ),
                    label=f"tab:appendix-spi-mf-{size}-{_slug(lambda_)}",
                ),
                f"appendix_spi_mf_{size}_{_slug(lambda_)}",
                EXPERIMENT,
            )

    choose(rows)


def choose(rows):
    """Print a ``SIZE_HYPERPARAMS["spi_mf"]`` snippet."""
    print('\n    "spi_mf": {')
    for size in SIZES:
        buckets = {
            k: b for k, b in group(
                [r for r in rows if r.get("size") == size], ("lambda", "stepsize")
            ).items()
            if k[0] in settings.SPI_MF_LAMBDAS[size]
            and k[1] in settings.SPI_MF_STEPSIZES[size]
        }
        if not buckets:
            continue

        best = min(median(b, "training_subopt_final") for b in buckets.values())
        allowed = {
            k: b for k, b in buckets.items()
            if median(b, "training_subopt_final") <= SUBOPT_TOLERANCE * best
        }
        lambda_, stepsize = min(
            allowed,
            key=lambda k: (median(allowed[k], "rank_final"),
                           median(allowed[k], "training_subopt_final")),
        )
        print(f'        "{size}": {{"lambda_": {lambda_:g}, "stepsize": {stepsize:g}}},')
    print("    },")


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return

    print("wrote tables to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
