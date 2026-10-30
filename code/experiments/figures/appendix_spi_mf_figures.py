"""Tables and the per-size choice of the model-free S-PI's nuclear-norm weight and step size.

Run with::

    python -m experiments.figures.appendix_spi_mf_figures
"""

from itertools import product

import settings
from experiments.appendix_spi_mf import EXPERIMENT, SIZES
from experiments.figures.aggregate import group, label_for, latest_per_experiment, tidy
from experiments.figures.style import table_dir
from experiments.figures.sweeps import median
from experiments.figures.tables import _math, sweep_table, usable, write_table
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("training_subopt_final", "rank_final")

SUBOPT_TOLERANCE = 3.0


def _budget(rows) -> str:
    """The caption sentence on the iterations and env steps."""
    iterations = {r.get("iterations") for r in rows if usable(r)}
    steps = {r.get("total_env_steps") for r in rows if usable(r)}
    if len(iterations) != 1 or len(steps) != 1:
        return ""
    (iterations,), (steps,) = iterations, steps
    return (f" Every stable run takes ${iterations:.0f}$ iterations and "
            f"${_math(steps, '{:.3g}')}$ environment steps.")


def render():
    """One table per inflation size, with a row per (lambda, step size)."""
    rows = tidy(latest_per_experiment(load_experiment(EXPERIMENT)))

    for size in SIZES:
        at_size = [
            r for r in rows if r.get("size") == size
            and r.get("lambda") in settings.SPI_MF_LAMBDAS[size]
            and r.get("stepsize") in settings.SPI_MF_STEPSIZES[size]
        ]
        if not at_size:
            continue

        write_table(
            sweep_table(
                at_size, ("lambda", "stepsize"),
                metrics=METRICS,
                row_order=list(product(settings.SPI_MF_LAMBDAS[size],
                                       settings.SPI_MF_STEPSIZES[size])),
                row_header=(r"$\lambda$", r"Step size $\eta$"),
                census=True,
                caption=(
                    f"{label_for('spi_mf')} per nuclear-norm weight $\\lambda$ and step size "
                    f"$\\eta$ under \\emph{{{size}}} inflation, mean and standard deviation "
                    f"over the stable seeds.{_budget(at_size)}"
                ),
                label=f"tab:appendix-spi-mf-{size}",
            ),
            f"appendix_spi_mf_{size}",
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
