"""Shared tables and selection for the hyperparameter sweeps of the appendix.
"""

import numpy as np

from experiments.figures.aggregate import group, label_for
from experiments.figures.tables import metric_table, write_table

__all__ = ["sweep_tables", "median", "choose", "print_choice"]


def sweep_tables(rows, experiment, solvers, sizes, column, order, metrics, header):
    """One table per ``(solver, size)``, with the swept value as the column."""
    for solver in solvers:
        for size in sizes:
            cell = [r for r in rows if r["solver"] == solver and r.get("size") == size]
            if not cell:
                continue

            write_table(
                metric_table(
                    cell, column,
                    metrics=metrics,
                    column_order=order(solver) if callable(order) else order,
                    column_header=header,
                    caption=f"{label_for(solver)} under \\emph{{{size}}} inflation, over the seeds.",
                    label=f"tab:{experiment.replace('_', '-')}-{solver.replace('_', '-')}-{size}",
                ),
                f"{experiment}_{solver}_{size}",
                experiment,
            )


def median(bucket, metric) -> float:
    """Median of a summary metric, ``inf`` for missing or non-finite values."""
    values = [np.inf if r.get(metric) is None else float(r[metric]) for r in bucket]
    return float(np.median(values))


def choose(rows, solvers, sizes, column, admissible, score) -> dict:
    """The best swept value per ``(solver, size)``.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`.
    column : str
        The swept condition key.
    admissible : callable
        ``(solver, bucket) -> bool``, whether a value may be chosen at all.
    score : callable
        ``(solver, bucket) -> float``, lower is better.

    Returns
    -------
    dict
        ``{solver: {size: value}}``. Falls back to the lowest median training
        suboptimality when no value is admissible.
    """
    best = {}
    for solver in solvers:
        for size in sizes:
            buckets = group(
                [r for r in rows if r["solver"] == solver and r.get("size") == size],
                column,
            )
            if not buckets:
                continue

            allowed = {k: b for k, b in buckets.items() if admissible(solver, b)}
            if allowed:
                key = min(allowed, key=lambda k: score(solver, allowed[k]))
            else:
                print(f"  {solver}@{size}: no admissible value, falling back to lowest subopt")
                key = min(
                    buckets, key=lambda k: median(buckets[k], "training_subopt_final")
                )

            best.setdefault(solver, {})[size] = key[0]
    return best


def print_choice(best: dict, name: str):
    """Print a ready-to-paste ``settings.SIZE_HYPERPARAMS`` snippet."""
    print("\nSIZE_HYPERPARAMS = {")
    for solver, per_size in best.items():
        print(f'    "{solver}": {{')
        for size, value in per_size.items():
            print(f'        "{size}": {{"{name}": {value:g}}},')
        print("    },")
    print("}")
