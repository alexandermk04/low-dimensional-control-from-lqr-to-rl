"""Test the RL stop parameters to find the optimal settings.

Run with::

    python -m experiments.figures.appendix_rl_convergence_figures
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import settings
from evaluation.analysis import Plateau
from experiments.appendix_rl_convergence import EXPERIMENT, SOLVERS
from experiments.figures.aggregate import latest_per_experiment
from experiments.figures.panels import learning_bands
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import write_table
from experiments.store import MissingExperimentError, load_experiment

# Median and maximum regret needed
MEDIAN_REGRET = 0.01
MAX_REGRET = 0.05

# Settings shown in the table
TABLE_ROWS = 10


def replay(record, rel_tol: float, patience: int, window: int) -> tuple[float, float]:
    """Where one run would have stopped, and the regret of the iterate kept.

    Returns
    -------
    tuple[float, float]
        ``(env_steps at the stop, cost kept / best cost - 1)``.
    """
    signal = record.arrays["cost_estimate"]
    cost = record.arrays["cost"]
    steps = record.arrays["env_steps"]

    stopper = Plateau(rel_tol=rel_tol, patience=patience, window=window)
    stop = len(cost) - 1

    for index, value in enumerate(signal):
        # The first sample has no finished episodes yet
        if np.isnan(value):
            continue
        if stopper.update(value, index, index, int(steps[index])):
            stop = index
            break

    kept = stopper.best_iteration if stopper.best_iteration is not None else stop
    best = np.nanmin(np.where(np.isfinite(cost), cost, np.nan))

    return float(steps[stop]), float(cost[kept] / best - 1.0)


def _table(results: list[tuple]) -> str:
    """The best settings, and the current one, as a tabular."""
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"rel.\ tol. & patience & window & median steps & median regret & max regret \\",
        r"\midrule",
    ]
    for (rel_tol, patience, window), stops, regrets in results:
        lines.append(
            f"{rel_tol:g} & {patience} & {window} & {np.median(stops):.3g} & "
            f"{np.median(regrets):.3g} & {np.max(regrets):.3g} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def render():
    records = [
        r for r in latest_per_experiment(load_experiment(EXPERIMENT))
        if "cost_estimate" in r.arrays
    ]

    grid = settings.RL_CONVERGENCE_GRID
    results = []
    for setting in product(grid["rel_tol"], grid["patience"], grid["window"]):
        outcomes = np.array([replay(r, *setting) for r in records])
        results.append((setting, outcomes[:, 0], outcomes[:, 1]))

    admissible = [
        r for r in results
        if np.median(r[2]) <= MEDIAN_REGRET and np.max(r[2]) <= MAX_REGRET
    ]
    admissible.sort(key=lambda r: np.median(r[1]))

    current = tuple((settings.CONVERGENCE["rl"] or {}).get(k) for k in ("rel_tol", "patience", "window"))
    shown = admissible[:TABLE_ROWS] + [r for r in results if r[0] == current]

    write_table(_table(shown), EXPERIMENT, EXPERIMENT)

    order = tuple(s for s in SOLVERS if any(r.solver == s for r in records))
    for size in settings.CH5_INFLATION_SIZES:
        subset = [r for r in records if r.condition.get("size") == size]
        if subset:
            ax = learning_bands(subset, "solver", x="env_steps", order=order, logx=False)
            save(ax.figure, f"{EXPERIMENT}_{size}_curves_steps", EXPERIMENT)
    plt.close("all")

    for setting, stops, regrets in shown:
        print(f"  {setting}: median steps {np.median(stops):.3g}, "
              f"median regret {np.median(regrets):.3g}, max regret {np.max(regrets):.3g}")

    if admissible:
        rel_tol, patience, window = admissible[0][0]
        print(f'\nCONVERGENCE["rl"] = dict(rel_tol={rel_tol:g}, patience={patience}, window={window})')
    else:
        print("\nNo setting meets the regret bounds.")
        floor = np.array([replay(r, 0.0, 10**9, 1)[1] for r in records])
        print(f"Never stopping gives median regret {np.median(floor):.3g}, "
              f"max {np.max(floor):.3g}: the bounds fail on the iterate selection.")
        for solver in order:
            probes = [r.arrays["cost"] for r in records if r.solver == solver]
            diverged = sum(not np.isfinite(c).all() for c in probes)
            if diverged:
                print(f"  {solver}: {diverged}/{len(probes)} runs with a non-finite probe cost")


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return

    print("wrote to", table_dir(EXPERIMENT), "and", figure_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
