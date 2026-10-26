"""Why the RL solvers have no stopping rule.

Replays the stopping rule of the LQR solvers on the full-length RL runs, and
shows per solver and size how late the runs still improve.

Run with::

    python -m experiments.figures.appendix_rl_convergence_figures
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import settings
from evaluation.analysis import Plateau
from experiments.appendix_rl_convergence import EXPERIMENT, SOLVERS
from experiments.figures.aggregate import SOLVER_LABELS, latest_per_experiment
from experiments.figures.panels import learning_bands
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import write_table
from experiments.store import MissingExperimentError, load_experiment

# Median and maximum regret a stopping rule would have to keep
MEDIAN_REGRET = 0.01
MAX_REGRET = 0.05

# Settings shown in the table
TABLE_ROWS = 10

# Share of the budget the "still improving" column looks at
LATE = 0.75


def _cost(record) -> np.ndarray:
    """The probed cost, with every non-finite value as ``inf``."""
    cost = record.arrays["cost"]
    return np.where(np.isfinite(cost), cost, np.inf)


def replay(record, rel_tol: float, patience: int, window: int) -> tuple[float, float]:
    """Where one run would have stopped, and the regret of the iterate kept.

    Returns
    -------
    tuple[float, float]
        ``(share of the budget used at the stop, cost kept / best cost - 1)``.
    """
    signal = record.arrays["cost_estimate"]
    cost = _cost(record)
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

    return float(steps[stop] / steps[-1]), float(cost[kept] / cost.min() - 1.0)


def _regrets(record) -> dict:
    """Where a run is best, and the regret of the iterates one could keep."""
    cost = _cost(record)
    n = len(cost)
    return {
        "best_at": int(np.argmin(cost)) / (n - 1),
        "late": cost[int(LATE * (n - 1))] / cost.min() - 1.0,
        "final": cost[-1] / cost.min() - 1.0,
        "selected": replay(record, 0.0, 10**9, 1)[1],
    }


def _percent(value: float) -> str:
    if not np.isfinite(value):
        return r"$\infty$"
    value = 100 * value
    return f"${value:.0f}\\,\\%$" if value >= 10 else f"${value:.2g}\\,\\%$"


def _grid_table(results: list[tuple]) -> str:
    """The stopping-rule settings with the lowest median regret, as a tabular."""
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\varepsilon_{\mathrm{rel}}$ & $N_{\mathrm{p}}$ & $N_{\mathrm{w}}$ & Budget used & Median regret & Max.\ regret \\",
        r"\midrule",
    ]
    for (rel_tol, patience, window), used, regrets in results:
        lines.append(
            f"${rel_tol:g}$ & ${patience}$ & ${window}$ & {_percent(np.median(used))} & "
            f"{_percent(np.median(regrets))} & {_percent(np.max(regrets))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _cell_table(records, order) -> str:
    """Per solver and size, the median over seeds of :func:`_regrets`, as a tabular."""
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Solver & Size & Best policy at & Regret at "
        f"${100 * LATE:.0f}\\,\\%$"
        r" & Final regret & Regret by $\bar{g}_j$ \\",
        r"\midrule",
    ]
    for size in settings.CH5_INFLATION_SIZES:
        for solver in order:
            cell = [_regrets(r) for r in records
                    if r.solver == solver and r.condition.get("size") == size]
            if not cell:
                continue
            median = {k: np.median([c[k] for c in cell]) for k in cell[0]}
            lines.append(
                f"{SOLVER_LABELS[solver]} & \\emph{{{size}}} & {_percent(median['best_at'])} & "
                f"{_percent(median['late'])} & {_percent(median['final'])} & "
                f"{_percent(median['selected'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render():
    records = [
        r for r in latest_per_experiment(load_experiment(EXPERIMENT))
        if "cost_estimate" in r.arrays
    ]
    order = tuple(s for s in SOLVERS if any(r.solver == s for r in records))

    # Diverged runs have infinite regret under every setting
    diverged = [r for r in records if not np.isfinite(r.arrays["cost"]).all()]
    stable = [r for r in records if r not in diverged]

    grid = settings.RL_CONVERGENCE_GRID
    results = []
    for setting in product(grid["rel_tol"], grid["patience"], grid["window"]):
        outcomes = np.array([replay(r, *setting) for r in stable])
        results.append((setting, outcomes[:, 0], outcomes[:, 1]))
    results.sort(key=lambda r: np.median(r[2]))

    write_table(_grid_table(results[:TABLE_ROWS]), EXPERIMENT, EXPERIMENT)
    write_table(_cell_table(records, order), f"{EXPERIMENT}_cells", EXPERIMENT)

    for size in settings.CH5_INFLATION_SIZES:
        subset = [r for r in records if r.condition.get("size") == size]
        if subset:
            ax = learning_bands(subset, "solver", x="env_steps", order=order, logx=False)
            save(ax.figure, f"{EXPERIMENT}_{size}_curves_steps", EXPERIMENT)
    plt.close("all")

    admissible = [
        r for r in results
        if np.median(r[2]) <= MEDIAN_REGRET and np.max(r[2]) <= MAX_REGRET
    ]
    print(f"{len(admissible)}/{len(results)} settings meet the regret bounds "
          f"on the {len(stable)} stable runs.")
    for setting, used, regrets in results[:TABLE_ROWS]:
        print(f"  {setting}: budget used {np.median(used):.2f}, "
              f"median regret {np.median(regrets):.3g}, max regret {np.max(regrets):.3g}")

    for solver in order:
        n = sum(r.solver == solver for r in diverged)
        if n:
            print(f"  {solver}: {n}/{sum(r.solver == solver for r in records)} "
                  "runs with a non-finite probe cost")

    for name in ("final", "selected"):
        values = [_regrets(r)[name] for r in stable]
        print(f"Keeping the {name} iterate: median regret {np.median(values):.3g}, "
              f"max {np.max(values):.3g}")


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return

    print("wrote to", table_dir(EXPERIMENT), "and", figure_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
