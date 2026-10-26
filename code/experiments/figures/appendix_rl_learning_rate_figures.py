"""Tables, learning curves and the per-size choice of the RL learning rates.

Run with::

    python -m experiments.figures.appendix_rl_learning_rate_figures
"""

import matplotlib.pyplot as plt
import numpy as np

import settings
from experiments.appendix_rl_learning_rate import EXPERIMENT, SIZES, SOLVERS
from experiments.figures.aggregate import label_for, latest_per_experiment, tidy
from experiments.figures.panels import learning_bands
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.sweeps import median, choose, print_choice, sweep_tables
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("training_subopt_final", "best_env_steps")


def render():
    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = tidy(records)

    sweep_tables(rows, EXPERIMENT, SOLVERS, SIZES, "lr",
                 None, METRICS, r"Learning rate $\eta_0$")

    for solver in SOLVERS:
        for size in SIZES:
            cell = [
                r for r in records
                if r.solver == solver and r.condition.get("size") == size
            ]
            if not cell:
                continue

            ax = learning_bands(
                cell, "lr", x="env_steps",
                order=[(lr,) for lr in settings.RL_LEARNING_RATES[size]],
                logx=False,
            )
            ax.set_title(f"{label_for(solver)}, {size} inflation")
            save(ax.figure, f"{EXPERIMENT}_{solver}_{size}", EXPERIMENT)
            plt.close(ax.figure)

    # Learning rate is unstable
    unstable = {
        (r.solver, r.condition.get("size"), r.condition.get("lr"))
        for r in records if not np.isfinite(r.arrays["cost"]).all()
    }

    # Lowest suboptimality among the stable learning rates
    best = choose(
        rows, SOLVERS, SIZES, "lr",
        admissible=lambda solver, bucket: not any(
            (r["solver"], r["size"], r["lr"]) in unstable for r in bucket
        ),
        score=lambda solver, bucket: median(bucket, "training_subopt_final"),
    )
    print_choice(best, "learning_rate")


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return

    print("wrote tables to", table_dir(EXPERIMENT), "and figures to", figure_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
