"""Tables, learning curves and the per-size choice of the policy gradient learning rates.

Run with::

    python -m experiments.figures.appendix_learning_rate_figures
"""

import matplotlib.pyplot as plt

import settings
from experiments.appendix_learning_rate import EXPERIMENT, SIZES, SOLVERS
from experiments.figures.aggregate import latest_per_experiment, tidy
from experiments.figures.panels import learning_bands
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.sweeps import median, choose, print_choice, sweep_tables
from experiments.store import MissingExperimentError, load_experiment

METRICS = ("J_final", "training_subopt_final", "test_cost_mean", "iterations",
           "iters_to_thresh_1pct", "steps_to_thresh_1pct", "total_env_steps")

# Suboptimality a learning rate has to reach to be selected
TOLERANCE = 0.01


def _speed(solver: str) -> str:
    """What a solver's speed is counted in; model-based ones take no env steps."""
    return "steps_to_thresh_1pct" if solver == "mf_npg" else "iters_to_thresh_1pct"


def render():
    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = tidy(records)

    sweep_tables(rows, EXPERIMENT, SOLVERS, SIZES, "lr",
                 lambda solver: settings.PG_LEARNING_RATES[solver], METRICS,
                 "Learning rate")

    for solver in SOLVERS:
        for size in SIZES:
            cell = [
                r for r in records
                if r.solver == solver and r.condition.get("size") == size
            ]
            if not cell:
                continue

            ax = learning_bands(
                cell, "lr",
                x="env_steps" if solver == "mf_npg" else "iteration",
                order=[(lr,) for lr in settings.PG_LEARNING_RATES[solver]],
            )
            ax.set_title(f"{solver}, {size} inflation")
            save(ax.figure, f"{EXPERIMENT}_{solver}_{size}", EXPERIMENT)
            plt.close(ax.figure)

    # Fastest learning rate among those that are stable and reach the tolerance
    best = choose(
        rows, SOLVERS, SIZES, "lr",
        admissible=lambda solver, bucket: (
            median(bucket, "training_subopt_final") < TOLERANCE
            and all(
                r.get("training_subopt_final") is not None
                and r["training_subopt_final"] < float("inf")
                for r in bucket
            )
        ),
        score=lambda solver, bucket: median(bucket, _speed(solver)),
    )
    print_choice(best, "gradient_lr")


def main():
    try:
        render()
    except MissingExperimentError as error:
        print(f"skipping {EXPERIMENT}: {error}")
        return

    print("wrote tables to", table_dir(EXPERIMENT), "and figures to", figure_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
