"""Chapter 4: no access to the model.

Run with::

    python -m experiments.ch4_identified_model
"""

import settings
from experiments.ch4_budget import budget, condition
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "ch4_identified_model"


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x seeds.
    """
    specs = []

    for solver in settings.CH4_IDENTIFIED_SOLVERS:
        for size in settings.CH4_INFLATION_SIZES:
            rollout_length, n_trajectories = budget(solver, size)

            specs += [
                RunSpec(
                    experiment=EXPERIMENT,
                    solver=solver,
                    seed=seed,
                    inflation=settings.INFLATION_SIZES[size],
                    rollout_length=rollout_length,
                    n_trajectories=n_trajectories,
                    iterations=settings.SOLVER_ITERATIONS,
                    condition=condition(solver, size),
                )
                for seed in settings.SEEDS
            ]

    return specs


def main():
    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": settings.CH4_IDENTIFIED_SOLVERS,
            "inflation_sizes": {
                name: settings.INFLATION_SIZES[name]
                for name in settings.CH4_INFLATION_SIZES
            },
            "chosen_sample_cell": settings.CHOSEN_SAMPLE_CELL,
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "iterations": settings.SOLVER_ITERATIONS,
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH4_IDENTIFIED_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
