"""Chapter 5: reinforcement-learning policies across inflation sizes.

Run with::

    python -m experiments.ch5_inflation_sizes
"""

import settings
from experiments.ch4_budget import budget, condition
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "ch5_inflation_sizes"


def supported(solver: str, size: str) -> bool:
    """Whether this arm can be run at this inflation size at all."""
    return (solver, size) not in settings.UNSUPPORTED_CELLS


def build_specs(solvers=None, experiment: str = EXPERIMENT) -> list[RunSpec]:
    """Solvers x inflation sizes x seeds, minus unsupported run combinations.

    Parameters
    ----------
    solvers : iterable or None, optional
        Defaults to ``settings.CH5_SOLVERS``.
    experiment : str, optional
        Name the runs are stored under.

    Returns
    -------
    list[RunSpec]
    """
    solvers = settings.CH5_SOLVERS if solvers is None else solvers

    specs = []
    for solver in solvers:
        for size in settings.CH5_INFLATION_SIZES:
            if not supported(solver, size):
                continue

            rollout_length, n_trajectories = budget(solver, size)

            specs += [
                RunSpec(
                    experiment=experiment,
                    solver=solver,
                    seed=seed,
                    inflation=settings.INFLATION_SIZES[size],
                    rollout_length=rollout_length,
                    n_trajectories=n_trajectories,
                    iterations=settings.SOLVER_MAX_ITERATIONS,
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
            "solvers": settings.CH5_SOLVERS,
            "inflation_sizes": {
                name: settings.INFLATION_SIZES[name]
                for name in settings.CH5_INFLATION_SIZES
            },
            "unsupported_cells": sorted(
                f"{solver}@{size}" for solver, size in settings.UNSUPPORTED_CELLS
            ),
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "rl_timesteps": settings.RL_MAX_TIMESTEPS,
            "train_horizon": settings.TRAIN_HORIZON,
            "size_hyperparams": {s: settings.SIZE_HYPERPARAMS.get(s, {}) for s in settings.CH5_SOLVERS},
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH5_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
