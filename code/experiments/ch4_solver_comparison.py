"""Chapter 4: compare the different solves at different inflation sizes

Run with::

    python -m experiments.ch4_solver_comparison
"""

import settings
from experiments.problem import RunSpec
from experiments.runner import run_grid
from experiments.settings_guard import require_per_size

EXPERIMENT = "ch4_solver_comparison"

SAMPLE_SET_BY = "python -m experiments.ch4_sample_size"

IDENTIFICATION_SET_BY = "python -m experiments.ch4_identification_length"


def _budget(solver: str, size: str) -> tuple[int | None, int | None]:
    """The training budget used for given solver.

    Three cases exist:

    - identification solver with only trajectory length and no
      trajectory count
    - rollout-based solver with both;
    - exact-gradient solver which doesn't use samples
    """
    if solver in settings.CH4_IDENTIFICATION_SOLVERS:
        chosen = require_per_size(
            settings.CHOSEN_IDENTIFICATION,
            "CHOSEN_IDENTIFICATION",
            IDENTIFICATION_SET_BY,
        )
        trajectory_length, _ = chosen[size]
        return trajectory_length, None

    if solver in settings.CH4_SAMPLE_SOLVERS:
        chosen = require_per_size(
            settings.CHOSEN_SAMPLE_CELL, "CHOSEN_SAMPLE_CELL", SAMPLE_SET_BY
        )
        return chosen[size]

    return None, None


def _condition(solver: str, size: str) -> dict:
    """Getting the specific setting for the solver and inflation size."""
    if solver in settings.CH4_IDENTIFICATION_SOLVERS:
        _, horizon = settings.CHOSEN_IDENTIFICATION[size]
        return {"size": size, "hankel": horizon}

    return {"size": size}


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x seeds."""
    specs = []

    for solver in settings.CH4_SOLVERS:
        for size in settings.INFLATION_SIZES:
            rollout_length, n_trajectories = _budget(solver, size)

            specs += [
                RunSpec(
                    experiment=EXPERIMENT,
                    solver=solver,
                    seed=seed,
                    inflation=settings.INFLATION_SIZES[size],
                    rollout_length=rollout_length,
                    n_trajectories=n_trajectories,
                    iterations=settings.SOLVER_ITERATIONS,
                    condition=_condition(solver, size),
                )
                for seed in settings.SEEDS
            ]

    return specs


def main():
    specs = build_specs()

    run_grid(
        specs,
        EXPERIMENT,
        used_settings={
            "solvers": settings.CH4_SOLVERS,
            "inflation_sizes": settings.INFLATION_SIZES,
            "chosen_sample_cell": settings.CHOSEN_SAMPLE_CELL,
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "iterations": settings.SOLVER_ITERATIONS,
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH4_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
