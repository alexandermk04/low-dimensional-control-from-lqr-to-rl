"""Chapter 5: Collection of solvers compared at pixel size inflation.

Run with::

    python -m experiments.ch5_solver_comparison
"""

import settings
from experiments.ch4_budget import budget, condition
from experiments.ch5_inflation_sizes import supported
from experiments.problem import RunSpec
from experiments.runner import run_grid
from experiments.settings_guard import require_choice

EXPERIMENT = "ch5_solver_comparison"


def build_specs() -> list[RunSpec]:
    """Solvers x seeds, at the pixle size inflation."""
    size = require_choice(
        settings.CH5_PIXEL_INFLATION, "CH5_PIXEL_INFLATION",
        settings.INFLATION_SIZES, "pick one of settings.INFLATION_SIZES",
    )

    specs = []
    for solver in settings.CH5_COMPARISON_SOLVERS:
        if not supported(solver, size):
            continue

        rollout_length, n_trajectories = budget(solver, size)

        specs += [
            RunSpec(
                experiment=EXPERIMENT,
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
            "solvers": settings.CH5_COMPARISON_SOLVERS,
            "inflation": settings.CH5_REFERENCE_INFLATION,
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "rl_timesteps": settings.RL_MAX_TIMESTEPS,
        },
    )


if __name__ == "__main__":
    main()
