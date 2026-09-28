"""Appendix: the learning rate of the policy gradient solvers, per inflation size.

Run with::

    python -m experiments.appendix_learning_rate
"""

import settings
from experiments.ch4_budget import budget, condition
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "appendix_learning_rate"

SOLVERS = tuple(settings.PG_LEARNING_RATES)

SIZES = settings.CH4_INFLATION_SIZES


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x learning rates x seeds."""
    specs = []
    for solver in SOLVERS:
        for size in SIZES:
            rollout_length, n_trajectories = budget(solver, size)

            for lr in settings.PG_LEARNING_RATES[solver]:
                specs += [
                    RunSpec(
                        experiment=EXPERIMENT,
                        solver=solver,
                        seed=seed,
                        inflation=settings.INFLATION_SIZES[size],
                        rollout_length=rollout_length,
                        n_trajectories=n_trajectories,
                        iterations=settings.SOLVER_MAX_ITERATIONS,
                        condition={**condition(solver, size), "lr": lr},
                        hyperparam_overrides={"gradient_lr": lr},
                    )
                    for seed in settings.SEEDS
                ]

    return specs


def main():
    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": SOLVERS,
            "sizes": SIZES,
            "learning_rates": settings.PG_LEARNING_RATES,
            "chosen_sample_cell": settings.CHOSEN_SAMPLE_CELL,
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "hyperparams": {s: settings.SOLVER_HYPERPARAMS.get(s, {}) for s in SOLVERS},
        },
    )


if __name__ == "__main__":
    main()
