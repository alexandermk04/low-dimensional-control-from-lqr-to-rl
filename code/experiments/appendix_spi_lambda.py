"""Appendix: the nuclear-norm weight of the model-based S-PI, per inflation size.

Run with::

    python -m experiments.appendix_spi_lambda
"""

import settings
from experiments.ch4_budget import budget, condition
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "appendix_spi_lambda"

SOLVERS = ("spi", "spi_est")

SIZES = settings.CH4_INFLATION_SIZES


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x nuclear-norm weights x seeds."""
    specs = []
    for solver in SOLVERS:
        for size in SIZES:
            rollout_length, n_trajectories = budget(solver, size)

            for lambda_ in settings.SPI_LAMBDAS:
                specs += [
                    RunSpec(
                        experiment=EXPERIMENT,
                        solver=solver,
                        seed=seed,
                        inflation=settings.INFLATION_SIZES[size],
                        rollout_length=rollout_length,
                        n_trajectories=n_trajectories,
                        iterations=settings.SOLVER_MAX_ITERATIONS,
                        condition={**condition(solver, size), "lambda": lambda_},
                        hyperparam_overrides={"lambda_": lambda_},
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
            "lambdas": settings.SPI_LAMBDAS,
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "hyperparams": {s: settings.SOLVER_HYPERPARAMS.get(s, {}) for s in SOLVERS},
        },
    )


if __name__ == "__main__":
    main()
