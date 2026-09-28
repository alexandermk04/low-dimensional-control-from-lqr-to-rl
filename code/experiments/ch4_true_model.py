"""Chapter 4: True model.

Run with::

    python -m experiments.ch4_true_model
"""

import settings
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "ch4_true_model"


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x seeds.
    """
    return [
        RunSpec(
            experiment=EXPERIMENT,
            solver=solver,
            seed=seed,
            inflation=settings.INFLATION_SIZES[size],
            iterations=settings.SOLVER_MAX_ITERATIONS,
            condition={"size": size},
        )
        for solver in settings.CH4_TRUE_SOLVERS
        for size in settings.CH4_INFLATION_SIZES
        for seed in settings.SEEDS
    ]


def main():
    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": settings.CH4_TRUE_SOLVERS,
            "inflation_sizes": {
                name: settings.INFLATION_SIZES[name]
                for name in settings.CH4_INFLATION_SIZES
            },
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH4_TRUE_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
