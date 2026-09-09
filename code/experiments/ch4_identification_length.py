"""Chapter 4: how much data the system identifier needs to work correctly

In here, we compare over a grid with two axis:
- how long the singular excitation trajectory is (T)
- how large future and past horizon for the Hankel matrices should be

We run this grid for all possible inflation sizes, then evaluate the trained
controllers on these identified systems via our test protocol.


Run with::

    python -m experiments.ch4_identification_length
"""

import settings
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "ch4_identification_length"


def build_specs() -> list[RunSpec]:
    """Trajectory lengths x Hankel horizons x inflation sizes x seeds.

    Returns
    -------
    list[RunSpec]
    """
    return [
        RunSpec(
            experiment=EXPERIMENT,
            solver=solver,
            seed=seed,
            inflation=inflation,
            rollout_length=trajectory_length,
            iterations=settings.SOLVER_ITERATIONS,
            condition={"size": name, "T": trajectory_length, "hankel": horizon},
        )
        for solver in settings.CH4_IDENTIFICATION_SOLVERS
        for name, inflation in settings.INFLATION_SIZES.items()
        for trajectory_length in settings.IDENTIFICATION_LENGTHS
        for horizon in settings.IDENTIFICATION_HORIZONS
        for seed in settings.SEEDS
    ]


def main():
    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": settings.CH4_IDENTIFICATION_SOLVERS,
            "identification_lengths": settings.IDENTIFICATION_LENGTHS,
            "identification_horizons": settings.IDENTIFICATION_HORIZONS,
            "inflation_sizes": settings.INFLATION_SIZES,
            "iterations": settings.SOLVER_ITERATIONS,
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH4_IDENTIFICATION_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
