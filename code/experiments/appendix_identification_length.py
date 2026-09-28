"""Chapter 4: how much data the system identifier needs to work correctly

In here, we compare over a grid with two axis:
- how long the singular excitation trajectory is (T)
- how large future and past horizon for the Hankel matrices should be

We run this grid for all possible inflation sizes, then evaluate the trained
controllers on these identified systems via our test protocol.


Run with::

    python -m experiments.appendix_identification_length
"""

import settings
from experiments.problem import RunSpec
from experiments.runner import run_grid

def _sizes():
    """The (name, (p, q)) pairs chapter 4 sweeps.

    ``settings.INFLATION_SIZES`` also carries chapter 5's ``"pixel"`` cell,
    which no identification or rollout solver can reach; see
    ``settings.CH4_INFLATION_SIZES``.
    """
    return [(name, settings.INFLATION_SIZES[name])
            for name in settings.CH4_INFLATION_SIZES]


EXPERIMENT = "appendix_identification_length"


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
            iterations=settings.SOLVER_MAX_ITERATIONS,
            condition={"size": name, "T": trajectory_length, "hankel": horizon},
        )
        for solver in settings.CH4_IDENTIFICATION_SOLVERS
        for name, inflation in _sizes()
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
            "inflation_sizes": dict(_sizes()),
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH4_IDENTIFICATION_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
