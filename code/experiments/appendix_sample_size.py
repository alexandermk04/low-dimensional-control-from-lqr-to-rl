"""Chapter 4: how much data the model-free policy gradient needs to work

Similar to the identification length, we check here the different
combinations of:
- Number of trajectories
- Length of the rollout
during the training process, and then evaluate via our test protocol.

Run with::

    python -m experiments.appendix_sample_size
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


# Base fixed LR
GRADIENT_LR = 1e-2

EXPERIMENT = "appendix_sample_size"


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x (l, m) cells x seeds."""
    return [
        RunSpec(
            experiment=EXPERIMENT,
            solver=solver,
            seed=seed,
            inflation=inflation,
            rollout_length=rollout_length,
            n_trajectories=n_trajectories,
            iterations=settings.SOLVER_MAX_ITERATIONS,
            hyperparam_overrides={"gradient_lr": GRADIENT_LR},
            condition={"size": name, "l": rollout_length, "m": n_trajectories},
        )
        for solver in settings.CH4_SAMPLE_SOLVERS
        for name, inflation in _sizes()
        for rollout_length, n_trajectories in settings.SAMPLE_GRID
        for seed in settings.SEEDS
    ]


def main():
    specs = build_specs()

    run_grid(
        specs,
        EXPERIMENT,
        used_settings={
            "solvers": settings.CH4_SAMPLE_SOLVERS,
            "sample_grid": settings.SAMPLE_GRID,
            "inflation_sizes": dict(_sizes()),
            "iterations": settings.SOLVER_MAX_ITERATIONS,
            "convergence": settings.CONVERGENCE,
            "gradient_lr": GRADIENT_LR,
            "hyperparams": {
                s: settings.SOLVER_HYPERPARAMS.get(s, {})
                for s in settings.CH4_SAMPLE_SOLVERS
            },
        },
    )


if __name__ == "__main__":
    main()
