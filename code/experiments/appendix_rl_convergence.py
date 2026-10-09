"""Appendix: whether the RL runs reach a plateau a stopping rule could detect.

Run with::

    python -m experiments.appendix_rl_convergence
"""

import settings
from experiments.ch4_budget import budget, condition
from experiments.ch5_inflation_sizes import supported
from experiments.problem import RunSpec
from experiments.runner import run_grid

EXPERIMENT = "appendix_rl_convergence"

SOLVERS = tuple(s for s in settings.CH5_SOLVERS if s != "baseline")

SIZES = settings.CH5_INFLATION_SIZES

SEEDS = settings.SEEDS[:3]


def build_specs() -> list[RunSpec]:
    """Solvers x inflation sizes x seeds, minus unsupported combinations."""
    specs = []
    for solver in SOLVERS:
        for size in SIZES:
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
                for seed in SEEDS
            ]

    return specs


def main():
    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": SOLVERS,
            "sizes": SIZES,
            "seeds": SEEDS,
            "timesteps": settings.RL_MAX_TIMESTEPS,
            "eval_freq": settings.RL_EVAL_FREQ,
            "train_horizon": settings.TRAIN_HORIZON,
            "chosen_identification": settings.CHOSEN_IDENTIFICATION,
            "size_hyperparams": {s: settings.SIZE_HYPERPARAMS.get(s, {}) for s in SOLVERS},
        },
    )


if __name__ == "__main__":
    main()
