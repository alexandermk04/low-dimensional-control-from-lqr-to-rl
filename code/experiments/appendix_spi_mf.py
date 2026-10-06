"""Appendix: nuclear-norm weight and step size of the model-free S-PI, per inflation size.

Run with::

    python -m experiments.appendix_spi_mf
"""

import settings
from experiments.problem import RunSpec
from experiments.runner import run_grid
from experiments.settings_guard import require_per_size

EXPERIMENT = "appendix_spi_mf"

SET_BY = "python -m experiments.appendix_sample_size"

# Shortened protocol to keep runtime shorter
ITERATIONS = 50
TEST_EPISODES = 20
TEST_HORIZON = 400

# The inflation sizes checked
SIZES = settings.CH4_INFLATION_SIZES


def build_specs() -> list[RunSpec]:
    """Nuclear-norm weights x step sizes x inflation sizes x seeds."""
    chosen = require_per_size(
        settings.CHOSEN_SAMPLE_CELL, "CHOSEN_SAMPLE_CELL", SET_BY, sizes=SIZES
    )

    specs = []
    for size in SIZES:
        rollout_length, n_trajectories = chosen[size]

        for lambda_ in settings.SPI_MF_LAMBDAS[size]:
            for stepsize in settings.SPI_MF_STEPSIZES[size]:
                cell = {"lambda": lambda_, "stepsize": stepsize}

                specs += [
                    RunSpec(
                        experiment=EXPERIMENT,
                        solver="spi_mf",
                        seed=seed,
                        inflation=settings.INFLATION_SIZES[size],
                        rollout_length=rollout_length,
                        n_trajectories=n_trajectories,
                        iterations=ITERATIONS,
                        condition={"size": size, **cell},
                        hyperparam_overrides={
                            "lambda_": lambda_, "stepsize": stepsize
                        },
                    )
                    for seed in settings.SEEDS
                ]

    return specs


def main():
    run_grid(
        build_specs(),
        EXPERIMENT,
        used_settings={
            "solvers": ("spi_mf",),
            "sizes": SIZES,
            "chosen_sample_cell": settings.CHOSEN_SAMPLE_CELL,
            "lambdas": settings.SPI_MF_LAMBDAS,
            "stepsizes": settings.SPI_MF_STEPSIZES,
            "iterations": ITERATIONS,
            "test_episodes": TEST_EPISODES,
            "test_horizon": TEST_HORIZON,
            "hyperparams": settings.SOLVER_HYPERPARAMS.get("spi_mf", {}),
        },
        test_kwargs={"episodes": TEST_EPISODES, "horizon": TEST_HORIZON},
    )


if __name__ == "__main__":
    main()
