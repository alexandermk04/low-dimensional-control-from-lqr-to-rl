"""Figures and tables for chapter 6's robustness evaluation.

Run with::

    python -m experiments.figures.ch6_noise_figures
"""

from functools import partial

import numpy as np

import config as cfg
import settings
from experiments.figures.aggregate import latest_per_experiment, with_degradation
from experiments.figures.ch4_noise_figures import render
from experiments.store import load_experiment

EXPERIMENT = "ch6_noise"

PREFIX = "ch6_noise"

# Where the oracle ran, as it is not evaluated under noise
SOURCE = "ch6_perception"

REFERENCE = "oracle_coarse"


def oracle_clean_cost() -> float:
    """Test cost of the oracle in the clean loop, pooled over its finite episodes."""
    episodes = np.concatenate([
        np.asarray(r.arrays["test_costs"], dtype=float)
        for r in latest_per_experiment(load_experiment(SOURCE))
        if r.solver == REFERENCE
    ])
    return float(np.mean(episodes[np.isfinite(episodes)]))


def main():
    # Trajectories are sampled at the agent rate, exclude measurement noise calibration
    render(
        EXPERIMENT, PREFIX, settings.CH6_NOISE_SOLVERS,
        set_by="python -m experiments.ch6_noise",
        dt=cfg.DEFAULT_DT * cfg.DEFAULT_ACTION_REPEAT,
        scorer=partial(with_degradation, reference_cost=oracle_clean_cost()),
        calibrated=("control", "parametric", "disturbance"),
    )


if __name__ == "__main__":
    main()
