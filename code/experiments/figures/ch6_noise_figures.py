"""Figures and tables for chapter 6's robustness evaluation.

Run with::

    python -m experiments.figures.ch6_noise_figures
"""

import config as cfg
import settings
from experiments.figures.aggregate import with_clean_ratio
from experiments.figures.ch4_noise_figures import render

EXPERIMENT = "ch6_noise"

PREFIX = "ch6_noise"


def main():
    # Trajectories are sampled at the agent rate
    render(
        EXPERIMENT, PREFIX, settings.CH6_NOISE_SOLVERS,
        set_by="python -m experiments.ch6_noise",
        dt=cfg.DEFAULT_DT * cfg.DEFAULT_ACTION_REPEAT,
        metric="clean_ratio", scorer=with_clean_ratio, log=True,
    )


if __name__ == "__main__":
    main()
