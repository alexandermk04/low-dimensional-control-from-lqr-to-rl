"""Figures and tables for chapter 5's robustness evaluation.

Run with::

    python -m experiments.figures.ch5_noise_figures
"""

import settings
from experiments.ch5_noise import SIZES
from experiments.figures.ch4_noise_figures import render

EXPERIMENT = "ch5_noise"

PREFIX = "ch5_noise"


def main():
    # sizes imported from the experiment
    for size in SIZES:
        render(
            EXPERIMENT, f"{PREFIX}_{size}", settings.CH5_SOLVERS,
            set_by="python -m experiments.ch5_noise",
            size=size,
        )


if __name__ == "__main__":
    main()
