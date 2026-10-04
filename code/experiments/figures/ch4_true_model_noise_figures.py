"""Figures and tables for chapter 4's robustness evaluation on the true model.

Run with::

    python -m experiments.figures.ch4_true_model_noise_figures
"""

import settings
from experiments.ch4_true_model_noise import SIZES
from experiments.figures.ch4_noise_figures import render

EXPERIMENT = "ch4_true_model_noise"

PREFIX = "ch4_true_model_noise"


def main():
    for size in SIZES:
        render(
            EXPERIMENT, f"{PREFIX}_{size}", settings.CH4_TRUE_SOLVERS,
            set_by="python -m experiments.ch4_true_model_noise",
            size=size,
        )


if __name__ == "__main__":
    main()
