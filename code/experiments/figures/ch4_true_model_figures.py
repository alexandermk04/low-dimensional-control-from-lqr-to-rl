"""Figures and tables for chapter 4 true-model.

Run with::

    python -m experiments.figures.ch4_true_model_figures
"""

import settings
from experiments.figures.ch4_tracks import render_track

EXPERIMENT = "ch4_true_model"

ORDER = tuple(settings.CH4_TRUE_SOLVERS)


def main():
    render_track(
        EXPERIMENT,
        ORDER,
        stem_prefix="ch4_true",
        track_caption=(
            "Policy architectures optimized by the exact gradients on the true model"
        ),
        across_caption=(
            "Policy architectures on the true model, across every inflation size."
        ),
    )


if __name__ == "__main__":
    main()
