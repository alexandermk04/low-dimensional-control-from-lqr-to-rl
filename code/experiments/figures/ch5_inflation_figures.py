"""Figures and tables for chapter 5's inflation-size experiment.

Run with::

    python -m experiments.figures.ch5_inflation_figures
"""

import settings
from experiments.figures.ch4_tracks import render_track

EXPERIMENT = "ch5_inflation_sizes"

ORDER = tuple(settings.CH5_SOLVERS)


def main():
    render_track(
        EXPERIMENT,
        ORDER,
        stem_prefix="ch5_inflation",
        track_caption="PPO variants",
        across_caption=(
            "Size of the PPO policies, across every inflation size."
        ),
        sizes=settings.CH5_INFLATION_SIZES,
    )


if __name__ == "__main__":
    main()
