"""Figures and tables for chapter 4's model-free setting

Run with::

    python -m experiments.figures.ch4_identified_model_figures
"""

import settings
from experiments.figures.ch4_tracks import render_track

EXPERIMENT = "ch4_identified_model"

ORDER = tuple(settings.CH4_IDENTIFIED_SOLVERS)


def main():
    render_track(
        EXPERIMENT,
        ORDER,
        stem_prefix="ch4_identified",
        track_caption="Solvers learning without the model",
        across_caption=(
            "Solvers learning without the model, across every inflation size."
        ),
        curves=False,
    )


if __name__ == "__main__":
    main()
