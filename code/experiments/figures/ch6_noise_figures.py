"""Figures and tables for chapter 6's robustness evaluation.

Run with::

    python -m experiments.figures.ch6_noise_figures
"""

from functools import partial

import numpy as np

import config as cfg
import settings
from experiments.figures.aggregate import (
    aggregate,
    label_for,
    latest_per_experiment,
    tidy,
    with_degradation,
)
from experiments.figures.ch4_noise_figures import render
from experiments.figures.tables import usable, write_table
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


def _percent(stats: dict) -> str:
    """Mean and standard deviation of a share, in whole percent."""
    if stats is None or stats["n"] == 0:
        return r"\multicolumn{1}{c}{--}"

    text = f"{100 * stats['mean']:.0f}"
    spread = f"{100 * stats['std']:.0f}"

    return f"${text} \\pm {spread}$" if float(spread) != 0 else f"${text}$"


def terminated_table() -> str:
    """Share of test episodes that end with the mass leaving the view, per channel and intensity.

    Unstable runs are excluded, the column ``none`` repeats the clean loop in every channel.
    """
    rows = [r for r in tidy(latest_per_experiment(load_experiment(EXPERIMENT))) if usable(r)]
    cells = aggregate(rows, ("solver", "channel", "level"), "terminated_rate")

    levels = settings.NOISE_LEVELS
    lines = [
        r"\begin{tabular}{@{}ll" + "r" * len(levels) + "@{}}",
        r"\toprule",
        " & ".join(["Channel", "Solver"] + [rf"\emph{{{level}}}" for level in levels]) + r" \\",
    ]
    for channel in settings.NOISE_CHANNELS:
        lines.append(r"\midrule")
        for i, solver in enumerate(settings.CH6_NOISE_SOLVERS):
            lines.append(" & ".join(
                [channel.capitalize() if i == 0 else "", label_for(solver)]
                + [
                    _percent(cells.get((solver, "clean", level) if level == "none"
                                       else (solver, channel, level)))
                    for level in levels
                ]
            ) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    caption = (
        r"Share of test episodes in the camera environment that end with the mass leaving the "
        r"view, in percent, per noise channel and intensity. Mean $\pm$ standard deviation over "
        r"the seeds; \emph{none} is the clean loop."
    )

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:perc-noise-terminated}",
        *lines,
        r"\end{table}",
    ])


def main():
    # Trajectories are sampled at the agent rate, exclude measurement noise calibration
    render(
        EXPERIMENT, PREFIX, settings.CH6_NOISE_SOLVERS,
        set_by="python -m experiments.ch6_noise",
        dt=cfg.DEFAULT_DT * cfg.DEFAULT_ACTION_REPEAT,
        scorer=partial(with_degradation, reference_cost=oracle_clean_cost()),
        calibrated=("control", "parametric", "disturbance"),
    )

    write_table(terminated_table(), "ch6_noise_terminated", EXPERIMENT)


if __name__ == "__main__":
    main()
