"""Figures and tables for chapter 5's robustness evaluation.

Run with::

    python -m experiments.figures.ch5_noise_figures
"""

import settings
from experiments.ch5_noise import SIZES
from experiments.figures.aggregate import (
    aggregate,
    label_for,
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.ch4_noise_figures import render
from experiments.figures.tables import _fit, _math, usable, write_table
from experiments.store import load_experiment

EXPERIMENT = "ch5_noise"

PREFIX = "ch5_noise"

# Channel of the worst-case table
WORST_CASE_CHANNEL = "parametric"


def worst_case_table(size: str, channel: str = WORST_CASE_CHANNEL):
    """Median and worst test episode of :math:`S` per solver and intensity.

    Parameters
    ----------
    size : str
        Inflation size.
    channel : str, optional
        Noise channel.
    """
    records = [
        r for r in latest_per_experiment(load_experiment(EXPERIMENT))
        if r.condition.get("size") == size and r.condition.get("channel") == channel
    ]
    rows = [r for r in with_suboptimality(tidy(records)) if usable(r)]
    stats = aggregate(rows, ("solver", "level"), "subopt")

    levels = [lvl for lvl in settings.NOISE_LEVELS if lvl != "none"]
    solvers = [
        s for s in settings.CH5_SOLVERS
        if s != "baseline" and any((s, lvl) in stats for lvl in levels)
    ]
    n = len(levels)

    def cell(solver, level, key):
        stat = stats.get((solver, level))
        if stat is None or stat["n"] == 0:
            return r"\multicolumn{1}{c}{--}"
        return f"${_math(stat[key], '{:+.2g}')}$"

    lines = [
        r"\begin{tabular}{l" + "r" * (2 * n) + "}",
        r"\toprule",
        rf" & \multicolumn{{{n}}}{{c}}{{Median}} & \multicolumn{{{n}}}{{c}}{{Worst episode}} \\",
        rf"\cmidrule(lr){{2-{1 + n}}}\cmidrule(lr){{{2 + n}-{1 + 2 * n}}}",
        " & ".join(["Solver"] + [rf"\emph{{{lvl}}}" for lvl in levels] * 2) + r" \\",
        r"\midrule",
    ]
    for solver in solvers:
        lines.append(" & ".join(
            [label_for(solver)]
            + [cell(solver, lvl, "median") for lvl in levels]
            + [cell(solver, lvl, "max") for lvl in levels]
        ) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    noise = "parametric uncertainty" if channel == "parametric" else f"{channel} noise"
    caption = (
        rf"Median and worst test episode of the suboptimality $S$ under \emph{{{size}}} "
        rf"inflation and {noise} of increasing intensity, over all episodes of all seeds."
    )
    write_table(
        "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{PREFIX.replace('_', '-')}-{size}-{channel}-worst}}",
            _fit("\n".join(lines)),
            r"\end{table}",
        ]),
        f"{PREFIX}_{size}_{channel}_worst",
        EXPERIMENT,
    )


def main():
    # sizes imported from the experiment
    for size in SIZES:
        render(
            EXPERIMENT, f"{PREFIX}_{size}", settings.CH5_SOLVERS,
            set_by="python -m experiments.ch5_noise",
            size=size,
        )
        worst_case_table(size)


if __name__ == "__main__":
    main()
