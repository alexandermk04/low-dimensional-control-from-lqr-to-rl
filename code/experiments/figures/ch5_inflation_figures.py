"""Figures and tables for chapter 5's inflation-size experiment.

Run with::

    python -m experiments.figures.ch5_inflation_figures
"""

import settings
from experiments.figures.aggregate import label_for, latest_per_experiment, tidy
from experiments.figures.ch4_tracks import render_track
from experiments.figures.tables import _fit, usable, write_table
from experiments.store import load_experiment

EXPERIMENT = "ch5_inflation_sizes"

ORDER = tuple(settings.CH5_SOLVERS)

PREFIX = "ch5_inflation"


def _size_table(rows, columns, cells, caption: str, name: str):
    """A table with one row per PPO variant and a group of columns per size.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`.
    columns : sequence of str
        Header of the columns inside each size group.
    cells : callable
        Maps the runs of one solver and size to one cell per column.
    caption : str
        Caption of the table.
    name : str
        Suffix of the file stem and the label.
    """
    sizes = settings.CH5_INFLATION_SIZES
    k = len(columns)

    lines = [
        r"\begin{tabular}{l" + "r" * (k * len(sizes)) + "}",
        r"\toprule",
        " & " + " & ".join(
            rf"\multicolumn{{{k}}}{{c}}{{\emph{{{size}}}}}" for size in sizes
        ) + r" \\",
        "".join(
            rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}"
            for i in range(len(sizes))
        ),
        " & ".join(["Solver"] + list(columns) * len(sizes)) + r" \\",
        r"\midrule",
    ]

    for solver in ORDER[1:]:
        line = [label_for(solver)]
        for size in sizes:
            line += cells([
                r for r in rows if r["solver"] == solver and r["size"] == size
            ])
        lines.append(" & ".join(line) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    write_table(
        "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{tab:{PREFIX.replace('_', '-')}-{name}}}",
            _fit("\n".join(lines)),
            r"\end{table}",
        ]),
        f"{PREFIX}_{name}",
        EXPERIMENT,
    )


def parameter_table(rows):
    """Trainable, frozen and total actor parameters per solver and size."""

    def count(n: int) -> str:
        return "$" + f"{n:,}".replace(",", r"\,") + "$"

    def cells(bucket):
        # Same architecture for every seed
        total = int(bucket[0]["policy_complexity"])
        frozen = int(bucket[0].get("policy_complexity_frozen") or 0)
        return [count(total - frozen), count(frozen), count(total)]

    _size_table(
        rows, ("Trainable", "Frozen", "Total"), cells,
        caption=(
            "Actor parameters of the PPO variants for each inflation size, split "
            "into those trained by PPO and those frozen after pretraining."
        ),
        name="parameters",
    )


def stability_table(rows):
    """Stable seeds and runaway test episodes per solver and size."""
    episodes = max(
        sum(r["test_episodes"] for r in rows
            if r["solver"] == solver and r["size"] == size)
        for solver in ORDER for size in settings.CH5_INFLATION_SIZES
    )

    def cells(bucket):
        stable = f"${sum(usable(r) for r in bucket)}/{len(bucket)}$"
        counts = [r.get("runaway_episodes") for r in bucket]

        # A linear gain is judged by its spectral radius only
        if all(c is None for c in counts):
            return [stable, r"\multicolumn{1}{c}{--}"]
        return [stable, f"${sum(c or 0 for c in counts)}$"]

    _size_table(
        rows, ("Stable", "Runaway"), cells,
        caption=(
            "Stability of the PPO variants for each inflation size, without noise. "
            "Stable counts the seeds whose controller kept the plant stable and "
            "contained the mass in every test episode, runaway the test episodes "
            f"(out of ${episodes}$) that end with the mass outside the range of "
            "initial positions. A linear gain is evaluated by its spectral radius only."
        ),
        name="stability",
    )


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

    rows = tidy(latest_per_experiment(load_experiment(EXPERIMENT)))
    parameter_table(rows)
    stability_table(rows)


if __name__ == "__main__":
    main()
