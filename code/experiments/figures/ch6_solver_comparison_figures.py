"""Figures and tables for chapter 6's cross-solver comparison at pixel size.

Run with::

    python -m experiments.figures.ch6_solver_comparison_figures
"""

import numpy as np

import settings
from experiments.figures.aggregate import (
    aggregate,
    label_for,
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import (
    complexity_bars,
    learning_bands,
    trajectory_overlay,
    box_metric,
    _symlog_axis,
)
from experiments.figures.style import figure_dir, new_figure, save, save_variants, table_dir
from experiments.figures.tables import _fit, _math, metric_table, usable, write_table
from experiments.store import load_experiment

EXPERIMENT = "ch6_solver_comparison"
LARGE_EXPERIMENT = "ch5_inflation_sizes"

ORDER = tuple(settings.CH6_COMPARISON_SOLVERS)


def spread_table(rows):
    """Median, best and worst test episode of the suboptimality per solver.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.with_suboptimality`.
    """
    solvers = [
        s for s in ORDER
        if s != "baseline" and any(r["solver"] == s for r in rows)
    ]
    kept = [r for r in rows if usable(r)]
    stats = aggregate(kept, "solver", "subopt")

    def cell(solver, key):
        stat = stats.get((solver,))
        if stat is None or stat["n"] == 0:
            return r"\multicolumn{1}{c}{--}"
        return f"${_math(stat[key], '{:.3g}')}$"

    lines = [
        r"\begin{tabular}{l" + "r" * len(solvers) + "}",
        r"\toprule",
        " & ".join(["Solver"] + [label_for(s) for s in solvers]) + r" \\",
        r"\midrule",
    ]

    seeds = {
        s: (
            sum(1 for r in kept if r["solver"] == s),
            sum(1 for r in rows if r["solver"] == s),
        )
        for s in solvers
    }
    if any(stable != total for stable, total in seeds.values()):
        lines += [
            " & ".join(
                ["Stable"] + [f"${seeds[s][0]}/{seeds[s][1]}$" for s in solvers]
            ) + r" \\",
            r"\midrule",
        ]

    for key, name in (
        ("median", "Median"), ("min", "Best episode"), ("max", "Worst episode")
    ):
        lines.append(
            " & ".join([name] + [cell(s, key) for s in solvers]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]

    caption = (
        r"Median, best and worst test episode of the suboptimality $S$ of the "
        rf"PPO variants under \emph{{{settings.CH6_PIXEL_INFLATION}}} inflation, "
        "without noise, pooled over the stable seeds."
    )
    write_table(
        "\n".join([
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:ch6-solvers-spread}",
            _fit("\n".join(lines)),
            r"\end{table}",
        ]),
        "ch6_solvers_spread",
        EXPERIMENT,
    )


def large_vs_pixel(learned):
    """Test suboptimality under large and pixel inflation.

    Parameters
    ----------
    learned : list[dict]
        Pixel rows without the baseline, from
        :func:`~experiments.figures.aggregate.with_suboptimality`.
    """
    large = [
        r for r in latest_per_experiment(load_experiment(LARGE_EXPERIMENT))
        if r.condition.get("size") == "large"
    ]
    large_rows = with_suboptimality(tidy(large), by=("seed", "size"))
    # Only the variants that also run at pixel size
    large_learned = [r for r in large_rows if r["solver"] in ORDER[1:]]

    _, axes = new_figure(1.0, 0.5, ncols=2, sharey=True)
    panels = ((axes[0], large_learned, "large"), (axes[1], learned, "pixel"))
    for ax, rows, size in panels:
        box_metric(rows, "solver", "subopt", order=ORDER, log="symlog", ax=ax, width=0.48)
        ax.set_title(rf"\emph{{{size}}}")
    axes[1].set_ylabel("")

    # Shared axis
    cells = [
        stat["values"]
        for rows in (large_learned, learned)
        for stat in aggregate([r for r in rows if usable(r)], "solver", "subopt").values()
    ]
    _symlog_axis(axes[0], cells, linscale=1.0)
    axes[0].set_ylim(top=2.0 * max(float(np.max(values)) for values in cells if np.size(values)))
    axes[0].yaxis.get_major_locator().set_params(numticks=8)

    save(axes[0].figure, "ch6_solvers_subopt_large_pixel", EXPERIMENT)


def main():
    records = latest_per_experiment(load_experiment(EXPERIMENT))
    rows = with_suboptimality(tidy(records), by=("seed",))

    learned = [r for r in rows if r["solver"] != "baseline"]
    learned_records = [r for r in records if r.solver != "baseline"]

    ax = box_metric(
        learned, "solver", "training_subopt_final", order=ORDER, log="symlog"
    )
    save(ax.figure, "ch6_solvers_training_subopt", EXPERIMENT)

    ax = learning_bands(learned_records, "solver", x="env_steps", order=ORDER)
    save(ax.figure, "ch6_solvers_curves_steps", EXPERIMENT)

    axes = trajectory_overlay(records, "solver", dt=0.01, order=ORDER)
    save(axes[0].figure, "ch6_solvers_trajectories", EXPERIMENT)

    # Half-widths
    save_variants(
        lambda **size: box_metric(
            learned, "solver", "subopt", order=ORDER, log="symlog", **size
        ),
        "ch6_solvers_subopt",
        EXPERIMENT,
    )
    large_vs_pixel(learned)

    ax = complexity_bars(learned, "solver", order=ORDER)
    save(ax.figure, "ch6_solvers_complexity", EXPERIMENT)

    write_table(
        metric_table(
            rows, "solver",
            metrics=("J_final", "training_subopt_final", "subopt",
                     "policy_complexity", "rank_final",
                     "iterations", "total_env_steps", "best_env_steps", "total_time"),
            column_order=ORDER,
            column_header="Solver",
            caption=(
                f"PPO variants under \\emph{{{settings.CH6_PIXEL_INFLATION}}} inflation, "
                "without noise."
            ),
            label="tab:ch6-solvers",
        ),
        "ch6_solvers",
        EXPERIMENT,
    )

    spread_table(rows)

    print("wrote figures to", figure_dir(EXPERIMENT))
    print("wrote tables  to", table_dir(EXPERIMENT))


if __name__ == "__main__":
    main()
