"""The shared figure and table set behind both chapter-4 version with- and without model.
"""

import settings
from experiments.figures.aggregate import (
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import (
    complexity_bars,
    learning_bands,
    trajectory_overlay,
    box_metric,
)
from experiments.figures.style import figure_dir, save, save_variants, table_dir
from experiments.figures.tables import metric_table, setting_table, write_table
from config import DEFAULT_DT
from experiments.store import load_experiment

__all__ = ["render_track"]

# What to show on x-axis
CURVE_SUFFIX = {"env_steps": "steps", "iteration": "iterations", "total_time": "time"}


def render_size(
    records, size: str, order, stem_prefix: str, track_caption: str,
    experiment: str, curve_x: str = "env_steps", curves: bool = True,
    trajectories: bool = True,
):
    """The per-size figures and table, grouped by the solver.
    """
    subset = [r for r in records if r.condition.get("size") == size]
    if not subset:
        print(f"  no runs at {size}, skipping")
        return

    rows = with_suboptimality(tidy(subset), by=("seed", "size"))
    stem = f"{stem_prefix}_{size}"

    # baseline is 0 for suboptimality
    learned = [r for r in rows if r["solver"] != "baseline"]
    ax = box_metric(
        learned, "solver", "training_subopt_final", order=order, log="symlog"
    )
    save(ax.figure, f"{stem}_training_subopt", experiment)

    # Training process
    if curves:
        save_variants(
            lambda **size: learning_bands(
                subset, "solver", x=curve_x, order=order,
                logx=curve_x != "iteration", **size,
            ),
            f"{stem}_curves_{CURVE_SUFFIX.get(curve_x, curve_x)}",
            experiment,
        )

    # Trajectories
    if trajectories:
        axes = trajectory_overlay(subset, "solver", dt=DEFAULT_DT, order=order)
        save(axes[0].figure, f"{stem}_trajectories", experiment)

    # Paired test suboptimality
    save_variants(
        lambda **size: box_metric(
            rows, "solver", "subopt", order=order, log="symlog", **size
        ),
        f"{stem}_subopt",
        experiment,
    )

    save_variants(
        lambda **size: complexity_bars(rows, "solver", order=order, **size),
        f"{stem}_complexity",
        experiment,
    )

    # Table 2
    write_table(
        metric_table(
            rows, "solver",
            metrics=("J_final", "training_subopt_final", "subopt",
                     "iterations", "total_env_steps", "best_env_steps", "policy_complexity", "rank_final"),
            column_order=order,
            column_header="Solver",
            caption=f"{track_caption} under \\emph{{{size}}} inflation, without noise.",
            label=f"tab:{stem.replace('_', '-')}",
        ),
        stem,
        experiment,
    )


def render_across_sizes(
    records, solvers, stem_prefix: str, caption: str, experiment: str,
):
    """How a solver degrades as the dimension grows.
    """
    rows = with_suboptimality(tidy(records), by=("seed", "size"))

    order = [
        (size, solver)
        for size in settings.CH4_INFLATION_SIZES
        for solver in solvers
    ]

    stem = f"{stem_prefix}_across_sizes"

    ax = box_metric(
        rows, ("size", "solver"), "subopt", order=order, log="symlog", color_by=1
    )
    ax.set_xlabel("Inflation size")
    save(ax.figure, f"{stem}_subopt", experiment)

    ax = complexity_bars(rows, ("size", "solver"), order=order, color_by=1)
    ax.set_xlabel("Inflation size")
    save(ax.figure, f"{stem}_complexity", experiment)

    write_table(
        setting_table(
            rows, "solver",
            metric=("policy_complexity", "rank_final"),
            column_by="size",
            row_order=solvers,
            column_order=settings.CH4_INFLATION_SIZES,
            row_header="Solver",
            caption=caption,
            label=f"tab:{stem.replace('_', '-')}",
        ),
        stem,
        experiment,
    )


def render_track(
    experiment: str, solvers, stem_prefix: str,
    track_caption: str, across_caption: str, curve_x: str = "env_steps",
    curves: bool = True, trajectories: bool = True,
):
    """Draw one chapter-4 setting.

    Parameters
    ----------
    experiment : str
        Name the track's runs were stored under.
    solvers : tuple[str, ...]
        Solver order
    stem_prefix : str
        Prefix of every written file name.
    track_caption : str
        Clause of the per-size table captions.
    across_caption : str
        Full caption of the cross-size table.
    curve_x : str, optional
        x-axis of the learning curves.
    curves : bool, optional
        Whether to draw the learning curves.
    trajectories : bool, optional
        Whether to create the trajectory curves.
    """
    records = latest_per_experiment(load_experiment(experiment))
    order = tuple(solvers)

    for size in settings.CH4_INFLATION_SIZES:
        render_size(
            records, size, order, stem_prefix, track_caption, experiment,
            curve_x, curves=curves, trajectories=trajectories,
        )

    render_across_sizes(
        records, order, stem_prefix, across_caption, experiment
    )

    print("wrote figures to", figure_dir(experiment))
    print("wrote tables  to", table_dir(experiment))
