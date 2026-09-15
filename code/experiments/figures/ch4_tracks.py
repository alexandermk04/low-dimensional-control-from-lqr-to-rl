"""The shared figure and table set behind both chapter-4 version with- and without model.
"""

import settings
from experiments.figures.aggregate import (
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import (
    learning_bands,
    trajectory_overlay,
    box_metric,
)
from experiments.figures.style import save
from experiments.figures.tables import metric_table, write_table
from config import DEFAULT_DT
from experiments.store import load_experiment

__all__ = ["render_track"]


def render_size(records, size: str, order, stem_prefix: str, track_caption: str):
    """The per-size figures and table, grouped by the solver."""
    subset = [r for r in records if r.condition.get("size") == size]
    if not subset:
        print(f"  no runs at {size}, skipping")
        return

    rows = with_suboptimality(tidy(subset), by=("seed", "size"))
    stem = f"{stem_prefix}_{size}"

    ax = box_metric(rows, "solver", "training_subopt_final", order=order)
    save(ax.figure, f"{stem}_training_subopt")

    # Training process
    ax = learning_bands(subset, "solver", x="env_steps", order=order)
    save(ax.figure, f"{stem}_curves_steps")

    # Trajectories
    axes = trajectory_overlay(subset, "solver", dt=DEFAULT_DT, order=order)
    save(axes[0].figure, f"{stem}_trajectories")

    # Test costs and suboptimality
    ax = box_metric(rows, "solver", "test_cost_mean", order=order)
    save(ax.figure, f"{stem}_testcost")

    ax = box_metric(rows, "solver", "subopt", order=order, log=False)
    save(ax.figure, f"{stem}_subopt")

    # Table 2, without policy complexity.
    write_table(
        metric_table(
            rows, "solver",
            metrics=("J_final", "training_subopt_final", "test_cost_mean",
                     "subopt", "total_env_steps"),
            column_order=order,
            column_header="Solver",
            caption=f"{track_caption} at the {size} inflation, without noise.",
            label=f"tab:{stem.replace('_', '-')}",
        ),
        stem,
    )


def render_across_sizes(records, solvers, stem_prefix: str, caption: str):
    """How a solver degrades as the dimension grows.
    """
    rows = with_suboptimality(tidy(records), by=("seed", "size"))

    order = [
        (size, solver)
        for size in settings.CH4_INFLATION_SIZES
        for solver in solvers
    ]

    stem = f"{stem_prefix}_across_sizes"

    ax = box_metric(rows, ("size", "solver"), "test_cost_mean", order=order)
    ax.set_xlabel("Inflation size / solver")
    save(ax.figure, f"{stem}_testcost")

    ax = box_metric(rows, ("size", "solver"), "subopt", order=order, log=False)
    ax.set_xlabel("Inflation size / solver")
    save(ax.figure, f"{stem}_subopt")

    write_table(
        metric_table(
            rows, ("size", "solver"),
            metrics=("training_subopt_final", "test_cost_mean", "subopt"),
            column_order=order,
            column_header="(size, solver)",
            caption=caption,
            label=f"tab:{stem.replace('_', '-')}",
        ),
        stem,
    )


def render_track(
    experiment: str, solvers, stem_prefix: str,
    track_caption: str, across_caption: str,
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
    """
    records = latest_per_experiment(load_experiment(experiment))
    order = tuple(solvers)

    for size in settings.CH4_INFLATION_SIZES:
        render_size(records, size, order, stem_prefix, track_caption)

    render_across_sizes(records, order, stem_prefix, across_caption)

    print("wrote figures to", settings.FIGURE_OUTPUT)
    print("wrote tables  to", settings.TABLE_OUTPUT)
