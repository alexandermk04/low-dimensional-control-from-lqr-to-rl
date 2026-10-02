"""Figures and tables for chapter 4's robustness evaluation.

Run with::

    python -m experiments.figures.ch4_noise_figures
"""

import settings
from experiments.figures.aggregate import (
    label_for,
    latest_per_experiment,
    tidy,
    with_suboptimality,
)
from experiments.figures.panels import trajectory_overlay, box_metric
from experiments.figures.style import figure_dir, save, table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.settings_guard import require_choice
from experiments.ch4_noise import SIZES
from experiments.store import load_experiment

EXPERIMENT = "ch4_noise"

PREFIX = "ch4_noise"

SET_BY = "python -m experiments.ch4_noise"


def render(experiment: str, prefix: str, solvers, set_by: str = SET_BY, size=None):
    """Draw the noise figures and tables of one chapter.

    Parameters
    ----------
    experiment : str
        Which noise experiment to read.
    prefix : str
        File stem prefix for the generated files.
    solvers : sequence of str
        Solver order along the x-axis.
    set_by : str
        For guard messages.
    size : str or None, optional
        Restrict to one inflation size.
    """
    level = require_choice(
        settings.CHOSEN_NOISE_LEVEL, "CHOSEN_NOISE_LEVEL",
        settings.NOISE_LEVELS, set_by,
    )
    channel = require_choice(
        settings.CHOSEN_NOISE_CHANNEL, "CHOSEN_NOISE_CHANNEL",
        settings.NOISE_CHANNELS, set_by,
    )

    slug = prefix.replace("_", "-")

    records = latest_per_experiment(load_experiment(experiment))
    if size is not None:
        records = [r for r in records if r.condition.get("size") == size]
        if not records:
            print(f"  no {experiment} runs at size {size!r}, skipping")
            return

    rows = with_suboptimality(tidy(records))

    level_order = tuple(settings.NOISE_LEVELS)
    channel_order = ("clean",) + tuple(settings.NOISE_CHANNELS) + ("all",)

    cells = [("clean", "none")] + [
        (ch, lvl)
        for lvl in level_order if lvl != "none"
        for ch in channel_order if ch != "clean"
    ]

    for cell_channel, cell_level in cells:
        subset = [
            r for r in rows
            if r.get("channel") == cell_channel and r.get("level") == cell_level
        ]
        if not subset:
            continue

        if cell_channel == "clean":
            title, stem = "No noise", "clean"
        else:
            title = f"Channel: {cell_channel}, intensity: {cell_level}"
            stem = f"{cell_channel}-{cell_level}"

        # Suboptimality can be negative
        ax = box_metric(subset, "solver", "subopt", order=solvers, log="symlog")
        ax.set_title(title)
        save(ax.figure, f"{prefix}_subopt_{stem}", experiment)

    reference = "baseline" if "baseline" in solvers else solvers[0]
    reference_records = [r for r in records if r.solver == reference]

    reference_levels = [
        r for r in reference_records
        if r.condition.get("channel") in (channel, "clean")
    ]
    if reference_levels:
        axes = trajectory_overlay(
            reference_levels, "level", dt=0.01, order=level_order
        )
        save(axes[0].figure, f"{prefix}_traj_levels_{reference}", experiment)

    reference_channels = [
        r for r in reference_records
        if r.condition.get("level") in (level, "none")
    ]
    if reference_channels:
        axes = trajectory_overlay(
            reference_channels, "channel", dt=0.01, order=channel_order
        )
        save(axes[0].figure, f"{prefix}_traj_channels_{reference}", experiment)

    for solver in solvers:
        solver_rows = [r for r in rows if r["solver"] == solver]
        if not solver_rows:
            continue

        short = solver.replace("_", "-")

        # Table 3
        write_table(
            metric_table(
                [r for r in solver_rows if r.get("channel") in (channel, "clean")],
                "level",
                metrics=("subopt",),
                column_order=level_order,
                column_header="Intensity",
                caption=f"{label_for(solver)} under {channel} noise of increasing intensity.",
                label=f"tab:{slug}-levels-{short}",
            ),
            f"{prefix}_levels_{solver}",
            experiment,
        )

        # Table 4
        write_table(
            metric_table(
                [r for r in solver_rows if r.get("level") in (level, "none")],
                "channel",
                metrics=("subopt",),
                column_order=channel_order,
                column_header="Channel",
                caption=(
                    f"{label_for(solver)} under each noise channel at intensity \\emph{{{level}}}."
                ),
                label=f"tab:{slug}-channels-{short}",
            ),
            f"{prefix}_channels_{solver}",
            experiment,
        )

    print("wrote figures to", figure_dir(experiment))
    print("wrote tables  to", table_dir(experiment))


def main():
    for size in SIZES:
        render(EXPERIMENT, f"{PREFIX}_{size}", settings.CH4_IDENTIFIED_SOLVERS, size=size)


if __name__ == "__main__":
    main()
