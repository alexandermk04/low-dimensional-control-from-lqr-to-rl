"""Figures and tables for chapter 4's robustness evaluation.

Run with::

    python -m experiments.figures.ch4_noise_figures
"""

import settings
from experiments.figures.aggregate import (
    group,
    label_for,
    latest_per_experiment,
    tidy,
    with_degradation,
)
from experiments.figures.panels import box_metric, degradation_curves, trajectory_overlay
from experiments.figures.style import figure_dir, save, save_variants, table_dir
from experiments.figures.tables import METRIC_ROWS, metric_table, setting_table, write_table
from experiments.settings_guard import require_choice
from experiments.ch4_noise import SIZES
from experiments.store import load_experiment

EXPERIMENT = "ch4_noise"

PREFIX = "ch4_noise"

SET_BY = "python -m experiments.ch4_noise"


def render(experiment: str, prefix: str, solvers, set_by: str = SET_BY, size=None,
           dt: float = 0.01, metric: str = "relative_degradation", scorer=with_degradation,
           log: bool | str = True, calibrated=tuple(settings.NOISE_CHANNELS),
           extra_metrics=("absolute_degradation",)):
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
    dt : float, optional
        Time between two trajectory samples, in seconds.
    metric : str, optional
        Column that is plotted and tabulated.
    scorer : callable, optional
        Adds ``metric`` to the tidy rows.
    log : bool or str, optional
        Axis scaling of the box plots.
    calibrated : sequence of str, optional
        Channels whose intensities were calibrated on the optimal gain. Only
        drawn as a reference when the baseline is not among ``solvers``.
    extra_metrics : sequence of str, optional
        Columns tabulated next to ``metric``.
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

    rows = scorer(tidy(records))
    table_metrics = (metric, *extra_metrics)
    name = " and ".join(METRIC_ROWS[m][0] for m in table_metrics)

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
            title = short_title = "No noise"
            stem = "clean"
        else:
            title = f"Channel: {cell_channel}, intensity: {cell_level}"
            short_title = f"{cell_channel}, {cell_level}"
            stem = f"{cell_channel}-{cell_level}"

        def draw(subset=subset, title=title, short_title=short_title, **size):
            ax = box_metric(
                subset, "solver", metric, order=solvers, log=log, **size
            )
            ax.set_title(short_title if size else title)
            return ax

        save_variants(draw, f"{prefix}_{metric}_{stem}", experiment)

    # Degradation over the intensities
    for cell_channel in channel_order[1:]:
        subset = [r for r in rows if r.get("channel") in (cell_channel, "clean")]
        if not any(r.get("channel") == cell_channel for r in subset):
            continue

        reference = (
            settings.NOISE_COST_TARGETS
            if "baseline" not in solvers and cell_channel in calibrated else None
        )
        for absolute, kind in ((False, "tradeoff_relative"), (True, "tradeoff_absolute")):
            save_variants(
                lambda subset=subset, reference=reference, absolute=absolute, **size:
                degradation_curves(
                    subset, solvers, level_order, reference=reference,
                    absolute=absolute, **size,
                ),
                f"{prefix}_{kind}_{cell_channel}",
                experiment,
            )

    reference = "baseline" if "baseline" in solvers else solvers[0]
    reference_records = [r for r in records if r.solver == reference]

    reference_levels = [
        r for r in reference_records
        if r.condition.get("channel") in (channel, "clean")
    ]
    if reference_levels:
        axes = trajectory_overlay(
            reference_levels, "level", dt=dt, order=level_order
        )
        save(axes[0].figure, f"{prefix}_traj_levels_{reference}", experiment)

    reference_channels = [
        r for r in reference_records
        if r.condition.get("level") in (level, "none")
    ]
    if reference_channels:
        axes = trajectory_overlay(
            reference_channels, "channel", dt=dt, order=channel_order
        )
        save(axes[0].figure, f"{prefix}_traj_channels_{reference}", experiment)

    inflation = f" under \\emph{{{size}}} inflation and" if size else " under"

    write_table(
        setting_table(
            [r for r in rows if r.get("channel") in (channel, "clean")],
            "level",
            metric=table_metrics,
            row_order=level_order,
            column_order=solvers,
            row_header="Intensity",
            caption=f"{name}{inflation} {channel} noise of increasing intensity.",
            label=f"tab:{slug}-levels",
        ),
        f"{prefix}_levels",
        experiment,
    )

    write_table(
        setting_table(
            [r for r in rows if r.get("level") in (level, "none")],
            "channel",
            metric=table_metrics,
            row_order=channel_order,
            column_order=solvers,
            row_header="Channel",
            caption=(
                f"{name}{inflation} each noise channel at intensity \\emph{{{level}}}."
            ),
            label=f"tab:{slug}-channels",
        ),
        f"{prefix}_channels",
        experiment,
    )

    # Solvers by channels, one table per intensity
    for cell_level in level_order:
        if cell_level == "none":
            continue

        write_table(
            setting_table(
                [
                    r for r in rows
                    if r["solver"] in solvers and r.get("level") in (cell_level, "none")
                ],
                "solver",
                metric=table_metrics,
                column_by="channel",
                row_order=solvers,
                column_order=channel_order,
                row_header="Solver",
                caption=(
                    f"{name}{inflation} each noise channel at "
                    f"intensity \\emph{{{cell_level}}}."
                ),
                label=f"tab:{slug}-solver-channels-{cell_level}",
            ),
            f"{prefix}_channels_{cell_level}",
            experiment,
            subdir="channels",
        )

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
                metrics=table_metrics,
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
                metrics=table_metrics,
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


def render_sizes(experiment: str, prefix: str, solvers, sizes):
    """Tables comparing the inflation sizes.

    Parameters
    ----------
    experiment : str
        Which noise experiment to read.
    prefix : str
        File stem prefix for the generated files.
    solvers : sequence of str
        Solver order along the columns.
    sizes : sequence of str
        Inflation sizes along the rows.
    """
    slug = prefix.replace("_", "-")

    records = latest_per_experiment(load_experiment(experiment))
    rows = with_degradation(tidy(records), by=("solver", "seed", "size"))

    for (cell_channel, cell_level), subset in group(rows, ("channel", "level")).items():
        stem = "clean" if cell_channel == "clean" else f"{cell_channel}-{cell_level}"
        noise = (
            "without noise" if cell_channel == "clean"
            else f"under {cell_channel} noise at intensity \\emph{{{cell_level}}}"
        )

        write_table(
            setting_table(
                subset,
                "size",
                metric=("relative_degradation", "absolute_degradation"),
                row_order=sizes,
                column_order=solvers,
                row_header="Inflation",
                caption=(
                    f"{METRIC_ROWS['relative_degradation'][0]} and {METRIC_ROWS['absolute_degradation'][0]} "
                    f"per inflation size {noise}."
                ),
                label=f"tab:{slug}-sizes-{stem}",
            ),
            f"{prefix}_sizes_{stem}",
            experiment,
            subdir="sizes",
        )


def main():
    for size in SIZES:
        render(EXPERIMENT, f"{PREFIX}_{size}", settings.CH4_IDENTIFIED_SOLVERS, size=size)
    render_sizes(EXPERIMENT, PREFIX, settings.CH4_IDENTIFIED_SOLVERS, SIZES)


if __name__ == "__main__":
    main()
