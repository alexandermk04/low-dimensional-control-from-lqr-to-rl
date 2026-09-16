"""Figures and tables for the appendix identification length experiment.

Reads ``experiment_results/appendix_identification_length``.

Run with::

    python -m experiments.figures.appendix_identification_figures
"""

import matplotlib
import numpy as np

import settings
from experiments.figures.aggregate import latest_per_experiment, tidy
from experiments.figures.style import figure_dir, new_figure, save, table_dir
from experiments.figures.tables import metric_table, write_table
from experiments.store import load_experiment

LENGTH_EXPERIMENT = "appendix_identification_length"

# Suboptimality below this counts as "system identified exactly"
EXACT = 100 * np.finfo(float).eps

# Used for a cell in the grid without an identified system to be validated against
NO_DATA_COLOR = "#DDDDDD"
NO_DATA_HATCH = "///"

# Fixed log10 suboptimality color range
LOG_SUBOPT_MIN = -16
LOG_SUBOPT_MAX = -1


def _cell_value(rows: list[dict]) -> tuple[float, int, int, int]:
    """``(median suboptimality, no-id, unstable, total)`` for one grid cell.

    No-id, unstable, total refer to the amount of seeds per cell which
    either failed to identify the system, or produced an unstable controller.
    """
    identified = [r for r in rows if not r.get("identification_failed")]
    no_id = len(rows) - len(identified)

    values = [
        r["training_subopt_final"] for r in identified
        if r.get("training_subopt_final") is not None and np.isfinite(r["training_subopt_final"])
    ]
    unstable = len(identified) - len(values)

    median = float(np.median(values)) if values else np.inf

    return median, no_id, unstable, len(rows)


def _cell_label(median: float, no_id: int, unstable: int, total: int) -> tuple[str, str]:
    """``(text, weight)`` for one cell.
    """
    if no_id:
        return f"{no_id}/{total} no id", "bold" if no_id == total else "normal"

    if unstable:
        return (
            f"{unstable}/{total} unstable",
            "bold" if unstable == total else "normal",
        )

    if median <= EXACT:
        return "exact", "normal"

    return f"{median:.0e}", "normal"


def _text_color(rgba) -> str:
    """Black or white color for labels, depending on ``rgba`` value.
    """
    channels = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in rgba[:3]
    ]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    # Contrast against white is (1.05)/(L+0.05), against black (L+0.05)/0.05.
    return "black" if (luminance + 0.05) / 0.05 > 1.05 / (luminance + 0.05) else "white"


def identification_map(rows: list[dict], size: str):
    """The ``T`` x ``hankel horizon`` map for one specific inflation size.

    Returns
    -------
    matplotlib.axes.Axes
    """
    lengths = list(settings.IDENTIFICATION_LENGTHS)
    horizons = list(settings.IDENTIFICATION_HORIZONS)

    figure, ax = new_figure(width=0.9, aspect=0.7)
    ax.grid(False)

    grid = np.full((len(horizons), len(lengths)), np.nan)
    notes = {}

    for row_index, horizon in enumerate(horizons):
        for column, length in enumerate(lengths):
            cell = [
                r for r in rows
                if r.get("T") == length and r.get("hankel") == horizon
            ]
            if not cell:
                continue

            median, no_id, unstable, total = _cell_value(cell)
            grid[row_index, column] = (
                np.log10(max(median, 10.0 ** LOG_SUBOPT_MIN)) if np.isfinite(median) else np.nan
            )
            notes[(row_index, column)] = (median, no_id, unstable, total)

    colormap = matplotlib.colormaps["viridis_r"].with_extremes(bad=NO_DATA_COLOR)
    image = ax.imshow(
        grid, aspect="auto", origin="lower", cmap=colormap,
        vmin=LOG_SUBOPT_MIN, vmax=LOG_SUBOPT_MAX,
    )
    figure.colorbar(
        image, ax=ax, extend="max", label=r"$\log_{10}$ training suboptimality",
    )

    for (row_index, column), note in notes.items():
        value = grid[row_index, column]
        color = (
            _text_color(matplotlib.colors.to_rgba(NO_DATA_COLOR))
            if np.isnan(value) else _text_color(image.cmap(image.norm(value)))
        )

        if np.isnan(value):
            ax.add_patch(matplotlib.patches.Rectangle(
                (column - 0.5, row_index - 0.5), 1, 1,
                facecolor=NO_DATA_COLOR, edgecolor="white",
                hatch=NO_DATA_HATCH, linewidth=0,
            ))

        text, weight = _cell_label(*note)
        ax.text(
            column, row_index, text,
            ha="center", va="center", fontsize=6, fontweight=weight,
            color=color,
        )

    ax.set_xticks(range(len(lengths)), [str(length) for length in lengths])
    ax.set_yticks(range(len(horizons)), [str(horizon) for horizon in horizons])
    ax.set_xlabel(r"Trajectory length $T$")
    ax.set_ylabel("Hankel horizon")
    ax.set_title(f"{size.capitalize()} inflation")

    return ax


def render_length_sweep():
    """The map and table per inflation size."""
    records = latest_per_experiment(load_experiment(LENGTH_EXPERIMENT))

    for solver in settings.CH4_IDENTIFICATION_SOLVERS:
        for size in settings.INFLATION_SIZES:
            subset = [
                r for r in records
                if r.solver == solver and r.condition.get("size") == size
            ]
            if not subset:
                print(f"  no runs for {solver} at {size}, skipping")
                continue

            rows = tidy(subset)
            stem = f"appendix_identification_{solver}_{size}"

            ax = identification_map(rows, size)
            save(ax.figure, f"{stem}_map", LENGTH_EXPERIMENT)

            # One table per horizon & size
            for horizon in settings.IDENTIFICATION_HORIZONS:
                at_horizon = [r for r in rows if r.get("hankel") == horizon]
                if not at_horizon:
                    continue

                write_table(
                    metric_table(
                        at_horizon, "T",
                        metrics=("training_subopt_final", "test_cost_mean", "total_time"),
                        column_order=settings.IDENTIFICATION_LENGTHS,
                        column_header=r"Trajectory length $T$",
                        caption=(
                            f"{solver} on a system with {size} inflation with a Hankel "
                            f"horizon of {horizon}."
                        ),
                        label=(
                            "tab:appendix-identification-"
                            f"{solver.replace('_', '-')}-{size}-h{horizon}"
                        ),
                    ),
                    f"{stem}_h{horizon}",
                    LENGTH_EXPERIMENT,
                )


def main():
    render_length_sweep()

    print("wrote figures to", figure_dir(LENGTH_EXPERIMENT))
    print("wrote tables  to", table_dir(LENGTH_EXPERIMENT))


if __name__ == "__main__":
    main()
