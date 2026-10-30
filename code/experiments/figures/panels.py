r"""Different plots used for visualization.

Functions are reusable for different experiments.
"""

import textwrap

import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, MaxNLocator, NullLocator

from evaluation.records import RunRecord
from experiments.figures.aggregate import (
    aggregate,
    clip_roundoff,
    curve_band,
    group,
    label_for,
    latest_per_experiment,
)
from experiments.figures.style import BASELINE_COLOR, color_for, new_figure
from experiments.figures.tables import usable

__all__ = [
    "box_metric",
    "degradation_curves",
    "learning_bands",
    "trajectory_overlay",
    "complexity_bars",
]

METRIC_LABELS = {
    "J_final": r"True cost $J$",
    "training_subopt_final": r"Training suboptimality $(J - J^*)/J^*$",
    "subopt": r"Suboptimality $S$",
    "subopt_episodes": r"Suboptimality $S$",
    "relative_degradation": r"Relative degradation $\kappa$",
    "absolute_degradation": r"Absolute degradation $\Delta \hat{J} / \hat{J}^{*}$",
    "r2_position": r"Linear probe $R^2$, position",
    "r2_velocity": r"Linear probe $R^2$, velocity",
    "r2_control": r"Linear probe $R^2$, optimal control",
    "off_screen_rate": r"Fraction of steps off screen",
    "terminated_rate": r"Share of episodes exiting the screen",
    "null_waste_mean": r"Thrust wasted in the null space",
    "thruster_energy_mean": r"Thruster energy per step",
    "policy_complexity": r"Policy parameters",
    "rank_final": r"Gain rank",
    "total_env_steps": r"Environment steps",
    "total_time": r"Run time (s)",
}

ABSOLUTE_LINTHRESH = 1e-2

# Gap from when on we split the y axis
SPLIT_GAP = 1.0


def _ordered_keys(buckets: dict, order=None) -> list:
    """Cell keys in the requested order, unknown ones appended as encountered."""
    keys = list(buckets)
    if order is None:
        return keys

    wanted = [(o,) if not isinstance(o, tuple) else o for o in order]
    known = [k for k in wanted if k in buckets]
    return known + [k for k in keys if k not in known]


def _tick_label(key: tuple, wrap: int | None = None) -> str:
    """A short axis label for a cell key.

    Parameters
    ----------
    key : tuple
        A cell key, e.g. ``("large", "spi_est")``.
    wrap : int or None, optional
        Break a part longer than this many characters over several lines.
    """
    parts = [label_for(part) for part in key if part is not None]

    if wrap:
        parts = [
            "\n".join(
                textwrap.wrap(
                    part, wrap, break_long_words=False,
                ) or [part]
            )
            for part in parts
        ]

    return "\n".join(parts)


def _wrap_width(n_labels: int, width: float = 1.0) -> int | None:
    """How wide a tick label may be before it is broken over several lines.
    """
    wrap = int(70 * width / max(n_labels, 1))
    if width < 1.0 and wrap < 8:
        return None
    return max(8, wrap)


def _angle(ax):
    """Angle the x tick labels."""
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")


def _linthresh_for(cells: list[np.ndarray]) -> float:
    """Where a symlog axis should switch from log to linear.

    Parameters
    ----------
    cells : list of np.ndarray
        The values of each box.

    Returns
    -------
    float
    """
    smallest = []
    for values in cells:
        magnitudes = np.abs(values)
        nonzero = magnitudes[magnitudes > 0]
        if nonzero.size:
            smallest.append(float(np.percentile(nonzero, 5)))

    if not smallest:
        return 1.0

    return float(10.0 ** np.floor(np.log10(min(smallest))))


def _symlog_axis(ax, cells: list[np.ndarray], linscale: float, floor: float = 0.0):
    """Symlog y-axis fitted to the shown values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    cells : list of np.ndarray
        The values of each box or curve.
    linscale : float
        Height of the linear part, in decades.
    floor : float, optional
        Smallest threshold of the linear part.
    """
    linthresh = max(_linthresh_for(cells), floor)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)

    lowest = min((float(np.min(values)) for values in cells if np.size(values)), default=0.0)
    if lowest > -linthresh:
        ax.set_ylim(bottom=min(-0.5 * linthresh, 1.2 * lowest))
    else:
        # default margin
        ax.set_ylim(bottom=2.0 * lowest)

    if linscale < 1.0:
        # less labels here
        default = ax.yaxis.get_major_formatter()
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, pos: "" if np.isclose(abs(y), linthresh) else default(y, pos)
        ))


class _PlainLogFormatter(LogFormatterSciNotation):
    """Labels the same ticks as the default log formatter, as plain numbers."""

    def __call__(self, x, pos=None):
        if not super().__call__(x, pos):
            return ""
        return rf"$\mathdefault{{{x:.3g}}}$"


def _plain_log_ticks(ax):
    """Plain numbers instead of powers of ten on a log y-axis of at most two decades."""
    low, high = ax.get_ylim()
    if low <= 0 or high / low > 100:
        return

    if high / low < 2:
        # Rather space evenly
        inches = ax.get_position().height * ax.figure.get_figheight()
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(2, round(inches / 0.3))))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"$\mathdefault{{{x:g}}}$"))
        return

    ax.yaxis.set_major_formatter(_PlainLogFormatter())
    ax.yaxis.set_minor_formatter(_PlainLogFormatter(labelOnlyBase=False))


def _split_ranges(data: list[np.ndarray], log: bool):
    """Value ranges of two far apart groups of boxes.

    Parameters
    ----------
    data : list of np.ndarray
        The values of each box.
    log : bool
        Whether distances are measured in decades.

    Returns
    -------
    tuple or None
        ``((low, high), (low, high))`` of the lower and the upper group, or
        ``None`` if no gap is larger than ``SPLIT_GAP`` times both groups.
    """
    scale = np.log10 if log else (lambda value: value)
    spans = sorted((scale(np.min(values)), scale(np.max(values))) for values in data)
    best = None

    for cut in range(1, len(spans)):
        lower_top = max(span[1] for span in spans[:cut])
        upper_bottom = spans[cut][0]
        upper_top = max(span[1] for span in spans[cut:])
        gap = upper_bottom - lower_top
        extent = (lower_top - spans[0][0]) + (upper_top - upper_bottom)
        if gap > SPLIT_GAP * extent and (best is None or gap > best[0]):
            best = (gap, (spans[0][0], lower_top), (upper_bottom, upper_top))

    if best is None:
        return None

    unscale = (lambda value: 10.0 ** value) if log else (lambda value: value)
    ranges = []
    for low, high in best[1:]:
        pad = max(0.1 * (high - low), 0.01 if log else 0.05 * max(abs(low), abs(high), 1e-12))
        ranges.append((unscale(low - pad), unscale(high + pad)))
    return tuple(ranges)


def _broken_axes(width: float, aspect: float, n_upper: int, n_lower: int):
    """Two stacked axes with break marks, the upper one for the upper group of boxes."""

    ratio = min(max(n_upper, n_lower) / max(min(n_upper, n_lower), 1), 3.0)
    shares = (ratio, 1.0) if n_upper >= n_lower else (1.0, ratio)
    figure, (upper, lower) = new_figure(
        width, aspect, nrows=2, sharex=True,
        gridspec_kw={"height_ratios": shares},
    )
    figure.get_layout_engine().set(h_pad=0.3)

    upper.spines["bottom"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    marks = dict(
        marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle="none",
        color="black", mec="black", mew=0.8, clip_on=False,
    )
    upper.plot([0], [0], transform=upper.transAxes, **marks)
    lower.plot([0], [1], transform=lower.transAxes, **marks)

    return figure, upper, lower


def box_metric(
    rows: list[dict],
    group_by,
    metric: str = "training_subopt_final",
    order=None,
    log: bool | str = True,
    ax=None,
    width: float = 1.0,
    aspect: float = 0.55,
    color_by: int | None = None,
    split: bool = False,
):
    """Box plot for a specific metric, one box per setting combination.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`.
    group_by : str or tuple of str
        Column(s) defining one box, e.g. ``"solver"`` or ``("l", "m")``.
    metric : str, optional
        Which summary column to draw.
    order : sequence or None, optional
        Settings order along the x-axis.
    log : bool or str, optional
        ``True`` log-scales the y-axis, ``"symlog"`` log-scales both signs
        around a linear window, ``False`` keeps the axis linear.
    ax : matplotlib.axes.Axes or None, optional
    width, aspect : float, optional
        Figure size, as a fraction of the text width, when ``ax`` is created.
    color_by : int or None, optional
        Position of the key part that picks the colour, e.g. ``1`` for the
        solver in ``("size", "solver")``.
    split : bool, optional
        Break the y-axis into two panels when the boxes form two far apart
        groups. Only without ``ax``, ``color_by`` and a symlog axis.

    Returns
    -------
    matplotlib.axes.Axes
        The upper panel if the axis is broken.
    """
    stats = aggregate([row for row in rows if usable(row)], group_by, metric)
    keys = _ordered_keys(stats, order)

    if color_by is None:
        box_colors = [color_for(i, key[0]) for i, key in enumerate(keys)]
    else:
        hues = list(dict.fromkeys(key[color_by] for key in keys))
        box_colors = [
            color_for(hues.index(key[color_by]), key[color_by]) for key in keys
        ]

    positions, data, labels, notes = [], [], [], []
    per_episode = any(stats[key]["episodes"] for key in keys)

    symlog = log == "symlog"
    wrap = _wrap_width(len(keys), width)

    for position, key in enumerate(keys, start=1):
        cell = stats[key]
        labels.append(_tick_label(key, wrap))

        values = cell["values"]

        if values.size == 0:
            # All diverged.
            notes.append((
                position,
                f"all {cell['n_episodes'] or cell['n_total']} diverged",
            ))
            continue

        if log and not symlog:
            # A log axis cannot show a non-positive value.
            positive = values[values > 0]
            if positive.size == 0:
                raise ValueError(
                    f"All values for {key} are non-positive, cannot log-scale."
                )
            dropped = values.size - positive.size
            if dropped:
                raise ValueError(
                    f"Non-positive values for {key} detected, cannot log-scale."
                )
            values = positive

        positions.append(position)
        data.append(values)

    ranges = (
        _split_ranges(data, log=bool(log))
        if split and ax is None and color_by is None and not symlog and len(data) > 1
        else None
    )
    if ranges is not None:
        lower_range, upper_range = ranges
        n_upper = sum(np.min(values) >= upper_range[0] for values in data)
        _, ax, bottom = _broken_axes(width, aspect, n_upper, len(data) - n_upper)
        panels = ((ax, upper_range), (bottom, lower_range))
    else:
        if ax is None:
            _, ax = new_figure(width, aspect)
        bottom = ax
        panels = ((ax, None),)

    for panel, limits in panels:
        if data:
            artists = panel.boxplot(
                data, positions=positions, widths=0.6,
                patch_artist=True, showfliers=True, whis=(5, 95),
                medianprops=dict(color="black", linewidth=1.2),
                flierprops=dict(
                    marker=".", markersize=2, markerfacecolor="black",
                    markeredgecolor="none", alpha=0.35,
                ),
                capprops=dict(linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
            )
            for box, position in zip(artists["boxes"], positions):
                box.set_facecolor(box_colors[position - 1])
                box.set_alpha(0.45)
                box.set_linewidth(0.8)

            if not per_episode:
                jitter_rng = np.random.default_rng(0)
                for position, values in zip(positions, data):
                    jitter = (jitter_rng.random(values.size) - 0.5) * 0.18
                    panel.scatter(
                        position + jitter, values, s=6, color="black", alpha=0.5,
                        zorder=3, linewidths=0,
                    )

        if symlog:
            if width < 1.0:
                _symlog_axis(panel, data, linscale=1.0)
                panel.yaxis.get_major_locator().set_params(numticks=8)
            else:
                _symlog_axis(panel, data, linscale=0.35)
        elif log:
            panel.set_yscale("log")

        if limits is not None:
            panel.set_ylim(*limits)
        if log and not symlog:
            _plain_log_ticks(panel)

    for position, note in notes:
        bottom.annotate(
            note, xy=(position, 0.04), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=6,
        )

    if color_by is None:
        bottom.set_xticks(range(1, len(keys) + 1))
        bottom.set_xticklabels(labels)
        if wrap is None:
            _angle(bottom)
    else:
        _label_runs(ax, keys, color_by, box_colors)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlim(0.4, len(keys) + 0.6)

    return ax


def degradation_curves(
    rows: list[dict],
    order,
    levels,
    reference: dict | None = None,
    absolute: bool = False,
    ax=None,
    width: float = 1.0,
    aspect: float = 0.55,
):
    """Pooled relative degradation of every solver over the intensities of one channel.

    With ``absolute``, the pooled absolute degradation instead, the difference
    to the clean cost in units of ``reference_cost`` from
    :func:`~experiments.figures.aggregate.with_degradation`.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`, the clean runs and
        the runs of one channel.
    order : sequence of str
        Solvers to draw, in legend order.
    levels : sequence of str
        Intensities along the x-axis, ``"none"`` being the clean loop.
    reference : dict or None, optional
        Relative degradation of the optimal gain per intensity, drawn dashed.
    absolute : bool, optional
        Draw the absolute rather than the relative degradation.
    ax : matplotlib.axes.Axes or None, optional
    width, aspect : float, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    stats = aggregate(
        [row for row in rows if usable(row)], ("solver", "level"), "test_cost_mean"
    )
    positions = np.arange(len(levels))

    if ax is None:
        _, ax = new_figure(width, aspect)

    def pooled(solver, level):
        return stats.get((solver, level), {}).get("mean", np.nan)

    j_ref = next((r["reference_cost"] for r in rows if r.get("reference_cost")), np.nan)

    drawn = []
    for index, solver in enumerate(order):
        costs = np.array([pooled(solver, level) for level in levels])
        clean = pooled(solver, "none")
        values = (costs - clean) / j_ref if absolute else costs / clean
        if not np.any(np.isfinite(values)):
            continue
        drawn.append(values[np.isfinite(values)])
        ax.plot(
            positions, values, color=color_for(index, solver), marker="o",
            markersize=3.5, linewidth=1.2, label=label_for(solver),
        )

    if reference is not None:
        kappa = np.array([reference.get(level, 1.0) for level in levels])
        ax.plot(
            positions, kappa - 1.0 if absolute else kappa,
            color=BASELINE_COLOR, linestyle="--", linewidth=1.0,
            label=r"$\kappa_\ell - 1$" if absolute else r"$\kappa_\ell$",
        )

    if absolute:
        _symlog_axis(ax, drawn, linscale=0.35, floor=ABSOLUTE_LINTHRESH)
    else:
        ax.set_yscale("log")
        _plain_log_ticks(ax)
    ax.set_xticks(positions)
    ax.set_xticklabels([rf"\emph{{{level}}}" for level in levels])
    if width < 1.0:
        _angle(ax)
    ax.set_xlabel("Noise intensity")
    ax.set_ylabel(METRIC_LABELS["absolute_degradation" if absolute else "relative_degradation"])
    handles, labels = ax.get_legend_handles_labels()

    # based on rendered label
    short = max((len(label) for label in labels[:len(drawn)]), default=0) <= 10
    ncol = min(len(handles), (7 if short else 3) if width >= 1.0 else (
        (4 if reference is None else 3) if short else 1
    ))

    # Fill by row
    order = [i for c in range(ncol) for i in range(c, len(handles), ncol)]
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol,
        columnspacing=0.6, handlelength=1.2,
    )

    return ax


def _label_runs(
    ax, keys: list, color_by: int, box_colors: list, extra: tuple = (),
    alpha: float = 0.45,
):
    """Coloring for x-ticks, with ``extra`` legend handles appended.

    ``alpha`` matches the legend patches to the drawn boxes or bars.
    """
    from matplotlib.patches import Patch

    rest = [key[:color_by] + key[color_by + 1:] for key in keys]
    ticks, runs, start = [], [], 0
    for end in range(1, len(keys) + 1):
        if end == len(keys) or rest[end] != rest[start]:
            ticks.append((start + end + 1) / 2)
            runs.append(rest[start])
            if end < len(keys):
                ax.axvline(end + 0.5, color="black", linewidth=0.5, alpha=0.3)
            start = end

    wrap = _wrap_width(len(runs))

    ax.set_xticks(ticks)
    ax.set_xticklabels([_tick_label(run, wrap) for run in runs])
    ax.grid(False, axis="x")

    handles = {}
    for key, color in zip(keys, box_colors):
        handles.setdefault(
            key[color_by], Patch(facecolor=color, alpha=alpha, label=label_for(key[color_by]))
        )
    handles = list(handles.values()) + list(extra)
    ax.legend(
        handles=handles, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=min(len(handles), 3),
    )


def learning_bands(
    records: list[RunRecord],
    group_by,
    x: str = "env_steps",
    normalize: bool = True,
    order=None,
    ax=None,
    width: float = 1.0,
    aspect: float = 0.55,
    logx: bool = False,
):
    """Plot the training process, median and range over the ten seeds.

    Parameters
    ----------
    records : list[RunRecord]
        With ``cost`` and the chosen ``x`` column in ``arrays``.
    group_by : str or tuple of str
        Column(s) defining one band.
    x : str, optional
        ``"env_steps"``, ``"iteration"`` or ``"total_time"``.
    normalize : bool, optional
        Plot suboptimality rather than raw cost.
    order : sequence or None, optional
    ax : matplotlib.axes.Axes or None, optional
    width, aspect : float, optional
    logx : bool, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    bands = curve_band(records, group_by, x=x)
    keys = _ordered_keys(bands, order)

    j_star = next(
        (float(r.summary["J_star"]) for r in records
         if r.summary.get("J_star") is not None),
        None,
    )

    flat_drawn = 0

    drawn = []

    def rescale(values):
        if normalize and j_star:
            scaled = clip_roundoff((np.asarray(values) - j_star) / j_star)
        else:
            scaled = np.asarray(values)
        drawn.append(np.atleast_1d(scaled))
        return scaled

    if ax is None:
        _, ax = new_figure(width, aspect)

    for index, key in enumerate(keys):
        band = bands[key]
        color = color_for(index, key[0])
        label = _tick_label(key).replace("\n", ", ")

        if band["flat"]:
            median = float(rescale(band["median"]))
            if not np.isfinite(median):
                continue
            exact = median <= 0.0
            suffix = "no iterations"
            ax.axhline(
                median, linestyle=(2 * flat_drawn % 8, (2, 6)), linewidth=1.5,
                color=color, label=f"{label} ({suffix})",
            )
            flat_drawn += 1
            low, high = float(rescale(band["low"])), float(rescale(band["high"]))
            if np.isfinite(low) and np.isfinite(high) and high > low:
                ax.axhspan(low, high, color=color, alpha=0.12, linewidth=0)
            continue

        ax.plot(band["x"], rescale(band["median"]), color=color, linewidth=1.4,
                label=label)
        ax.fill_between(
            band["x"], rescale(band["low"]), rescale(band["high"]),
            color=color, alpha=0.18, linewidth=0,
        )

    if logx:
        ax.set_xscale("log")

    if normalize:
        _symlog_axis(ax, drawn, linscale=0.35)
    else:
        ax.set_yscale("log")

    ax.set_xlabel({
        "env_steps": "Environment steps",
        "total_time": "Run time (s)",
        "iteration": "Solver iteration",
    }.get(x, x))
    ax.set_ylabel(
        r"Training suboptimality $(J - J^*)/J^*$" if normalize else r"True cost $J$"
    )
    ax.legend(frameon=False)

    return ax


def trajectory_overlay(
    records: list[RunRecord],
    group_by,
    dt: float = 0.01,
    seed: int | None = None,
    order=None,
    axes=None,
    width: float = 1.0,
    aspect: float = 0.62,
):
    """Comparison of different trajectories per settings combination.

    Parameters
    ----------
    records : list[RunRecord]
        With ``x_traj`` and ``u_traj`` in ``arrays``.
    group_by : str or tuple of str
        Column(s) defining one line.
    dt : float, optional
        Sampling interval, for the time axis.
    seed : int or None, optional
        Draw only this seed
    order : sequence or None, optional
    axes : array of Axes or None, optional
        Four axes. Created if
        omitted.
    width, aspect : float, optional

    Returns
    -------
    np.ndarray
        The four axes, the last of which holds the legend.
    """
    if seed is None:
        seed = min(r.seed for r in records)
    records = [r for r in records if r.seed == seed]

    rows = [
        {"solver": r.solver, "seed": r.seed, "_record": r, **r.condition}
        for r in latest_per_experiment(records)
    ]
    buckets = group(rows, group_by)
    keys = _ordered_keys(buckets, order)

    if axes is None:
        import matplotlib.pyplot as plt
        from experiments.figures.style import figure_size
        _, axes = plt.subplots(2, 2, figsize=figure_size(width, aspect))
    axes = np.asarray(axes).reshape(-1)

    for index, key in enumerate(keys):
        record = buckets[key][0]["_record"]
        x_traj = record.arrays.get("x_traj")
        u_traj = record.arrays.get("u_traj")
        if x_traj is None:
            continue

        color = color_for(index, key[0])
        label = _tick_label(key).replace("\n", ", ")
        time = np.arange(x_traj.shape[0]) * dt

        axes[0].plot(time, x_traj[:, 0], color=color, linewidth=1.1, label=label)
        axes[1].plot(time, x_traj[:, 1], color=color, linewidth=1.1, label=label)
        if u_traj is not None:
            axes[2].plot(
                np.arange(u_traj.shape[0]) * dt, u_traj[:, 0],
                color=color, linewidth=1.1, label=label,
            )

    for ax, ylabel in zip(axes, ("Position", "Velocity", "Control")):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)

    # for the legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[3].axis("off")
    if handles:
        axes[3].legend(
            handles, labels, frameon=False, loc="center",
            ncol=1 if len(handles) <= 6 else 2,
        )

    return axes


def complexity_bars(
    rows: list[dict],
    group_by,
    metric: str = "policy_complexity",
    order=None,
    ax=None,
    width: float = 1.0,
    aspect: float = 0.5,
    log: bool = True,
    color_by: int | None = None,
    frozen: str | None = "policy_complexity_frozen",
):
    """Policy complexity as a bar plot.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`.
    group_by : str or tuple of str
    metric : str, optional
        ``"policy_complexity"``, the policy size
        :math:`\\min\\{mn,\\, r(m + n + 1)\\}`. 
    order : sequence or None, optional
    ax : matplotlib.axes.Axes or None, optional
    width, aspect : float, optional
    log : bool, optional
    color_by : int or None, optional
        Position of the key part that picks the colour, e.g. ``1`` for the
        solver in ``("size", "solver")``.
    frozen : str or None, optional
        Column with the part of ``metric`` held fixed during training.

    Returns
    -------
    matplotlib.axes.Axes
    """
    stats = aggregate(rows, group_by, metric)
    frozen_stats = aggregate(rows, group_by, frozen) if frozen else {}
    keys = _ordered_keys(stats, order)

    if color_by is None:
        bar_colors = [color_for(i, key[0]) for i, key in enumerate(keys)]
    else:
        hues = list(dict.fromkeys(key[color_by] for key in keys))
        bar_colors = [
            color_for(hues.index(key[color_by]), key[color_by]) for key in keys
        ]

    if ax is None:
        _, ax = new_figure(width, aspect)

    frozen_bars = []
    for position, key in enumerate(keys, start=1):
        cell = stats[key]
        if not np.isfinite(cell["mean"]):
            continue
        ax.bar(
            position, cell["mean"], width=0.6,
            color=bar_colors[position - 1], alpha=0.8,
        )

        frozen_mean = frozen_stats.get(key, {}).get("mean", np.nan)
        if np.isfinite(frozen_mean) and frozen_mean > 0:
            frozen_bars.append((position, frozen_mean))

        # Only show if complexity varies across seeds
        if cell["max"] > cell["min"]:
            ax.errorbar(
                position, cell["mean"],
                yerr=[[cell["mean"] - cell["min"]], [cell["max"] - cell["mean"]]],
                fmt="none", ecolor="black", capsize=3, linewidth=1,
            )

    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0)

    low, high = ax.get_ylim()
    bottom = low if log else 0.0
    for position, frozen_mean in frozen_bars:
        ax.bar(
            position, frozen_mean - bottom, bottom=bottom, width=0.6, fill=False,
            hatch="///", edgecolor="white", linewidth=0,
        )
    ax.set_ylim(low, high)

    from matplotlib.patches import Patch
    extra = [
        Patch(facecolor="0.6", edgecolor="white", hatch="///", linewidth=0,
              label="Frozen during training")
    ] if frozen_bars else []

    if color_by is None:
        wrap = _wrap_width(len(keys), width)
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels([_tick_label(k, wrap) for k in keys])
        if wrap is None:
            _angle(ax)
        if extra:
            ax.legend(handles=extra, frameon=False, loc="best")
    else:
        _label_runs(ax, keys, color_by, bar_colors, extra, alpha=0.8)

    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlim(0.4, len(keys) + 0.6)

    return ax
