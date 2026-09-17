r"""Different plots used for visualization.

Functions are reusable for different experiments.
"""

import numpy as np

from evaluation.records import RunRecord
from experiments.figures.aggregate import (
    aggregate,
    curve_band,
    group,
    label_for,
    latest_per_experiment,
)
from experiments.figures.style import color_for, new_figure

__all__ = [
    "box_metric",
    "learning_bands",
    "trajectory_overlay",
    "complexity_bars",
]

METRIC_LABELS = {
    "J_final": r"True cost $J$",
    "training_subopt_final": r"Training suboptimality $(J - J^*)/J^*$",
    "subopt": r"Suboptimality $S$",
    "subopt_episodes": r"Suboptimality $S$",
    "test_cost_mean": r"Accumulated test cost",
    "test_costs": r"Accumulated test cost",
    "r2_position": r"Linear probe $R^2$, position",
    "r2_velocity": r"Linear probe $R^2$, velocity",
    "off_screen_rate": r"Fraction of steps off screen",
    "null_waste_mean": r"Thrust wasted in the null space",
    "thruster_energy_mean": r"Thruster energy per step",
    "policy_complexity": r"Policy parameters",
    "rank_final": r"Gain rank",
    "total_env_steps": r"Environment steps",
    "total_time": r"Run time (s)",
}


def _ordered_keys(buckets: dict, order=None) -> list:
    """Cell keys in the requested order, unknown ones appended as encountered."""
    keys = list(buckets)
    if order is None:
        return keys

    wanted = [(o,) if not isinstance(o, tuple) else o for o in order]
    known = [k for k in wanted if k in buckets]
    return known + [k for k in keys if k not in known]


def _tick_label(key: tuple) -> str:
    """A short axis label for a cell key."""
    return "\n".join(label_for(part) for part in key if part is not None)


def box_metric(
    rows: list[dict],
    group_by,
    metric: str = "training_subopt_final",
    order=None,
    log: bool = True,
    ax=None,
    width: float = 1.0,
    aspect: float = 0.55,
    color_by: int | None = None,
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
    log : bool, optional
        Log-scale the y-axis. Must be ``False`` for a non-strictly
        positive metric
    ax : matplotlib.axes.Axes or None, optional
    width, aspect : float, optional
        Figure size, as a fraction of the text width, when ``ax`` is created.
    color_by : int or None, optional
        Position of the key part that picks the colour, e.g. ``1`` for the
        solver in ``("size", "solver")``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    stats = aggregate(rows, group_by, metric)
    keys = _ordered_keys(stats, order)

    if color_by is None:
        box_colors = [color_for(i, key[0]) for i, key in enumerate(keys)]
    else:
        hues = list(dict.fromkeys(key[color_by] for key in keys))
        box_colors = [
            color_for(hues.index(key[color_by]), key[color_by]) for key in keys
        ]

    if ax is None:
        _, ax = new_figure(width, aspect)

    positions, data, labels, notes = [], [], [], []
    per_episode = any(stats[key]["episodes"] for key in keys)

    for position, key in enumerate(keys, start=1):
        cell = stats[key]
        labels.append(_tick_label(key))

        values = cell["values"]

        if values.size == 0:
            # All diverged.
            notes.append((
                position,
                f"all {cell['n_episodes'] or cell['n_total']} diverged",
            ))
            continue

        if log:
            # A log axis cannot show a non-positive value. 
            positive = values[values > 0]
            if positive.size == 0:
                raise ValueError(
                    f"All values for {key} are non-positive, cannot log-scale."
                )
                continue
            dropped = values.size - positive.size
            if dropped:
                raise ValueError(
                    f"Non-positive values for {key} detected, cannot log-scale."
                )
            values = positive

        positions.append(position)
        data.append(values)

    if data:
        artists = ax.boxplot(
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
                ax.scatter(
                    position + jitter, values, s=6, color="black", alpha=0.5,
                    zorder=3, linewidths=0,
                )

    for position, note in notes:
        ax.annotate(
            note, xy=(position, 0.04), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=6,
        )

    if log:
        ax.set_yscale("log")
    if color_by is None:
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels(labels)
    else:
        _label_runs(ax, keys, color_by, box_colors)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlim(0.4, len(keys) + 0.6)

    return ax


def _label_runs(ax, keys: list, color_by: int, box_colors: list):
    """Coloring for x-ticks"""
    from matplotlib.patches import Patch

    rest = [key[:color_by] + key[color_by + 1:] for key in keys]
    ticks, names, start = [], [], 0
    for end in range(1, len(keys) + 1):
        if end == len(keys) or rest[end] != rest[start]:
            ticks.append((start + end + 1) / 2)
            names.append(_tick_label(rest[start]))
            if end < len(keys):
                ax.axvline(end + 0.5, color="black", linewidth=0.5, alpha=0.3)
            start = end
    ax.set_xticks(ticks)
    ax.set_xticklabels(names)
    ax.grid(False, axis="x")

    handles = {}
    for key, color in zip(keys, box_colors):
        handles.setdefault(
            key[color_by], Patch(facecolor=color, alpha=0.45, label=label_for(key[color_by]))
        )
    ax.legend(
        handles=list(handles.values()), frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
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
    logx: bool = True,
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

    def rescale(values):
        if normalize and j_star:
            return (np.asarray(values) - j_star) / j_star
        return np.asarray(values)

    if ax is None:
        _, ax = new_figure(width, aspect)

    for index, key in enumerate(keys):
        band = bands[key]
        color = color_for(index, key[0])
        label = _tick_label(key).replace("\n", ", ")

        if band["flat"]:
            if normalize and not float(rescale(band["median"])) > 0:
                continue
            ax.axhline(
                float(rescale(band["median"])), linestyle=":", linewidth=1.5,
                color=color, label=f"{label} (no samples)",
            )
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
    aspect: float = 0.30,
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
        Three axes. Created if omitted.
    width, aspect : float, optional

    Returns
    -------
    np.ndarray
        The three axes.
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
        _, axes = plt.subplots(1, 3, figsize=figure_size(width, aspect))
    axes = np.asarray(axes)

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

    axes[0].legend(frameon=False)

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

    Returns
    -------
    matplotlib.axes.Axes
    """
    stats = aggregate(rows, group_by, metric)
    keys = _ordered_keys(stats, order)

    if ax is None:
        _, ax = new_figure(width, aspect)

    for position, key in enumerate(keys, start=1):
        cell = stats[key]
        if not np.isfinite(cell["mean"]):
            continue
        ax.bar(
            position, cell["mean"], width=0.6,
            color=color_for(position - 1, key[0]), alpha=0.8,
        )

        # Only show if complexity varies across seeds
        if cell["max"] > cell["min"]:
            ax.errorbar(
                position, cell["mean"],
                yerr=[[cell["mean"] - cell["min"]], [cell["max"] - cell["mean"]]],
                fmt="none", ecolor="black", capsize=3, linewidth=1,
            )

    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([_tick_label(k) for k in keys])
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_xlim(0.4, len(keys) + 0.6)

    return ax
