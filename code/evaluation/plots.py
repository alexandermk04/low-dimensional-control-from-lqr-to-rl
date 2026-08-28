"""Figures built from saved runs rather than from objects in memory.

This module reads :class:`~evaluation.records.RunRecord` from disk and 
creates its plots from there.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from evaluation.records import RunRecord, SUMMARY_KEYS

__all__ = [
    "FIGURE_ROOT",
    "learning_curves",
    "budget_table",
    "format_table",
    "trajectory_panel",
    "figure_from_records",
    "save_figure",
]

# Where figures are written to
FIGURE_ROOT = Path("results/figures")

X_LABELS = {
    "env_steps": "Environment steps",
    "wall_time": "Clock time (s)",
    "iteration": "Solver iteration",
}


def save_figure(figure, name: str, root: Path | str = FIGURE_ROOT) -> Path:
    """Write a figure, creating the directory if needed.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    name : str
        File stem; ``.png`` is appended.
    root : Path or str, optional
        Output directory.

    Returns
    -------
    Path
        The file written.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _curve(record: RunRecord, key: str) -> np.ndarray | None:
    """Fetch one learning-curve column, or ``None`` if the run doesn't contain it."""
    value = record.arrays.get(key)
    return None if value is None or value.size == 0 else value


def learning_curves(
    records: list[RunRecord],
    x: str = "env_steps",
    j_star: float | None = None, 
    normalize: bool = True,
    ax=None,
    logx: bool = True,
):
    """Plot cost during learning process, one line per run.

    Parameters
    ----------
    records : list[RunRecord]
        Runs to draw. Each needs ``cost`` and the chosen ``x`` column.
    x : str, optional
        ``"env_steps"``, ``"wall_time"``
        or ``"iteration"``.
    j_star : float or None, optional
        Optimal cost. Uses record's ``summary["J_star"]`` when
        omitted.
    normalize : bool, optional
        Plot suboptimality ``(J - J*) / J*`` instead of raw cost. Defaults to
        ``True``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; created if omitted.
    logx : bool, optional
        Log-scale the x-axis. Defaults to ``True`` since solvers might take different time.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if j_star is None:
        for record in records:
            if record.summary.get("J_star") is not None:
                j_star = float(record.summary["J_star"])
                break

    for record in records:
        cost = _curve(record, "cost")
        axis = _curve(record, x)
        if cost is None or axis is None:
            continue

        y = cost
        if normalize and j_star:
            y = (cost - j_star) / j_star

        finite = np.isfinite(y)
        if not finite.any():
            continue

        if np.all(axis == 0):
            # Aproach for model-based solvers without samples
            ax.axhline(y[finite][-1], linestyle=":", linewidth=2,
                       label=f"{record.label()} (no samples)")
            continue

        ax.plot(axis[finite], y[finite], label=record.label(), linewidth=1.6)

    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(X_LABELS.get(x, x))
    ax.set_ylabel("Suboptimality $(J - J^*)/J^*$" if normalize else "Cost $J$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    return ax


def budget_table(records: list[RunRecord]) -> list[dict]:
    """Collect the summary of every run into rows for a table.

    Parameters
    ----------
    records : list[RunRecord]
        Runs to summarize.

    Returns
    -------
    list[dict]
        One row per run: ``solver``, ``seed`` and every key in
        :data:`~evaluation.records.SUMMARY_KEYS`. Suitable for
        ``pandas.DataFrame`` or for :func:`format_table`.
    """
    return [
        {"solver": r.solver, "seed": r.seed, **{k: r.summary.get(k) for k in SUMMARY_KEYS}}
        for r in records
    ]


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render summary rows as a fixed-width text table.

    Parameters
    ----------
    rows : list[dict]
        As returned by :func:`budget_table`.
    columns : list[str] or None, optional
        Columns to show, in order. Defaults to a relevant preselection.

    Returns
    -------
    str
        The table for printing.
    """
    if not rows:
        return "(no runs)"

    if columns is None:
        columns = ["solver", "J_final", "subopt_final", "total_env_steps",
                   "total_wall_time", "rank_final", "policy_complexity", "converged"]

    def render(value):
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    table = [[c for c in columns]] + [[render(row.get(c)) for c in columns] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]

    lines = [
        "  ".join(cell.ljust(w) for cell, w in zip(table[0], widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in table[1:]]

    return "\n".join(lines)


def trajectory_panel(records: list[RunRecord], dt: float = 1.0, axes=None):
    """Draw position, velocity and control for every run that stored one.

    Parameters
    ----------
    records : list[RunRecord]
        Runs whose ``x_traj`` / ``u_traj`` arrays are drawn.
    dt : float, optional
        Sampling interval.
    axes : array of matplotlib.axes.Axes or None, optional
        Three axes to draw on; created if omitted.

    Returns
    -------
    np.ndarray
        The three axes.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(16, 4))

    for record in records:
        x_traj = record.arrays.get("x_traj")
        u_traj = record.arrays.get("u_traj")
        if x_traj is None:
            continue

        time = np.arange(x_traj.shape[0]) * dt
        axes[0].plot(time, x_traj[:, 0], label=record.label(), linewidth=1.3)
        axes[1].plot(time, x_traj[:, 1], label=record.label(), linewidth=1.3)
        if u_traj is not None:
            axes[2].plot(np.arange(u_traj.shape[0]) * dt, u_traj[:, 0],
                         label=record.label(), linewidth=1.3)

    for ax, title, ylabel in zip(
        axes, ("Position", "Velocity", "Control"), ("Position", "Velocity", "Control")
    ):
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    return axes


def figure_from_records(
    records: list[RunRecord],
    dt: float = 1.0,
    save_to: str | None = None,
    show: bool = False,
):
    """The comparison figure: trajectories, two different time axes, and the table.

    Parameters
    ----------
    records : list[RunRecord]
        Runs to compare.
    dt : float, optional
        Sampling interval, for the trajectory time axis.
    save_to : str or None, optional
        File stem under :data:`FIGURE_ROOT`. Saved if given.
    show : bool, optional
        Call ``plt.show()``. Defaults to ``False``.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure.
    """
    figure = plt.figure(figsize=(18, 10))

    trajectory_panel(
        records, dt=dt,
        axes=np.array([figure.add_subplot(2, 3, i) for i in (1, 2, 3)]),
    )

    learning_curves(records, x="env_steps", ax=figure.add_subplot(2, 3, 4))
    figure.axes[-1].set_title("Cost vs. environment steps")

    learning_curves(records, x="wall_time", ax=figure.add_subplot(2, 3, 5), logx=False)
    figure.axes[-1].set_title("Cost vs. wall time")

    ax = figure.add_subplot(2, 3, 6)
    ax.axis("off")
    ax.text(0.0, 1.0, format_table(budget_table(records)),
            family="monospace", fontsize=7, va="top", ha="left")

    figure.tight_layout()

    if save_to is not None:
        save_figure(figure, save_to)
    if show:
        plt.show()

    return figure
