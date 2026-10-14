r"""
Metrics tables for comparison, emitted as .tex files.
"""

import numpy as np

from experiments.figures.aggregate import aggregate, group, label_for
from experiments.figures.style import table_dir

__all__ = [
    "metric_table",
    "setting_table",
    "write_table",
]

# Counts for system identification tables
CENSUS_ROWS = ("Identified", "Stable")

# Label of the row with the divergence rate of the metric above it.
DIVERGED_ROW = r"\quad diverged"

# Label of the row counting episodes for which the baseline diverged
UNPAIRED_ROW = r"\quad no reference"

# Appended to the caption of a table that carries the census rows.
CENSUS_NOTE = (
    " Identified counts the seeds a model was obtained from and stable those "
    "whose controller left the plant stable."
)
CENSUS_NOTE_STABLE_ONLY = (
    " Stable counts the seeds whose controller left the plant stable."
)

# Row label and formatting per metric.
METRIC_ROWS = {
    "J_final": (r"True cost $J$", "{:.4g}"),
    "training_subopt_final": (r"Training suboptimality $(J-J^*)/J^*$", "{:.3g}"),
    "subopt": (r"Suboptimality $S$", "{:+.3g}"),
    "clean_ratio": (r"Cost ratio to the clean loop", "{:.3g}"),
    "policy_complexity": (r"Policy parameters", "{:.0f}"),
    "rank_final": (r"Gain rank", "{:.1f}"),
    "iterations": (r"Iterations", "{:.0f}"),
    "total_env_steps": (r"Env.\ steps to convergence", "{:.3g}"),
    "best_env_steps": (r"Env.\ steps to best policy", "{:.3g}"),
    "total_time": (r"Wall time (s)", "{:.1f}"),
    "spectral_radius_final": (r"Spectral radius", "{:.4f}"),
    "test_cost_mean": (r"Test cost", "{:.4g}"),
    "steps_to_thresh_1pct": (r"Env.\ steps to 1\,\%", "{:.3g}"),
    "iters_to_thresh_1pct": (r"Iterations to 1\,\%", "{:.0f}"),
}

DEFAULT_METRICS = ("J_final", "training_subopt_final", "subopt")


def _escape(text: str) -> str:
    """Escape the LaTeX specials that turn up in solver and condition names."""
    for char, replacement in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ):
        text = text.replace(char, replacement)
    return text


def usable(row: dict) -> bool:
    """Whether a run produced a usable controller.

    Depends on ``J_final`` where a run has one.
    """
    if row.get("identification_failed"):
        return False

    cost = row.get("J_final")
    if cost is not None:
        return bool(np.isfinite(float(cost)))

    episodes = row.get("test_costs")
    if episodes is not None:
        return bool(np.any(np.isfinite(np.asarray(episodes, dtype=float))))

    cost = row.get("test_cost_mean")

    return cost is not None and bool(np.isfinite(float(cost)))


def identifies(rows: list[dict]) -> bool:
    """Whether these runs came from a solver that uses N4SID.
    """
    return any(
        "n_hat" in row or row.get("identification_failed") for row in rows
    )


def _census(rows: list[dict], column_by) -> dict[tuple, tuple[int, int, int]]:
    """``(identified, stable, total)`` per column."""
    out = {}

    for key, bucket in group(rows, column_by).items():
        identified = [r for r in bucket if not r.get("identification_failed")]
        out[key] = (
            len(identified),
            sum(1 for r in identified if usable(r)),
            len(bucket),
        )

    return out


def _math(value: float, fmt: str) -> str:
    """Format a number for math mode.

    ``"{:.4g}".format`` produces ``5.624e+04``, which typesets as literal
    letters in math mode. This turns it into ``5.624 \\times 10^{4}``.
    """
    text = fmt.format(value)

    if "e" not in text:
        return text

    mantissa, exponent = text.split("e")
    power = int(exponent)

    if mantissa in ("1", "1.0"):
        return f"10^{{{power}}}"

    return f"{mantissa} \\times 10^{{{power}}}"


def _fit(tabular: str) -> str:
    r"""Shrink a tabular to the text width if it is too wide.
    """
    return "\n".join([
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        tabular,
        "}",
    ])


def _cell(stats: dict, fmt: str, show_spread: bool, scale: float = 1.0) -> str:
    """One table cell, containing its mean, optionally with its spread.

    Parameters
    ----------
    stats : dict or None
        One cell of :func:`~experiments.figures.aggregate.aggregate`.
    fmt : str
        Format of the number.
    show_spread : bool
        Append the standard deviation, unless it is exactly zero.
    scale : float, optional
        Factor applied to mean and spread before formatting.
    """
    if stats is None or stats["n"] == 0:
        # A cell where every seed diverged
        return r"\multicolumn{1}{c}{--}"

    mean = _math(scale * stats["mean"], fmt)

    if not show_spread or stats["n"] < 2 or stats["std"] == 0:
        return f"${mean}$"

    return f"${mean} \\pm {_math(scale * stats['std'], fmt.replace('+', ''))}$"


def _share(count: int, total: int) -> str:
    """A share of episodes, rounded to whole percent
    """
    if not total:
        return r"\multicolumn{1}{c}{--}"

    if count == 0:
        return r"$0\,\%$"

    percent = 100.0 * count / total

    # Use less than one instead of 0
    if f"{percent:.0f}" == "0":
        return r"$<1\,\%$"

    return f"${percent:.0f}\\,\\%$"


def _diverged(stats: dict) -> str:
    """The share of episodes that diverged."""
    if stats is None:
        return r"\multicolumn{1}{c}{--}"

    return _share(stats.get("n_diverged", 0), stats.get("n_episodes", 0))


def _unpaired(stats: dict) -> str:
    """The share of episodes for which reference diverged.
    """
    if stats is None:
        return r"\multicolumn{1}{c}{--}"

    return _share(
        stats.get("n_reference_diverged", 0), stats.get("n_episodes", 0)
    )


def metric_table(
    rows: list[dict],
    column_by,
    metrics=DEFAULT_METRICS,
    column_order=None,
    caption: str | None = None,
    label: str | None = None,
    show_spread: bool = True,
    column_header: str | None = None,
    fit: bool = True,
    census: bool | None = None,
) -> str:
    """Render metrics-by-conditions as a table.

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`.
    column_by : str or tuple of str
        Columns defining one table column.
    metrics : sequence of str, optional
        Rows of the table, in order. Keys of :data:`METRIC_ROWS`.
    column_order : sequence or None, optional
        Column order.
    caption, label : str or None, optional
        Wrap the tabular in a ``table`` float with this caption and
        ``\\label``.
    show_spread : bool, optional
        Append ``\\pm`` the standard deviation across the sample.
    column_header : str or None, optional
        Text over the metric column, e.g. ``"Solver"``.
    fit : bool, optional
        Whether to wrap the tabular in the shrink-to-fit box of :func:`_fit`.
    census : bool or None, optional
        Print how many seeds were identified and how many stayed stable, above
        the metrics.

    Returns
    -------
    str
        Ready to ``\\input``.
    """
    counts = _census(rows, column_by)
    with_identification = identifies(rows)
    show_census = any(
        identified != total or stable != total
        for identified, stable, total in counts.values()
    ) if census is None else census

    kept = [row for row in rows if usable(row)]

    stats = {
        metric: aggregate(kept, column_by, metric) for metric in metrics
    }

    keys = list(counts)
    if column_order is not None:
        wanted = [(c,) if not isinstance(c, tuple) else c for c in column_order]
        known = [k for k in wanted if k in counts]
        keys = known + [k for k in keys if k not in known]

    headers = [
        _escape(", ".join(str(label_for(part)) for part in key if part is not None))
        for key in keys
    ]

    lines = [
        r"\begin{tabular}{l" + "r" * len(keys) + "}",
        r"\toprule",
        " & ".join([column_header or ""] + headers) + r" \\",
        r"\midrule",
    ]

    # Census rows
    if show_census:
        identified_label, stable_label = CENSUS_ROWS
        if with_identification:
            lines.append(" & ".join(
                [identified_label]
                + [f"${counts[key][0]}/{counts[key][2]}$" for key in keys]
            ) + r" \\")
        lines.append(" & ".join(
            [stable_label]
            + [f"${counts[key][1]}/{counts[key][2]}$" for key in keys]
        ) + r" \\")
        lines.append(r"\midrule")

    for metric in metrics:
        row_label, fmt = METRIC_ROWS.get(metric, (_escape(metric), "{:.4g}"))

        cells = [
            _cell(stats[metric].get(key), fmt, show_spread) for key in keys
        ]
        lines.append(" & ".join([row_label] + cells) + r" \\")

        # Divergence rate of the solver
        if any(
            cell.get("n_diverged", 0) for cell in stats[metric].values()
        ):
            lines.append(" & ".join(
                [DIVERGED_ROW] + [_diverged(stats[metric].get(key)) for key in keys]
            ) + r" \\")

        # Baseline divergence report
        if any(
            cell.get("n_reference_diverged", 0)
            for cell in stats[metric].values()
        ):
            lines.append(" & ".join(
                [UNPAIRED_ROW] + [_unpaired(stats[metric].get(key)) for key in keys]
            ) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    tabular = "\n".join(lines)

    if fit:
        tabular = _fit(tabular)

    if caption is None and label is None:
        return tabular

    full_caption = (
        caption
        + ((CENSUS_NOTE if with_identification else CENSUS_NOTE_STABLE_ONLY)
           if show_census else "")
    )

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{full_caption}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        tabular,
        r"\end{table}",
    ])


def setting_table(
    rows: list[dict],
    row_by: str,
    metric: str | tuple[str, ...] = "subopt",
    column_by: str = "solver",
    row_order=None,
    column_order=None,
    caption: str | None = None,
    label: str | None = None,
    show_spread: bool = True,
    row_header: str | None = None,
    fit: bool = True,
    fmt: str | None = None,
    scale: float = 1.0,
) -> str:
    """Render a metric as a table of settings by solvers (or ``column_by``).

    Parameters
    ----------
    rows : list[dict]
        From :func:`~experiments.figures.aggregate.tidy`.
    row_by : str
        Column defining one table row, e.g. ``"level"`` or ``"size"``.
    metric : str or tuple of str, optional
        Key of :data:`METRIC_ROWS`, the value of every cell. Several keys
        place one group of columns per metric side by side, headed by its
        label.
    column_by : str, optional
        Column defining one table column, ``"solver"`` by default.
    row_order, column_order : sequence or None, optional
        Order of the rows and of the columns.
    caption, label : str or None, optional
        Wrap the tabular in a ``table`` float with this caption and
        ``\\label``.
    show_spread : bool, optional
        Append ``\\pm`` the standard deviation across the sample.
    row_header : str or None, optional
        Text over the settings column, e.g. ``"Intensity"``.
    fit : bool, optional
        Whether to wrap the tabular in the shrink-to-fit box of :func:`_fit`.
    fmt : str or None, optional
        Number format replacing the one of :data:`METRIC_ROWS`.
    scale : float, optional
        Factor applied to every value, e.g. ``1e3`` to show in
        :math:`10^{-3}`.

    Returns
    -------
    str
        Ready to ``\\input``.
    """
    metrics = (metric,) if isinstance(metric, str) else tuple(metric)
    kept = [row for row in rows if usable(row)]
    stats = {m: aggregate(kept, (row_by, column_by), m) for m in metrics}
    keys = [key for m in metrics for key in stats[m]]

    def ordered(values, order):
        values = list(dict.fromkeys(values))
        if order is None:
            return values
        return [v for v in order if v in values] + [v for v in values if v not in order]

    settings = ordered((key[0] for key in keys), row_order)
    columns = ordered((key[1] for key in keys), column_order)

    labels = {m: METRIC_ROWS.get(m, (_escape(m), "{:.4g}")) for m in metrics}
    n = len(columns)

    lines = [
        r"\begin{tabular}{l" + "r" * (n * len(metrics)) + "}",
        r"\toprule",
    ]

    # One group header per metric
    if len(metrics) > 1:
        lines.append(" & ".join(
            [""] + [rf"\multicolumn{{{n}}}{{c}}{{{labels[m][0]}}}" for m in metrics]
        ) + r" \\")
        lines.append("".join(
            rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}"
            for i in range(len(metrics))
        ))

    lines += [
        " & ".join(
            [row_header or ""]
            + [_escape(str(label_for(column))) for column in columns] * len(metrics)
        ) + r" \\",
        r"\midrule",
    ]

    def seeds(source, column):
        return {r["seed"] for r in source if r.get(column_by) == column}

    census = {column: (len(seeds(kept, column)), len(seeds(rows, column))) for column in columns}
    show_census = column_by == "solver" and any(
        stable != total for stable, total in census.values()
    )
    if show_census:
        lines += [
            " & ".join(
                [CENSUS_ROWS[1]]
                + [f"${census[column][0]}/{census[column][1]}$" for column in columns] * len(metrics)
            ) + r" \\",
            r"\midrule",
        ]

    for setting in settings:
        cells = [
            stats[m].get((setting, column)) for m in metrics for column in columns
        ]
        formats = [fmt or labels[m][1] for m in metrics for _ in columns]
        lines.append(" & ".join(
            [_escape(str(label_for(setting)))]
            + [
                _cell(cell, cell_fmt, show_spread, scale)
                for cell, cell_fmt in zip(cells, formats)
            ]
        ) + r" \\")

        # Divergence rate of the solvers in this setting
        if any(cell and cell.get("n_diverged", 0) for cell in cells):
            lines.append(" & ".join(
                [DIVERGED_ROW] + [_diverged(cell) for cell in cells]
            ) + r" \\")

        if any(cell and cell.get("n_reference_diverged", 0) for cell in cells):
            lines.append(" & ".join(
                [UNPAIRED_ROW] + [_unpaired(cell) for cell in cells]
            ) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    tabular = "\n".join(lines)

    if fit:
        tabular = _fit(tabular)

    if caption is None and label is None:
        return tabular

    full_caption = (caption or "") + (CENSUS_NOTE_STABLE_ONLY if show_census else "")

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{full_caption}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        tabular,
        r"\end{table}",
    ])


def write_table(text: str, name: str, experiment: str, subdir: str | None = None):
    """Write a table fragment to the experiment's table directory.

    Parameters
    ----------
    text : str
        As returned by :func:`metric_table`.
    name : str
        File stem. ``\\input{generated/<experiment>/tables/<name>.tex}`` to use
        in latex.
    experiment : str
        Name for the output directory
    subdir : str or None, optional
        Subdirectory of the table directory to write into.

    Returns
    -------
    Path
        The file written.
    """
    directory = table_dir(experiment)
    if subdir is not None:
        directory = directory / subdir
        directory.mkdir(exist_ok=True)
    path = directory / f"{name}.tex"
    path.write_text(text + "\n")
    return path
