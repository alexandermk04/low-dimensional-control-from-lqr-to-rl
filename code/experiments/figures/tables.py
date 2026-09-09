r"""
Metrics tables for comparison, emitted as .tex files.
"""

import numpy as np

from experiments.figures.aggregate import aggregate, group, label_for
from experiments.figures.style import table_dir

__all__ = [
    "metric_table",
    "write_table",
    "DEFAULT_METRICS",
    "METRIC_ROWS",
    "POOLED_MARKER",
    "CENSUS_ROWS",
    "usable",
]

# Counts for system identification tables
CENSUS_ROWS = ("Identified", "Stable")

# Marks a row whose statistics are over test episodes rather than over seeds.
POOLED_MARKER = r"$^{\dagger}$"

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
    "J_best": (r"Best true cost $J$", "{:.4g}"),
    "subopt_final": (r"Suboptimality $(J-J^*)/J^*$", "{:.3f}"),
    "subopt_best": (r"Best suboptimality", "{:.3f}"),
    "test_cost_mean": (r"Test cost", "{:.4g}"),
    "policy_complexity": (r"Policy parameters", "{:.0f}"),
    "n_params_dense": (r"Dense parameters", "{:.0f}"),
    "rank_final": (r"Gain rank", "{:.1f}"),
    "total_env_steps": (r"Environment steps", "{:.3g}"),
    "total_time": (r"Wall time (s)", "{:.1f}"),
    "spectral_radius_final": (r"Spectral radius", "{:.4f}"),
}

DEFAULT_METRICS = ("J_final", "subopt_final", "test_cost_mean")


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
    """
    if row.get("identification_failed"):
        return False

    cost = row.get("J_final")

    return cost is not None and np.isfinite(float(cost))


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


def _provenance(stats: dict, pooled_metrics: list, show_spread: bool) -> str:
    """Sentence stating what the numbers were taken over.
    """
    if not show_spread:
        return ""

    cells = [cell for metric in stats.values() for cell in metric.values()]
    if not cells:
        return ""

    runs = max(cell["n_total"] for cell in cells)
    per_seed = [m for m in stats if m not in pooled_metrics]

    if not pooled_metrics:
        return f" Mean $\\pm$ standard deviation over {runs} seeds."

    episodes = max(
        (cell["n"] for metric in pooled_metrics
         for cell in stats[metric].values() if cell["pooled"]),
        default=0,
    )
    per_run = episodes // runs if runs else episodes

    pooled_clause = (
        f" {POOLED_MARKER} mean $\\pm$ standard deviation over all {episodes} test "
        f"episodes ({per_run} per seed, {runs} seeds)"
    )

    if not per_seed:
        return pooled_clause + "."

    return pooled_clause + f"; every other row over the {runs} seeds."


def _fit(tabular: str) -> str:
    r"""Shrink a tabular to the text width if it is too wide.
    """
    return "\n".join([
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        tabular,
        "}",
    ])


def _cell(stats: dict, fmt: str, show_spread: bool) -> str:
    """One table cell, containing its mean, optionally with its spread.
    """
    if stats is None or stats["n"] == 0:
        # A cell where every seed diverged is a result, not a gap.
        return r"\multicolumn{1}{c}{--}"

    mean = _math(stats["mean"], fmt)

    if not show_spread or stats["n"] < 2:
        return f"${mean}$"

    return f"${mean} \\pm {_math(stats['std'], fmt)}$"


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
    pool: bool | None = None,
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
        Append ``\\pm`` the standard deviation across seeds.
    column_header : str or None, optional
        Text over the metric column, e.g. ``"Solver"``.
    fit : bool, optional
        Whether to wrap the tabular in the shrink-to-fit box of :func:`_fit`.
    pool : bool or None, optional
        Passed to :func:`~experiments.figures.aggregate.aggregate`. ``None``
        summarizes the test cost over test episodes and everything else over
        seeds.
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
        metric: aggregate(kept, column_by, metric, pool=pool) for metric in metrics
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

    pooled_metrics = []

    for metric in metrics:
        row_label, fmt = METRIC_ROWS.get(metric, (_escape(metric), "{:.4g}"))

        if any(cell["pooled"] for cell in stats[metric].values()):
            pooled_metrics.append(metric)
            row_label += POOLED_MARKER

        cells = [_cell(stats[metric].get(key), fmt, show_spread) for key in keys]
        lines.append(" & ".join([row_label] + cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    tabular = "\n".join(lines)

    if fit:
        tabular = _fit(tabular)

    if caption is None and label is None:
        return tabular

    # Escaped first, then the provenance is appended.
    full_caption = (
        _escape(caption)
        + _provenance(stats, pooled_metrics, show_spread)
        + ((CENSUS_NOTE if with_identification else CENSUS_NOTE_STABLE_ONLY)
           if show_census else "")
    )

    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        tabular,
        rf"\caption{{{full_caption}}}" if caption else "",
        rf"\label{{{label}}}" if label else "",
        r"\end{table}",
    ])


def write_table(text: str, name: str):
    """Write a table fragment to ``settings.TABLE_OUTPUT``.

    Parameters
    ----------
    text : str
        As returned by :func:`metric_table`.
    name : str
        File stem. ``\\input{generated/tables/<name>.tex}`` to use in latex.

    Returns
    -------
    Path
        The file written.
    """
    path = table_dir() / f"{name}.tex"
    path.write_text(text + "\n")
    return path
