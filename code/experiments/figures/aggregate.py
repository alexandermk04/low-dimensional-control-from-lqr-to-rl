"""
Functions for aggregating saved runs into usable experiment statistics.
"""

from collections import defaultdict

import numpy as np

from evaluation.records import RunRecord

__all__ = [
    "tidy",
    "pooled_values",
    "EPISODE_ARRAYS",
    "group",
    "group_key",
    "aggregate",
    "curve_band",
    "latest_per_experiment",
    "SOLVER_LABELS",
    "label_for",
]

# How a solver is named in a legend or a table header.
SOLVER_LABELS = {
    "baseline": "Baseline",
    "mb_pg": "Model-based PG",
    "mb_est": "Model-based PG + N4SID",
    "mf_npg": "Model-free NPG",
    "spi": "S-PI",
    "spi_mf": "S-PI (model-free)",
    "ppo_linear": "Linear PPO",
    "ppo_mlp": "Fully-connected PPO",
    "ppo_encoder": "State encoder + PPO",
    "pve": "Position-Velocity Encoder",
}

# Summary metrics that also have a per-episode array behind them
EPISODE_ARRAYS = {"test_cost_mean": "test_costs"}

# How a condition value is named, mapped to a more readable format.
VALUE_LABELS = {
    "clean": "none",
    "all": "all channels",
}


def label_for(value) -> str:
    """A presentable name for a solver key or a condition value."""
    text = SOLVER_LABELS.get(value, None)

    if text is None:
        text = VALUE_LABELS.get(value, str(value))
    return text


def tidy(records: list[RunRecord]) -> list[dict]:
    """Flatten records into one row each.

    Parameters
    ----------
    records : list[RunRecord]

    Returns
    -------
    list[dict]
        ``solver``, ``seed``, ``run_id``, every condition key, and every summary
        key.

    Raises
    ------
    ValueError
        If a condition key collides with a summary key.
    """
    rows = []

    for record in records:
        clash = set(record.condition) & set(record.summary)
        if clash:
            raise ValueError(
                f"Run {record.run_id!r} has condition keys that collide with the "
                f"summary schema: {sorted(clash)}. "
                f"Rename the condition key in the experiment that emits it."
            )

        row = {
            "solver": record.solver,
            "seed": record.seed,
            "run_id": record.run_id,
            **record.condition,
            **record.summary,
        }

        # Additional episode arrays
        for column in EPISODE_ARRAYS.values():
            array = record.arrays.get(column)
            if array is not None and array.size:
                row[column] = array
        rows.append(row)

    return rows


def latest_per_experiment(records: list[RunRecord], keys=("solver", "seed")) -> list[RunRecord]:
    """Keep only the newest run of each experiment setting.

    Parameters
    ----------
    records : list[RunRecord]
        As loaded, i.e. sorted oldest first.
    keys : tuple of str, optional
        What identifies an experiment, beyond its condition.

    Returns
    -------
    list[RunRecord]
    """
    newest = {}
    for record in records:
        identity = tuple(getattr(record, k) for k in keys) + tuple(
            sorted(record.condition.items())
        )
        newest[identity] = record
    return list(newest.values())


def group_key(row: dict, by) -> tuple:
    """The value of ``by`` in one row, as a tuple."""
    if isinstance(by, str):
        by = (by,)
    return tuple(row.get(k) for k in by)


def group(rows: list[dict], by) -> dict[tuple, list[dict]]:
    """Bucket rows by one or more columns."""
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(group_key(row, by), []).append(row)
    return buckets


def _finite(values) -> np.ndarray:
    """The finite entries of a metric column, as floats."""
    array = np.array(
        [np.nan if v is None else float(v) for v in values], dtype=float
    )
    return array[np.isfinite(array)]


def pooled_values(rows: list[dict], by, column: str) -> dict[tuple, np.ndarray]:
    """Concatenate a per-episode array across the seeds of each cell.

    Parameters
    ----------
    rows : list[dict]
        From :func:`tidy`.
    by : str or tuple of str
        What defines a cell.
    column : str
        Which array column to pool, e.g. ``"test_costs"``.

    Returns
    -------
    dict[tuple, np.ndarray]
        Every episode of every seed of the cell, finite entries only.
    """
    out = {}

    for key, bucket in group(rows, by).items():
        arrays = [
            np.asarray(row[column], dtype=float)
            for row in bucket
            if row.get(column) is not None
        ]
        if not arrays:
            continue

        stacked = np.concatenate(arrays)
        out[key] = stacked[np.isfinite(stacked)]

    return out


def aggregate(rows: list[dict], by, metric: str, pool: bool | None = None) -> dict[tuple, dict]:
    """Aggregate metrics over all the seeds for one experiment setting.

    Parameters
    ----------
    rows : list[dict]
        From :func:`tidy`.
    by : str or tuple of str
        What defines an experiment setting, e.g. ``"solver"`` or ``("solver", "level")``.
    metric : str
        Column to summarize, e.g. ``"subopt_final"``.
    pool : bool or None, optional
        Force the per-episode array on or off. ``None`` uses it whenever the
        metric has one and the rows contain it.

    Returns
    -------
    dict[tuple, dict]
        Per setting: ``n`` (values that went in), ``n_total`` (runs in this setting),
        ``pooled`` (whether ``n`` counts episodes rather than runs), ``mean``,
        ``std``, ``median``, ``min``, ``max`` and ``values``. A setting where every
        seed diverged has ``n = 0`` and ``nan`` statistics.
    """
    episodes = {}
    if pool is not False and metric in EPISODE_ARRAYS:
        episodes = pooled_values(rows, by, EPISODE_ARRAYS[metric])

    out = {}

    for key, bucket in group(rows, by).items():
        pooled = key in episodes
        values = episodes[key] if pooled else _finite(
            [row.get(metric) for row in bucket]
        )

        out[key] = {
            "n": int(values.size),
            "n_total": len(bucket),
            "pooled": pooled,
            "values": values,
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)) if values.size else float("nan"),
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
        }

    return out


def curve_band(
    records: list[RunRecord], by, x: str = "env_steps", y: str = "cost", points: int = 200
) -> dict[tuple, dict]:
    """Median learning curve and min-max band of every experiment setting, over seeds.

    Parameters
    ----------
    records : list[RunRecord]
        Must carry the ``x`` and ``y`` columns in ``arrays``.
    by : str or tuple of str
        What defines a setting. Read from ``record.solver`` and
        ``record.condition``.
    x : str, optional
        ``"env_steps"``, ``"iteration"`` or ``"total_time"``.
    y : str, optional
        Usually ``"cost"``.
    points : int, optional
        Resolution of the shared axis.

    Returns
    -------
    dict[tuple, dict]
        Per setting: ``x``, ``median``, ``low``, ``high``, ``n_seeds``, and
        ``flat`` - ``True`` when the solver reports no sample size (model-based).
    """
    if isinstance(by, str):
        by = (by,)

    buckets = defaultdict(list)
    for record in records:
        key = tuple(
            record.solver if k == "solver" else record.condition.get(k) for k in by
        )
        buckets[key].append(record)

    out = {}

    for key, bucket in buckets.items():
        curves = []
        finals = []

        for record in bucket:
            axis = record.arrays.get(x)
            values = record.arrays.get(y)
            if axis is None or values is None or values.size == 0:
                continue

            finite = np.isfinite(values) & np.isfinite(axis)
            if not finite.any():
                continue

            finals.append(values[finite][-1])

            if np.all(axis[finite] == 0):
                continue

            curves.append((axis[finite], values[finite]))

        if not curves:
            out[key] = {
                "flat": True,
                "x": None,
                "median": float(np.median(finals)) if finals else float("nan"),
                "low": float(np.min(finals)) if finals else float("nan"),
                "high": float(np.max(finals)) if finals else float("nan"),
                "n_seeds": len(finals),
            }
            continue

        lo = max(float(a[0]) for a, _ in curves)
        hi = min(float(a[-1]) for a, _ in curves)
        if not hi > lo:
            lo, hi = min(float(a[0]) for a, _ in curves), max(float(a[-1]) for a, _ in curves)

        shared = np.linspace(lo, hi, points)
        stacked = np.vstack([np.interp(shared, a, v) for a, v in curves])

        out[key] = {
            "flat": False,
            "x": shared,
            "median": np.median(stacked, axis=0),
            "low": np.min(stacked, axis=0),
            "high": np.max(stacked, axis=0),
            "n_seeds": stacked.shape[0],
        }

    return out
