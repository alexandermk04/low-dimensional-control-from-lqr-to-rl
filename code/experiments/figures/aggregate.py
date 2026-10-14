"""
Functions for aggregating saved runs into usable experiment statistics.
"""

from collections import defaultdict

import numpy as np

from evaluation.records import RunRecord

__all__ = [
    "tidy",
    "with_suboptimality",
    "group",
    "aggregate",
    "curve_band",
    "latest_per_experiment",
    "label_for",
    "clip_roundoff",
]

# How a solver is named in a legend or a table header.
SOLVER_LABELS = {
    "baseline": "Baseline",
    "mb_pg": "Model-based PG",
    "mb_est": "N4SID + Model-based PG",
    "mf_npg": "Model-free NPG",
    "spi": "S-PI",
    "spi_mf": "S-PI (model-free)",
    "ce_dare": "N4SID + DARE",
    "spi_est": "N4SID + S-PI",
    "ppo_linear": "Linear PPO",
    "ppo_bottleneck": "Bottleneck PPO",
    "ppo_mlp": "Fully-connected PPO",
    "ppo_encoder": "State encoder + PPO",
    "ppo_encoder_scalable": "Scalable encoder + PPO",
    "ppo_fc_extractor": "FC-Extractor + PPO",
    # Chapter 6
    "oracle_coarse": "Oracle (true state)",
    "ppo_pixel_fc": "Pixels + PPO",
    "ppo_pixel_posvel": "Position-Velocity Encoder + PPO",
    "ppo_pixel_ortho": "Bottleneck PPO",
    "ppo_pixel_estimator": "Scalable encoder + PPO",
}

# Summary metrics whose per-episode array is stored with the run
EPISODE_ARRAYS = {"test_cost_mean": "test_costs"}

# Metrics whose per-episode array is computed from other rows
DERIVED_ARRAYS = {"subopt": "subopt_episodes"}

# Every metric that is summarized over pooled episodes
CELL_ARRAYS = {**EPISODE_ARRAYS, **DERIVED_ARRAYS}

# Round below this at suboptimlaity
ROUNDOFF_TOL = 1e-12


def clip_roundoff(values):
    """Set relative cost differences within round-off to exactly zero."""
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < ROUNDOFF_TOL, 0.0, values)


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
        If a condition key collides with a summary key of a different value.
    """
    rows = []

    for record in records:
        clash = {
            key for key in set(record.condition) & set(record.summary)
            if record.condition[key] != record.summary[key]
        }
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

        if row.get("training_subopt_final") is not None:
            row["training_subopt_final"] = float(
                clip_roundoff(row["training_subopt_final"])
            )

        # Additional episode arrays
        for column in EPISODE_ARRAYS.values():
            array = record.arrays.get(column)
            if array is not None and array.size:
                row[column] = array
        rows.append(row)

    return rows


def with_suboptimality(
    rows: list[dict],
    reference: str = "baseline",
    by=("seed", "size", "channel", "level"),
    metric: str = "test_cost_mean",
) -> list[dict]:
    """Score every row against the reference solver, episode by episode.

    Adds the suboptimality :math:`S` of the thesis, the relative excess cost
    over the reference solver under the *same* condition. Because the test
    protocol draws the same initial states and the same noise realizations for
    every solver of a cell, episode :math:`i` of one solver and episode
    :math:`i` of the reference differ only in the controller. They are therefore
    paired, and

    .. math:: S_i = (J_i - J_i^{\\mathrm{base}}) / J_i^{\\mathrm{base}}

    isolates what the choice of controller contributed, with the initial state
    cancelling instead of inflating the spread. It may legitimately be negative,
    since the nominal LQR gain is not optimal once the loop is noisy.

    Two columns are written: ``subopt_episodes``, the paired values, which
    :func:`aggregate` pools across the seeds of a cell, and ``subopt``, their
    mean, for the rows that carry no episodes.

    Episodes are censored in two distinct ways. Where the solver's own episode
    diverged there is no cost to compare, and the pair is dropped and counted by
    :func:`aggregate` as diverged. Where the *reference* episode diverged the
    ratio has no meaning at all, so the pair is dropped and counted separately -
    the reference is one fixed gain under one condition, so that censoring
    applies to every solver of the cell alike and must not be read as a failure
    of the solver being scored.

    Parameters
    ----------
    rows : list[dict]
        From :func:`tidy`.
    reference : str, optional
        Solver key of the reference. Defaults to ``"baseline"``.
    by : tuple of str, optional
        What identifies a cell, and hence what has to agree between a row and
        its reference. Must not contain ``"solver"``.
    metric : str, optional
        Cost column to compare. Its per-episode array is what gets paired.

    Returns
    -------
    list[dict]
        The same rows. Where no reference was found under the same condition,
        ``subopt`` is ``None`` and ``subopt_episodes`` is absent.
    """
    column = EPISODE_ARRAYS.get(metric)

    # The reference episodes of each cell, kept unfiltered: which episode
    # diverged has to stay visible, so the pairing below can tell a reference
    # failure apart from a solver failure. Position within the array is the
    # pairing key, so nothing may be dropped here.
    references = {}
    for row in rows:
        if row.get("solver") != reference:
            continue
        episodes = None if column is None else row.get(column)
        if episodes is not None:
            references[group_key(row, by)] = np.asarray(episodes, dtype=float)

    # Falls back to the scalar cost where a run carries no episodes at all.
    scalar_references = {}
    for row in rows:
        if row.get("solver") != reference:
            continue
        key = group_key(row, by)
        if key in references:
            continue
        cost = row.get(metric)
        if cost is not None and np.isfinite(float(cost)):
            scalar_references[key] = float(cost)

    for row in rows:
        key = group_key(row, by)
        episodes = None if column is None else row.get(column)
        base = references.get(key)

        if episodes is not None and base is not None:
            episodes = np.asarray(episodes, dtype=float)

            if episodes.shape != base.shape:
                raise ValueError(
                    f"Run {row.get('run_id')!r} has {episodes.size} test "
                    f"episodes but its {reference!r} reference has {base.size}. "
                    f"Paired suboptimality needs the protocol to have been run "
                    f"identically; re-run the cell rather than comparing these."
                )

            paired = np.full(episodes.shape, np.nan)
            # Where the reference diverged or is zero the ratio is undefined,
            # and nan keeps it out of every statistic downstream.
            usable = np.isfinite(base) & (base != 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                paired[usable] = (
                    episodes[usable] - base[usable]
                ) / base[usable]

            paired = clip_roundoff(paired)
            row["subopt_episodes"] = paired
            finite = paired[np.isfinite(paired)]
            row["subopt"] = float(np.mean(finite)) if finite.size else None
            continue

        # No episodes on either side: fall back to the scalar costs.
        cost = row.get(metric)
        j_ref = scalar_references.get(key)
        finite = cost is not None and np.isfinite(float(cost))
        row["subopt"] = (
            float(clip_roundoff((float(cost) - j_ref) / j_ref))
            if finite and j_ref else None
        )
        row.pop("subopt_episodes", None)

    return rows


def with_clean_ratio(
    rows: list[dict],
    by=("solver", "seed"),
    metric: str = "test_cost_mean",
) -> list[dict]:
    """Score every controller against themselves without noise.

    Parameters
    ----------
    rows : list[dict]
        From :func:`tidy`.
    by : tuple of str, optional
        What has to agree between a row and its clean counterpart.
    metric : str, optional
        Cost column to compare.

    Returns
    -------
    list[dict]
        The same rows. Where no clean cost was found, ``clean_ratio`` is ``None``.
    """
    clean = {
        group_key(row, by): float(row[metric])
        for row in rows
        if row.get("channel") == "clean" and row.get(metric) is not None
    }

    for row in rows:
        cost = row.get(metric)
        j_clean = clean.get(group_key(row, by))
        usable = (
            cost is not None and np.isfinite(float(cost))
            and j_clean is not None and np.isfinite(j_clean) and j_clean != 0.0
        )
        row["clean_ratio"] = float(cost) / j_clean if usable else None

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


def _pooled_episodes(rows: list[dict], by, column: str) -> dict[tuple, dict]:
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
    dict[tuple, dict]
        Per cell: ``values`` (the finite episodes), ``n_episodes``,
        ``n_diverged`` (non-finite values) and ``n_reference_diverged``.
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
        finite = np.isfinite(stacked)

        # Count how many times the baseline solution diverged
        reference_diverged = int(np.count_nonzero(np.isnan(stacked)))

        out[key] = {
            "values": stacked[finite],
            "n_episodes": int(stacked.size),
            "n_diverged": int(stacked.size - np.count_nonzero(finite))
            - reference_diverged,
            "n_reference_diverged": reference_diverged,
        }

    return out


def aggregate(rows: list[dict], by, metric: str) -> dict[tuple, dict]:
    """Aggregate one metric over all the runs of each experiment setting.

    Parameters
    ----------
    rows : list[dict]
        From :func:`tidy`.
    by : str or tuple of str
        What defines an experiment setting, e.g. ``"solver"`` or
        ``("solver", "level")``.
    metric : str
        Column to summarize, e.g. ``"subopt"``.

    Returns
    -------
    dict[tuple, dict]
        Per setting: ``n`` (values that went in), ``n_total`` (runs in this
        setting), ``episodes`` (whether ``n`` counts episodes rather than runs),
        ``mean``, ``std``, ``median``, ``min``, ``max``, ``values``, ``n_episodes``,
        ``n_diverged``,``n_reference_diverged`` and ``divergence_rate``.
    """
    column = CELL_ARRAYS.get(metric)
    pooled = _pooled_episodes(rows, by, column) if column is not None else {}

    out = {}

    for key, bucket in group(rows, by).items():
        cell = pooled.get(key)

        if cell is None:
            values = _finite([row.get(metric) for row in bucket])
            counts = {
                "n_episodes": 0,
                "n_diverged": 0,
                "n_reference_diverged": 0,
            }
        else:
            values = cell["values"]
            counts = {k: v for k, v in cell.items() if k != "values"}

        seen = counts["n_episodes"]

        out[key] = {
            "n": int(values.size),
            "n_total": len(bucket),
            "episodes": cell is not None,
            "values": values,
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)) if values.size else float("nan"),
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
            **counts,
            "divergence_rate": (counts["n_diverged"] / seen) if seen else 0.0,
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

            if np.unique(axis[finite]).size < 2:
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
