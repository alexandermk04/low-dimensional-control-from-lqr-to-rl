"""Save experiment data locally.

The :class:`RunRecord` introduces:

- ``record.json`` - configuration, seed, timestamp and the scalar summary.
  Human-readable part.
- ``arrays.npz`` - the learning curve, the trajectories and the gains.

The ``summary`` keys are fixed across different runs so a
comparison table can easily be built.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.serialization import to_jsonable

__all__ = [
    "SCHEMA_VERSION",
    "SUMMARY_KEYS",
    "RunRecord",
    "condition_slug",
    "save_record",
    "load_record",
    "load_records",
    "append_index",
    "DEFAULT_ROOT",
]

SCHEMA_VERSION = "2"

# Where the records are saved to
DEFAULT_ROOT = Path("results")

# TODO: Check whether descriptions are accurate
# Every summary contains these keys, is ``None`` if it is not applicable
SUMMARY_KEYS = (
    "J_star", # optimal cost
    "J_final",
    "J_best", # closest one
    "subopt_final", # cost difference to optimal
    "subopt_best",
    "steps_to_thresh_1pct",
    "steps_to_thresh_10pct",
    "iters_to_thresh_1pct",
    "seconds_to_thresh_1pct",
    "total_env_steps", 
    "total_time", # overall time that has passed
    "converged", # boolean
    "spectral_radius_final",
    "rank_final", # rank of gain matrix
    "policy_complexity", # complexity in number of parameters
    "n_params_dense",
    # RL-only stats below - None otherwise
    "action_clips", # should be 0 unless constrained problem is solved
    "bias_ratio", # whether extracted gain describes policy accurately
    "mc_cost_mean",
    "mc_cost_stderr",
    "horizon",
    "n_mc_episodes",
    # Test evaluation metrics
    "test_cost_mean", # accumulated cost, averaged over test episodes
    "test_cost_std", # spread across test episodes
    "test_cost_stderr",
    "test_episodes",
    "test_horizon",
)


def condition_slug(condition: dict) -> str:
    """Turn a condition dict into a filesystem-safe, stable identifier.

    Parameters
    ----------
    condition : dict
        Flat mapping of cell name to value.

    Returns
    -------
    str
        e.g. ``"l500-m200"`` for ``{"l": 500, "m": 200}``. Empty for an empty
        condition.
    """
    def clean(value):
        if isinstance(value, (list, tuple)):
            return "x".join(str(v) for v in value)
        return str(value).replace(" ", "").replace("/", "-")

    return "-".join(f"{k}{clean(v)}" for k, v in sorted(condition.items()))


def _timestamp() -> str:
    """ISO-8601 basic timestamp.
    """
    return time.strftime("%Y%m%dT%H%M%S")


@dataclass
class RunRecord:
    """Everything one experiment produced.

    Attributes
    ----------
    experiment : str
        Which experiment this belongs to, e.g. ``"comparison"`` or
        ``"inflation"``. Becomes a directory.
    solver : str
        Which method produced it, e.g. ``"mf_npg"``, ``"spi_mf"``,
        ``"ppo_linear"``. Becomes the legend label.
    seed : int
        The global seed the run was produced under.
    condition : dict
        Specific condition settings of this run, e.g. ``{"l": 500, "m": 200}``.
        Used to name the run.
    config : dict
        Nested configuration, passed through
        :func:`~evaluation.serialization.to_jsonable`.
    summary : dict
        Scalar metrics. Missing entries are filled with ``None`` on
        construction, so every record carries every key in
        :data:`SUMMARY_KEYS`.
    arrays : dict[str, np.ndarray]
        Learning-curve columns, trajectories and gains.
    schema_version : str
        Version of this layout.
    timestamp : str
        When the run finished.
    run_id : str
        ``<experiment>__<solver>[__<condition slug>]__<timestamp>__seed<seed>``,
        and the directory name.
    """

    experiment: str
    solver: str
    seed: int
    condition: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=_timestamp)
    run_id: str = ""

    def __post_init__(self):
        if not self.run_id:
            parts = [self.experiment, self.solver]
            if self.condition:
                parts.append(condition_slug(self.condition))
            parts += [self.timestamp, f"seed{self.seed}"]
            self.run_id = "__".join(parts)
        for key in SUMMARY_KEYS:
            self.summary.setdefault(key, None)

    def label(self) -> str:
        """Human-readable name for a plot legend."""
        return self.solver.replace("_", " ")


def save_record(record: RunRecord, root: Path | str = DEFAULT_ROOT) -> Path:
    """Write a record to ``<root>/<experiment>/<run_id>/``.

    Parameters
    ----------
    record : RunRecord
        The record to write.
    root : Path or str, optional
        Root path for the results - created if missing.

    Returns
    -------
    Path
        The directory written to, containing ``record.json`` and ``arrays.npz``.
    """
    directory = Path(root) / record.experiment / record.run_id
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": record.schema_version,
        "run_id": record.run_id,
        "experiment": record.experiment,
        "solver": record.solver,
        "timestamp": record.timestamp,
        "seed": record.seed,
        "condition": to_jsonable(record.condition),
        "config": to_jsonable(record.config),
        "summary": to_jsonable(record.summary),
        "array_keys": sorted(record.arrays),
    }

    (directory / "record.json").write_text(json.dumps(payload, indent=2))

    if record.arrays:
        np.savez_compressed(directory / "arrays.npz", **record.arrays)

    return directory


def load_record(path: Path | str) -> RunRecord:
    """Load a record.

    Parameters
    ----------
    path : Path or str
        Either the run directory or the ``record.json`` inside it.

    Returns
    -------
    RunRecord
        With ``arrays`` populated from the NPZ file if it exists.

    Raises
    ------
    FileNotFoundError
        If no ``record.json`` is found.
    """
    path = Path(path)
    directory = path.parent if path.name == "record.json" else path
    json_path = directory / "record.json"

    if not json_path.exists():
        raise FileNotFoundError(f"No record.json in {directory}")

    payload = json.loads(json_path.read_text())

    arrays: dict[str, np.ndarray] = {}
    npz_path = directory / "arrays.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}

    return RunRecord(
        experiment=payload["experiment"],
        solver=payload.get("solver"),
        seed=payload["seed"],
        condition=payload.get("condition") or {},
        config=payload["config"],
        summary=payload["summary"],
        arrays=arrays,
        schema_version=payload["schema_version"],
        timestamp=payload["timestamp"],
        run_id=payload["run_id"],
    )


def load_records(
    root: Path | str = DEFAULT_ROOT,
    experiment: str | None = None,
    solver: str | None = None,
) -> list[RunRecord]:
    """Load every record under ``root``, optionally filtered.

    Parameters
    ----------
    root : Path or str, optional
        Root of the results tree.
    experiment : str or None, optional
        Filter for experiment type.
    solver : str or None, optional
        Filter for solver method.

    Returns
    -------
    list[RunRecord]
        Sorted by timestamp, oldest first. Empty if ``root`` does not exist.
    """
    root = Path(root)
    if not root.exists():
        return []

    records = [load_record(p.parent) for p in sorted(root.glob("*/*/record.json"))]

    if experiment is not None:
        records = [r for r in records if r.experiment == experiment]
    if solver is not None:
        records = [r for r in records if r.solver == solver]

    return sorted(records, key=lambda r: r.timestamp)


def append_index(record: RunRecord, root: Path | str = DEFAULT_ROOT) -> Path:
    """Append a one-line summary of a run to ``<root>/index.jsonl``.

    A simple log of every run ever made, so we can easily keep track
    of available experiments, without loading the records first.

    Parameters
    ----------
    record : RunRecord
        The record to log.
    root : Path or str, optional
        Root of the results tree.

    Returns
    -------
    Path
        The index file.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = root / "index.jsonl"

    line = {
        "run_id": record.run_id,
        "experiment": record.experiment,
        "solver": record.solver,
        "timestamp": record.timestamp,
        "seed": record.seed,
        "condition": to_jsonable(record.condition),
        "J_final": to_jsonable(record.summary.get("J_final")),
        "subopt_final": to_jsonable(record.summary.get("subopt_final")),
        "total_env_steps": to_jsonable(record.summary.get("total_env_steps")),
        "total_time": to_jsonable(record.summary.get("total_time")),
        "test_cost_mean": to_jsonable(record.summary.get("test_cost_mean")),
    }

    with index.open("a") as handle:
        handle.write(json.dumps(line) + "\n")

    return index
