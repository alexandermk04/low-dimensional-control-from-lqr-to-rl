"""
How we save the data of an experiment, and we read it back.

Layout, one subfolder per experiment::

    experiment_results/<experiment>/
        manifest.json         what was run, and under which settings
        index.jsonl           one line per run
        <run_id>/record.json
        <run_id>/arrays.npz
"""

import json
import time
from pathlib import Path

import numpy as np

import settings
from evaluation.records import RunRecord, append_index, load_records, save_record
from evaluation.serialization import to_jsonable

__all__ = [
    "experiment_root",
    "write_manifest",
    "read_manifest",
    "store_record",
    "load_experiment",
    "load_trained_gains",
    "MissingExperimentError",
]

ROOT = Path(settings.EXPERIMENT_ROOT)


class MissingExperimentError(RuntimeError):
    """An experiment this script depends on has not been run yet."""


def experiment_root(experiment: str) -> Path:
    """Directory holding one experiment's runs, created if missing."""
    directory = ROOT / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_manifest(experiment: str, used_settings: dict, grid: list | None = None) -> Path:
    """Record what this experiment ran and under which settings.

    Parameters
    ----------
    experiment : str
        Experiment name, and the subdirectory.
    used_settings : dict
        Every :mod:`settings` value the experiment read.
    grid : list or None, optional
        The conditions that will be run.

    Returns
    -------
    Path
        The manifest file.
    """
    path = experiment_root(experiment) / "manifest.json"

    payload = {
        "experiment": experiment,
        "created": time.strftime("%Y%m%dT%H%M%S"),
        "seeds": list(settings.SEEDS),
        "test_protocol": {
            "episodes": settings.TEST_EPISODES,
            "horizon": settings.TEST_HORIZON,
        },
        "settings": to_jsonable(used_settings),
        "grid": to_jsonable(grid) if grid is not None else None,
    }

    path.write_text(json.dumps(payload, indent=2))

    return path


def read_manifest(experiment: str) -> dict:
    """Load an experiment's manifest.

    Raises
    ------
    MissingExperimentError
        If the experiment path doesn't exist.
    """
    path = ROOT / experiment / "manifest.json"

    if not path.exists():
        raise MissingExperimentError(
            f"No manifest at {path}. Run `python -m experiments.{experiment}` first."
        )

    return json.loads(path.read_text())


def store_record(record: RunRecord) -> Path:
    """Save one run and append it to its experiment's index."""
    directory = save_record(record, ROOT)
    append_index(record, experiment_root(record.experiment))
    return directory


def load_experiment(experiment: str, solver: str | None = None) -> list[RunRecord]:
    """Load every run of one experiment.

    Parameters
    ----------
    experiment : str
        Experiment name.
    solver : str or None, optional
        Restrict to one solver.

    Returns
    -------
    list[RunRecord]
        Sorted oldest first.

    Raises
    ------
    MissingExperimentError
        If the experiment folder holds no runs.
    """
    directory = ROOT / experiment

    if not directory.exists():
        raise MissingExperimentError(
            f"No results at {directory}. "
            f"Run `python -m experiments.{experiment}` first."
        )

    records = load_records(ROOT, experiment=experiment, solver=solver)

    if not records:
        raise MissingExperimentError(
            f"{directory} exists but holds no runs. "
            f"Run `python -m experiments.{experiment}` first."
        )

    return records


def load_trained_gains(experiment: str) -> dict[tuple[str, int], np.ndarray]:
    """The final gain of every run of a training experiment.

    Parameters
    ----------
    experiment : str
        Name of the training experiment to read.

    Returns
    -------
    dict[tuple[str, int], np.ndarray]
        ``(solver, seed) -> K_final``, in the controller's own space.

    Raises
    ------
    MissingExperimentError
        If the experiment has not been run, or produced no usable gains.
    """
    gains = {}

    for record in load_experiment(experiment):
        K = record.arrays.get("K_final")
        if K is None or K.size == 0:
            continue
        gains[(record.solver, record.seed)] = K

    if not gains:
        raise MissingExperimentError(
            f"{experiment} holds no runs with a stored K_final. "
            "Non-linear policies cannot be reused this way."
        )

    return gains


def load_run_index(experiment: str) -> dict[tuple[str, int], RunRecord]:
    """``(solver, seed) -> the most recent record`` for that cell."""
    index = {}
    for record in load_experiment(experiment):
        index[(record.solver, record.seed)] = record
    return index
