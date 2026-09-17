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
import os
import time
from pathlib import Path

import settings
from evaluation.records import (
    RunRecord,
    append_index,
    condition_slug,
    load_records,
    record_dir,
    save_record,
)
from evaluation.serialization import to_jsonable

__all__ = [
    "experiment_root",
    "write_manifest",
    "read_manifest",
    "store_record",
    "load_experiment",
    "load_run_index",
    "run_key",
    "completed_runs",
    "in_shard",
    "MissingExperimentError",
    "SettingsChangedError",
]

ROOT = Path(settings.EXPERIMENT_ROOT)


class MissingExperimentError(RuntimeError):
    """An experiment this script depends on has not been run yet."""


class SettingsChangedError(RuntimeError):
    """Finished runs exist, but were produced under different settings."""


# Manifest entries used to identify a started run
_RESUME_KEYS = ("seeds", "test_protocol", "settings")

# Version to not mix breaking changes in experiments
PROTOCOL_VERSION = 3


def experiment_root(experiment: str) -> Path:
    """Directory holding one experiment's runs, created if missing."""
    directory = ROOT / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _changed_settings(previous: dict, current: dict) -> list[str]:
    """Names of the manifest entries that differ between two manifests."""
    changed = [
        key for key in _RESUME_KEYS
        if key != "settings" and previous.get(key) != current.get(key)
    ]

    old, new = previous.get("settings") or {}, current.get("settings") or {}
    changed += [
        f"settings.{key}" for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    ]

    return changed


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

    Raises
    ------
    SettingsChangedError
        If finished runs exist and were produced under different settings.
    """
    directory = experiment_root(experiment)
    path = directory / "manifest.json"
    now = time.strftime("%Y%m%dT%H%M%S")

    payload = {
        "experiment": experiment,
        "created": now,
        "seeds": list(settings.SEEDS),
        "test_protocol": {
            "episodes": settings.TEST_EPISODES,
            "horizon": settings.TEST_HORIZON,
            "version": PROTOCOL_VERSION,
            "x0_stream": "sub_seed(seed, 'plant', 'test')",
        },
        "settings": to_jsonable(used_settings),
        "grid": to_jsonable(grid) if grid is not None else None,
    }

    payload = json.loads(json.dumps(payload))

    has_runs = any(directory.glob("*/record.json"))

    if path.exists() and has_runs:
        previous = json.loads(path.read_text())
        changed = _changed_settings(previous, payload)

        if changed:
            raise SettingsChangedError(
                f"{directory} holds finished runs made under different settings "
                f"({', '.join(changed)}). Revert settings.py to resume, or delete contents of "
                f"{directory} to restart {experiment!r}."
            )

        payload["created"] = previous.get("created", now)
        payload["resumed"] = previous.get("resumed", []) + [now]

    # Atomic writing to allow sharding
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)

    return path


def run_key(solver: str, seed: int, condition: dict) -> tuple[str, int, str]:
    """What identifies one run within an experiment."""
    return solver, int(seed), condition_slug(condition)


def in_shard(position: int) -> bool:
    """Whether the grid entry at ``position`` belongs to this SLURM array task.

    Always ``True`` outside of a job array.
    """
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    return position % count == task


def completed_runs(experiment: str) -> dict[tuple[str, int, str], dict]:
    """``run_key -> record.json payload`` of every finished run of an experiment.

    Returns
    -------
    dict[tuple[str, int, str], dict]
        Empty if the experiment has not been run yet. If a run exists twice,
        the most recent one is used.
    """
    done = {}

    # Directory names end in <timestamp>__seed<seed>
    for path in sorted((ROOT / experiment).glob("*/record.json")):
        payload = json.loads(path.read_text())
        key = run_key(payload["solver"], payload["seed"], payload.get("condition") or {})
        done[key] = payload

    return done


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
    """Save one run, its policy and its index entry.
    """
    directory = record_dir(record, ROOT)
    directory.mkdir(parents=True, exist_ok=True)

    saver = record.save_policy
    if saver is not None:
        record.policy = saver(directory)
        record.save_policy = None

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


def load_run_index(experiment: str) -> dict[tuple[str, int, str], RunRecord]:
    """``(solver, seed, condition) -> the most recent record`` for the combination
    of this specific experiment settings.

    Returns
    -------
    dict[tuple[str, int, str], RunRecord]
        Keyed by ``(solver, seed, condition_slug(condition))``. The slug is the
        same one :func:`evaluation.records.condition_slug` builds run directory
        names from.
    """
    index = {}
    for record in load_experiment(experiment):
        index[(record.solver, record.seed, condition_slug(record.condition))] = record
    return index
