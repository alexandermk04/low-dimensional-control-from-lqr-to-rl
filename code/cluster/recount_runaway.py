"""Count the runaway test episodes of stored runs, by replaying their clean test protocol.

Run from ``code/``::

    python cluster/recount_runaway.py <experiment> [<experiment> ...] [--write] [--workers N]
    python cluster/recount_runaway.py ch5_inflation_sizes ch5_noise --write

"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings

ROOT = Path(settings.EXPERIMENT_ROOT)

KEY = "runaway_episodes"


def replay(path: Path) -> int | None:
    """Runaway episodes of one run holding a policy.

    ``None`` if it holds none, or a linear gain.
    """
    import torch

    from evaluation.records import load_record
    from experiments.evaluate import evaluate_policy
    from experiments.policies import MissingPolicyError, load_policy
    from experiments.problem import make_problem

    torch.set_num_threads(1)

    record = load_record(path)
    try:
        policy = load_policy(record)
    except MissingPolicyError:
        return None

    size = record.condition.get("size")
    inflation = (
        settings.INFLATION_SIZES[size] if size in settings.INFLATION_SIZES
        else tuple(record.config["inflation"])
    )
    test = evaluate_policy(make_problem(record.seed, inflation), policy)

    stored = record.arrays.get("test_costs")
    if stored is not None and not np.allclose(test.returns, stored, rtol=1e-3, equal_nan=True):
        raise RuntimeError(f"{record.run_id}: replayed test costs differ from the stored ones")

    return test.summary[KEY]


def recount(path: Path, payload: dict) -> int | None:
    """The new count of one run."""
    source = (payload.get("config") or {}).get("source_run")
    if source:
        source_path = ROOT / source.split("__")[0] / source / "record.json"
        return json.loads(source_path.read_text())["summary"].get(KEY)
    return replay(path)


def _recount(path: Path):
    return path, recount(path, json.loads(path.read_text()))


def main(experiment: str, write: bool, workers: int) -> None:
    paths = sorted((ROOT / experiment).glob("*/record.json"))

    changed = runaway = 0
    with ProcessPoolExecutor(workers) as pool:
        for path, new in pool.map(_recount, paths, chunksize=4):
            payload = json.loads(path.read_text())
            runaway += bool(new)
            if payload["summary"].get(KEY) == new:
                continue

            changed += 1
            if new:
                print(f"{payload['run_id']}: {new} runaway episodes")

            if write:
                payload["summary"][KEY] = new
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2))
                os.replace(tmp, path)

    print(
        f"{experiment}: {runaway} runs with runaway episodes, {changed} runs "
        f"{'rewritten' if write else 'to rewrite (dry run)'}"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    workers = os.cpu_count() or 1
    if "--workers" in args:
        at = args.index("--workers")
        workers = int(args[at + 1])
        del args[at:at + 2]
    write = "--write" in args
    for name in (a for a in args if a != "--write"):
        main(name, write, workers)
