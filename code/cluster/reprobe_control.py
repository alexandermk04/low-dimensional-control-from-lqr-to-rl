"""Add ``r2_control`` to stored chapter 6 runs.

Run from ``code/``::

    python cluster/reprobe_control.py [--write] [--workers N]

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

EXPERIMENT = "ch6_perception"

KEY = "r2_control"


def reprobe(path: Path) -> float | None:
    """Probe score of the oracle's control for one run.

    ``None`` if the run holds no learned policy.
    """
    import torch
    from stable_baselines3 import PPO

    from evaluation.latent_probe import latent_extra
    from evaluation.pixel_metrics import SB3EnvPolicy
    from evaluation.records import load_record, record_dir
    from experiments.registry import _pixel_env, _pixel_setup, oracle_force

    torch.set_num_threads(1)

    record = load_record(path)
    spec = record.policy or {}
    policy_path = record_dir(record, ROOT) / spec.get("file", "policy.zip")
    if spec.get("kind") != "sb3" or not policy_path.exists():
        return None

    K, _, horizon, _ = _pixel_setup(None)
    model = PPO.load(policy_path, device="cpu")
    env = _pixel_env(record.seed, f"probe:{record.solver}", horizon)
    scores = latent_extra(model, env, SB3EnvPolicy(model), control=oracle_force(K, env.c))

    for name in ("r2_position", "r2_velocity"):
        stored = record.summary.get(name)
        if stored is not None and not np.isclose(scores[name], stored, atol=1e-3):
            raise RuntimeError(
                f"{record.run_id}: replayed {name} {scores[name]:.4f} differs from the stored {stored:.4f}"
            )

    return scores[KEY]


def _reprobe(path: Path):
    return path, reprobe(path)


def main(write: bool, workers: int) -> None:
    paths = sorted((ROOT / EXPERIMENT).glob("*/record.json"))

    changed = 0
    with ProcessPoolExecutor(workers) as pool:
        for path, new in pool.map(_reprobe, paths):
            if new is None:
                continue

            payload = json.loads(path.read_text())
            print(f"{payload['solver']:22s} seed {payload['seed']:6d}: {KEY} {new:.3f}", flush=True)
            if payload["summary"].get(KEY) == new:
                continue

            changed += 1
            if write:
                payload["summary"][KEY] = new
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2))
                os.replace(tmp, path)

    print(f"{EXPERIMENT}: {changed} runs {'rewritten' if write else 'to rewrite (dry run)'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    workers = os.cpu_count() or 1
    if "--workers" in args:
        at = args.index("--workers")
        workers = int(args[at + 1])
        del args[at:at + 2]
    main("--write" in args, workers)
