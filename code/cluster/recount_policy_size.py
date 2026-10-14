"""Recount the policy size of stored PPO runs, with the feature extractor included.

Run from ``code/``::

    python cluster/recount_policy_size.py <experiment> [<experiment> ...] [--write]
    python cluster/recount_policy_size.py ch5_inflation_sizes ch5_noise --write

"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings
from evaluation.analysis import policy_complexity_mlp

ROOT = Path(settings.EXPERIMENT_ROOT)

KEYS = ("policy_complexity", "policy_complexity_frozen")


def recount(directory: Path, payload: dict, presets: dict) -> dict | None:
    """The new counts of one run, or ``None`` if it holds no PPO policy."""
    source = (payload.get("config") or {}).get("source_run")
    if source:
        path = ROOT / source.split("__")[0] / source / "record.json"
        summary = json.loads(path.read_text())["summary"]
        return {key: summary.get(key) for key in KEYS}

    policy = payload.get("policy") or {}
    if policy.get("kind") != "sb3":
        return None

    from stable_baselines3 import PPO

    model = PPO.load(directory / policy.get("file", "policy.zip"), device="cpu")
    encoder = policy.get("encoder")
    complexity = policy_complexity_mlp(
        model, presets[payload["solver"]].freeze_extractor,
        n_frozen_extra=int(np.load(directory / encoder).size) if encoder else 0,
    )
    return {"policy_complexity": complexity["n_params"],
            "policy_complexity_frozen": complexity["n_params_frozen"]}


def main(experiment: str, write: bool) -> None:
    from reinforcement_learning.solvers import pixel_presets, vector_presets

    presets = {**vector_presets(), **pixel_presets()}

    changed = 0
    for path in sorted((ROOT / experiment).glob("*/record.json")):
        payload = json.loads(path.read_text())
        new = recount(path.parent, payload, presets)
        if new is None:
            continue

        old = {key: payload["summary"].get(key) for key in KEYS}
        if old == new:
            continue

        changed += 1
        print(f"{payload['run_id']}\n    {old}\n -> {new}")

        if write:
            payload["summary"].update(new)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, path)

    print(f"{experiment}: {changed} runs {'rewritten' if write else 'to rewrite (dry run)'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--write"]
    for name in args:
        main(name, write="--write" in sys.argv)
