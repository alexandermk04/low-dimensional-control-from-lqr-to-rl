"""Bring manifest entries up to date with settings.py

Run from ``code/``::

    python cluster/patch_manifest.py <experiment> <entry> [<entry> ...] [--write]
    python cluster/patch_manifest.py ch4_identified_model chosen_identification size_hyperparams

"""

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings
from evaluation.serialization import to_jsonable


def current_settings(experiment: str) -> dict:
    """The ``used_settings`` the experiment would pass to ``run_grid`` now, without running it."""
    module = importlib.import_module(f"experiments.{experiment}")
    captured = {}
    module.run_grid = lambda specs, name, used_settings, **kwargs: captured.update(used_settings)
    module.main()
    return json.loads(json.dumps(to_jsonable(captured)))


def main(experiment: str, entries: list[str], write: bool) -> None:
    path = Path(settings.EXPERIMENT_ROOT) / experiment / "manifest.json"
    manifest = json.loads(path.read_text())
    current = current_settings(experiment)

    for entry in entries:
        old, new = manifest["settings"].get(entry), current[entry]

        if isinstance(old, dict) and isinstance(new, dict):
            changed = [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
            for key in changed:
                print(f"{experiment}: {entry}[{key}]\n    {old.get(key)}\n -> {new.get(key)}")
        else:
            changed = [entry] if old != new else []
            if changed:
                print(f"{experiment}: {entry}\n    {old}\n -> {new}")

        if not changed:
            print(f"{experiment}: {entry} unchanged")

        manifest["settings"][entry] = new

    if write:
        path.write_text(json.dumps(manifest, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--write"]
    main(args[0], args[1:], write="--write" in sys.argv)
