"""Chapter 4: comparison of robustness of the trained controllers.

Run with::

    python -m experiments.ch4_noise
"""

import time

import numpy as np

import settings
from experiments.policies import MissingPolicyError, load_policy
from evaluation.records import RunRecord
from evaluation.serialization import to_jsonable
from experiments.keep_running import keep_running
from experiments.evaluate import evaluate_policy, figure_trajectory
from experiments.noise_builder import build_noise, noise_conditions
from experiments.problem import make_problem
from experiments.settings_guard import require_noise
from experiments.store import (
    MissingExperimentError,
    completed_runs,
    in_shard,
    load_run_index,
    run_key,
    store_record,
    write_manifest,
)

EXPERIMENT = "ch4_noise"

SOURCE = "ch4_identified_model"

SIZES = ("base", settings.CH4_REFERENCE_INFLATION)


@keep_running()
def evaluate_grid(source: str, experiment: str, sizes=None, verbose: bool = True):
    """Score every trained controller of ``source`` on every noise cell.

    Parameters
    ----------
    source : str
        Experiment whose policies are reused.
    experiment : str
        Where the records are written to.
    sizes : iterable or None, optional
        Which inflation sizes to include, as names of
        ``settings.INFLATION_SIZES``.
    verbose : bool, optional
        Print a line per finished cell.

    Returns
    -------
    list[RunRecord]
        One record per ``(solver, seed, size, channel, level)``.
    """
    require_noise(settings.NOISE_CHANNELS)

    wanted = None if sizes is None else set(sizes)

    source_runs = {
        key: record
        for key, record in load_run_index(source).items()
        if wanted is None or record.condition.get("size") in wanted
    }

    if not source_runs:
        raise MissingExperimentError(
            f"No runs of {source!r} match sizes {sorted(map(str, wanted))}. "
            f"Run `python -m experiments.{source}` first."
        )

    conditions = list(noise_conditions())

    write_manifest(
        experiment,
        used_settings={
            "source_experiment": source,
            "sizes": sorted(str(size) for size in wanted) if wanted else "all",
            "noise_levels": settings.NOISE_LEVELS,
            "noise_channels": settings.NOISE_CHANNELS,
            "noise_std": settings.NOISE_STD,
            "parametric_time_invariant": settings.PARAMETRIC_TIME_INVARIANT,
            "disturbance_type": settings.DISTURBANCE_TYPE,
        },
        grid=[
            {
                "solver": record.solver,
                "seed": record.seed,
                "size": record.condition.get("size"),
                "channel": name,
                "level": level,
            }
            for record in source_runs.values()
            for name, _, level in conditions
        ],
    )

    done = completed_runs(experiment)

    def already_scored(source_record, condition: dict) -> bool:
        """Whether this cell has already been scored and stored.
        """
        previous = done.get(
            run_key(source_record.solver, source_record.seed, condition)
        )
        return (
            previous is not None
            and previous["config"].get("source_run") == source_record.run_id
        )

    total = len(source_runs) * len(conditions)
    n_done = sum(
        already_scored(
            record,
            {"size": record.condition.get("size"), "channel": name, "level": level},
        )
        for record in source_runs.values()
        for name, _, level in conditions
    )
    if verbose:
        print(
            f"[{experiment}] {total} evaluations, no training, {n_done} already "
            f"done, {total - n_done} to go"
        )

    records = []
    index = 0

    for position, key in enumerate(sorted(source_runs)):
        if not in_shard(position):
            index += len(conditions)
            continue

        source_record = source_runs[key]
        solver, seed = source_record.solver, source_record.seed

        try:
            policy = load_policy(source_record)
        except MissingPolicyError as error:
            if verbose:
                print(f"  skipping {solver} seed={seed}: {error}")
            index += len(conditions)
            continue

        K = source_record.arrays.get("K_final")

        size = source_record.condition.get("size")
        inflation = settings.INFLATION_SIZES[size]

        problem = make_problem(seed, inflation)

        for name, channels, level in conditions:
            index += 1

            if already_scored(
                source_record, {"size": size, "channel": name, "level": level}
            ):
                continue

            started = time.perf_counter()

            noise = build_noise(channels, level)

            test = evaluate_policy(problem, policy, noise_settings=noise)
            summary = test.summary
            summary["J_star"] = float(problem.j_star)
            summary["rank_final"] = (
                int(np.linalg.matrix_rank(K)) if K is not None and K.size
                else source_record.summary.get("rank_final")
            )
            summary["policy_complexity"] = source_record.summary.get(
                "policy_complexity"
            )

            trajectory = figure_trajectory(
                problem, policy, noise_settings=noise,
            )

            record = RunRecord(
                experiment=experiment,
                solver=solver,
                seed=seed,
                condition={"size": size, "channel": name, "level": level},
                config={
                    "source_run": source_record.run_id,
                    "source_policy": source_record.policy,
                    "size": size,
                    "inflation": list(inflation) if inflation else None,
                    "noise": to_jsonable(noise),
                    "channels": list(channels),
                    "level": level,
                },
                summary=summary,
                arrays={
                    "test_costs": test.returns,
                    "x_traj": trajectory.x_traj,
                    "u_traj": trajectory.u_traj,
                    **({"K_final": K} if K is not None and K.size else {}),
                },
            )
            store_record(record)
            records.append(record)

            if verbose:
                print(
                    f"  [{index:>4d}/{total}] {solver:12s} seed={seed:<6d}"
                    f" {str(size):8s} {name:12s} {level:10s}"
                    f" test={summary['test_cost_mean']:.4e}"
                    f" ({time.perf_counter() - started:.0f}s)"
                )

    return records


def main():
    evaluate_grid(SOURCE, EXPERIMENT, SIZES)


if __name__ == "__main__":
    main()
