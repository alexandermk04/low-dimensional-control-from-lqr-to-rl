"""Chapter 6: robustness of the perception-based controllers.

Run with::

    python -m experiments.ch6_noise
"""

import time

import settings
from evaluation.pixel_metrics import SB3EnvPolicy
from evaluation.records import RunRecord, record_dir
from evaluation.serialization import to_jsonable
from experiments.keep_running import keep_running
from experiments.noise_builder import build_noise, noise_conditions
from experiments.policies import MissingPolicyError
from experiments.registry import _pixel_protocol, _pixel_setup
from experiments.settings_guard import require_noise
from experiments.store import (
    ROOT,
    MissingExperimentError,
    completed_runs,
    in_shard,
    load_run_index,
    run_key,
    store_record,
    write_manifest,
)
from seeding import sub_seed

EXPERIMENT = "ch6_noise"

SOURCE = "ch6_perception"


def pixel_policy(record, env, K):
    """Rebuild a ch6 controller so that it acts on ``env``.
    """
    if record.solver == "oracle_coarse":
        from lqr_base.thruster_ring import ThrusterOraclePolicy

        oracle = ThrusterOraclePolicy(K, env.c)
        return lambda observation, info: oracle(env.lqr.observe())

    from stable_baselines3 import PPO

    spec = record.policy or {}
    path = record_dir(record, ROOT) / spec.get("file", "policy.zip")
    if spec.get("kind") != "sb3" or not path.exists():
        raise MissingPolicyError(f"{record.run_id} has no policy file at {path}.")

    return SB3EnvPolicy(PPO.load(path, device="cpu"))


@keep_running()
def evaluate_grid(verbose: bool = True):
    """Score every trained ch6 controller on every noise cell.

    Returns
    -------
    list[RunRecord]
        One record per ``(solver, seed, channel, level)``.
    """
    from reinforcement_learning.pixel_env import build_pixel_env

    require_noise(settings.NOISE_CHANNELS, table="CH6_NOISE_STD")

    source_runs = {
        key: record for key, record in load_run_index(SOURCE).items()
        if record.solver in settings.CH6_NOISE_SOLVERS
    }
    if not source_runs:
        raise MissingExperimentError(
            f"No runs of {SOURCE!r}. Run `python -m experiments.{SOURCE}` first."
        )

    conditions = list(noise_conditions())

    write_manifest(
        EXPERIMENT,
        used_settings={
            "source_experiment": SOURCE,
            "solvers": settings.CH6_NOISE_SOLVERS,
            "noise_levels": settings.NOISE_LEVELS,
            "noise_channels": settings.NOISE_CHANNELS,
            "noise_std": settings.CH6_NOISE_STD,
            "parametric_time_invariant": settings.PARAMETRIC_TIME_INVARIANT,
            "disturbance_type": settings.DISTURBANCE_TYPE,
        },
        grid=[
            {"solver": record.solver, "seed": record.seed,
             "channel": name, "level": level}
            for record in source_runs.values()
            for name, _, level in conditions
        ],
    )

    done = completed_runs(EXPERIMENT)

    def already_scored(source_record, condition: dict) -> bool:
        previous = done.get(
            run_key(source_record.solver, source_record.seed, condition)
        )
        return (
            previous is not None
            and previous["config"].get("source_run") == source_record.run_id
        )

    # The spec argument is unused, the setup only reads config
    K, j_star, horizon, _ = _pixel_setup(None)

    total = len(source_runs) * len(conditions)
    if verbose:
        print(f"[{EXPERIMENT}] {total} evaluations, no training")

    records = []

    for position, key in enumerate(sorted(source_runs)):
        if not in_shard(position):
            continue

        source_record = source_runs[key]
        solver, seed = source_record.solver, source_record.seed

        for name, channels, level in conditions:
            condition = {"channel": name, "level": level}
            if already_scored(source_record, condition):
                continue

            started = time.perf_counter()

            env = build_pixel_env(sub_seed(seed, "pixel-score", solver), horizon=horizon)
            noise = build_noise(
                channels, level,
                {channel: settings.CH6_NOISE_STD[channel][level] for channel in channels},
            )
            env.lqr.setup_noise(noise)

            try:
                policy = pixel_policy(source_record, env, K)
            except MissingPolicyError as error:
                if verbose:
                    print(f"  skipping {solver} seed={seed}: {error}")
                break

            evaluate_test, trajectory = _pixel_protocol(env, policy, horizon)
            test = evaluate_test()
            rollout = trajectory()

            summary = test.summary
            summary["J_star"] = j_star
            summary["rank_final"] = source_record.summary.get("rank_final")
            summary["policy_complexity"] = source_record.summary.get(
                "policy_complexity"
            )

            record = RunRecord(
                experiment=EXPERIMENT,
                solver=solver,
                seed=seed,
                condition=condition,
                config={
                    "source_run": source_record.run_id,
                    "source_policy": source_record.policy,
                    "noise": to_jsonable(noise),
                    "channels": list(channels),
                    "level": level,
                },
                summary=summary,
                arrays={
                    "test_costs": test.returns,
                    "x_traj": rollout.x_traj,
                    "u_traj": rollout.u_traj,
                },
            )
            store_record(record)
            records.append(record)

            if verbose:
                print(
                    f"  {solver:20s} seed={seed:<6d} {name:12s} {level:10s}"
                    f" test={summary['test_cost_mean']:.4e}"
                    f" ({time.perf_counter() - started:.0f}s)"
                )

    return records


def main():
    evaluate_grid()


if __name__ == "__main__":
    main()
