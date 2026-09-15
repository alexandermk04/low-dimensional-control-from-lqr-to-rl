"""Train one or more solvers based on a grid of settings.

Single entrypoint for experiments.
"""

import time

import numpy as np

import settings
from evaluation.analysis import converged, suboptimality, time_to_threshold
from evaluation.metrics import spectral_radius
from evaluation.probe import IterationRecord
from evaluation.records import RunRecord
from evaluation.serialization import to_jsonable
from experiments.keep_running import keep_running
from experiments.evaluate import evaluate_policy, figure_trajectory
from experiments.problem import Problem, RunSpec, make_problem
from experiments.registry import SolverResult, run_solver
from experiments.store import completed_runs, run_key, store_record, write_manifest

__all__ = ["train", "run_grid"]


def _curve_arrays(records: list[IterationRecord]) -> dict[str, np.ndarray]:
    """The learning curve as one float array per field, ``None`` becoming NaN."""
    if not records:
        return {}

    fields = ("iteration", "env_steps", "total_time", "cost", "cost_estimate", "rank")

    return {
        field: np.array(
            [
                np.nan if getattr(r, field) is None else float(getattr(r, field))
                for r in records
            ]
        )
        for field in fields
    }


def _summarize(
    result: SolverResult, problem: Problem, spec: RunSpec, test: dict
) -> dict:
    """Collapse a finished run into a fixed summary schema."""
    records = result.records

    j_final = records[-1].cost if records else np.inf

    j_star = problem.j_star if result.j_star is None else result.j_star

    one_pct = time_to_threshold(records, j_star, 0.01)

    K_eff = (
        result.scorer.effective_gain(result.K_final)
        if result.K_final is not None else None
    )

    summary = {
        "J_star": float(j_star),
        "J_final": float(j_final),
        "training_subopt_final": suboptimality(j_final, j_star),
        "steps_to_thresh_1pct": one_pct["env_steps"],
        "total_env_steps": int(records[-1].env_steps) if records else 0,
        "total_time": float(records[-1].total_time) if records else 0.0,
        "converged": converged(records),
        "spectral_radius_final": (
            spectral_radius(result.scorer, K_eff) if K_eff is not None else None
        ),
        "rank_final": result.complexity.get("rank"),
        "policy_complexity": result.complexity.get("n_params"),
        **test,
        **result.extra,
    }

    return summary


def _config_dict(spec: RunSpec, problem: Problem, result: SolverResult) -> dict:
    """The data stored in every record for reproducibility."""
    return {
        "plant": to_jsonable(problem.model),
        "cost": {"Q": to_jsonable(problem.Q), "R": to_jsonable(problem.R)},
        "dt": problem.dt,
        "max_initial_state": to_jsonable(problem.max_initial_state),
        "inflation": list(spec.inflation) if spec.inflation else None,
        "alpha": settings.K0_ALPHA,
        "sampling": {
            "rollout_length": spec.rollout_length,
            "n_trajectories": spec.n_trajectories,
            "iterations": spec.iterations,
        },
        "evaluation": {
            "figure_horizon": settings.FIGURE_HORIZON,
            "probe_horizon": settings.PROBE_HORIZON,
            "action_limit": problem.action_limit,
            "test_episodes": settings.TEST_EPISODES,
            "test_horizon": settings.TEST_HORIZON,
        },
        "solver": to_jsonable(result.solver_settings),
    }


def train(spec: RunSpec, test_kwargs: dict | None = None) -> RunRecord:
    """Run one solver on one seed and package the result.

    Parameters
    ----------
    spec : RunSpec
        What to run.
    test_kwargs : dict or None, optional
        Potential overrides for the test protocol.

    Returns
    -------
    RunRecord
        With the learning curve, the final gain and a fixed-``x_0`` trajectory
        in ``arrays``, and the scalar metrics in ``summary``.
    """
    problem = make_problem(spec.seed, spec.inflation)

    result = run_solver(spec, problem)

    test = (
        result.evaluate_test(**(test_kwargs or {}))
        if result.evaluate_test is not None
        else evaluate_policy(problem, result.policy, **(test_kwargs or {}))
    )

    arrays = _curve_arrays(result.records)
    arrays["test_costs"] = test.returns

    trajectory = (
        result.trajectory() if result.trajectory is not None
        else figure_trajectory(problem, result.policy)
    )
    arrays["x_traj"] = trajectory.x_traj
    arrays["u_traj"] = trajectory.u_traj

    if result.K_final is not None:
        arrays["K_final"] = result.K_final
        arrays["K_eff_final"] = result.scorer.effective_gain(result.K_final)

    return RunRecord(
        experiment=spec.experiment,
        solver=spec.solver,
        seed=spec.seed,
        condition=dict(spec.condition),
        config=_config_dict(spec, problem, result),
        summary=_summarize(result, problem, spec, test.summary),
        arrays=arrays,
        policy={"kind": "linear", "array": "K_final"}
        if result.K_final is not None
        else {},
        save_policy=result.save_policy,
    )


@keep_running()
def run_grid(
    specs: list[RunSpec],
    experiment: str,
    used_settings: dict,
    verbose: bool = True,
    test_kwargs: dict | None = None,
) -> list[RunRecord]:
    """Train every spec, saving each record as soon as it is finished.

    Can be resumed if interrupted during runs.

    Parameters
    ----------
    specs : list[RunSpec]
        The grid, already expanded over seeds and conditions.
    experiment : str
        Name of the experiment, and its results subdirectory.
    used_settings : dict
        The :mod:`settings` values this experiment read.
    verbose : bool, optional
        Print one line per finished run.
    test_kwargs : dict or None, optional
        Passed through to :func:`train`.

    Returns
    -------
    list[RunRecord]
        The runs trained in this call. Runs skipped as already done are not
        reloaded.
    """
    write_manifest(
        experiment,
        used_settings,
        grid=[{"solver": s.solver, "seed": s.seed, **s.condition} for s in specs],
    )

    done = completed_runs(experiment)
    pending = [
        spec for spec in specs
        if run_key(spec.solver, spec.seed, spec.condition) not in done
    ]

    if verbose:
        print(
            f"[{experiment}] {len(specs)} runs, {len(specs) - len(pending)} "
            f"already done, {len(pending)} to go"
        )

    records = []
    for index, spec in enumerate(specs, start=1):
        if run_key(spec.solver, spec.seed, spec.condition) in done:
            continue

        started = time.perf_counter()

        record = train(spec, test_kwargs=test_kwargs)
        store_record(record)
        records.append(record)

        if verbose:
            s = record.summary
            cell = " ".join(f"{k}={v}" for k, v in spec.condition.items())
            print(
                f"  [{index:>3d}/{len(specs)}] {spec.solver:12s} seed={spec.seed:<6d}"
                f" {cell:24s} subopt={s.get('training_subopt_final', float('nan')):+.4f}"
                f" test={s['test_cost_mean']:.4e}"
                f" rank={s['rank_final']} ({time.perf_counter() - started:.0f}s)"
            )

    return records
