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
from experiments.evaluate import evaluate_policy, figure_trajectory
from experiments.problem import Problem, RunSpec, make_problem
from experiments.registry import SolverResult, run_solver
from experiments.store import store_record, write_manifest

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
    finite = [r.cost for r in records if np.isfinite(r.cost)]
    j_best = min(finite) if finite else np.inf

    one_pct = time_to_threshold(records, problem.j_star, 0.01)
    ten_pct = time_to_threshold(records, problem.j_star, 0.10)

    K_eff = (
        result.scorer.effective_gain(result.K_final)
        if result.K_final is not None else None
    )

    summary = {
        "J_star": float(problem.j_star),
        "J_final": float(j_final),
        "J_best": float(j_best),
        "subopt_final": suboptimality(j_final, problem.j_star),
        "subopt_best": suboptimality(j_best, problem.j_star),
        "steps_to_thresh_1pct": one_pct["env_steps"],
        "steps_to_thresh_10pct": ten_pct["env_steps"],
        "iters_to_thresh_1pct": one_pct["iterations"],
        "seconds_to_thresh_1pct": one_pct["total_time"],
        "total_env_steps": int(records[-1].env_steps) if records else 0,
        "total_time": float(records[-1].total_time) if records else 0.0,
        "converged": converged(records),
        "spectral_radius_final": (
            spectral_radius(result.scorer, K_eff) if K_eff is not None else None
        ),
        "rank_final": result.complexity.get("rank"),
        "policy_complexity": result.complexity.get("n_params"),
        "n_params_dense": result.complexity.get("n_params_dense"),
        "horizon": settings.FIGURE_HORIZON,
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
            "horizon": settings.FIGURE_HORIZON,
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

    test = evaluate_policy(
        problem, result.policy, purpose=spec.solver, **(test_kwargs or {})
    )

    arrays = _curve_arrays(result.records)
    arrays["test_costs"] = test.returns

    trajectory = figure_trajectory(problem, result.policy, purpose=spec.solver)
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
    )


def run_grid(
    specs: list[RunSpec],
    experiment: str,
    used_settings: dict,
    verbose: bool = True,
    test_kwargs: dict | None = None,
) -> list[RunRecord]:
    """Train every spec, saving each record as soon as it is finished.

    Allows for re-running with inbetween saves.

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
    """
    write_manifest(
        experiment,
        used_settings,
        grid=[{"solver": s.solver, "seed": s.seed, **s.condition} for s in specs],
    )

    if verbose:
        print(f"[{experiment}] {len(specs)} runs")

    records = []
    for index, spec in enumerate(specs, start=1):
        started = time.perf_counter()

        record = train(spec, test_kwargs=test_kwargs)
        store_record(record)
        records.append(record)

        if verbose:
            s = record.summary
            cell = " ".join(f"{k}={v}" for k, v in spec.condition.items())
            print(
                f"  [{index:>3d}/{len(specs)}] {spec.solver:12s} seed={spec.seed:<6d}"
                f" {cell:24s} subopt={s['subopt_final']:+.4f}"
                f" test={s['test_cost_mean']:.4e}"
                f" rank={s['rank_final']} ({time.perf_counter() - started:.0f}s)"
            )

    return records
