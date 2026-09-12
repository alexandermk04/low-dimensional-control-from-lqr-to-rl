"""Comparison run using the new evaluation framework with probes, etc.
For now only comparing the LQR part, for later extension with the RL part.

Run it from ``code/`` with::

    python -m evaluation.comparison
"""

import dataclasses
import time
from dataclasses import dataclass, field

import numpy as np

from config import (
    DEFAULT_DT,
    DEFAULT_MAX_INITIAL_STATE,
    default_costs,
    double_integrator_model,
    make_full_rank_K0,
    make_lifts,
)
from basetypes import StateSpaceModel
from seeding import create_seed_sequences, rng
from settings import TEST_SEED

from lqr_base.simple_lqr import (
    LQRBase,
    LQRDiscrete,
    LQRDiscreteSettings,
    simulate,
    get_optimal_K_and_cost_discrete as baseline_solution,
)
from lqr_base.inflated_lqr import LQRInflated, LQRInflatedSettings
from lqr_base.noise import LQRNoiseSettings

from lqr_solvers.mb_policy_gradient import (
    ModelBasedPolicyGradientSettings,
    gradient_descent_discrete,
)
from lqr_solvers.mf_policy_gradient import (
    ModelFreePolicyGradientSettings,
    model_free_natural_policy_gradient,
)
from lqr_solvers.structured_policy_iteration import (
    MFStructuredPolicySettings,
    StructuredPolicySettings,
    structured_policy_iteration,
    structured_policy_iteration_mf,
)

from evaluation.analysis import (
    converged,
    policy_complexity_linear,
    policy_complexity_mlp,
    suboptimality,
    time_to_threshold,
)
from evaluation.metrics import (
    LinearPolicy,
    choose_horizon,
    expected_true_cost,
    finite_horizon_expected_cost,
    monte_carlo_cost,
    spectral_radius,
)
from evaluation.probe import Probe
from evaluation.records import RunRecord, append_index, save_record
from evaluation.serialization import to_jsonable

__all__ = ["ComparisonConfig", "build_lqr", "run_comparison"]

RL_SOLVERS = ("ppo_linear", "ppo_mlp", "sac_mlp")

LQR_SOLVERS = ("mb_pg", "spi", "mf_npg", "spi_mf")


@dataclass
class ComparisonConfig:
    """One singular, same problem, and the set of solvers to run on it.

    Attributes
    ----------
    model : StateSpaceModel
        Continuous-time plant.
    Q, R : np.ndarray
        Cost weights.
    dt : float
        Sampling interval.
    max_initial_state : np.ndarray
        Initial states limits.
    resample_initial_state : bool
        Must stay ``True`` for a comparison against RL.
    noise_settings : LQRNoiseSettings or None
        Noise condition. ``None`` runs clean.
    inflation : tuple[int, int] or None
        ``(p, q)`` observation and action dimensions, or ``None`` for the plain
        system.
    alpha : float
        Size of the full-rank component of ``K_0``.
    horizon : int or None
        Evaluation horizon. Derived from the optimal closed loop when omitted.
    train_horizon : int
        Episode length during RL training. May be shorter than ``horizon``.
    reward_scale : float or None
        Defaults to ``dt``, making the return a Riemann sum of the continuous
        cost integral.
    action_limit : float or None
        Half-width of the action box. Derived from the optimal policy's peak
        output when omitted, with a safety factor.
    n_mc_episodes : int
        Rollouts per Monte-Carlo score.
    solvers : tuple[str, ...]
        Which solvers to run.
    experiment : str
        Name of the results subdirectory.
    label : str
        Short tag for the condition, appended to each solver name.
    """

    model: StateSpaceModel = field(default_factory=double_integrator_model)
    Q: np.ndarray = None
    R: np.ndarray = None
    dt: float = DEFAULT_DT
    max_initial_state: np.ndarray = field(
        default_factory=lambda: DEFAULT_MAX_INITIAL_STATE.copy()
    )
    resample_initial_state: bool = True
    noise_settings: LQRNoiseSettings | None = None
    inflation: tuple[int, int] | None = (50, 25)
    alpha: float = 0.5
    horizon: int | None = None
    train_horizon: int = 400
    reward_scale: float | None = None
    action_limit: float | None = None
    n_mc_episodes: int = 300
    solvers: tuple[str, ...] = LQR_SOLVERS + ("ppo_linear", "ppo_mlp", "sac_mlp")
    experiment: str = "comparison"
    label: str = "pure"

    # ----- Solver parameters -----
    solver_iterations: int = 500
    mf_trajectories: int = 200
    mf_rollout_length: int = 800
    mf_smoothing: float = 0.5
    gradient_lr: float = 1.2e-3
    mf_gradient_lr: float = 1e-3
    spi_lambda: float = 50
    spi_mf_lambda: float = 4e4
    spi_mf_stepsize: float = 3.5e-7
    rl_timesteps: int = 120_000_000
    rl_eval_freq: int = 25_000
    # ----------------------------

    def __post_init__(self):
        if self.Q is None or self.R is None:
            default_Q, default_R = default_costs()
            self.Q = default_Q if self.Q is None else self.Q
            self.R = default_R if self.R is None else self.R
        if self.reward_scale is None:
            self.reward_scale = self.dt


def build_lqr(config: ComparisonConfig, seed_seq, U_1=None, U_2=None) -> LQRBase:
    """Construct the plant for one solver, plain or inflated.

    Parameters
    ----------
    config : ComparisonConfig
        The problem.
    seed_seq : np.random.SeedSequence
        Seed for this solver's copy of the system.
    U_1, U_2 : np.ndarray or None
        The lifts. Required when ``config.inflation`` is set.

    Returns
    -------
    LQRBase
        An :class:`~lqr_base.inflated_lqr.LQRInflated` when inflation is
        configured, otherwise an :class:`~lqr_base.simple_lqr.LQRDiscrete`.
    """
    common = dict(
        model=config.model,
        Q=config.Q,
        R=config.R,
        max_initial_state=config.max_initial_state,
        resample_initial_state=config.resample_initial_state,
        noise_settings=config.noise_settings,
        seed_seq=seed_seq,
        dt=config.dt,
    )

    if config.inflation is None:
        return LQRDiscrete(settings=LQRDiscreteSettings(**common))

    return LQRInflated(settings=LQRInflatedSettings(**common, U_1=U_1, U_2=U_2))


def _summarize(
    records,
    j_star: float,
    K_final: np.ndarray | None,
    lqr,
    config: ComparisonConfig,
    mc: tuple[float, float] | None,
    complexity: dict,
) -> dict:
    """Collapse a finished run into the fixed summary schema."""
    j_final = records[-1].cost if records else np.inf
    finite = [r.cost for r in records if np.isfinite(r.cost)]
    j_best = min(finite) if finite else np.inf

    one_pct = time_to_threshold(records, j_star, 0.01)
    ten_pct = time_to_threshold(records, j_star, 0.10)

    K_eff = lqr.effective_gain(K_final) if K_final is not None else None

    return {
        "J_star": float(j_star),
        "J_final": float(j_final),
        "J_best": float(j_best),
        "training_subopt_final": suboptimality(j_final, j_star),
        "training_subopt_best": suboptimality(j_best, j_star),
        "steps_to_thresh_1pct": one_pct["env_steps"],
        "steps_to_thresh_10pct": ten_pct["env_steps"],
        "iters_to_thresh_1pct": one_pct["iterations"],
        "seconds_to_thresh_1pct": one_pct["total_time"],
        "total_env_steps": int(records[-1].env_steps) if records else 0,
        "total_time": float(records[-1].total_time) if records else 0.0,
        "converged": converged(records),
        "spectral_radius_final": (
            spectral_radius(lqr, K_eff) if K_eff is not None else None
        ),
        "rank_final": complexity.get("rank"),
        "policy_complexity": complexity.get("n_params"),
        "mc_cost_mean": None if mc is None else mc[0],
        "mc_cost_stderr": None if mc is None else mc[1],
        "horizon": config.horizon,
        "n_mc_episodes": config.n_mc_episodes,
    }


def _base_config_dict(config: ComparisonConfig, extra: dict | None = None) -> dict:
    """The config part stored in every record of a run."""
    payload = {
        "plant": to_jsonable(config.model),
        "cost": {"Q": to_jsonable(config.Q), "R": to_jsonable(config.R)},
        "dt": config.dt,
        "max_initial_state": to_jsonable(config.max_initial_state),
        "resample_initial_state": config.resample_initial_state,
        "noise": to_jsonable(config.noise_settings),
        "inflation": list(config.inflation) if config.inflation else None,
        "alpha": config.alpha,
        "evaluation": {
            "horizon": config.horizon,
            "n_mc_episodes": config.n_mc_episodes,
            "reward_scale": config.reward_scale,
            "action_limit": config.action_limit,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _run_lqr_solver(solver, config, K_0, U_1, U_2, j_star, seeds) -> RunRecord:
    """Run one LQR solver and package the result."""
    lqr = build_lqr(config, seeds[f"lqr_{solver}"], U_1, U_2)
    lqr.reset()

    # A separate system for scoring
    scorer = build_lqr(config, seeds[f"eval_{solver}"], U_1, U_2)
    scorer.reset()

    probe = Probe(lqr, evaluate=lambda K: expected_true_cost(scorer, scorer.effective_gain(K)))
    probe.start()

    if solver == "mb_pg":
        settings = ModelBasedPolicyGradientSettings(
            K_0=K_0, gradient_steps=config.solver_iterations,
            gradient_lr=config.gradient_lr, probe=probe)
        ks, _ = gradient_descent_discrete(lqr, settings)
    elif solver == "spi":
        settings = StructuredPolicySettings(
            K_0=K_0, beta=0.5, lambda_=config.spi_lambda, epsilon=1e-8,
            limit=config.solver_iterations, max_linesearch_iter=10, probe=probe)
        ks, _ = structured_policy_iteration(lqr, settings)
    elif solver == "mf_npg":
        settings = ModelFreePolicyGradientSettings(
            K_0=K_0, m=config.mf_trajectories, l=config.mf_rollout_length,
            r=config.mf_smoothing, gradient_steps=config.solver_iterations,
            gradient_lr=config.mf_gradient_lr, seed_seq=seeds[f"solve_{solver}"], probe=probe)
        ks, _ = model_free_natural_policy_gradient(lqr, settings)
    elif solver == "spi_mf":
        settings = MFStructuredPolicySettings(
            K_0=K_0, beta=0.5, lambda_=config.spi_mf_lambda, epsilon=1e-8,
            limit=config.solver_iterations, max_linesearch_iter=10,
            m=config.mf_trajectories, l=config.mf_rollout_length,
            r=config.mf_smoothing, stepsize=config.spi_mf_stepsize,
            seed_seq=seeds[f"solve_{solver}"], probe=probe)
        ks, _ = structured_policy_iteration_mf(lqr, settings)
    else:
        raise ValueError(f"Unknown LQR solver {solver!r}")

    K_final = ks[-1]
    complexity = policy_complexity_linear(K_final)

    mc = monte_carlo_cost(
        scorer, LinearPolicy(K_final),
        n_episodes=config.n_mc_episodes, horizon=config.horizon,
    )

    trajectory = simulate(lqr=scorer, K=K_final, n_steps=config.horizon)

    record = RunRecord(
        experiment=config.experiment,
        solver=f"{solver}_{config.label}",
        seed=TEST_SEED,
        config=_base_config_dict(config, {"solver": to_jsonable(settings)}),
        summary=_summarize(probe.records, j_star, K_final, scorer, config, mc, complexity),
        arrays={
            **probe.to_arrays(),
            "K_final": K_final,
            "K_eff_final": scorer.effective_gain(K_final),
            "x_traj": trajectory.x_traj,
            "u_traj": trajectory.u_traj,
        },
    )
    record.summary["n_params_dense"] = complexity.get("n_params_dense")

    return record


def _run_rl_solver(solver, config, K_0, U_1, U_2, j_star, seeds) -> RunRecord:
    """Train one RL solver and package the result."""
    import dataclasses as dc

    from reinforcement_learning.solvers import (
        LINEAR_PPO, MLP_PPO, MLP_SAC, SB3Policy,
        build_model, extract_linear_gain, is_linear_solver,
    )
    from reinforcement_learning.callbacks import ProbeCallback
    from reinforcement_learning.lqr_env import LQREnv

    presets = {"ppo_linear": LINEAR_PPO, "ppo_mlp": MLP_PPO, "sac_mlp": MLP_SAC}
    settings = dc.replace(
        presets[solver],
        total_timesteps=config.rl_timesteps,
        eval_freq=config.rl_eval_freq,
        initial_gain=K_0 if is_linear_solver(presets[solver]) else None,
    )

    train = build_lqr(config, seeds[f"lqr_{solver}"], U_1, U_2)
    env = LQREnv(
        train, horizon=config.train_horizon,
        reward_scale=config.reward_scale, action_limit=config.action_limit,
    )

    scorer = build_lqr(config, seeds[f"eval_{solver}"], U_1, U_2)
    scorer.reset()

    linear = is_linear_solver(settings)

    def evaluate(model):
        if linear:
            K, _ = extract_linear_gain(model)
            return (
                expected_true_cost(scorer, scorer.effective_gain(K)),
                int(np.linalg.matrix_rank(K)),
            )
        mean, _ = monte_carlo_cost(
            scorer, SB3Policy(model),
            n_episodes=max(20, config.n_mc_episodes // 10), horizon=config.horizon,
        )
        return mean, None

    callback = ProbeCallback(evaluate, eval_freq=settings.eval_freq)
    model = build_model(env, settings, seed=TEST_SEED)
    model.learn(total_timesteps=settings.total_timesteps, callback=callback,
                progress_bar=False)

    if linear:
        K_final, bias = extract_linear_gain(model)
        complexity = policy_complexity_linear(K_final)
        policy = LinearPolicy(K_final)
    else:
        K_final, bias = None, None
        complexity = policy_complexity_mlp(model)
        policy = SB3Policy(model)

    mc = monte_carlo_cost(
        scorer, policy, n_episodes=config.n_mc_episodes, horizon=config.horizon
    )

    arrays = {**callback.to_arrays()}
    if K_final is not None:
        arrays["K_final"] = K_final
        arrays["K_eff_final"] = scorer.effective_gain(K_final)
        arrays["bias"] = bias
        trajectory = simulate(lqr=scorer, K=K_final, n_steps=config.horizon)
        arrays["x_traj"] = trajectory.x_traj
        arrays["u_traj"] = trajectory.u_traj

    record = RunRecord(
        experiment=config.experiment,
        solver=f"{solver}_{config.label}",
        seed=TEST_SEED,
        config=_base_config_dict(config, {"solver": to_jsonable(settings)}),
        summary=_summarize(callback.records, j_star, K_final, scorer, config, mc, complexity),
    )
    record.arrays = arrays
    record.summary["n_params_dense"] = complexity.get("n_params_dense")
    record.summary["action_clips"] = env.n_action_clips
    if bias is not None:
        record.summary["bias_ratio"] = float(
            np.linalg.norm(bias) / max(np.linalg.norm(K_final), 1e-12)
        )

    return record


def run_comparison(config: ComparisonConfig, root="results", verbose: bool = True):
    """Run every configured solver and save one record each.

    Parameters
    ----------
    config : ComparisonConfig
        The problem and the solvers.
    root : str, optional
        Results tree to write into.
    verbose : bool, optional
        Print a line per solver as it finishes.

    Returns
    -------
    list[evaluation.records.RunRecord]
        One record per solver, in the order they were run.
    """
    names = (
        ["matrices", "baseline"]
        + [f"lqr_{a}" for a in config.solvers]
        + [f"eval_{a}" for a in config.solvers]
        + [f"solve_{a}" for a in config.solvers]
    )
    seeds = create_seed_sequences(names, TEST_SEED)

    U_1 = U_2 = None
    if config.inflation is not None:
        matrix_rng = rng(seeds["matrices"])
        U_1, U_2 = make_lifts(*config.inflation, config.model, matrix_rng)
        K_0 = make_full_rank_K0(U_1, U_2, matrix_rng, alpha=config.alpha)
    else:
        K_0 = np.array([[0.1, 0.5]])

    # The optimum, and the evaluation settings derived from it.
    reference = build_lqr(config, seeds["baseline"], U_1, U_2)
    plain = LQRDiscrete(settings=LQRDiscreteSettings(
        model=config.model, Q=config.Q, R=config.R,
        max_initial_state=config.max_initial_state,
        resample_initial_state=config.resample_initial_state,
        noise_settings=None, seed_seq=seeds["baseline"], dt=config.dt))
    K_star, _ = baseline_solution(plain)
    reference.reset()
    j_star = expected_true_cost(reference, K_star)

    if config.horizon is None:
        config.horizon = choose_horizon(reference, K_star)
    if config.action_limit is None:
        to_action = K_0 @ U_1 if U_1 is not None else K_star
        peak = float(np.max(np.abs(to_action) @ np.abs(config.max_initial_state)))
        config.action_limit = float(10.0 * max(peak, 1.0))

    if verbose:
        print(f"[{config.label}] J* = {j_star:.6e}   H = {config.horizon}   "
              f"action_limit = {config.action_limit:.1f}")
        print(f"[{config.label}] J(K_0) = {expected_true_cost(reference, reference.effective_gain(K_0)):.6e}")

    records = []
    for solver in config.solvers:
        started = time.perf_counter()
        runner = _run_rl_solver if solver in RL_SOLVERS else _run_lqr_solver
        record = runner(solver, config, K_0, U_1, U_2, j_star, seeds)

        save_record(record, root)
        append_index(record, root)
        records.append(record)

        if verbose:
            s = record.summary
            print(f"  {solver:12s} J={s['J_final']:.5e}  subopt={s['training_subopt_final']:+.4f}  "
                  f"steps={s['total_env_steps']:>9d}  rank={s['rank_final']}  "
                  f"params={s['policy_complexity']}  ({time.perf_counter()-started:.0f}s)")

    return records


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from evaluation.plots import figure_from_records, format_table, budget_table

    config = ComparisonConfig(
        inflation=(50, 25),
        solvers=("mb_pg", "spi", "mf_npg"), # "spi_mf", "ppo_linear", "ppo_mlp", "sac_mlp"
        noise_settings=None,
        solver_iterations=500,
        rl_timesteps=80_000_000,
        label="pure_adjusted",
    )
    results = run_comparison(config)

    print("\n" + format_table(budget_table(results)))
    figure_from_records(results, dt=config.dt, save_to=f"comparison_{config.label}")
    print(f"\nfigure -> results/figures/comparison_{config.label}.png")
