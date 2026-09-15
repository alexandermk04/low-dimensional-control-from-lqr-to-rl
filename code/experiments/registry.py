"""Contains all the available problem solvers.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import settings
from evaluation.analysis import policy_complexity_linear, policy_complexity_mlp
from evaluation.metrics import LinearPolicy, Policy, expected_true_cost
from evaluation.probe import IterationRecord, Probe
from experiments.problem import Problem, RunSpec, build_plant
from seeding import sub_seed
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

__all__ = ["SolverResult", "SOLVER_REGISTRY", "run_solver", "known_solvers"]

# Solvers whose training happens through stable-baselines3.
RL_SOLVERS = ("ppo_linear", "ppo_mlp", "ppo_encoder")


@dataclass
class SolverResult:
    """What one solver produced.

    Attributes
    ----------
    K_final : np.ndarray or None
        Final gain in the *controller's* space, or ``None`` for a policy that is
        not a linear gain.
    policy : Policy
        The trained controller.
    records : list[IterationRecord]
        The learning curve.
    complexity : dict
        From :func:`~evaluation.analysis.policy_complexity_linear` or
        :func:`~evaluation.analysis.policy_complexity_mlp`.
    solver_settings : Any
        The settings object handed to the solver.
    scorer : Any
        The plant the learning curve was scored against.
    extra : dict
        Solver-specific summary entries, e.g. ``action_clips``.
    save_policy : callable or None
        Given the run's directory, writes what is needed to reload the
        policy later.
    j_star : float or None
        Reference cost the learning curve is scored against, when it is not
        ``problem.j_star``. Chapter 6 sets it: its plant is the thruster ring
        driven at the agent rate, so the chapter-5 optimum is the wrong
        yardstick.
    evaluate_test : callable or None
        ``() -> TestResult``.
    trajectory : callable or None
        ``() -> SimulationResult``.
    """

    K_final: np.ndarray | None
    policy: Policy
    records: list[IterationRecord]
    complexity: dict
    solver_settings: Any
    scorer: Any
    extra: dict = field(default_factory=dict)
    save_policy: Any = None
    j_star: float | None = None
    evaluate_test: Any = None
    trajectory: Any = None


def _train_and_score_plants(spec: RunSpec, problem: Problem, purpose: str = "train"):
    """A plant to learn on and an independent, clean plant to score against.
    """
    lqr = build_plant(problem, f"{purpose}:{spec.solver}")
    lqr.reset()

    scorer = build_plant(problem, f"score:{spec.solver}")
    scorer.reset()

    return lqr, scorer


def _probe(lqr, scorer) -> Probe:
    """The per-iteration instrumentation shared by every LQR solver."""
    return Probe(
        lqr,
        evaluate=lambda K: expected_true_cost(scorer, scorer.effective_gain(K)),
    )


def _require_samples(spec: RunSpec) -> tuple[int, int]:
    """The ``(l, m)`` a sample-based solver needs, or a clear error."""
    if spec.rollout_length is None or spec.n_trajectories is None:
        raise ValueError(
            f"Solver {spec.solver!r} is sample-based but the run spec carries "
            "no rollout_length / n_trajectories. Set the solver's entry in "
            "settings.CHOSEN_SAMPLE_CELL for this inflation size."
        )
    return spec.rollout_length, spec.n_trajectories


def _require_trajectory_length(spec: RunSpec) -> int:
    """The single trajectory length for N4SID.
    """
    if spec.rollout_length is None:
        raise ValueError(
            f"Solver {spec.solver!r} identifies from one trajectory but the run "
            "spec carries no rollout_length. Set "
            "the solver's entry in settings.CHOSEN_IDENTIFICATION for this inflation size."
        )
    return spec.rollout_length


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _run_baseline(spec: RunSpec, problem: Problem) -> SolverResult:
    """The optimal controller, so the optimum appears as a row in every figure.
    """
    scorer = build_plant(problem, f"score:{spec.solver}")
    scorer.reset()

    if problem.inflation is None:
        K = problem.K_star.copy()
    else:
        K = np.linalg.pinv(problem.U_2) @ problem.K_star @ np.linalg.pinv(problem.U_1)

    records = [IterationRecord(
        iteration=0, env_steps=0, total_time=0.0, cost=problem.j_star,
        rank=int(np.linalg.matrix_rank(K)),
    )]

    return SolverResult(
        K_final=K,
        policy=LinearPolicy(K),
        records=records,
        complexity=policy_complexity_linear(K),
        solver_settings={"solver": "baseline", "method": "discrete algebraic Riccati"},
        scorer=scorer,
    )


# ---------------------------------------------------------------------------
# LQR solvers
# ---------------------------------------------------------------------------

def _run_mb_pg(spec: RunSpec, problem: Problem) -> SolverResult:
    """Exact-gradient policy descent (Fazel et al., 2018)."""
    lqr, scorer = _train_and_score_plants(spec, problem)
    probe = _probe(lqr, scorer)
    probe.start()

    solver_settings = ModelBasedPolicyGradientSettings(
        K_0=problem.K_0,
        gradient_steps=spec.iterations,
        gradient_lr=spec.hyperparams()["gradient_lr"],
        probe=probe,
    )
    ks, _ = gradient_descent_discrete(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

@dataclass
class _Identification:
    """Class for an identification run.

    Attributes
    ----------
    lqr : LQRBase
        The plant the trajectory was drawn from.
    scorer : LQRBase
        Independent clean plant every iterate is scored against.
    settings : IdentificationSettings
        Identification settings
    probe : Probe
        The probe to be passed to the inner solver.
    model : IdentifiedModel or None
        The identified system, or ``None`` when identification failed.
    failure : IdentificationError or None
        The reason for failure, if happened.
    """

    lqr: Any
    scorer: Any
    settings: Any
    probe: Probe
    model: Any = None
    failure: Any = None

    def failed_result(self, problem: Problem) -> SolverResult:
        """The run's result when identification did not produce a model."""
        return _identification_failed(
            problem, self.scorer, self.settings, self.failure
        )


def _identify_for(
    spec: RunSpec, problem: Problem, to_controller=None, purpose: str = "train",
    reduce_input: bool = True,
) -> _Identification:
    """Run the excitation trajectory and identify the plant of one run.

    Parameters
    ----------
    spec : RunSpec
        The run settings.
    problem : Problem
        Its problem.
    to_controller : callable or None, optional
        ``(IdentifiedModel, K) -> np.ndarray`` mapping a gain as the inner
        solver produces it onto the controller's ``(q, p)`` space. ``None`` builds no
        probe.
    purpose : str, optional
        Seed-stream label.
    reduce_input : bool, optional
        Passed through to
        :class:`~lqr_solvers.system_identification.IdentificationSettings`.

    Returns
    -------
    _Identification
    """
    from lqr_solvers.system_identification import (
        IdentificationError,
        IdentificationSettings,
        identify,
    )

    lqr, scorer = _train_and_score_plants(spec, problem, purpose=purpose)
    hyper = spec.hyperparams()

    id_settings = IdentificationSettings(
        trajectory_length=_require_trajectory_length(spec),
        excitation=hyper["excitation"],
        seed_seq=sub_seed(spec.seed, "identify", spec.solver),
        max_order=hyper["max_order"],
        horizon=spec.condition.get("hankel", hyper["horizon"]),
        order_threshold=hyper["order_threshold"],
        input_rank_tol=hyper.get("input_rank_tol", 1e-3), # only for reducing input
        reduce_input=reduce_input,
    )

    context = _Identification(
        lqr=lqr, scorer=scorer, settings=id_settings, probe=None
    )

    if to_controller is not None:
        context.probe = Probe(
            lqr,
            evaluate=lambda K: expected_true_cost(
                scorer, scorer.effective_gain(to_controller(context.model, K))
            ),
        )
        context.probe.start()

    try:
        context.model = identify(lqr, problem.K_0, id_settings)
    except IdentificationError as failure:
        context.failure = failure

    return context


def _identification_extra(identified) -> dict:
    """The additional summary entries from the identification."""
    return {
        "n_hat": identified.n_hat,
        "m_hat": identified.m_hat,
        "effective_max_order": identified.effective_max_order,
        "cost_residual": identified.cost_residual,
    }


def _identification_failed(problem, scorer, id_settings, failure) -> SolverResult:
    """Report an under-determined identification as a result.
    """
    K = problem.K_0
    cost = expected_true_cost(scorer, scorer.effective_gain(K))

    records = [IterationRecord(
        iteration=0,
        env_steps=id_settings.trajectory_length,
        total_time=0.0,
        cost=cost,
        rank=int(np.linalg.matrix_rank(K)),
    )]

    return SolverResult(
        K_final=K,
        policy=LinearPolicy(K),
        records=records,
        complexity=policy_complexity_linear(K),
        solver_settings=id_settings,
        scorer=scorer,
        extra={"identification_failed": str(failure)},
    )


def _run_mb_est(spec: RunSpec, problem: Problem) -> SolverResult:
    """Exact-gradient policy descent on an identified model.
    """

    from lqr_solvers.system_identification import to_inflated_plant

    context = _identify_for(spec, problem, lambda model, K: K, reduce_input=False)
    if context.model is None:
        return context.failed_result(problem)

    identified = context.model
    surrogate = to_inflated_plant(
        identified, sub_seed(spec.seed, "surrogate", spec.solver)
    )

    solver_settings = ModelBasedPolicyGradientSettings(
        K_0=problem.K_0,
        gradient_steps=spec.iterations,
        gradient_lr=spec.hyperparams()["gradient_lr"],
        probe=context.probe,
    )
    ks, _ = gradient_descent_discrete(surrogate, solver_settings)

    result = _linear_result(ks[-1], context.probe, solver_settings, context.scorer)

    result.extra.update(_identification_extra(identified))
    return result


def _run_ce_dare(spec: RunSpec, problem: Problem) -> SolverResult:
    """Certainty equivalence: identify the plant, then solve its Riccati equation.
    """

    from lqr_base.simple_lqr import get_optimal_K_and_cost_discrete
    from lqr_solvers.system_identification import to_plant

    context = _identify_for(
        spec, problem, lambda model, K: model.to_controller_gain(K)
    )
    if context.model is None:
        return context.failed_result(problem)

    identified = context.model
    surrogate = to_plant(identified, sub_seed(spec.seed, "surrogate", spec.solver))

    K_identified, _ = get_optimal_K_and_cost_discrete(surrogate)
    K_final = identified.to_controller_gain(K_identified)

    context.probe.record(0, K_identified)

    result = _linear_result(
        K_final, context.probe,
        {"solver": "ce_dare", "trajectory_length": context.settings.trajectory_length},
        context.scorer,
    )

    result.extra.update(_identification_extra(identified))
    return result


def _run_spi_est(spec: RunSpec, problem: Problem) -> SolverResult:
    """S-PI on an identified model.
    """
    from lqr_solvers.system_identification import to_inflated_plant

    context = _identify_for(
        spec, problem, lambda model, K: K, reduce_input=False
    )
    if context.model is None:
        return context.failed_result(problem)

    identified = context.model
    surrogate = to_inflated_plant(
        identified, sub_seed(spec.seed, "surrogate", spec.solver)
    )

    hyper = spec.hyperparams()
    solver_settings = StructuredPolicySettings(
        K_0=problem.K_0,
        beta=hyper["beta"],
        lambda_=hyper["lambda_"],
        epsilon=hyper["epsilon"],
        limit=spec.iterations,
        max_linesearch_iter=hyper["max_linesearch_iter"],
        probe=context.probe,
    )
    ks, _ = structured_policy_iteration(surrogate, solver_settings)

    result = _linear_result(ks[-1], context.probe, solver_settings, context.scorer)

    result.extra.update(_identification_extra(identified))
    return result


def _run_mf_npg(spec: RunSpec, problem: Problem) -> SolverResult:
    """Rollout-based natural policy gradient."""
    rollout_length, n_trajectories = _require_samples(spec)

    lqr, scorer = _train_and_score_plants(spec, problem)
    probe = _probe(lqr, scorer)
    probe.start()

    hyper = spec.hyperparams()
    solver_settings = ModelFreePolicyGradientSettings(
        K_0=problem.K_0,
        m=n_trajectories,
        l=rollout_length,
        r=hyper["smoothing"],
        gradient_steps=spec.iterations,
        gradient_lr=hyper["gradient_lr"],
        seed_seq=sub_seed(spec.seed, "solve", spec.solver),
        probe=probe,
    )
    ks, _ = model_free_natural_policy_gradient(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


def _run_spi(spec: RunSpec, problem: Problem) -> SolverResult:
    """Model-based low-rank structured policy iteration (Park et al., 2020)."""
    lqr, scorer = _train_and_score_plants(spec, problem)
    probe = _probe(lqr, scorer)
    probe.start()

    hyper = spec.hyperparams()
    solver_settings = StructuredPolicySettings(
        K_0=problem.K_0,
        beta=hyper["beta"],
        lambda_=hyper["lambda_"],
        epsilon=hyper["epsilon"],
        limit=spec.iterations,
        max_linesearch_iter=hyper["max_linesearch_iter"],
        probe=probe,
    )
    ks, _ = structured_policy_iteration(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


def _run_spi_mf(spec: RunSpec, problem: Problem) -> SolverResult:
    """Model-free structured policy iteration."""
    rollout_length, n_trajectories = _require_samples(spec)

    lqr, scorer = _train_and_score_plants(spec, problem)
    probe = _probe(lqr, scorer)
    probe.start()

    hyper = spec.hyperparams()
    solver_settings = MFStructuredPolicySettings(
        K_0=problem.K_0,
        beta=hyper["beta"],
        lambda_=hyper["lambda_"],
        epsilon=hyper["epsilon"],
        limit=spec.iterations,
        max_linesearch_iter=hyper["max_linesearch_iter"],
        m=n_trajectories,
        l=rollout_length,
        r=hyper["smoothing"],
        stepsize=hyper["stepsize"],
        seed_seq=sub_seed(spec.seed, "solve", spec.solver),
        probe=probe,
    )
    ks, _ = structured_policy_iteration_mf(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


def _linear_result(K_final, probe, solver_settings, scorer) -> SolverResult:
    """Package a solver that produced a gain matrix."""
    return SolverResult(
        K_final=K_final,
        policy=LinearPolicy(K_final),
        records=probe.records,
        complexity=policy_complexity_linear(K_final),
        solver_settings=solver_settings,
        scorer=scorer,
    )


# ---------------------------------------------------------------------------
# Reinforcement learning
# ---------------------------------------------------------------------------

def _run_rl(
    spec: RunSpec,
    problem: Problem,
    encoder: np.ndarray | None = None,
    env_steps_offset: int = 0,
) -> SolverResult:
    """Train one stable-baselines3 policy on the LQR environment.

    Parameters
    ----------
    spec, problem
        The run.
    encoder : np.ndarray or None, optional
        ``(k, p)`` matrix the policy observes through.
    env_steps_offset : int, optional
        Added to every recorded ``env_steps``.
    """
    import dataclasses as dc

    from reinforcement_learning.callbacks import ProbeCallback
    from reinforcement_learning.lqr_env import LQREnv
    from reinforcement_learning.solvers import (
        ENCODER_PPO,
        LINEAR_PPO,
        MLP_PPO,
        SB3Policy,
        build_model,
        extract_linear_gain,
        is_linear_solver,
    )
    from evaluation.metrics import monte_carlo_cost

    presets = {
        "ppo_linear": LINEAR_PPO,
        "ppo_mlp": MLP_PPO,
        "ppo_encoder": ENCODER_PPO,
    }
    preset = presets[spec.solver]
    linear = is_linear_solver(preset)

    solver_settings = dc.replace(
        preset,
        total_timesteps=settings.RL_TIMESTEPS,
        eval_freq=settings.RL_EVAL_FREQ,
        initial_gain=problem.K_0 if linear else None,
    )

    train = build_plant(problem, f"train:{spec.solver}")
    env = LQREnv(
        train,
        horizon=settings.TRAIN_HORIZON,
        reward_scale=problem.dt,
        action_limit=problem.action_limit,
        encoder=encoder,
    )

    scorer = build_plant(problem, f"score:{spec.solver}")
    scorer.reset()

    def evaluate(model):
        if linear:
            K, _ = extract_linear_gain(model)
            return (
                expected_true_cost(scorer, scorer.effective_gain(K)),
                int(np.linalg.matrix_rank(K)),
            )
        mean, _ = monte_carlo_cost(
            scorer, SB3Policy(model, encoder=encoder),
            n_episodes=max(20, settings.TEST_EPISODES // 10),
            horizon=settings.PROBE_HORIZON,
        )
        return mean, None

    callback = ProbeCallback(
        evaluate,
        eval_freq=solver_settings.eval_freq,
        env_steps_offset=env_steps_offset,
    )
    model = build_model(env, solver_settings, seed=spec.seed)
    model.learn(
        total_timesteps=solver_settings.total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    extra = {"action_clips": env.n_action_clips}

    if linear:
        K_final, bias = extract_linear_gain(model)
        complexity = policy_complexity_linear(K_final)
        policy = LinearPolicy(K_final)
        extra["bias_ratio"] = float(
            np.linalg.norm(bias) / max(np.linalg.norm(K_final), 1e-12)
        )
    else:
        K_final = None
        complexity = policy_complexity_mlp(model)
        policy = SB3Policy(model, encoder=encoder)

    return SolverResult(
        K_final=K_final,
        policy=policy,
        records=callback.records,
        complexity=complexity,
        solver_settings=solver_settings,
        scorer=scorer,
        extra=extra,
        save_policy=None if linear else _sb3_saver(model, solver_settings, encoder),
    )


def _run_ppo_encoder(spec: RunSpec, problem: Problem) -> SolverResult:
    """PPO behind the linear encoder of an identified model.
    """
    context = _identify_for(spec, problem, purpose="identify")

    if context.model is None:
        return context.failed_result(problem)

    result = _run_rl(
        spec, problem,
        encoder=context.model.decode,
        env_steps_offset=context.settings.trajectory_length,
    )

    result.extra.update(_identification_extra(context.model))
    return result


def _sb3_saver(model, solver_settings, encoder=None) -> Callable[[Any], dict]:
    """A saver saving a stable-baselines3 model next to its record.
    """
    def save(directory) -> dict:
        model.save(directory / "policy.zip")

        manifest = {
            "kind": "sb3",
            "algo": solver_settings.algo,
            "file": "policy.zip",
            "deterministic": True,
        }

        if encoder is not None:
            np.save(directory / "encoder.npy", np.asarray(encoder))
            manifest["encoder"] = "encoder.npy"

        return manifest

    return save


def _missing(name: str, what: str) -> Callable[[RunSpec, Problem], SolverResult]:
    """A registry entry that explains itself instead of raising a KeyError."""
    def raise_missing(spec: RunSpec, problem: Problem) -> SolverResult:
        raise NotImplementedError(
            f"Solver {name!r} is registered but not implemented yet: {what}\n"
            f"  Add it in experiments/registry.py; nothing else has to change."
        )
    return raise_missing


SOLVER_REGISTRY: dict[str, Callable[[RunSpec, Problem], SolverResult]] = {
    "baseline": _run_baseline,
    "mb_pg": _run_mb_pg,
    "mb_est": _run_mb_est,
    "mf_npg": _run_mf_npg,
    "spi": _run_spi,
    "ce_dare": _run_ce_dare,
    "spi_est": _run_spi_est,
    "spi_mf": _run_spi_mf,
    "ppo_linear": _run_rl,
    "ppo_mlp": _run_rl,
    "ppo_encoder": _run_ppo_encoder,
    # Not yet implemented
    "ppo_bottleneck": _missing(
        "ppo_bottleneck",
        "PPO with an orthonomal projection layer"
    ),
    "ppo_encoder_scalable": _missing(
        "ppo_encoder_scalable",
        "observation encoder that replaces N4SID at large scale"
    ),
}


def known_solvers() -> tuple[str, ...]:
    """Every registered solver name."""
    return tuple(SOLVER_REGISTRY)


def run_solver(spec: RunSpec, problem: Problem) -> SolverResult:
    """Dispatch one run to its solver.

    Raises
    ------
    KeyError
        If the solver is not registered at all.
    NotImplementedError
        If it is registered but not written yet.
    """
    if spec.solver not in SOLVER_REGISTRY:
        raise KeyError(
            f"Unknown solver {spec.solver!r}. Known: {', '.join(known_solvers())}"
        )
    return SOLVER_REGISTRY[spec.solver](spec, problem)
