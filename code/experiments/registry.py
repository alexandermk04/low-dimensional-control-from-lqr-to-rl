"""Contains all the available problem solvers.

References
----------
.. [1] M. Fazel, R. Ge, S. M. Kakade, M. Mesbahi, "Global Convergence of Policy
       Gradient Methods for the Linear Quadratic Regulator", ICML 2018.
       https://arxiv.org/abs/1801.05039
       The exact and the rollout-based policy gradient, in
       :mod:`lqr_solvers.mb_policy_gradient` and
       :mod:`lqr_solvers.mf_policy_gradient`.
.. [2] Y. Park, R. A. Rossi, Z. Wen, G. Wu, H. Zhao, "Structured Policy
       Iteration for Linear Quadratic Regulator", ICML 2020.
       https://arxiv.org/abs/2007.06202
       S-PI, model-based and model-free, in
       :mod:`lqr_solvers.structured_policy_iteration`.
.. [3] P. Van Overschee, B. De Moor, "N4SID: Subspace algorithms for the
       identification of combined deterministic-stochastic systems",
       Automatica 30(1), 1994.
       The identification we use for ``*_est`` solvers, in
       :mod:`lqr_solvers.system_identification`, through the SIPPY package.
.. [4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal
       Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347
       Every ``ppo_*`` solver, through Stable-Baselines3.
.. [5] The chapter-5 and chapter-6 policy architectures, cited individually in
       :mod:`reinforcement_learning.features`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import settings
from basetypes import SimulationResult
from evaluation.analysis import Plateau, policy_complexity_linear, policy_complexity_mlp
from evaluation.metrics import (
    LinearPolicy,
    Policy,
    expected_true_cost,
    monte_carlo_cost,
)
from evaluation.latent_probe import latent_extra
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

# Solvers who are trained through stable-baselines3.
RL_SOLVERS = (
    "ppo_linear", "ppo_mlp", "ppo_encoder", "ppo_bottleneck", "ppo_encoder_scalable",
)

# Chapter 6: trained through stable-baselines3 too, but on a pixel
# environment and a thruster ring rather than on an inflated LQR.
PIXEL_SOLVERS = (
    "ppo_pixel_fc", "ppo_pixel_ortho", "ppo_pixel_estimator", "ppo_pixel_posvel",
)


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
        ``() -> TestResult``
    trajectory : callable or None
        ``() -> SimulationResult``
    j_final : float or None
        True cost of the returned policy.
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
    j_final: float | None = None


def _train_and_score_plants(spec: RunSpec, problem: Problem, purpose: str = "train"):
    """A plant to learn on and an independent, clean plant to score against.
    """
    lqr = build_plant(problem, f"{purpose}:{spec.solver}")
    lqr.reset()

    scorer = build_plant(problem, "score")
    scorer.reset()

    return lqr, scorer


def _probe(lqr, scorer) -> Probe:
    """The per-iteration instrumentation shared by every LQR solver."""
    return Probe(
        lqr,
        evaluate=lambda K: expected_true_cost(scorer, scorer.effective_gain(K)),
    )


def _stopper(kind: str, overrides: dict | None = None) -> Plateau:
    """The convergence criterion of ``settings.CONVERGENCE[kind]``, with ``overrides`` on top."""
    return Plateau(**{**settings.CONVERGENCE[kind], **(overrides or {})})


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
    scorer = build_plant(problem, "score")
    scorer.reset()

    K = optimal_controller_gain(problem)

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
    """Exact-gradient policy descent on the true model.

    The algorithm of Fazel et al. [1]_.
    """
    lqr, scorer = _train_and_score_plants(spec, problem)
    probe = _probe(lqr, scorer)
    probe.start()

    solver_settings = ModelBasedPolicyGradientSettings(
        K_0=problem.K_0,
        gradient_steps=spec.iterations,
        gradient_lr=spec.hyperparams()["gradient_lr"],
        probe=probe,
        stopper=_stopper("model_based"),
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


def optimal_controller_gain(problem: Problem) -> np.ndarray:
    """The optimal gain expressed in the controller's own space.

    Parameters
    ----------
    problem : Problem
        The run's problem.

    Returns
    -------
    np.ndarray
        ``K_star`` itself for the plain plant, its lift ``pinv(U_2) K_star
        pinv(U_1)`` of shape ``(q, p)`` for an inflated one.
    """
    if problem.inflation is None:
        return problem.K_star.copy()

    return np.linalg.pinv(problem.U_2) @ problem.K_star @ np.linalg.pinv(problem.U_1)


def probe_reference_cost(scorer, problem: Problem) -> float:
    """The optimal policy's cost measured during training.

    Parameters
    ----------
    scorer : LQRBase
        The plant the learning curve is probed against.
    problem : Problem
        The run's problem, for the optimal gain.

    Returns
    -------
    float
        Reference cost for :attr:`SolverResult.j_star`.
    """
    mean, _ = monte_carlo_cost(
        scorer, LinearPolicy(optimal_controller_gain(problem)),
        n_episodes=settings.PROBE_EPISODES,
        horizon=settings.PROBE_HORIZON,
    )
    return float(mean)


def _run_mb_est(spec: RunSpec, problem: Problem) -> SolverResult:
    """Exact-gradient policy descent on an identified model.

    Fazel et al. [1]_ run on a plant identified by N4SID [3]_.
    """
    from lqr_solvers.system_identification import to_inflated_plant

    context = _identify_for(spec, problem, lambda model, K: K)
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
        stopper=_stopper("model_based"),
    )
    ks, _ = gradient_descent_discrete(surrogate, solver_settings)

    result = _linear_result(ks[-1], context.probe, solver_settings, context.scorer)

    result.extra.update(_identification_extra(identified))
    return result


def _run_ce_dare(spec: RunSpec, problem: Problem) -> SolverResult:
    """Certainty equivalence: identify the plant, then solve its Riccati equation.

    Identification by N4SID [3]_.
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

    Park et al. [2]_ run on a plant identified by N4SID [3]_.
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
        stopper=_stopper("model_based"),
    )
    ks, _ = structured_policy_iteration(surrogate, solver_settings)

    result = _linear_result(ks[-1], context.probe, solver_settings, context.scorer)

    result.extra.update(_identification_extra(identified))
    return result


def _run_mf_npg(spec: RunSpec, problem: Problem) -> SolverResult:
    """Rollout-based natural policy gradient.

    The model-free variant of Fazel et al. [1]_.
    """
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
        stopper=_stopper("model_free"),
    )
    ks, _ = model_free_natural_policy_gradient(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


def _run_spi(spec: RunSpec, problem: Problem) -> SolverResult:
    """Model-based low-rank structured policy iteration.

    The algorithm of Park et al. [2]_.
    """
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
        stopper=_stopper("model_based"),
    )
    ks, _ = structured_policy_iteration(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


def _run_spi_mf(spec: RunSpec, problem: Problem) -> SolverResult:
    """Model-free structured policy iteration.

    The model-free variant of Park et al. [2]_.
    """
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
        stopper=_stopper("model_free"),
    )
    ks, _ = structured_policy_iteration_mf(lqr, solver_settings)

    return _linear_result(ks[-1], probe, solver_settings, scorer)


def _linear_result(K_final, probe, solver_settings, scorer) -> SolverResult:
    """Package a solver that produced a gain matrix.
    """
    stopper = getattr(solver_settings, "stopper", None)
    extra, j_final = {}, None

    if stopper is not None and stopper.best_payload is not None:
        K_final = stopper.best_payload
        extra = stopper.extra()
        j_final = float(expected_true_cost(scorer, scorer.effective_gain(K_final)))

    return SolverResult(
        K_final=K_final,
        policy=LinearPolicy(K_final),
        records=probe.records,
        complexity=policy_complexity_linear(K_final),
        solver_settings=solver_settings,
        scorer=scorer,
        extra=extra,
        j_final=j_final,
    )


# ---------------------------------------------------------------------------
# Reinforcement learning
# ---------------------------------------------------------------------------

def _explore(control, seed: int, scale) -> Callable:
    """A random policy that collects observations for a pretrained extractor.

    Parameters
    ----------
    control : callable
        ``control(observation, info) -> action``.
    seed : int
        Seeds the mixture.
    scale : array-like or float
        Half-width of the box the random actions are drawn from.

    Returns
    -------
    callable
        ``(observation, info) -> action``.
    """
    rng = np.random.default_rng(seed)

    def act(observation, info):
        if rng.random() < 0.3:
            return rng.uniform(-scale, scale, size=np.shape(control(observation, info)))
        return control(observation, info)

    return act


def _obs_scale(problem: Problem, encoder: np.ndarray | None) -> float:
    """``sqrt(p/n)`` for a lifted observation the policy sees raw, else ``1``."""
    if encoder is not None or problem.U_1 is None:
        return 1.0
    return float(np.sqrt(problem.U_1.shape[0] / problem.U_1.shape[1]))


def _run_rl(
    spec: RunSpec,
    problem: Problem,
    encoder: np.ndarray | None = None,
    env_steps_offset: int = 0,
) -> SolverResult:
    """Train one stable-baselines3 policy on the LQR environment.

    PPO [4]_ used throughout.

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
    from reinforcement_learning.features import fit_extractor_weights
    from reinforcement_learning.lqr_env import LQREnv
    from reinforcement_learning.solvers import (
        SB3Policy,
        build_model,
        extract_linear_gain,
        is_linear_solver,
        vector_presets,
    )
    from evaluation.metrics import LinearPolicy, monte_carlo_cost
    from evaluation.pixel_metrics import SB3EnvPolicy

    preset = vector_presets()[spec.solver]
    linear = is_linear_solver(preset)
    obs_scale = _obs_scale(problem, encoder)

    def make_env(purpose: str) -> LQREnv:
        return LQREnv(
            build_plant(problem, f"{purpose}:{spec.solver}"),
            horizon=settings.TRAIN_HORIZON,
            reward_scale=problem.dt,
            action_limit=problem.action_limit,
            encoder=encoder,
            obs_scale=obs_scale,
        )

    env = make_env("train")

    weights = None
    if preset.freeze_extractor:
        stabilizing = LinearPolicy(problem.K_0 / obs_scale)
        weights = fit_extractor_weights(
            preset, make_env("fit"),
            _explore(lambda y, info: stabilizing(y), spec.seed, problem.action_limit / 10.0),
            n_steps=settings.PRETRAIN_STEPS,
            epochs=settings.PRETRAIN_EPOCHS,
        )
        env_steps_offset += settings.PRETRAIN_STEPS

    max_timesteps = settings.RL_MAX_TIMESTEPS[spec.condition["size"]]
    solver_settings = dc.replace(
        preset,
        total_timesteps=spec.hyperparams().get("total_timesteps", max_timesteps),
        anneal_timesteps=max_timesteps,
        learning_rate=spec.hyperparams().get("learning_rate", preset.learning_rate),
        eval_freq=settings.RL_EVAL_FREQ,
        initial_gain=problem.K_0 / obs_scale if linear else None,
        pretrained_extractor=weights,
    )

    scorer = build_plant(problem, "score")
    scorer.reset()

    def evaluate(model):
        if linear:
            K = obs_scale * extract_linear_gain(model)[0]
            return (
                expected_true_cost(scorer, scorer.effective_gain(K)),
                int(np.linalg.matrix_rank(K)),
            )
        mean, _ = monte_carlo_cost(
            scorer, SB3Policy(model, encoder=encoder, obs_scale=obs_scale),
            n_episodes=settings.PROBE_EPISODES,
            horizon=settings.PROBE_HORIZON,
        )
        return mean, None

    callback = ProbeCallback(
        evaluate,
        eval_freq=solver_settings.eval_freq,
        env_steps_offset=env_steps_offset,
        stopper=_stopper("rl", spec.hyperparams().get("convergence")),
    )
    model = build_model(env, solver_settings, seed=spec.seed)

    try:
        model.learn(
            total_timesteps=solver_settings.total_timesteps,
            callback=callback,
            progress_bar=False,
        )
    except ValueError as error:
        if _policy_is_finite(model):
            raise
        return _training_diverged(
            spec, problem, scorer, solver_settings, callback.records, error,
            j_star=None if linear else probe_reference_cost(scorer, problem),
        )

    callback.restore_best()
    stopper = callback.stopper

    extra = {
        "action_clips": env.n_action_clips,
        "obs_scale": obs_scale,
        **stopper.extra(),
    }

    if preset.features_extractor_class is not None:
        extra.update(latent_extra(
            model, make_env("probe"), SB3EnvPolicy(model), horizon=settings.PROBE_HORIZON,
        ))

    if linear:
        K_final, bias = extract_linear_gain(model)
        K_final = obs_scale * K_final
        complexity = policy_complexity_linear(K_final)
        policy = LinearPolicy(K_final)
        extra["bias_ratio"] = float(
            np.linalg.norm(bias) / max(np.linalg.norm(K_final), 1e-12)
        )
    else:
        K_final = None
        complexity = policy_complexity_mlp(model)
        policy = SB3Policy(model, encoder=encoder, obs_scale=obs_scale)

    return SolverResult(
        K_final=K_final,
        policy=policy,
        records=callback.records,
        complexity=complexity,
        solver_settings=solver_settings,
        scorer=scorer,
        extra=extra,
        save_policy=None if linear else _sb3_saver(model, solver_settings, encoder, obs_scale),
        j_star=None if linear else probe_reference_cost(scorer, problem),
        j_final=_best_record_cost(callback.records, stopper),
    )


def _best_record_cost(records, stopper) -> float | None:
    """The probed cost of the policy the stopper kept, ``None`` if it kept none."""
    if stopper.best_iteration is None:
        return None
    return float(records[stopper.best_iteration].cost)


def _policy_is_finite(model) -> bool:
    """Whether every parameter of an SB3 policy is a finite number.

    Parameters
    ----------
    model : stable_baselines3.common.base_class.BaseAlgorithm
        The model that raised.

    Returns
    -------
    bool
    """
    import torch

    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.policy.parameters()
    )


def _training_diverged(
    spec: RunSpec, problem: Problem, scorer, solver_settings, records, failure,
    j_star: float | None = None,
) -> SolverResult:
    """Report a run whose policy diverged to ``nan``.

    Parameters
    ----------
    spec, problem : RunSpec, Problem
        The run.
    scorer : LQRBase
        The plant the curve was probed against.
    solver_settings : RLSolverSettings
        What was being trained.
    records : list[IterationRecord]
        Whatever the probe collected before the blow-up.
    failure : Exception
        The error raised.
    j_star : float or None, optional
        Reference cost, as for :attr:`SolverResult.j_star`.

    Returns
    -------
    SolverResult
    """
    from experiments.evaluate import TestResult

    steps = records[-1].env_steps if records else 0

    records = list(records) + [IterationRecord(
        iteration=len(records),
        env_steps=steps,
        total_time=records[-1].total_time if records else 0.0,
        cost=float("inf"),
        rank=None,
    )]

    def evaluate_test(**_) -> TestResult:
        """An infinite test cost."""
        return TestResult(
            summary={
                "test_cost_mean": float("inf"),
                "test_episodes": settings.TEST_EPISODES,
                "test_horizon": settings.TEST_HORIZON,
            },
            returns=np.full(settings.TEST_EPISODES, np.inf),
        )

    def trajectory() -> SimulationResult:
        """A blank trajectory."""
        return SimulationResult(
            x_traj=np.full((settings.FIGURE_HORIZON, problem.model.A.shape[0]), np.nan),
            u_traj=np.full((settings.FIGURE_HORIZON, problem.model.B.shape[1]), np.nan),
        )

    return SolverResult(
        K_final=None,
        policy=LinearPolicy(optimal_controller_gain(problem)),
        records=records,
        complexity={"rank": -1, "n_params": -1},
        solver_settings=solver_settings,
        scorer=scorer,
        extra={"training_diverged": str(failure).splitlines()[0], "converged": False},
        j_star=j_star,
        evaluate_test=evaluate_test,
        trajectory=trajectory,
    )


def _run_ppo_encoder(spec: RunSpec, problem: Problem) -> SolverResult:
    """PPO [4]_ behind a plant identified by N4SID [3]_.
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


# --- Chapter 6 ---------------------------------------


def _pixel_setup(spec: RunSpec):
    """Chapter 6 setup

    Returns
    -------
    tuple
        ``(K, j_star, horizon, action_repeat)``. ``K`` is the exact optimal
        gain at the agent rate.

    Raises
    ------
    ValueError
        If the configured ring cannot deliver what ``K`` demands.
    """
    import config as cfg
    from evaluation.pixel_metrics import agent_horizon
    from lqr_base.thruster_ring import check_thrust_calibration, optimal_coarse_gain

    K, P, rho = optimal_coarse_gain(
        cfg.double_integrator_model(), *cfg.default_costs(),
        cfg.DEFAULT_DT, cfg.DEFAULT_ACTION_REPEAT,
    )
    check_thrust_calibration(cfg.thruster_lifts()[1][0], K, cfg.DEFAULT_MAX_INITIAL_STATE)

    horizon = (
        settings.TRAIN_HORIZON if settings.TRAIN_HORIZON is not None
        else agent_horizon(rho)
    )

    # Exact expected infinite-horizon cost under perfect state feedback
    sigma_0 = np.diag(np.asarray(cfg.DEFAULT_MAX_INITIAL_STATE, dtype=float) ** 2 / 3.0)
    j_star = float(np.trace(P @ sigma_0))

    return K, j_star, int(horizon), int(cfg.DEFAULT_ACTION_REPEAT)


def _pixel_extra(action_clips: int, diagnostics: dict) -> dict:
    """Additional summary entries for pixel runs"""
    import config as cfg

    return {
        "action_clips": int(action_clips),
        **diagnostics,
        "action_repeat": int(cfg.DEFAULT_ACTION_REPEAT),
        "n_thrusters": int(cfg.DEFAULT_N_THRUSTERS),
        "thrust_scale": float(cfg.DEFAULT_THRUST_SCALE),
    }


def _pixel_protocol(env, policy, horizon: int):
    """The test protocol and the figure rollout for chapter 6"""
    from evaluation.pixel_metrics import env_monte_carlo_cost, rollout_env
    from experiments.evaluate import TestResult

    def evaluate_test(episodes: int | None = None, horizon_override: int | None = None,
                      **_ignored) -> TestResult:
        episodes = settings.TEST_EPISODES if episodes is None else episodes
        steps = horizon if horizon_override is None else horizon_override

        mean, stderr, _ = env_monte_carlo_cost(
            env, policy, n_episodes=episodes, horizon=steps
        )

        return TestResult(
            summary={
                "test_cost_mean": float(mean),
                "test_cost_stderr": float(stderr),
                "test_episodes": int(episodes),
                "test_horizon": int(steps),
            },
            returns=np.full(1, float(mean)),
        )

    return evaluate_test, lambda: rollout_env(env, policy, horizon)


def _run_oracle_coarse(spec: RunSpec, problem: Problem) -> SolverResult:
    """Chapter 6's baseline, the exact optimum at the agent's decision rate.
    """
    from evaluation.pixel_metrics import StatePolicy, env_monte_carlo_cost
    from lqr_base.thruster_ring import ThrusterOraclePolicy
    from reinforcement_learning.pixel_env import build_pixel_env

    K, j_star, horizon, repeat = _pixel_setup(spec)

    env = build_pixel_env(sub_seed(spec.seed, "pixel", spec.solver), horizon=horizon)
    policy = StatePolicy(ThrusterOraclePolicy(K, env.c))

    mean, _, diagnostics = env_monte_carlo_cost(
        env, policy, n_episodes=settings.PIXEL_PROBE_EPISODES, horizon=horizon
    )

    # No training
    records = [IterationRecord(
        iteration=0, env_steps=0, total_time=0.0,
        cost=float(mean), cost_estimate=None, rank=int(np.linalg.matrix_rank(K)),
    )]

    evaluate_test, trajectory = _pixel_protocol(env, policy, horizon)

    return SolverResult(
        K_final=None,
        policy=policy,
        records=records,
        complexity=policy_complexity_linear(K),
        solver_settings={"solver": "oracle_coarse", "action_repeat": repeat, "K": K},
        scorer=env,
        extra=_pixel_extra(env.n_action_clips, diagnostics),
        j_star=j_star,
        evaluate_test=evaluate_test,
        trajectory=trajectory,
    )


def _run_pixel_rl(spec: RunSpec, problem: Problem) -> SolverResult:
    """Train a PPO policy with different feature extractors on the pixel environment.

    PPO [4]_ used throughout.
    """
    import dataclasses as dc

    from evaluation.pixel_metrics import SB3EnvPolicy, env_monte_carlo_cost
    from evaluation.latent_probe import latent_extra
    from reinforcement_learning.callbacks import ProbeCallback
    from reinforcement_learning.pixel_env import build_pixel_env
    from reinforcement_learning.features import fit_extractor_weights
    from reinforcement_learning.solvers import build_model, pixel_presets

    K, j_star, horizon, repeat = _pixel_setup(spec)

    # Train env
    train_env = build_pixel_env(sub_seed(spec.seed, "pixel-train", spec.solver), horizon=horizon)

    # Test env
    score_env = build_pixel_env(sub_seed(spec.seed, "pixel-score", spec.solver), horizon=horizon)

    preset = pixel_presets()[spec.solver]

    weights = None
    if preset.freeze_extractor:
        from lqr_base.thruster_ring import ThrusterOraclePolicy

        fit_env = build_pixel_env(sub_seed(spec.seed, "pixel-fit", spec.solver), horizon=horizon)
        oracle = ThrusterOraclePolicy(K, fit_env.c)
        weights = fit_extractor_weights(
            preset, fit_env,
            _explore(lambda observation, info: oracle(info["state"]), spec.seed, 1.0),
            n_steps=settings.PRETRAIN_STEPS,
            epochs=settings.PRETRAIN_EPOCHS,
        )

    solver_settings = dc.replace(
        preset,
        total_timesteps=settings.PIXEL_RL_MAX_AGENT_STEPS,
        eval_freq=settings.RL_EVAL_FREQ,
        pretrained_extractor=weights,
    )

    def evaluate(model):
        mean, _, _ = env_monte_carlo_cost(
            score_env, SB3EnvPolicy(model),
            n_episodes=settings.PIXEL_PROBE_EPISODES,
            horizon=horizon,
        )
        return mean, None

    callback = ProbeCallback(
        evaluate,
        eval_freq=solver_settings.eval_freq,
        env_steps_scale=repeat,
        env_steps_offset=settings.PRETRAIN_STEPS * repeat if weights is not None else 0,
        stopper=_stopper("rl"),
    )

    model = build_model(train_env, solver_settings, seed=spec.seed)
    model.learn(
        total_timesteps=solver_settings.total_timesteps,
        callback=callback,
        progress_bar=False,
    )
    callback.restore_best()

    policy = SB3EnvPolicy(model)

    _, _, diagnostics = env_monte_carlo_cost(
        score_env, policy,
        n_episodes=settings.PIXEL_PROBE_EPISODES,
        horizon=horizon,
    )

    extra = _pixel_extra(callback.n_action_clips, diagnostics)
    extra.update(latent_extra(model, score_env, policy))
    extra.update(callback.stopper.extra())

    evaluate_test, trajectory = _pixel_protocol(score_env, policy, horizon)

    return SolverResult(
        K_final=None,
        policy=policy,
        records=callback.records,
        complexity=policy_complexity_mlp(model),
        solver_settings=solver_settings,
        scorer=score_env,
        extra=extra,
        save_policy=_sb3_saver(model, solver_settings, None),
        j_star=j_star,
        evaluate_test=evaluate_test,
        trajectory=trajectory,
        j_final=_best_record_cost(callback.records, callback.stopper),
    )


def _sb3_saver(model, solver_settings, encoder=None, obs_scale: float = 1.0) -> Callable[[Any], dict]:
    """A saver saving a stable-baselines3 model next to its record.
    """
    def save(directory) -> dict:
        model.save(directory / "policy.zip")

        manifest = {
            "kind": "sb3",
            "algo": solver_settings.algo,
            "file": "policy.zip",
            "deterministic": True,
            "obs_scale": float(obs_scale),
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
    "ppo_bottleneck": _run_rl,
    "ppo_encoder_scalable": _run_rl,
    "oracle_coarse": _run_oracle_coarse,
    "ppo_pixel_fc": _run_pixel_rl,
    "ppo_pixel_posvel": _run_pixel_rl,
    "ppo_pixel_ortho": _run_pixel_rl,
    "ppo_pixel_estimator": _run_pixel_rl,
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
