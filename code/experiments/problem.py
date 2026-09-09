"""The problem one run is posed on, and the plants it needs.

Contains dataclasses and functions to offer a unified entrypoint.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

import settings
from basetypes import StateSpaceModel
from config import (
    DEFAULT_DT,
    DEFAULT_MAX_INITIAL_STATE,
    default_costs,
    double_integrator_model,
    make_full_rank_K0,
    make_lifts,
)
from evaluation.metrics import choose_horizon, expected_true_cost
from seeding import sub_rng, sub_seed
from lqr_base.inflated_lqr import LQRInflated, LQRInflatedSettings
from lqr_base.noise import LQRNoiseSettings
from lqr_base.simple_lqr import (
    LQRBase,
    LQRDiscrete,
    LQRDiscreteSettings,
    get_optimal_K_and_cost_discrete,
)

__all__ = ["RunSpec", "Problem", "build_plant", "make_problem", "PLAIN_K0"]


PLAIN_K0 = np.array([[0.1, 0.5]])


@dataclass
class RunSpec:
    """One training run: a solver, a seed, and the condition setting it belongs to.

    Attributes
    ----------
    experiment : str
        Name of the experiment.
    solver : str
        Key into :data:`experiments.registry.SOLVER_REGISTRY`.
    seed : int
        One of ``settings.SEEDS``.
    inflation : tuple[int, int] or None
        ``(p, q)``, or ``None`` for the base double integrator.
    rollout_length : int or None
        ``l``, steps per rollout.
    n_trajectories : int or None
        ``m``, rollouts per gradient step.
    iterations : int
        Solver allowed iterations.
    condition : dict
        The experiment condition settings, e.g. ``{"l": 500, "m": 200}``
    """

    experiment: str
    solver: str
    seed: int
    inflation: tuple[int, int] | None = None
    rollout_length: int | None = None
    n_trajectories: int | None = None
    iterations: int = settings.SOLVER_ITERATIONS
    condition: dict = field(default_factory=dict)

    def hyperparams(self) -> dict:
        """This solver's entry in ``settings.SOLVER_HYPERPARAMS``."""
        return dict(settings.SOLVER_HYPERPARAMS.get(self.solver, {}))


@dataclass
class Problem:
    """Data needed acroos solvers for one ``(seed, inflation)`` experiment setting.

    Attributes
    ----------
    seed : int
    inflation : tuple[int, int] or None
    model : StateSpaceModel
    Q, R : np.ndarray
    dt : float
    max_initial_state : np.ndarray
    U_1, U_2 : np.ndarray or None
        The lifts, ``None`` for the plain plant.
    K_0 : np.ndarray
        Stabilizing initial gain in the controller's space.
    K_star : np.ndarray
        Optimal gain on the true plant.
    j_star : float
        Optimal expected infinite-horizon cost.
    horizon : int
        At what point the closed-loop has decayed to a tolerance. 
    action_limit : float
        Half-width of the RL action box.
    """

    seed: int
    inflation: tuple[int, int] | None
    model: StateSpaceModel
    Q: np.ndarray
    R: np.ndarray
    dt: float
    max_initial_state: np.ndarray
    U_1: np.ndarray | None
    U_2: np.ndarray | None
    K_0: np.ndarray
    K_star: np.ndarray
    j_star: float
    horizon: int
    action_limit: float


def build_plant(
    problem: Problem,
    purpose: str,
    noise_settings: LQRNoiseSettings | None = None,
    resample_initial_state: bool = True,
) -> LQRBase:
    """Construct one plant of a problem.

    Parameters
    ----------
    problem : Problem
        The experiment this system belongs to.
    purpose : str
        What the plant is for, e.g. ``"train:mf_npg"``, used for seed creation.
    noise_settings : LQRNoiseSettings or None, optional
        Left ``None`` for training.
    resample_initial_state : bool, optional
        ``False`` pins the plant to a single initial state.

    Returns
    -------
    LQRBase
        :class:`~lqr_base.inflated_lqr.LQRInflated` when the problem is
        inflated, otherwise :class:`~lqr_base.simple_lqr.LQRDiscrete`.
    """
    common = dict(
        model=problem.model,
        Q=problem.Q,
        R=problem.R,
        max_initial_state=problem.max_initial_state,
        resample_initial_state=resample_initial_state,
        noise_settings=noise_settings,
        seed_seq=sub_seed(problem.seed, "plant", purpose),
        dt=problem.dt,
    )

    if problem.inflation is None:
        return LQRDiscrete(settings=LQRDiscreteSettings(**common))

    return LQRInflated(
        settings=LQRInflatedSettings(**common, U_1=problem.U_1, U_2=problem.U_2)
    )


@lru_cache(maxsize=None)
def make_problem(seed: int, inflation: tuple[int, int] | None = None) -> Problem:
    """Build the shared context of one ``(seed, inflation)`` experiment.

    Cached, so the solvers of one cell have the same data.

    Parameters
    ----------
    seed : int
        One of ``settings.SEEDS``.
    inflation : tuple[int, int] or None, optional
        ``(p, q)``, or ``None`` for the plain plant.

    Returns
    -------
    Problem
    """
    model = double_integrator_model()
    Q, R = default_costs()
    max_initial_state = DEFAULT_MAX_INITIAL_STATE.copy()

    U_1 = U_2 = None
    if inflation is None:
        K_0 = PLAIN_K0.copy()
    else:
        matrix_rng = sub_rng(seed, "lifts")
        U_1, U_2 = make_lifts(*inflation, model, matrix_rng)
        K_0 = make_full_rank_K0(U_1, U_2, matrix_rng, alpha=settings.K0_ALPHA)

    problem = Problem(
        seed=seed,
        inflation=inflation,
        model=model,
        Q=Q,
        R=R,
        dt=DEFAULT_DT,
        max_initial_state=max_initial_state,
        U_1=U_1,
        U_2=U_2,
        K_0=K_0,
        K_star=np.zeros((1, 1)),
        j_star=np.inf,
        horizon=0,
        action_limit=0.0,
    )

    # Creating an uninflated plant to calculate the optima
    plain = LQRDiscrete(settings=LQRDiscreteSettings(
        model=model, Q=Q, R=R,
        max_initial_state=max_initial_state,
        resample_initial_state=True,
        noise_settings=None,
        seed_seq=sub_seed(seed, "plant", "baseline"),
        dt=DEFAULT_DT,
    ))
    K_star, _ = get_optimal_K_and_cost_discrete(plain)

    reference = build_plant(problem, "reference")
    reference.reset()

    problem.K_star = K_star
    problem.j_star = expected_true_cost(reference, K_star)
    problem.horizon = choose_horizon(reference, K_star)

    to_action = K_0 @ U_1 if U_1 is not None else K_star
    peak = float(np.max(np.abs(to_action) @ np.abs(max_initial_state)))
    problem.action_limit = float(10.0 * max(peak, 1.0))

    return problem
