"""Shared cost metrics for both LQR and RL runs.

Conventions
-----------
Every function here takes the raw gain ``K`` in the controller's space, of
shape ``(q, p)``. The cost is charged in that space too, but all exact
quantities are computed on the true ``n``-dimensional state, using
:func:`~lqr_base.simple_lqr.closed_loop_matrix` and
:func:`~lqr_base.simple_lqr.stage_cost_matrix`, so they are
calculatable at larger scales.
"""

from typing import Callable, Protocol

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from lqr_base.simple_lqr import (
    LQRDiscrete,
    apply_weight,
    closed_loop_matrix,
    initial_state_covariance,
    stage_cost_matrix,
)

__all__ = [
    "initial_state_covariance",
    "spectral_radius",
    "closed_loop_value_matrix",
    "expected_true_cost",
    "finite_horizon_expected_cost",
    "process_noise_covariance",
    "measurement_cost_offset",
    "average_cost",
    "choose_horizon",
    "Policy",
    "LinearPolicy",
    "monte_carlo_returns",
    "monte_carlo_cost",
]


def _closed_loop(lqr: LQRDiscrete, K: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Closed-loop state matrix, optionally discounted.

    A ``gamma``-discounted LQR problem on ``(A, B, Q, R)`` is exactly the
    undiscounted problem on ``(sqrt(gamma) A, sqrt(gamma) B, Q, R)``, so a
    discount is applied here by scaling and nowhere else.
    """
    return np.sqrt(gamma) * closed_loop_matrix(lqr, K)


def spectral_radius(lqr: LQRDiscrete, K: np.ndarray, gamma: float = 1.0) -> float:
    """Largest closed-loop eigenvalue magnitude; ``< 1`` means stable.

    Parameters
    ----------
    lqr : LQRDiscrete
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.
    gamma : float, optional
        Discount factor. Defaults to ``1.0``, i.e. undiscounted.

    Returns
    -------
    float
        ``max |eig(A - B U_2 K U_1)|``, scaled by ``sqrt(gamma)``.
    """
    return float(np.max(np.abs(np.linalg.eigvals(_closed_loop(lqr, K, gamma)))))


def closed_loop_value_matrix(
    lqr: LQRDiscrete, K: np.ndarray, gamma: float = 1.0
) -> np.ndarray | None:
    """Value matrix ``P`` of the closed loop, or ``None`` if it is unstable.

    ``P`` solves ``P = Q + (K U_1)' R (K U_1) + A_cl' P A_cl``, so that the
    infinite-horizon cost from ``x_0`` is ``x_0' P x_0``.

    Parameters
    ----------
    lqr : LQRDiscrete
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.
    gamma : float, optional
        Discount factor. Defaults to ``1.0``.

    Returns
    -------
    np.ndarray or None
        ``P`` of shape ``(n, n)``, or ``None`` when the closed loop is not
        stable.
    """
    A_cl = _closed_loop(lqr, K, gamma)

    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1:
        return None

    return solve_discrete_lyapunov(A_cl.T, stage_cost_matrix(lqr, K))


def expected_true_cost(lqr: LQRDiscrete, K: np.ndarray, gamma: float = 1.0) -> float:
    """Expected infinite-horizon cost over the initial state distribution.

    Parameters
    ----------
    lqr : LQRDiscrete
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.
    gamma : float, optional
        Discount factor. Defaults to ``1.0``.

    Returns
    -------
    float
        ``trace(P @ Sigma_0)``, or ``np.inf`` if the closed loop is unstable.

    Notes
    -----
    Noise-free by construction: ``P`` describes the nominal closed loop. Under
    disturbance or measurement noise use
    :func:`finite_horizon_expected_cost` or
    :func:`average_cost` for the steady state.
    """
    P = closed_loop_value_matrix(lqr, K, gamma)

    if P is None:
        return np.inf

    return float(np.trace(P @ initial_state_covariance(lqr)))


def process_noise_covariance(
    lqr: LQRDiscrete, K: np.ndarray | None = None
) -> np.ndarray | None:
    """Covariance ``W`` of the additive state noise the closed loop sees.

    Parameters
    ----------
    lqr : LQRDiscrete
        The plant whose noise is processed.
    K : np.ndarray or None, optional
        The gain.

    Returns
    -------
    np.ndarray or None
        ``W`` of shape ``(n, n)``, or ``None`` when no closed form exists.

    Notes
    -----
    Only the channels that are additive and state-independent have a constant
    covariance:

    - **disturbance**, ``"gaussian"``: ``sigma**2 I``. The ``"sinusoidal"`` and
      ``"constant"`` variants are not zero-mean white noise and are reported as
      ``None``.
    - **measurement**: the observation error ``xi`` reaches the state as
      ``-B @ project_action(K) @ xi``, giving ``sigma**2 B M M' B'`` with
      ``M = project_action(K)``.
    - **control** noise is multiplicative and **parametric** noise perturbs
      ``A`` and ``B`` themselves, so both scale with the state and neither is a
      constant ``W``. Either one active returns ``None``; use
      :func:`monte_carlo_cost` there.
    """
    if not lqr.noise:
        return np.zeros((lqr.A.shape[0], lqr.A.shape[0]))

    if lqr.control_noise is not None or lqr.parametric_noise is not None:
        return None

    W = np.zeros((lqr.A.shape[0], lqr.A.shape[0]))

    if lqr.disturbance_noise is not None:
        if lqr.disturbance_noise.disturbance_type != "gaussian":
            return None
        W = W + lqr.disturbance_noise.disturbance_noise_std ** 2 * np.eye(lqr.A.shape[0])

    if lqr.measurement_noise is not None:
        if K is None:
            raise ValueError(
                "Measurement noise requires the raw gain K."
            )
        # project_action is linear, so applying it to the whole matrix maps
        # every observation direction at once: (q, p) -> (m, p).
        M = lqr.project_action(K)
        W = W + lqr.measurement_noise.measurement_noise_std ** 2 * (lqr.B @ M @ M.T @ lqr.B.T)

    return W


def measurement_cost_offset(lqr: LQRDiscrete, K: np.ndarray | None = None) -> float:
    """Constant per-step control cost added by measurement noise.

    Measurement noise does two things: it perturbs the state trajectory, which
    :func:`process_noise_covariance` describes. But it also inflates the
    control cost charged at every step: the controller output is
    ``v = -K U_1 x - K xi``, so ``v' R v`` picks up a term ``xi' K' R K xi``
    whose expectation is a constant, independent of the state and therefore
    never visible in the covariance recursion.

    Parameters
    ----------
    lqr : LQRDiscrete
        System whose noise channels are inspected.
    K : np.ndarray or None, optional
        Raw gain in the controller's space. Required when measurement noise is
        active.

    Returns
    -------
    float
        ``sigma**2 * trace(K' R K)``, or ``0.0`` when there is no measurement
        noise.

    Raises
    ------
    ValueError
        If measurement noise is active but ``K`` was not supplied.
    """
    if not lqr.noise or lqr.measurement_noise is None:
        return 0.0

    if K is None:
        raise ValueError(
            "Measurement noise requires the raw gain K."
        )

    return float(lqr.measurement_noise.measurement_noise_std ** 2 * np.sum(K * apply_weight(lqr.R, K)))


def finite_horizon_expected_cost(
    lqr: LQRDiscrete,
    K: np.ndarray,
    horizon: int,
    gamma: float = 1.0,
) -> float:
    """Expected cost accumulated over ``horizon`` steps.

    Propagates the state covariance forward and charges the stage cost against
    it at every step.

    Parameters
    ----------
    lqr : LQRDiscrete
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.
    horizon : int
        Number of steps to accumulate.
    gamma : float, optional
        Discount factor. Defaults to ``1.0``, i.e. undiscounted.

    Returns
    -------
    float
        ``sum_t trace(M Sigma_t)`` with the stage weight ``M`` of
        :func:`~lqr_base.simple_lqr.stage_cost_matrix` and
        ``Sigma_{t+1} = A_cl Sigma_t A_cl' + W``.

    Raises
    ------
    ValueError
        If the active noise channels admit no closed form. Use
        :func:`monte_carlo_cost` in that case.
    """
    W = process_noise_covariance(lqr, K)
    if W is None:
        raise ValueError(
            "The active noise channels are state-dependent and have no closed "
            "form. Estimate this with monte_carlo_cost() instead."
        )

    A_cl = _closed_loop(lqr, K, gamma)
    stage = stage_cost_matrix(lqr, K)

    # Measurement noise also inflates the control cost by a state-independent
    # constant, which the covariance recursion below cannot see.
    offset = measurement_cost_offset(lqr, K)

    sigma = initial_state_covariance(lqr)
    total = 0.0
    discount = 1.0

    for _ in range(horizon):
        total += discount * (float(np.trace(stage @ sigma)) + offset)
        sigma = A_cl @ sigma @ A_cl.T + W
        discount *= gamma

    return total


def average_cost(lqr: LQRDiscrete, K: np.ndarray) -> float:
    """Steady-state cost per step under noise.

    The infinite-horizon *sum* diverges once the loop is driven by process
    noise; the finite quantity is the long-run average, ``trace(P W)``.

    Parameters
    ----------
    lqr : LQRDiscrete
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.

    Returns
    -------
    float
        ``trace(P @ W)``, or ``np.inf`` if the closed loop is unstable.

    Raises
    ------
    ValueError
        If the active noise channels admit no closed form.
    """
    P = closed_loop_value_matrix(lqr, K)
    if P is None:
        return np.inf

    W = process_noise_covariance(lqr, K)
    if W is None:
        raise ValueError(
            "The active noise channels are state-dependent and have no closed form."
        )

    return float(np.trace(P @ W)) + measurement_cost_offset(lqr, K)


def choose_horizon(lqr: LQRDiscrete, K: np.ndarray, tol: float = 1e-3) -> int:
    """Horizon at which the closed loop has decayed to ``tol``.

    Parameters
    ----------
    lqr : LQRDiscrete
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.
    tol : float, optional
        Residual fraction of the initial transient. Defaults to ``1e-3``.

    Returns
    -------
    int
        Smallest ``H`` with ``rho**H < tol``, where ``rho`` is the closed-loop
        spectral radius.

    Raises
    ------
    ValueError
        If the closed loop is unstable.
    """
    rho = spectral_radius(lqr, K)

    if rho >= 1:
        raise ValueError(
            f"Closed loop is unstable (rho = {rho:.4f}); no decay horizon exists."
        )

    return int(np.ceil(np.log(tol) / np.log(rho)))


class Policy(Protocol):
    """Anything that maps an observation to a control.
    """

    def __call__(self, y: np.ndarray) -> np.ndarray:
        ...


class LinearPolicy:
    """Adapter presenting a gain matrix as a :class:`Policy`.

    Parameters
    ----------
    K : np.ndarray
        Gain in the controller's own space, applied as ``v = -K y``.

    Attributes
    ----------
    K : np.ndarray
        The gain, unchanged.
    """

    def __init__(self, K: np.ndarray):
        self.K = K

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Return ``-K @ y``."""
        return -self.K @ y


def monte_carlo_returns(
    lqr: LQRDiscrete,
    policy: Policy,
    n_episodes: int,
    horizon: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Accumulated cost of each individual rollout.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to roll out. Its RNG state and step counter are restored
        afterwards.
    policy : Policy
        Callable mapping an observation to a control in the controller's space.
    n_episodes : int
        Number of independent rollouts. Requires ``resample_initial_state=True``
        to actually sample the initial-state distribution.
    horizon : int
        Steps per rollout.
    gamma : float, optional
        Discount. Defaults to ``1.0``.

    Returns
    -------
    np.ndarray
        Accumulated cost per episode, of shape ``(n_episodes,)``.
    """
    state_snapshot = lqr.state_rng.bit_generator.state
    noise_snapshot = lqr.noise_rng.bit_generator.state
    steps_snapshot = lqr.env_steps
    episodes_snapshot = lqr.episodes

    returns = np.empty(n_episodes)

    try:
        for episode in range(n_episodes):
            lqr.reset()

            total = 0.0
            discount = 1.0

            for _ in range(horizon):
                result = lqr.apply_control(policy(lqr.observe()))
                total += discount * result.cost
                discount *= gamma

            returns[episode] = total
    finally:
        lqr.state_rng.bit_generator.state = state_snapshot
        lqr.noise_rng.bit_generator.state = noise_snapshot
        lqr.env_steps = steps_snapshot
        lqr.episodes = episodes_snapshot

    return returns


def monte_carlo_cost(
    lqr: LQRDiscrete,
    policy: Policy,
    n_episodes: int,
    horizon: int,
    gamma: float = 1.0,
) -> tuple[float, float]:
    """Mean and standard error of :func:`monte_carlo_returns`.

    Parameters
    ----------
    lqr, policy, n_episodes, horizon, gamma
        As for :func:`monte_carlo_returns`.

    Returns
    -------
    tuple[float, float]
        ``(mean, standard_error)`` of the per-episode accumulated cost.
    """
    returns = monte_carlo_returns(lqr, policy, n_episodes, horizon, gamma)

    mean = float(np.mean(returns))
    stderr = (
        float(np.std(returns, ddof=1) / np.sqrt(n_episodes))
    )

    return mean, stderr
