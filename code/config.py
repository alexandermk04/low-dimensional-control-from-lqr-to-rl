"""Definitions for shared experiment setup.

The single relevant file where the double integrator is built, to
allow comparisons between different evaluation scripts without needing to
copy them over.

Notes
-----
The matrices are *continuous* time. Every LQR class discretizes them itself.
"""

import numpy as np

from basetypes import StateSpaceModel

# Sampling interval, in seconds.
DEFAULT_DT = 0.01

# Elementwise bound of the box initial states are drawn from
DEFAULT_MAX_INITIAL_STATE = np.array([10.0, 1.0])

# Length of the trajectory rollouts the figures show.
DEFAULT_SIM_STEPS = 2500

DEFAULT_A = np.array([[0.0, 1.0],
                       [0.0, 0.0]])
DEFAULT_B = np.array([[0.0],
                       [1.0]])

def double_integrator_model() -> StateSpaceModel:
    """The standard double integrator as its state space model.

    A unit-mass point on a line: the state is ``[position, velocity]`` and the
    input is an acceleration.

    Returns
    -------
    StateSpaceModel
        Continuous-time ``(A, B)`` with ``n = 2``, ``m = 1``.
    """
    return StateSpaceModel(
        A=DEFAULT_A,
        B=DEFAULT_B,
    )


def default_costs(
    observation_dim: int,
    action_dim: int,
    state_weight: float = 1.0,
    input_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The cost weights every experiment is scored against.

    Parameters
    ----------
    observation_dim : int
        ``p``, the observation dimension (``n`` for the plain plant).
    action_dim : int
        ``q``, the action dimension (``m`` for the plain plant).
    state_weight, input_weight : float, optional
        Scalar multiples of the identity. Default to ``1.0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Q, R)`` as diagonals of shapes ``(p,)`` and ``(q,)``.
    """
    return (
        np.full(observation_dim, float(state_weight)),
        np.full(action_dim, float(input_weight)),
    )


def make_lifts(
    inflated_input_dim: int,
    inflated_output_dim: int,
    model: StateSpaceModel,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the observation and actuation lifts of an inflated system.

    The lifts are *orthonormal* as random ones make
    the model-free solvers diverge.

    Parameters
    ----------
    inflated_input_dim : int
        ``p``, the number of input sensors.
    inflated_output_dim : int
        ``q``, the number of output actuators.
    model : StateSpaceModel
        Plant being inflated, used only for its dimensions.
    rng : np.random.Generator
        Source of randomness, so the lifts are reproducible.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(U_1, U_2)`` of shapes ``(p, n)`` and ``(m, q)``, so that the
        controller observes ``y = U_1 x`` and actuates ``u = U_2 v``.
    """
    n = model.A.shape[0]
    m = model.B.shape[1]

    U_1, _ = np.linalg.qr(rng.standard_normal((inflated_input_dim, n)))
    U_2_full, _ = np.linalg.qr(rng.standard_normal((inflated_output_dim, m)))

    return U_1, U_2_full.T


def make_full_rank_K0(
    U_1: np.ndarray,
    U_2: np.ndarray,
    rng: np.random.Generator,
    K_start: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Build a stabilizing initial gain that is full rank.

    Parameters
    ----------
    U_1, U_2 : np.ndarray
        The lifts, as returned by :func:`make_lifts`.
    rng : np.random.Generator
        Source of randomness for the null component.
    K_start : np.ndarray or None, optional
        Stabilizing gain on the true plant, of shape ``(m, n)``. Defaults to
        ``[[0.1, 0.5]]``.
    alpha : float, optional
        Size of the full-rank component, in Frobenius norm. Defaults to ``0.5``.

    Returns
    -------
    np.ndarray
        ``K_0`` of shape ``(q, p)`` with
        ``U_2 @ K_0 @ U_1 == K_start`` up to numerical error.
    """
    if K_start is None:
        K_start = np.array([[0.1, 0.5]])

    K_low_rank = np.linalg.pinv(U_2) @ K_start @ np.linalg.pinv(U_1)

    G = rng.standard_normal((U_2.shape[1], U_1.shape[0]))

    # Create N, such that U_2 @ N @ U_1 == 0, so
    # adding it changes the representation without changing the effective gain.
    N = G - np.linalg.pinv(U_2) @ (U_2 @ G @ U_1) @ np.linalg.pinv(U_1)

    norm_N = np.linalg.norm(N, ord='fro')

    # Edge case for an already full rank K_low_rank, where the nullspace is empty
    if norm_N <= 1e-8 * max(np.linalg.norm(G, ord='fro'), 1.0):
        return K_low_rank

    return K_low_rank + alpha * N / norm_N


# --- Chapter 6: pixel observations and ring of thrusters ---------------------

DEFAULT_FRAME_SIZE = 96
DEFAULT_FRAME_STACK = 2

# Simulation steps one agent command is held for
DEFAULT_ACTION_REPEAT = 30

# Visible half-track
DEFAULT_VIEW_SCALE = 1.2

# Number of thrusters over a half circle
DEFAULT_N_THRUSTERS = 256
DEFAULT_THRUSTER_PHASE = 0.0

DEFAULT_THRUST_SCALE = 26 / 128


def thruster_lifts(
    n_thrusters: int = DEFAULT_N_THRUSTERS,
    phase: float = DEFAULT_THRUSTER_PHASE,
    thrust_scale: float = DEFAULT_THRUST_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Observation and actuation lifts of the chapter-6 thruster ring.

    Parameters
    ----------
    n_thrusters : int, optional
        Number of thrusters, ``N``. Defaults to :data:`DEFAULT_N_THRUSTERS`.
    phase : float, optional
        Rigid rotation of the ring, in radians. Defaults to
        :data:`DEFAULT_THRUSTER_PHASE`.
    thrust_scale : float, optional
        Force of a fully commanded thruster. Defaults to
        :data:`DEFAULT_THRUST_SCALE`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(U_1, U_2)`` of shapes ``(2, 2)`` and ``(1, N)``.
    """
    from lqr_base.thruster_ring import thruster_ring_lift

    return thruster_ring_lift(n_thrusters, phase, thrust_scale)
