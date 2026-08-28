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
        A=np.array([[0.0, 1.0],
                    [0.0, 0.0]]),
        B=np.array([[0.0],
                    [1.0]]),
    )


def default_costs() -> tuple[np.ndarray, np.ndarray]:
    """The cost weights every experiment is scored against.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Q, R)`` of shapes ``(n, n)`` and ``(m, m)``.
    """
    return np.eye(2), np.array([[1.0]])


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

    # Normalize
    N = N / np.linalg.norm(N, ord='fro')

    return K_low_rank + alpha * N
