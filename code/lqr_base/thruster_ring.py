"""Thruster-ring actuation and the coarse-rate LQR oracle.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

from basetypes import StateSpaceModel

from lqr_base.simple_lqr import continuous_to_discrete

__all__ = [
    "thruster_ring_lift",
    "allocate_thrust",
    "lifted_cost_weights",
    "optimal_coarse_gain",
    "check_thrust_calibration",
    "ThrusterOraclePolicy",
]


def thruster_ring_lift(
    n_thrusters: int,
    phase: float = 0.0,
    thrust_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift matrices of a ring of ``N`` thrusters at ``theta_i = pi * i / N + phase``.

    Parameters
    ----------
    n_thrusters : int
        Number of thrusters, ``N >= 1``.
    phase : float, optional
        Rotation of the ring, in radians.
    thrust_scale : float, optional
        Force produced by a fully commanded thruster.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(U_1, U_2)`` of shapes ``(2, 2)`` and ``(1, N)``, where ``U_2`` holds
        the along-rail effectiveness ``thrust_scale * cos(theta_i)``.
    """
    if n_thrusters < 1:
        raise ValueError(f"Need at least one thruster, got {n_thrusters}.")

    c = np.cos(np.pi * np.arange(n_thrusters) / n_thrusters + phase)
    c[np.abs(c) < 1e-12] = 0.0

    return np.eye(2), thrust_scale * c[None, :]


def allocate_thrust(a: float, c: np.ndarray) -> np.ndarray:
    """Spread a desired acceleration over the ring with minimum energy.

    Parameters
    ----------
    a : float
        Desired along-rail acceleration.
    c : np.ndarray
        Effectiveness vector of shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Least-norm commands ``a * c / ||c||^2``, clipped into ``[-1, 1]``.
    """
    return np.clip(a * c / (c @ c), -1.0, 1.0)


def lifted_cost_weights(
    A_d: np.ndarray,
    B_d: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact cost weights of one agent step under action repeat.

    Parameters
    ----------
    A_d, B_d : np.ndarray
        Discrete-time plant at the simulation rate.
    Q, R : np.ndarray
        Stage cost weights of the underlying problem.
    repeat : int
        Simulation steps one command is held for, ``k >= 1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(Q_bar, N_bar, R_bar, A_k, B_k)``, the accumulated weights together
        with the coarse-rate plant ``x_next = A_k x + B_k u``.
    """
    if repeat < 1:
        raise ValueError(f"Action repeat must be positive, got {repeat}.")

    n, m = B_d.shape

    Q_bar = np.zeros((n, n))
    N_bar = np.zeros((n, m))
    R_bar = np.zeros((m, m))

    A_j = np.eye(n)          # A_d ** j
    G_j = np.zeros_like(B_d)  # sum_{l < j} A_d ** l @ B_d

    for _ in range(repeat):
        Q_bar += A_j.T @ Q @ A_j
        N_bar += A_j.T @ Q @ G_j
        R_bar += G_j.T @ Q @ G_j + R

        G_j = A_d @ G_j + B_d
        A_j = A_d @ A_j

    return Q_bar, N_bar, R_bar, A_j, G_j


def optimal_coarse_gain(
    model: StateSpaceModel,
    Q: np.ndarray,
    R: np.ndarray,
    dt: float,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Optimal gain when every command is held for ``repeat`` simulation steps.

    Parameters
    ----------
    model : StateSpaceModel
        The plant, in continuous time.
    Q, R : np.ndarray
        Stage cost weights of the underlying problem.
    dt : float
        Simulation interval.
    repeat : int
        Simulation steps per agent step.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        The gain ``K`` of shape ``(m, n)``, the Riccati solution ``P`` and the
        closed-loop spectral radius at the agent rate.
    """
    A_d, B_d = continuous_to_discrete(model.A, model.B, dt=dt)
    Q_bar, N_bar, R_bar, A_k, B_k = lifted_cost_weights(A_d, B_d, Q, R, repeat)

    # The cross term N_bar is removed by a change of input variable.
    R_inv = np.linalg.inv(R_bar)
    P = solve_discrete_are(
        A_k - B_k @ R_inv @ N_bar.T, B_k, Q_bar - N_bar @ R_inv @ N_bar.T, R_bar,
    )
    K = np.linalg.solve(R_bar + B_k.T @ P @ B_k, B_k.T @ P @ A_k + N_bar.T)

    rho = float(np.max(np.abs(np.linalg.eigvals(A_k - B_k @ K))))

    return K, P, rho


def check_thrust_calibration(
    c: np.ndarray,
    K: np.ndarray,
    max_initial_state: np.ndarray,
    safety: float = 1.5,
) -> None:
    """Raise if the ring cannot deliver the demanded thrust without clipping.

    Parameters
    ----------
    c : np.ndarray
        Effectiveness vector of shape ``(N,)``.
    K : np.ndarray
        Gain of shape ``(1, n)``.
    max_initial_state : np.ndarray
        Bound of the initial-state box, of shape ``(n,)``.
    safety : float, optional
        Required ratio of the unclipped least-norm limit to the peak demand.
    """
    n = len(max_initial_state)
    corners = np.array(np.meshgrid(*([[-1.0, 1.0]] * n))).T.reshape(-1, n) * max_initial_state
    peak = float(np.max(np.abs(corners @ K.T)))

    # Largest acceleration the least-norm allocation reaches before clipping.
    limit = float(c @ c) / float(np.abs(c).max())

    if limit < safety * peak:
        raise ValueError(
            f"Thrust scale too small: the ring reaches {limit:.3f} without clipping, "
            f"the controller demands up to {peak:.3f} (safety factor {safety})."
        )


class ThrusterOraclePolicy:
    """The LQR solution ``a = -K x``, executed through the thruster ring.

    Parameters
    ----------
    K : np.ndarray
        Gain acting on the true state, of shape ``(1, n)``.
    c : np.ndarray
        Effectiveness vector of shape ``(N,)``.
    """

    def __init__(self, K: np.ndarray, c: np.ndarray):
        self.K = np.asarray(K, dtype=float)
        self.c = np.asarray(c, dtype=float)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Thruster commands in ``[-1, 1]`` for the true state ``x``."""
        return allocate_thrust(-(self.K @ x).item(), self.c)
