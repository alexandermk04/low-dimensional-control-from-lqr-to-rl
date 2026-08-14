"""Small dataclasses to unify the representation of LQR-related data.

Notes
-----
Throughout the repository the feedback convention is ``u = -K x``.
"""

from dataclasses import dataclass
import numpy as np

# Global seed for the initial state sampling of the LQR, can be changed
SEED = 2023

@dataclass
class StateSpaceModel:
    """Continuous-time linear system ``x' = A x + B u``.

    The matrices are stored in continuous time. Every LQR class in
    :mod:`lqr_base` discretizes them itself via
    :func:`lqr_base.simple_lqr.continuous_to_discrete`, so callers should not
    pass pre-discretized matrices.

    Attributes
    ----------
    A : np.ndarray
        State matrix of shape ``(n, n)``.
    B : np.ndarray
        Input matrix of shape ``(n, m)``.
    """

    A: np.ndarray
    B: np.ndarray

@dataclass
class StepResult:
    """One simulation step, returning *observations* by the controller, as well as 
    the actual *states*.

    The distinction matters as soon as noise or a dimension inflation is
    active: the controller only ever gets ``y_t`` and only ever emits ``v_t``,
    while the cost and the dynamics are evaluated on the true ``x_t`` / ``u_t``.

    Attributes
    ----------
    y_t : np.ndarray
        Observation handed to the policy. Equals ``x_t`` for a noise-free
        :class:`~lqr_base.simple_lqr.LQRDiscrete`, and ``U_1 @ x_t``
        for :class:`~lqr_base.inflated_lqr.LQRInflated`.
    v_t : np.ndarray
        Output of the policy, ``-K @ y_t``, after control noise. For the
        inflated system this lives in the inflated action space.
    x_t : np.ndarray
        True state of shape ``(n,)`` at the start of the step.
    u_t : np.ndarray
        Control actually applied to the true system, of shape ``(m,)``.
    cost : float
        Stage cost ``x_t' Q x_t + u_t' R u_t``.
    """

    y_t: np.ndarray
    v_t: np.ndarray
    x_t: np.ndarray
    u_t: np.ndarray
    cost: float

@dataclass
class SimulationResult:
    """Trajectory returned by :func:`lqr_base.simple_lqr.simulate`.

    Attributes
    ----------
    x_traj : np.ndarray
        True states over time, of shape ``(n_steps, n)``.
    u_traj : np.ndarray
        Applied controls over time, of shape ``(n_steps, m)``.
    """

    x_traj: np.ndarray
    u_traj: np.ndarray
