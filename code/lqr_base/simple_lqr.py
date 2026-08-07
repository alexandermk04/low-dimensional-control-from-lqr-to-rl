"""Discrete-time LQR system, plus the exact baseline solution and a simulator.

This module defines the system the solvers in :mod:`lqr_solvers` optimize over.
It is deliberately *not* a solver itself: :class:`LQRDiscrete` only knows how to
take a step given a gain, and the two free functions provide the analytic
reference the learned policies are measured against.

Everything here is discrete time. A continuous ``(A, B)`` pair together with a
sampling interval ``dt`` goes in; :func:`continuous_to_discrete` converts it on
construction. The discrete setting was chosen because it is the one an actual
embodied controller runs in.

Conventions
-----------
- Feedback is ``u = -K x``.
- ``reset()`` must be called before stepping. It samples (or restores) ``x_0``
  and clears the noise flag.
- Noise is off by default and must be switched on per rollout via
  :meth:`LQRBase.apply_noise`. See :mod:`lqr_base.noise`.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from scipy.signal import cont2discrete

from basetypes import StateSpaceModel, SimulationResult, StepResult, SEED
from lqr_base.noise import (
    ParametricNoiseSettings,
    ControlNoiseSettings,
    MeasurementNoiseSettings,
    DisturbanceNoiseSettings,
    LQRNoiseSettings
)

def continuous_to_discrete(A, B, dt):
    """Discretize ``x' = A x + B u`` with a zero-order hold.

    A zero-order hold assumes the control is held constant across each interval
    of length ``dt``, which is exactly what a digital controller does between
    two updates.

    Parameters
    ----------
    A : np.ndarray
        Continuous-time state matrix of shape ``(n, n)``.
    B : np.ndarray
        Continuous-time input matrix of shape ``(n, m)``.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(A_d, B_d)``, the discrete-time matrices, with the same shapes.
    """

    # y=Cx+Du; for LQR we have full state feedback and no feedthrough, so C=I and D=0
    system = (A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1])))

    A_d, B_d, _, _, _ = cont2discrete(system, dt, method='zoh')

    return A_d, B_d

@dataclass
class LQRBaseSettings:
    """Base settings for every LQR variant.

    Attributes
    ----------
    model : StateSpaceModel
        Continuous-time ``(A, B)``.
    Q : np.ndarray
        State cost of shape ``(n, n)``, positive semi-definite.
    R : np.ndarray
        Control cost of shape ``(m, m)``, positive definite.
    max_initial_state : np.ndarray
        Elementwise bound of shape ``(n,)``. Initial states are drawn uniformly
        from ``[-max_initial_state, max_initial_state]``.
    resample_initial_state : bool
        If ``False``, one ``x_0`` is drawn (seeded with
        :data:`basetypes.SEED`) and reused by every subsequent ``reset()``, so
        all solvers see the same problem. If ``True``, every ``reset()`` draws
        a fresh one.
    noise_settings : LQRNoiseSettings or None
        Noise channels to expose the system to, or ``None`` for a clean system.
    """

    model: StateSpaceModel
    Q: np.ndarray
    R: np.ndarray
    max_initial_state: np.ndarray
    resample_initial_state: bool
    noise_settings: LQRNoiseSettings | None

@dataclass
class LQRDiscreteSettings(LQRBaseSettings):
    """Extends :class:`LQRBaseSettings` by the sampling interval.

    Attributes
    ----------
    dt : float
        Sampling interval in seconds, used to discretize ``model``.
    """

    dt: float


class LQRBase:
    """Abstract discrete-time LQR system with an internal state.

    The class carries the plant, the cost weights and the current state, and
    exposes exactly one operation to a solver: :meth:`next_state`, which
    advances one step under a given gain. Subclasses decide what the controller
    observes and actuates.

    Attributes
    ----------
    A, B : np.ndarray
        Plant matrices, of shapes ``(n, n)`` and ``(n, m)``.
    Q, R : np.ndarray
        Cost weights, of shapes ``(n, n)`` and ``(m, m)``.
    x : np.ndarray
        Current state. Only valid after :meth:`reset`.
    x_0 : np.ndarray or None
        Initial state of the current episode, ``None`` before the first reset.
    initialized : bool
        Whether :meth:`reset` has been called. Guards :meth:`next_state`.
    noise : bool
        Runtime flag. When ``False`` the step is deterministic even if noise
        settings are present. Cleared by :meth:`reset`, set by
        :meth:`apply_noise`.
    parametric_noise, control_noise, measurement_noise, disturbance_noise
        The individual channels, or ``None`` when disabled.

    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    x: np.ndarray
    x_0: np.ndarray | None = None

    max_initial_state: np.ndarray
    resample_initial_state: bool
    initialized: bool = False

    noise: bool
    parametric_noise: ParametricNoiseSettings | None
    control_noise: ControlNoiseSettings | None
    measurement_noise: MeasurementNoiseSettings | None
    disturbance_noise: DisturbanceNoiseSettings | None

    def __init__(self, settings: LQRBaseSettings):
        """Store the plant and cost weights and wire up the noise channels.

        Parameters
        ----------
        settings : LQRBaseSettings
            Configuration.
        """
        self.A = settings.model.A
        self.B = settings.model.B

        self.Q = settings.Q
        self.R = settings.R

        self.max_initial_state = settings.max_initial_state
        self.resample_initial_state = settings.resample_initial_state
        self.x_0 = None
        self.initialized = False

        self.noise = False
        self.setup_noise(settings.noise_settings)

    def sample_initial_state(self):
        """Draw an initial state uniformly from the configured box.

        Returns
        -------
        np.ndarray
            A sample from ``U(-max_initial_state, max_initial_state)``, shaped
            like ``max_initial_state``.
        """
        return np.random.uniform(-self.max_initial_state, self.max_initial_state)

    def reset(self):
        """Return the system to its initial state and disable noise.

        Must be called before :meth:`next_state`. On the very first call the
        global RNG is seeded with :data:`basetypes.SEED` before sampling, so a
        fixed-``x_0`` experiment is reproducible across runs.

        With ``resample_initial_state=False`` the first sample is cached and
        every later reset restores it. With ``True``, a new ``x_0`` is drawn
        each time.
        """
        if self.resample_initial_state or self.x_0 is None:
            if self.x_0 is None:
                np.random.seed(SEED)
            self.x_0 = self.sample_initial_state()
        self.x = self.x_0.copy()
        self.noise = False
        self.initialized = True

    def setup_noise(self, noise_settings: LQRNoiseSettings | None):
        """Unpack a noise bundle into the four per-channel attributes.

        Parameters
        ----------
        noise_settings : LQRNoiseSettings or None
            The bundle, or ``None`` to disable every channel.
        """
        if noise_settings is None:
            self.parametric_noise = None
            self.control_noise = None
            self.measurement_noise = None
            self.disturbance_noise = None
            return
        self.parametric_noise = noise_settings.parametric_noise
        self.control_noise = noise_settings.control_noise
        self.measurement_noise = noise_settings.measurement_noise
        self.disturbance_noise = noise_settings.disturbance_noise

    def apply_noise(self):
        """Arm the configured noise channels for the steps that follow.

        Call this after :meth:`reset`, which clears the flag. Channels that
        were never configured stay inactive regardless.
        """
        self.noise = True

    def next_state(self, K: np.ndarray) -> StepResult:
        """Advance one step under the gain ``K``. Implemented by subclasses.

        Parameters
        ----------
        K : np.ndarray
            Feedback gain, applied as ``u = -K y``.

        Returns
        -------
        StepResult
            What was observed, commanded, and actually applied, plus the stage
            cost.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """
        raise NotImplementedError


class LQRDiscrete(LQRBase):
    """Discrete-time LQR with full state feedback.

    The controller observes the true state directly (``y = x``) and its output
    is applied directly (``u = v``), so the gain the solvers learn *is* the
    gain acting on the plant.

    Attributes
    ----------
    dt : float
        Sampling interval used for the discretization.
    """

    dt: float

    def __init__(self, settings: LQRDiscreteSettings):
        """Discretize the model and store the configuration.

        Parameters
        ----------
        settings : LQRDiscreteSettings
            Configuration; ``settings.model`` is expected in continuous time
            and is discretized here with ``settings.dt``.
        """
        self.dt = settings.dt

        A_d, B_d = continuous_to_discrete(settings.model.A, settings.model.B, dt=settings.dt)

        self.A = A_d
        self.B = B_d

        self.Q = settings.Q
        self.R = settings.R

        self.max_initial_state = settings.max_initial_state
        self.resample_initial_state = settings.resample_initial_state
        self.x_0 = None
        self.initialized = False

        self.noise = False
        self.setup_noise(settings.noise_settings)

    def effective_gain(self, K: np.ndarray) -> np.ndarray:
        """Map a gain from the controller's space to the true state space.

        For full state feedback the two spaces coincide, so this is the
        identity.

        Parameters
        ----------
        K : np.ndarray
            Gain of shape ``(m, n)``.

        Returns
        -------
        np.ndarray
            ``K`` unchanged.
        """
        return K

    def project_gradient(self, grad_K: np.ndarray) -> np.ndarray:
        """Map a gradient from the true state space back to the controller's.

        The counterpart of :meth:`effective_gain`, and again the identity here.

        Parameters
        ----------
        grad_K : np.ndarray
            Gradient with respect to the effective gain.

        Returns
        -------
        np.ndarray
            ``grad_K`` unchanged.
        """
        return grad_K

    def next_state(self, K: np.ndarray) -> StepResult:
        """Advance the plant one step under ``u = -K x``.

        Parameters
        ----------
        K : np.ndarray
            Feedback gain of shape ``(m, n)``.

        Returns
        -------
        StepResult
            What was observed, commanded, and actually applied, plus the stage
            cost.

        Raises
        ------
        ValueError
            If :meth:`reset` has not been called.
        """
        if not self.initialized:
            raise ValueError("LQRDiscrete must be initialized before calling next_state. Call reset() first.")

        if self.noise:
            x_t = self.x

            # Observation
            y_t = x_t
            if self.measurement_noise is not None:
                noise = self.measurement_noise.sample_measurement_noise(y_t)
                y_t = y_t + noise

            # Control input based on the observation
            u_t = - K @ y_t

            # The actual control input applied
            v_t = u_t
            if self.control_noise is not None:
                control_noise = self.control_noise.sample_control_noise(u_t)
                v_t = v_t + control_noise

            # Parametric noise applied to state space
            A = self.A
            B = self.B
            if self.parametric_noise is not None:
                noise_A, noise_B = self.parametric_noise.sample_parametric_noise(A, B)
                A = A + noise_A
                B = B + noise_B

            # Here we use the actual state, with our control
            x_next = A @ x_t + B @ v_t

            # Disturbance noise also lives in the state space, not the measured space
            if self.disturbance_noise is not None:
                disturbance_noise = self.disturbance_noise.sample_disturbance(x_next)
                x_next = x_next + disturbance_noise

            self.x = x_next

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t

            return StepResult(y_t, v_t, x_t, u_t, cost)

        else:
            x_t = self.x
            u_t = - K @ x_t

            self.x = self.A @ x_t + self.B @ u_t

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t

            return StepResult(x_t, u_t, x_t, u_t, cost)


def get_optimal_K_and_cost_discrete(lqr: LQRDiscrete):
    """Solve the LQR exactly, giving the baseline every solver is compared to.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to solve.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(K, cost)``: the optimal gain of shape ``(m, n)`` under the
        ``u = -K x`` convention, and the resulting infinite-horizon cost
        ``x_0' P x_0``.

    Notes
    -----
    ``P`` solves the discrete algebraic Riccati equation, and

    .. code-block:: text

        K = (B' P B + R)^-1 B' P A
    """

    lqr.reset()

    P = solve_discrete_are(lqr.A, lqr.B, lqr.Q, lqr.R)

    # SHOW: K = np.linalg.inv(lqr.R) @ lqr.B.T @ P

    part_1 = np.linalg.inv(lqr.B.T @ P @ lqr.B + lqr.R)

    # SHOW: Minus was here for both papers, but our convention is u(t) = - K @ x(t), so we don't need the minus here
    K = part_1 @ lqr.B.T @ P @ lqr.A

    cost = lqr.x_0.T @ P @ lqr.x_0

    return K, cost

def true_cost(lqr: LQRDiscrete, K: np.ndarray) -> float:
    """Exact infinite-horizon cost of a gain, without simulating anything.

    Parameters
    ----------
    lqr : LQRDiscrete
        System that has already been reset, so that ``lqr.x_0`` exists.
    K : np.ndarray
        Gain acting on the true state, of shape ``(m, n)``, applied as
        ``u = -K x``. For an inflated system pass the effective gain, i.e.
        ``lqr.effective_gain(K)``, not the raw inflated one.

    Returns
    -------
    float
        ``x_0' P x_0``, or ``np.inf`` if the closed loop ``A - B K`` is
        unstable.

    Raises
    ------
    ValueError
        If ``lqr`` has not been reset.
    """

    if not lqr.initialized:
        raise ValueError("LQR must be reset first to sample x_O.")

    x_0 = lqr.x_0

    A_cl = lqr.A - lqr.B @ K

    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1:
        return np.inf

    P = solve_discrete_lyapunov(A_cl.T, lqr.Q + K.T @ lqr.R @ K)
    cost = (x_0.T @ P @ x_0).flatten()[0]
    return float(cost)


def simulate(lqr: LQRDiscrete, K: np.ndarray, n_steps: int):
    """Simulate the closed loop forward for a given number of steps and record the trajectory.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to roll out. Reset here, so any previously running episode is
        discarded.
    K : np.ndarray
        Gain in the controller's own space - the raw learned gain, which for
        an inflated system means the inflated one, *not* the effective gain.
    n_steps : int
        Number of steps to simulate.

    Returns
    -------
    SimulationResult
        True states and applied controls, of shapes ``(n_steps, n)`` and
        ``(n_steps, m)``.
    """
    # Reset lqr state to initial state before simulation
    lqr.reset()
    lqr.apply_noise()

    x_traj = []
    u_traj = []

    for _ in range(n_steps):
        res = lqr.next_state(K=K)

        # Add the ACTUAL observation/actuation
        x_traj.append(res.x_t)
        u_traj.append(res.u_t)

    results = SimulationResult(x_traj=np.array(x_traj), u_traj=np.array(u_traj))

    return results
