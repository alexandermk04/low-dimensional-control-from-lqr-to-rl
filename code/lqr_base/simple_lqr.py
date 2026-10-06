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
- Feedback is ``v = -K y``, with ``y = U_1 x`` and ``u = U_2 v``. For the
  plain system both lifts are the identity.
- The cost is charged in the controller's space, ``y' Q y + v' R v`` on the
  clean ``y = U_1 x``, so ``Q`` and ``R`` have the controller's dimensions.
- ``reset()`` must be called before stepping. It samples (or restores) ``x_0``
"""

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from scipy.signal import cont2discrete

from basetypes import StateSpaceModel, SimulationResult, StepResult
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


def apply_weight(W: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Multiply by a cost weight given either as a matrix or as its diagonal.

    Parameters
    ----------
    W : np.ndarray
        Weight of shape ``(d, d)``, or its diagonal of shape ``(d,)``.
    M : np.ndarray
        Vector or matrix with ``d`` rows.

    Returns
    -------
    np.ndarray
        ``W @ M``, shaped like ``M``.
    """
    if W.ndim == 1:
        return W[:, None] * M if M.ndim == 2 else W * M
    return W @ M


def as_weight_matrix(W: np.ndarray) -> np.ndarray:
    """The weight as a dense matrix."""
    return np.diag(W) if W.ndim == 1 else W

class ControllerModel(NamedTuple):
    """The LQR problem in the controller's dimension.

    Attributes
    ----------
    A : np.ndarray
        ``U_1 A U_1^+``, of shape ``(p, p)``.
    B : np.ndarray
        ``U_1 B U_2``, of shape ``(p, q)``.
    Q, R : np.ndarray
        Dense cost weights, ``(p, p)`` and ``(q, q)``.
    sigma_0 : np.ndarray
        Initial observation covariance ``U_1 Sigma_x U_1'``, ``(p, p)``.
    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    sigma_0: np.ndarray


@dataclass
class LQRBaseSettings:
    """Base settings for every LQR variant.

    Attributes
    ----------
    model : StateSpaceModel
        Continuous-time ``(A, B)``.
    Q : np.ndarray
        Cost on the observation ``y``, of shape ``(p, p)`` or its diagonal
        ``(p,)``, positive semi-definite. ``p = n`` for the plain system.
    R : np.ndarray
        Cost on the controller output ``v``, of shape ``(q, q)`` or its
        diagonal ``(q,)``, positive definite. ``q = m`` for the plain system.
    max_initial_state : np.ndarray
        Elementwise bound of shape ``(n,)``. Initial states are drawn uniformly
        from ``[-max_initial_state, max_initial_state]``.
    resample_initial_state : bool
        If ``False``, one ``x_0`` is drawn (seeded with
        :data:`settings.TEST_SEED`) and reused by every subsequent ``reset()``, so
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
    seed_seq: np.random.SeedSequence

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
    U_1, U_2 : np.ndarray
        Observation lift ``(p, n)`` and actuation projection ``(m, q)``.
    Q, R : np.ndarray
        Cost weights in the controller's space, ``(p, p)`` and ``(q, q)``, or
        their diagonals.
    Q_x : np.ndarray
        ``U_1' Q U_1``, the state cost in the true state.
    x : np.ndarray
        Current state. Only valid after :meth:`reset`.
    x_0 : np.ndarray or None
        Initial state of the current episode, ``None`` before the first reset.
    initialized : bool
        Whether :meth:`reset` has been called. Guards :meth:`next_state`.
    noise : bool
        Runtime flag. When ``False`` the step is deterministic even if noise
        settings are present.
    parametric_noise, control_noise, measurement_noise, disturbance_noise
        The individual channels, or ``None`` when disabled.

    """

    A: np.ndarray
    B: np.ndarray
    U_1: np.ndarray
    U_2: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Q_x: np.ndarray

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

    state_rng: np.random.Generator
    state_rng_bit_state: dict
    noise_rng: np.random.Generator
    noise_rng_bit_state: dict

    def __init__(self, settings: LQRBaseSettings):
        """Store the plant and cost weights and wire up the noise channels.

        Parameters
        ----------
        settings : LQRBaseSettings
            Configuration.
        """
        self.A = settings.model.A
        self.B = settings.model.B

        self.U_1 = getattr(settings, "U_1", np.eye(self.A.shape[0]))
        self.U_2 = getattr(settings, "U_2", np.eye(self.B.shape[1]))

        self.Q = settings.Q
        self.R = settings.R
        self.validate_dimensions()
        self.Q_x = self.U_1.T @ apply_weight(self.Q, self.U_1)

        self.max_initial_state = settings.max_initial_state
        self.resample_initial_state = settings.resample_initial_state
        self.x_0 = None
        self.initialized = False

        # Observation cached so that measurement noise is only drawn once
        self._y_cache = None

        # Counts total steps of this model
        self.env_steps = 0
        # How many times it was reset
        self.episodes = 0

        # Seeding
        children = settings.seed_seq.spawn(2)
        self.state_rng = np.random.default_rng(children[0])
        self.state_rng_bit_state = self.state_rng.bit_generator.state
        self.noise_rng = np.random.default_rng(children[1])
        self.noise_rng_bit_state = self.noise_rng.bit_generator.state

        self.noise = False
        self.setup_noise(settings.noise_settings)

    def validate_dimensions(self):
        """Check that the lifts and cost weights fit dimensionally.

        Raises
        ------
        AssertionError
            If a lift or a cost weight has the wrong shape.
        """
        assert self.U_1.shape[1] == self.A.shape[0], "U_1 must have as many columns as A has rows"
        assert self.U_2.shape[0] == self.B.shape[1], "U_2 must have as many rows as B has columns"
        assert self.Q.shape[0] == self.U_1.shape[0] and self.Q.ndim in (1, 2), "Q must match the observation dimension"
        assert self.R.shape[0] == self.U_2.shape[1] and self.R.ndim in (1, 2), "R must match the action dimension"

    def controller_model(self) -> ControllerModel:
        """The LQR problem for the controller.

        Returns
        -------
        ControllerModel
            ``(A, B, Q, R, sigma_0)`` with ``A = U_1 A U_1^+`` of shape
            ``(p, p)``, ``B = U_1 B U_2`` of shape ``(p, q)``, the dense cost
            weights and ``Sigma_0 = U_1 Sigma_x U_1'``.
        """
        U_1_pinv = np.linalg.pinv(self.U_1)
        return ControllerModel(
            self.U_1 @ self.A @ U_1_pinv,
            self.U_1 @ self.B @ self.U_2,
            as_weight_matrix(self.Q),
            as_weight_matrix(self.R),
            self.U_1 @ initial_state_covariance(self) @ self.U_1.T,
        )

    def sample_initial_state(self):
        """Draw an initial state uniformly from the configured box.

        Returns
        -------
        np.ndarray
            A sample from ``U(-max_initial_state, max_initial_state)``, shaped
            like ``max_initial_state``.
        """
        initial =  self.state_rng.uniform(-self.max_initial_state, self.max_initial_state)
        return initial

    def reset_seeding(self):
        """Reset the random number generators to their initial state.
        """
        self.state_rng.bit_generator.state = self.state_rng_bit_state
        self.noise_rng.bit_generator.state = self.noise_rng_bit_state

    def reset_counters(self):
        """Zero the sample counters.
        """
        self.env_steps = 0
        self.episodes = 0

    def reset(self):
        """Return the system to its initial state.

        Must be called before :meth:`next_state`.

        With ``resample_initial_state=False`` the first sample is cached and
        every later reset restores it. With ``True``, a new ``x_0`` is drawn
        each time.
        """
        if self.resample_initial_state or self.x_0 is None:
            self.x_0 = self.sample_initial_state()
        self.x = self.x_0.copy()
        self._y_cache = None
        self.episodes += 1

        # Reset the noises carrying an internal state
        if self.disturbance_noise is not None:
            self.disturbance_noise.reset()
        if self.parametric_noise is not None:
            self.parametric_noise.reset()

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
        self.noise = True


    def lift_observation(self, x: np.ndarray) -> np.ndarray:
        """Map the true state into the space the controller observes.

        Parameters
        ----------
        x : np.ndarray
            True state of shape ``(n,)``.

        Returns
        -------
        np.ndarray
            ``U_1 @ x``, of shape ``(p,)``.
        """
        return self.U_1 @ x

    def project_action(self, v: np.ndarray) -> np.ndarray:
        """Map the controller's output into the true input space.

        Parameters
        ----------
        v : np.ndarray
            Controller output of shape ``(q,)``.

        Returns
        -------
        np.ndarray
            ``U_2 @ v``, of shape ``(m,)``.
        """
        return self.U_2 @ v

    def lift_action(self, u: np.ndarray) -> np.ndarray:
        """Map a true input back into the controller's space.

        Parameters
        ----------
        u : np.ndarray
            Input of shape ``(m,)``.

        Returns
        -------
        np.ndarray
            ``U_2' @ u``, of shape ``(q,)``. A right inverse of
            :meth:`project_action` when ``U_2`` has orthonormal rows.
        """
        return self.U_2.T @ u

    @property
    def observation_dim(self) -> int:
        """Dimension of the space the controller observes, ``p``."""
        return self.U_1.shape[0]

    @property
    def action_dim(self) -> int:
        """Dimension of the space the controller acts in, ``q``."""
        return self.U_2.shape[1]

    def observe(self) -> np.ndarray:
        """Observation of the current state, can be called multiple times within a step.
        The noise stays the same for all at a certain time.

        Returns
        -------
        np.ndarray
            ``y_t``, of shape ``(observation_dim,)``.

        Raises
        ------
        ValueError
            If :meth:`reset` has not been called.
        """
        if not self.initialized:
            raise ValueError(
                f"{type(self).__name__} must be initialized before observing. "
                "Call reset() first."
            )

        if self._y_cache is None:
            y = self.lift_observation(self.x)
            if self.noise and self.measurement_noise is not None:
                y = y + self.measurement_noise.sample_measurement_noise(y, self.noise_rng)
            self._y_cache = y

        return self._y_cache

    def apply_control(self, v: np.ndarray) -> StepResult:
        """Advance the plant one step given the controller's raw output.

        Parameters
        ----------
        v : np.ndarray
            Controller output in the controller's own space, of shape
            ``(action_dim,)``.

        Returns
        -------
        StepResult
            ``y_t`` as observed, ``v_t`` after control noise, the true
            ``x_t`` / ``u_t``, and the cost ``x' Q_x x + v' R v`` charged on
            the clean lifted state and the applied controller output.

        Raises
        ------
        ValueError
            If :meth:`reset` has not been called.
        """
        y_t = self.observe()
        x_t = self.x

        # Multiplicative control noise
        if self.noise and self.control_noise is not None:
            v = v + self.control_noise.sample_control_noise(v, self.noise_rng)

        u_t = self.project_action(v)

        # Parametric noise
        A = self.A
        B = self.B
        if self.noise and self.parametric_noise is not None:
            noise_A, noise_B = self.parametric_noise.sample_parametric_noise(A, B, self.noise_rng)
            A = A + noise_A
            B = B + noise_B

        x_next = A @ x_t + B @ u_t

        # Disturbances on the true successor state
        if self.noise and self.disturbance_noise is not None:
            x_next = x_next + self.disturbance_noise.sample_disturbance(x_next, self.noise_rng)

        self.x = x_next
        self._y_cache = None
        self.env_steps += 1

        cost = float(x_t @ self.Q_x @ x_t + v @ apply_weight(self.R, v))

        return StepResult(y_t, v, x_t, u_t, cost)

    def next_state(self, K: np.ndarray) -> StepResult:
        """Advance one step under the linear policy ``v = -K y``.

        Parameters
        ----------
        K : np.ndarray
            Feedback gain in the controller's space, applied as ``v = -K y``.

        Returns
        -------
        StepResult
            See :meth:`apply_control`.
        """
        return self.apply_control(-K @ self.observe())


class LQRDiscrete(LQRBase):
    """Discrete-time LQR with full state feedback.

    Without lifts in the settings the controller observes the true state
    directly (``y = x``) and its output is applied directly (``u = v``), so the
    gain the solvers learn is the gain acting on the plant.

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
        super().__init__(settings)

        # Only do the setup not covered by base class

        self.dt = settings.dt

        A_d, B_d = continuous_to_discrete(settings.model.A, settings.model.B, dt=settings.dt)

        self.A = A_d
        self.B = B_d

def closed_loop_matrix(lqr: LQRBase, K: np.ndarray) -> np.ndarray:
    """Closed-loop matrix ``A - B U_2 K U_1`` of the true state.

    Parameters
    ----------
    lqr : LQRBase
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.

    Returns
    -------
    np.ndarray
        Of shape ``(n, n)``.
    """
    return lqr.A - lqr.B @ lqr.U_2 @ K @ lqr.U_1


def stage_cost_matrix(lqr: LQRBase, K: np.ndarray) -> np.ndarray:
    """Stage cost of the closed loop as a weight on the true state.

    With ``y = U_1 x`` and ``v = -K y`` the stage cost ``y' Q y + v' R v``
    equals ``x' M x`` for the returned ``M``.

    Parameters
    ----------
    lqr : LQRBase
        System the gain acts on.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``.

    Returns
    -------
    np.ndarray
        ``Q_x + (K U_1)' R (K U_1)``, of shape ``(n, n)``.
    """
    K_x = K @ lqr.U_1
    return lqr.Q_x + K_x.T @ apply_weight(lqr.R, K_x)


def get_optimal_K_and_cost_discrete(lqr: LQRBase):
    """Solve the LQR exactly, giving the baseline every solver is compared to.

    Parameters
    ----------
    lqr : LQRBase
        System to solve.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(K, cost)``: the optimal gain in the controller's space, of shape
        ``(q, p)``, under the ``v = -K y`` convention, and the resulting
        infinite-horizon cost ``x_0' P x_0``.
    """

    lqr.reset()

    B_v = lqr.B @ lqr.U_2
    R = as_weight_matrix(lqr.R)

    P = solve_discrete_are(lqr.A, B_v, lqr.Q_x, R)

    # SHOW: Minus was here for both papers, but our convention is u(t) = - K @ x(t), so we don't need the minus here
    K_v = np.linalg.solve(R + B_v.T @ P @ B_v, B_v.T @ P @ lqr.A)

    K = K_v @ np.linalg.pinv(lqr.U_1)

    cost = lqr.x_0.T @ P @ lqr.x_0

    return K, cost

def initial_state_covariance(lqr: LQRDiscrete) -> np.ndarray:
    """Second moment ``Sigma_0 = E[x_0 x_0']`` of the reset distribution.

    Parameters
    ----------
    lqr : LQRDiscrete
        System whose reset distribution is described.

    Returns
    -------
    np.ndarray
        Of shape ``(n, n)``. For ``resample_initial_state=True`` the states are
        drawn from the uniform box ``U(-b, b)`` with independent coordinates,
        whose second moment is ``diag(b_i**2 / 3)``. For a fixed ``x_0`` it
        degenerates to ``outer(x_0, x_0)``, so that ``trace(P @ Sigma_0)``
        reduces to :func:`true_cost` exactly.

    Raises
    ------
    ValueError
        If the initial state is fixed but has not been sampled yet, i.e.
        ``reset()`` has not been called.
    """

    if lqr.resample_initial_state:
        bounds = np.asarray(lqr.max_initial_state, dtype=float)
        return np.diag(bounds ** 2 / 3.0)

    if lqr.x_0 is None:
        raise ValueError(
            "With resample_initial_state=False the covariance is determined by "
            "the single sampled x_0. Call reset() first."
        )

    return np.outer(lqr.x_0, lqr.x_0)


def true_cost(lqr: LQRDiscrete, K: np.ndarray) -> float:
    """Exact infinite-horizon cost of a gain, without simulating anything.

    Parameters
    ----------
    lqr : LQRDiscrete
        System that has already been reset, so that ``lqr.x_0`` exists.
    K : np.ndarray
        Gain in the controller's space, of shape ``(q, p)``, applied as
        ``v = -K y``.

    Returns
    -------
    float
        ``x_0' P x_0``, or ``np.inf`` if the closed loop ``A - B U_2 K U_1``
        is unstable.

    Raises
    ------
    ValueError
        If ``lqr`` has not been reset.
    """

    if not lqr.initialized:
        raise ValueError("LQR must be reset first to sample x_O.")

    x_0 = lqr.x_0

    A_cl = closed_loop_matrix(lqr, K)

    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1:
        return np.inf

    P = solve_discrete_lyapunov(A_cl.T, stage_cost_matrix(lqr, K))
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
        Gain in the controller's own space, of shape ``(q, p)``.
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

    x_traj = []
    u_traj = []

    for _ in range(n_steps):
        res = lqr.next_state(K=K)

        # Add the ACTUAL observation/actuation
        x_traj.append(res.x_t)
        u_traj.append(res.u_t)

    results = SimulationResult(x_traj=np.array(x_traj), u_traj=np.array(u_traj))

    return results
