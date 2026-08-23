"""LQR system whose controller sees and acts in artificially enlarged spaces.

The point of the inflation is to model an embodiment mismatch. A real agent
rarely gets the real state dimensions: it gets many noisy,
redundant sensor readings, and it drives many actuators that together produce a
much lower-dimensional effect. The plant stays small; the controller's inputs and outputs
get big.

:class:`LQRInflated` implements that with two fixed lift matrices:

.. code-block:: text

    x_t  --[ U_1 ]-->  y_t  --[ policy -K ]-->  v_t  --[ U_2 ]-->  u_t
    (n)               (p)                      (q)                (m)

so the learned gain ``K`` has shape ``(q, p)`` and can be far larger than the
``(m, n)`` gain of the underlying problem, while the gain that actually acts on
the plant is the *effective* gain ``U_2 @ K @ U_1``. Since ``U_1`` and ``U_2``
have rank at most ``n`` and ``m``, the optimal ``K`` is low rank -- which is
what makes this the right setting for the nuclear-norm solver in
:mod:`lqr_solvers.structured_policy_iteration`.

Notes
-----
The experiment scripts build ``U_1`` and ``U_2`` as orthonormal lifts via a QR
decomposition. Plain random matrices made the solvers diverge, 
so the orthonormality is relevant.
"""

from dataclasses import dataclass
import numpy as np

from basetypes import StepResult

from lqr_base.simple_lqr import (
    continuous_to_discrete,
    StateSpaceModel,
    LQRDiscrete,
    LQRDiscreteSettings,
)

from lqr_base.noise import (
    LQRNoiseSettings,
)

@dataclass
class LQRInflatedSettings(LQRDiscreteSettings):
    """:class:`~lqr_base.simple_lqr.LQRDiscreteSettings` plus the two lifts.

    Attributes
    ----------
    U_1 : np.ndarray
        Observation lift of shape ``(p, n)``, mapping the true state to what
        the controller sees. ``p >= n`` inflates the observation space.
    U_2 : np.ndarray
        Actuation projection of shape ``(m, q)``, mapping the controller's
        output down to the plant input. ``q >= m`` inflates the action space.
    """

    U_1: np.ndarray
    U_2: np.ndarray

class LQRInflated(LQRDiscrete):
    """Discrete LQR observed through ``U_1`` and actuated through ``U_2``.

    Behaves like :class:`~lqr_base.simple_lqr.LQRDiscrete` in every respect
    except the two spaces the controller lives in, which is why it only
    overrides :meth:`effective_gain`, :meth:`project_gradient` and
    :meth:`next_state`. Setting ``U_1 = I`` and ``U_2 = I`` recovers the plain
    system exactly.

    Attributes
    ----------
    U_1 : np.ndarray
        Observation lift of shape ``(p, n)``.
    U_2 : np.ndarray
        Actuation projection of shape ``(m, q)``.
    dt : float
        Sampling interval used for the discretization.
    """

    U_1: np.ndarray
    U_2: np.ndarray

    def __init__(self, settings: LQRInflatedSettings):
        """Discretize the model, store the lifts and check their shapes.

        Parameters
        ----------
        settings : LQRInflatedSettings
            Configuration; ``settings.model`` is continuous time and is
            discretized here.

        Raises
        ------
        AssertionError
            If the lift shapes do not match the plant, via
            :meth:`validate_dimensions`.
        """
        super().__init__(settings)

        self.U_1 = settings.U_1
        self.U_2 = settings.U_2
        self.validate_dimensions()

    def validate_dimensions(self):
        """Check that the lifts compose with the plant.

        Raises
        ------
        AssertionError
            If ``U_1`` does not have ``n`` columns, or ``U_2`` does not have
            ``m`` rows.
        """
        assert self.U_1.shape[1] == self.A.shape[0], "U_1 must have the same number of columns as A has rows"
        assert self.U_2.shape[0] == self.B.shape[1], "U_2 must have the same number of rows as B has columns"

    def effective_gain(self, K: np.ndarray) -> np.ndarray:
        """Map an inflated gain to the gain acting on the true plant.

        Parameters
        ----------
        K : np.ndarray
            Gain in the controller's space, of shape ``(q, p)``.

        Returns
        -------
        np.ndarray
            ``U_2 @ K @ U_1``, of shape ``(m, n)``. Its rank is bounded by
            ``min(m, n)`` no matter how large ``K`` is, so many different
            ``K`` produce identical closed-loop behaviour.

        Notes
        -----
        Pass this, not the raw ``K``, to
        :func:`~lqr_base.simple_lqr.true_cost`.
        """
        return self.U_2 @ K @ self.U_1

    def project_gradient(self, grad_K: np.ndarray) -> np.ndarray:
        """Pull a gradient on the effective gain back to the inflated space.

        Parameters
        ----------
        grad_K : np.ndarray
            Gradient with respect to the effective gain, of shape ``(m, n)``.

        Returns
        -------
        np.ndarray
            ``U_2' @ grad_K @ U_1'``, of shape ``(q, p)``: the chain rule
            applied to :meth:`effective_gain`.
        """
        return self.U_2.T @ grad_K @ self.U_1.T

    def next_state(self, K: np.ndarray) -> StepResult:
        """Advance one step with the controller confined to the lifted spaces.

        Parameters
        ----------
        K : np.ndarray
            Gain in the controller's space, of shape ``(q, p)``.

        Returns
        -------
        StepResult
            ``y_t`` is the (possibly noisy) inflated observation of shape
            ``(p,)``, ``v_t`` the inflated control of shape ``(q,)``, while
            ``x_t`` and ``u_t`` remain the true state and the true plant input.
            Unlike the plain system, these never coincide, so both branches
            fill all four fields.

        Raises
        ------
        ValueError
            If :meth:`reset` has not been called.
        """
        if not self.initialized:
            raise ValueError("LQRInflated must be initialized before calling next_state. Call reset() first.")

        if self.noise:
            x_t = self.x
            y_t = self.U_1 @ x_t

            if self.measurement_noise is not None:
                noise = self.measurement_noise.sample_measurement_noise(y_t, self.noise_rng)
                y_t = y_t + noise
            v_t = - K @ y_t

            if self.control_noise is not None:
                control_noise = self.control_noise.sample_control_noise(v_t, self.noise_rng)
                v_t = v_t + control_noise

            u_t = self.U_2 @ v_t

            # Parametric noise lives in the state space, not the measured space
            A = self.A
            B = self.B
            if self.parametric_noise is not None:
                noise_A, noise_B = self.parametric_noise.sample_parametric_noise(A, B, self.noise_rng)
                A = A + noise_A
                B = B + noise_B

            # Use the actual state, with the calculated control, based on the observations
            x_next = A @ x_t + B @ u_t

            # Disturbance noise also lives in the state space, not the measured space
            if self.disturbance_noise is not None:
                disturbance_noise = self.disturbance_noise.sample_disturbance(x_next, self.noise_rng)
                x_next = x_next + disturbance_noise

            self.x = x_next

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t

            return StepResult(y_t, v_t, x_t, u_t, cost)

        else:
            x_t = self.x
            y_t = self.U_1 @ x_t
            v_t = - K @ y_t

            u_t = self.U_2 @ v_t

            self.x = self.A @ x_t + self.B @ u_t

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t
            return StepResult(y_t, v_t, x_t, u_t, cost)
