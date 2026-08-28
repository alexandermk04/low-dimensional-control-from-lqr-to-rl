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

    @property
    def observation_dim(self) -> int:
        """Dimension of the inflated observation space, ``p``."""
        return self.U_1.shape[0]

    @property
    def action_dim(self) -> int:
        """Dimension of the inflated action space, ``q``."""
        return self.U_2.shape[1]

    def lift_observation(self, x: np.ndarray) -> np.ndarray:
        """Lift the true state into the inflated observation space.

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
        """Project the inflated control down onto the true input space.

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
