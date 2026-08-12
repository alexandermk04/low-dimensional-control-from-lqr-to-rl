"""Model-based policy gradient descent on the LQR cost.

This is the standard gradient solver: it has full access to the model 
and computes the exact policy gradient, so it converges cleanly and shows
what the model-free solvers in :mod:`lqr_solvers.mf_policy_gradient` are trying
to approximate from samples alone.

References
----------
.. [1] M. Fazel, R. Ge, S. M. Kakade, M. Mesbahi, "Global Convergence of Policy
       Gradient Methods for the Linear Quadratic Regulator", ICML 2018.
       https://arxiv.org/abs/1801.05039

"""

from dataclasses import dataclass
import numpy as np

from scipy.linalg import solve_discrete_lyapunov

from basetypes import StateSpaceModel

from lqr_base.simple_lqr import LQRDiscrete

@dataclass
class ModelBasedPolicyGradientSettings:
    """Hyperparameters for :func:`gradient_descent_discrete`.

    Attributes
    ----------
    K_0 : np.ndarray
        Initial gain of shape ``(m, n)``. Must be stabilizing.
    gradient_steps : int
        Number of descent steps.
    gradient_lr : float
        Fixed step size.
    """

    K_0: np.ndarray  # Initial policy
    gradient_steps: int  # Number of gradient steps
    gradient_lr: float  # Gradient step size

# Global Convergence... Paper, page 5
def calculate_gradient_model_based_discrete(lqr: LQRDiscrete, K_t: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact policy gradient of the LQR cost at ``K_t``.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over.
    K_t : np.ndarray
        Current gain of shape ``(m, n)``, applied as ``u = -K x``.

    Returns
    -------
    grad : np.ndarray
        ``2 E_k Sigma_k``, of shape ``(m, n)``.
    expected_cost : float
        ``x_0' P_k x_0``, the cost of the current policy from the sampled
        initial state.

    Notes
    -----
    .. code-block:: text

        E_k  = (R + B' P_k B) K_t - B' P_k A
        grad = 2 E_k Sigma_k

    The sign of ``E_k`` follows this repository's ``u = -K x`` convention and
    is flipped relative to the paper.
    """

    lya_a = lqr.A - lqr.B @ K_t

    lya_q = lqr.Q + K_t.T @ lqr.R @ K_t

    P_k = solve_discrete_lyapunov(lya_a.T, lya_q)

    # Initial state covariance
    sigma_0 = np.eye(lqr.A.shape[0])

    # Expected cost using current policy K_t
    expected_cost = lqr.x_0.T @ P_k @ lqr.x_0

    # Compute the state covariance under the current policy K_t
    sigma_k = solve_discrete_lyapunov(lya_a, sigma_0)

    E_k = (lqr.R + lqr.B.T @ P_k @ lqr.B) @ K_t - lqr.B.T @ P_k @ lqr.A

    grad = 2 * E_k @ sigma_k

    return grad, expected_cost

# Runs gradient descent for a defined number of steps, returns K after
def gradient_descent_discrete(lqr: LQRDiscrete, settings: ModelBasedPolicyGradientSettings) -> tuple[list[np.ndarray], list[float]]:
    """Run fixed-step gradient descent on the LQR cost.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over.
    settings : ModelBasedPolicyGradientSettings
        Settings for the gradient descent run.

    Returns
    -------
    ks : list[np.ndarray]
        Every iterate including ``K_0``, so ``len(ks) == gradient_steps + 1``.
    costs : list[float]
        Cost *before* each step, so ``len(costs) == gradient_steps``. Element
        ``i`` is the cost of ``ks[i]``, leaving the final iterate's cost
        uncomputed.
    """
    lqr.reset()

    K = settings.K_0

    costs = []
    ks = [K]

    for _ in range(settings.gradient_steps):
        grad, expected_cost = calculate_gradient_model_based_discrete(lqr, K)
        costs.append(expected_cost)

        K = K - settings.gradient_lr * grad
        ks.append(K)

    return ks, costs
