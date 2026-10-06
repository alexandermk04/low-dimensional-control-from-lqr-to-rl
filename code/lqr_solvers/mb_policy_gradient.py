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
from tqdm import tqdm

from scipy.linalg import solve_discrete_lyapunov

from basetypes import StateSpaceModel

from lqr_base.simple_lqr import ControllerModel, LQRDiscrete
from evaluation.analysis import Plateau
from evaluation.probe import Probe, record_iteration, should_stop

@dataclass
class ModelBasedPolicyGradientSettings:
    """Hyperparameters for :func:`gradient_descent_discrete`.

    Attributes
    ----------
    K_0 : np.ndarray
        Initial gain in the controller's space, of shape ``(q, p)``. Must be
        stabilizing.
    gradient_steps : int
        Number of descent steps.
    gradient_lr : float
        Fixed step size.
    probe : evaluation.probe.Probe or None, optional
        Instrumentation recording cost, sample budget and total time per
        iteration. Defaults to ``None`` 
    stopper : evaluation.analysis.Plateau or None, optional
        Stops on a plateau of the model cost and keeps the best gain.
        ``gradient_steps`` is then only a cap. Defaults to ``None``
    """

    K_0: np.ndarray  # Initial policy
    gradient_steps: int  # Number of gradient steps
    gradient_lr: float  # Gradient step size
    probe: Probe | None = None  # Optional per-iteration instrumentation
    stopper: Plateau | None = None  # Optional convergence criterion

# Global Convergence... Paper, page 5
def calculate_gradient_model_based_discrete(
    model: ControllerModel,
    K_t: np.ndarray,
):
    """Exact policy gradient of the LQR cost at ``K_t``.

    Parameters
    ----------
    model : ControllerModel
        The problem in the controller's space, from
        :meth:`~lqr_base.simple_lqr.LQRBase.controller_model`.
    K_t : np.ndarray
        Current gain in the controller's space, applied as ``v = -K y``, of
        shape ``(q, p)``.

    Returns
    -------
    grad : np.ndarray
        ``2 E_k Sigma_k``, shaped like ``K_t``.
    expected_cost : float
        ``trace(sigma_0 P_k)``, the expected cost of the current policy over
        the initial-state distribution; ``inf`` if the closed loop is unstable.

    Notes
    -----
    .. code-block:: text

        E_k   = (R + B' P_k B) K_t - B' P_k A
        grad  = 2 E_k Sigma_k

    on the controller-space model. The sign of
    ``E_k`` follows this thesis' ``u = -K x`` convention and is flipped
    relative to the paper.
    """
    A, B, Q, R, sigma_0 = model

    lya_a = A - B @ K_t

    P_k = solve_discrete_lyapunov(lya_a.T, Q + K_t.T @ R @ K_t)

    # Expected cost using current policy K_t
    expected_cost = np.trace(sigma_0 @ P_k)
    if np.max(np.abs(np.linalg.eigvals(lya_a))) >= 1:
        expected_cost = np.inf

    # Compute the state covariance under the current policy K_t
    sigma_k = solve_discrete_lyapunov(lya_a, sigma_0)

    E_k = (R + B.T @ P_k @ B) @ K_t - B.T @ P_k @ A

    return 2 * E_k @ sigma_k, expected_cost

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
    model = lqr.controller_model()

    K = settings.K_0

    costs = []
    ks = [K]

    progress_bar = tqdm(range(settings.gradient_steps), desc="Model-Based Policy Gradient", unit="step")

    for i in progress_bar:
        grad, expected_cost = calculate_gradient_model_based_discrete(model, K)
        costs.append(expected_cost)

        K = K - settings.gradient_lr * grad
        ks.append(K)

        # data collection
        record_iteration(settings, lqr, progress_bar, i, K, cost_estimate=expected_cost)

        if should_stop(settings, lqr, i, expected_cost, ks[-2]):
            break

    return ks, costs
