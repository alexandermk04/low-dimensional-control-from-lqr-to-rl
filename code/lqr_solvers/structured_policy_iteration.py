"""Structured Policy Iteration (S-PI): LQR with a low-rank gain.

S-PI adds a nuclear-norm penalty to the LQR cost, which drives the singular values of the
gain to zero and yields a low-rank controller - one that projects
its many observations onto a few internal coordinates.

The algorithm is proximal gradient descent: an ordinary gradient step on the
LQR cost, followed by a proximal step that soft-thresholds the singular values,
wrapped in a backtracking line search that rejects any unstable step.

Both a model-based and a model-free version are provided:

- :func:`structured_policy_iteration` - exact gradient via Lyapunov solver.
- :func:`structured_policy_iteration_mf` - gradient estimated from rollouts.

References
----------
.. [1] Y. Park, R. A. Rossi, Z. Wen, G. Wu, H. Zhao, "Structured Policy
       Iteration for Linear Quadratic Regulator", ICML 2020.
       https://arxiv.org/abs/2007.06202
       All of the implementation below is adapted from this paper.

"""

import numpy as np
from dataclasses import dataclass

from scipy.linalg import solve_discrete_lyapunov

from lqr_base.simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    LQRDiscreteSettings,
    simulate,
    true_cost,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from lqr_solvers.mf_policy_gradient import (
    estimate_gradient
)

from visualization import visualize_result, visualize_multiple_results

from lqr_base.noise import (
    ParametricNoiseSettings,
    ControlNoiseSettings,
    MeasurementNoiseSettings,
    DisturbanceNoiseSettings,
    LQRNoiseSettings
)


# All implementation adapted from Structured Policy Iteration paper

@dataclass
class StructuredPolicySettings:
    """Hyperparameters for the model-based :func:`structured_policy_iteration`.

    Attributes
    ----------
    K_0 : np.ndarray
        Initial gain in the controller's space. Must be stabilizing.
    beta : float
        Backtracking factor in ``(0, 1)``.
    lambda_ : float
        Weight of the nuclear-norm penalty. Larger values push harder towards
        low rank at the cost of optimality.
    epsilon : float
        Convergence tolerance.
    limit : int
        Hard cap on iterations, so a non-converging run still terminates.
    max_linesearch_iter : int
        Cap on backtracking steps per iteration. If it is exhausted, the last
        candidate is accepted regardless of whether it satisfied the criterion.
    """

    K_0: np.ndarray  # Initial policy
    beta: float # linesearch parameter
    lambda_: float # regularization parameter for low-rank structure
    epsilon: float # stopping criterion for convergence
    limit: int # maximum number of iterations to prevent infinite loop in case of non-convergence
    max_linesearch_iter: int

def solve_lyapunov(lqr: LQRDiscrete, K_t: np.ndarray, sigma_0: np.ndarray) -> np.ndarray:
    """Value matrix and state covariance of the closed loop under ``K_t``.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to be solved.
    K_t : np.ndarray
        Gain in the controller's space.
    sigma_0 : np.ndarray
        Initial-state covariance of shape ``(n, n)``.

    Returns
    -------
    P_k : np.ndarray
        Closed-loop value matrix; ``trace(sigma_0 @ P_k)`` is the LQR cost.
    sigma_k : np.ndarray
        Closed-loop state covariance, which weights the gradient.
    """

    K_eff = lqr.effective_gain(K_t)

    lya_a = lqr.A - lqr.B @ K_eff

    lya_q = lqr.Q + K_eff.T @ lqr.R @ K_eff

    P_k = solve_discrete_lyapunov(lya_a.T, lya_q)

    sigma_k = solve_discrete_lyapunov(lya_a, sigma_0)

    return P_k, sigma_k

def compute_gradient(lqr: LQRDiscrete, K_t: np.ndarray, P_k: np.ndarray, sigma_k: np.ndarray) -> np.ndarray:
    """Exact gradient of the LQR cost at ``K_t``.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to be optimized over.
    K_t : np.ndarray
        Current gain.
    P_k : np.ndarray
        Value matrix from :func:`solve_lyapunov`.
    sigma_k : np.ndarray
        State covariance from :func:`solve_lyapunov`.

    Returns
    -------
    np.ndarray
        ``2 ((R + B' P_k B) K_t - B' P_k A) sigma_k``.

    Notes
    -----
    The sign follows this repository's ``u = -K x`` convention.
    """

    # Minus, as we use u(t) = - K @ x(t)
    E_k = (lqr.R + lqr.B.T @ P_k @ lqr.B) @ K_t - lqr.B.T @ P_k @ lqr.A

    grad = 2 * E_k @ sigma_k

    return grad

def soft_thresholding(K: np.ndarray, threshold: float) -> np.ndarray:
    """Shrink each entry towards zero by ``threshold``, clamping at zero.

    Parameters
    ----------
    K : np.ndarray
        Values to shrink.
    threshold : float
        Amount to subtract from each magnitude.

    Returns
    -------
    np.ndarray
        ``sign(K) * max(|K| - threshold, 0)``.
    """
    return np.sign(K) * np.maximum(np.abs(K) - threshold, 0.0)

# Nuclear norm for low-rank matrices
def prox_grad_nuclear_norm(K: np.ndarray, grad: np.ndarray, stepsize: float, lambda_: float) -> np.ndarray:
    """One proximal gradient step with the nuclear norm.

    Takes an ordinary gradient step, then applies the proximal operator of the
    nuclear norm: soft-threshold the singular values and rebuild.

    Parameters
    ----------
    K : np.ndarray
        Current gain.
    grad : np.ndarray
        Gradient of the LQR cost at ``K``.
    stepsize : float
        Gradient step size.
    lambda_ : float
        Nuclear-norm weight.

    Returns
    -------
    np.ndarray
        The updated gain, of the same shape.

    Notes
    -----
    The threshold is ``lambda_ * stepsize``, fixed by the paper.
    """

    # Gradient step
    G = K - stepsize * grad

    # Proximal step
    U, sigma, V_h = np.linalg.svd(G, full_matrices=False)

    sigma_thresholded = soft_thresholding(sigma, lambda_ * stepsize)

    K_plus = U @ np.diag(sigma_thresholded) @ V_h

    return K_plus

def linesearch_criterion_satisfied(
        lqr: LQRDiscrete,
        K: np.ndarray,
        K_plus: np.ndarray,
        f_K: float,
        f_K_plus: float,
        grad_K: np.ndarray,
        stepsize: float
    ) -> bool:
    """Decide whether a step is acceptable.

    Two conditions must hold: the candidate must keep the closed loop stable,
    and it must decrease the cost by enough to justify its size.

    Parameters
    ----------
    lqr : LQRDiscrete
        System, used for the closed-loop stability check.
    K : np.ndarray
        Current gain.
    K_plus : np.ndarray
        Candidate gain.
    f_K : float
        Cost at ``K``.
    f_K_plus : float
        Cost at ``K_plus``.
    grad_K : np.ndarray
        Gradient at ``K``.
    stepsize : float
        Step size that produced the candidate.

    Returns
    -------
    bool
        ``True`` if the candidate is accepted.
    """

    # Check stability (with minus, due to our convention for u(t) = - K @ x(t))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(lqr.A - lqr.B @ lqr.effective_gain(K_plus))))

    if spectral_radius >= 1:
        return False

    # Sufficient decrease condition
    G_K = 1/stepsize * (K - K_plus)

    part_1 = f_K - stepsize * np.linalg.trace(grad_K.T @ G_K)

    part_2 = stepsize/2 * (np.linalg.norm(G_K, 'fro') ** 2)

    return f_K_plus <= part_1 + part_2

def calculate_next_K(
        lqr: LQRDiscrete,
        K_t: np.ndarray,
        sigma_0: np.ndarray,
        settings: StructuredPolicySettings
    ) -> np.ndarray:
    """One model-based S-PI iteration.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over.
    K_t : np.ndarray
        Current gain.
    sigma_0 : np.ndarray
        Initial-state covariance.
    settings : StructuredPolicySettings
        Hyperparameters.

    Returns
    -------
    np.ndarray
        The next gain.
    """

    P_k, sigma_k = solve_lyapunov(lqr, K_t, sigma_0=sigma_0)

    grad_K = compute_gradient(lqr, K_t, P_k, sigma_k)

    # Paper suggests O(1/lambda) for initial stepsize
    stepsize = 1 / settings.lambda_ * 0.5 # TODO: better here

    f_K = np.linalg.trace(sigma_0 @ P_k)

    linsearch_iter = 0
    while linsearch_iter < settings.max_linesearch_iter:
        K_plus = prox_grad_nuclear_norm(K_t, grad_K, stepsize, lambda_=settings.lambda_)

        P_plus, _ = solve_lyapunov(lqr, K_plus, sigma_0=sigma_0)
        f_K_plus = np.linalg.trace(sigma_0 @ P_plus)

        if linesearch_criterion_satisfied(lqr, K_t, K_plus, f_K, f_K_plus, grad_K, stepsize):
            break

        # TODO: Check if is correct to descrease AFTER first iteration, or if should be before
        linsearch_iter += 1
        stepsize *= settings.beta

    return K_plus


def structured_policy_iteration(lqr: LQRDiscrete, settings: StructuredPolicySettings) -> tuple[list[np.ndarray], list[float]]:
    """Run model-based S-PI to convergence or to the iteration cap.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over.
    settings : StructuredPolicySettings
        Hyperparameters.

    Returns
    -------
    ks : list[np.ndarray]
        Every iterate including ``K_0``.
    costs : list[float]
        Currently always empty, will either be added or removed in the future (TODO).
    """
    K_t = settings.K_0

    ks = [K_t]
    # TODO: Costs
    costs = []

    # Initial state covariance
    sigma_0 = np.eye(lqr.A.shape[0])

    i = 0

    while i < settings.limit:
        i += 1
        print(f"Iteration {i}")
        K_plus = calculate_next_K(lqr, K_t, sigma_0, settings)

        ks.append(K_plus)

        if np.linalg.norm(K_plus - K_t, ord='fro') < settings.epsilon:
            print(f"Converged after {i} iterations")
            break

        K_t = K_plus

    print(f"Ended after {i} iterations")

    return ks, costs


@dataclass
class MFStructuredPolicySettings:
    """Hyperparameters for :func:`structured_policy_iteration_mf`.

    Extends :class:`StructuredPolicySettings` with the rollout parameters of
    :class:`~lqr_solvers.mf_policy_gradient.ModelFreePolicyGradientSettings`.

    Attributes
    ----------
    K_0 : np.ndarray
        Initial gain; must be stabilizing.
    beta : float
        Backtracking factor.
    lambda_ : float
        Nuclear-norm weight.
    epsilon : float
        Frobenius convergence tolerance.
    limit : int
        Iteration cap.
    max_linesearch_iter : int
        Backtracking cap per iteration.
    m : int
        Trajectories per gradient estimate.
    l : int
        Steps per rollout.
    r : float
        Perturbation radius.
    stepsize : float, optional
        Fixed step size, disables backtracking if given.
    """

    K_0: np.ndarray  # Initial policy
    beta: float # linesearch parameter
    lambda_: float # regularization parameter for low-rank structure
    epsilon: float # stopping criterion for convergence
    limit: int # maximum number of iterations to prevent infinite loop in case of non-convergence
    max_linesearch_iter: int
    m: int  # Number of trajectories
    l: int  # Length of steps in each trajectory
    r: float  # Smoothing parameter
    stepsize: float = None  # Fixed stepsize disables linesearch

def calculate_next_K_mf(
        lqr: LQRDiscrete,
        K_t: np.ndarray,
        f_K: float,
        grad_K: np.ndarray,
        settings: MFStructuredPolicySettings
    ) -> np.ndarray:
    """One model-free S-PI iteration.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to sample from.
    K_t : np.ndarray
        Current gain.
    f_K : float
        Estimated cost at ``K_t``.
    grad_K : np.ndarray
        Estimated gradient at ``K_t``.
    settings : MFStructuredPolicySettings
        Hyperparameters.

    Returns
    -------
    np.ndarray
        The next gain.
    """

    # Paper suggests O(1/lambda) for initial stepsize
    step = 1.0 / settings.lambda_

    linesearch_enabled = True
    if settings.stepsize is not None:
        linesearch_enabled = False
        step = settings.stepsize

    linsearch_iter = 0
    while linsearch_iter < settings.max_linesearch_iter:
        K_plus = prox_grad_nuclear_norm(K_t, grad_K, step, lambda_=settings.lambda_)

        if not linesearch_enabled:
            return K_plus

        _, _, f_K_plus = estimate_gradient(lqr, K_plus, m=settings.m, l=settings.l, r=settings.r)
        #print(f"Line search step: {step}, f_K: {f_K}, f_K_plus: {f_K_plus}")

        # Sufficient decrease condition via estimated cost
        G_K = 1/step * (K_t - K_plus)

        part_1 = f_K - step * np.linalg.trace(grad_K.T @ G_K)

        part_2 = step/2 * (np.linalg.norm(G_K, 'fro') ** 2)

        #print(f"Line search criterion: f_K_plus <= {part_1 + part_2} ?")
        if f_K_plus <= part_1 + part_2:
            break

        # TODO: Check if is correct to descrease AFTER first iteration, or if should be before
        linsearch_iter += 1
        step *= settings.beta
        #print(f"Line search step decreased to {step}")

    return K_plus

def structured_policy_iteration_mf(lqr: LQRDiscrete, settings: MFStructuredPolicySettings) -> tuple[list[np.ndarray], list[float]]:
    """Run model-free S-PI to convergence or to the iteration cap.

    Parameters
    ----------
    lqr : LQRDiscrete
        System to optimize over.
    settings : MFStructuredPolicySettings
        Hyperparameters.

    Returns
    -------
    ks : list[np.ndarray]
        Every iterate including ``K_0``.
    costs : list[float]
        Rollout cost estimate per iteration.
    """

    K_t = settings.K_0

    ks = [K_t]
    costs = []

    i = 0

    while i < settings.limit:
        i += 1
        print(f"Iteration {i}")
        grad_K, _, f_K = estimate_gradient(lqr, K_t, settings.m, settings.l, settings.r)

        K_plus = calculate_next_K_mf(lqr, K_t, grad_K, f_K, settings)

        ks.append(K_plus)
        costs.append(f_K)
        #print(f"Iteration {i}, cost estimate: {mean_cost}")

        if np.linalg.norm(K_plus - K_t, ord='fro') < settings.epsilon:
            print(f"Converged after {i} iterations")
            return ks, costs

        K_t = K_plus

    print(f"Ended after {i} iterations")
    return ks, costs


if __name__ == "__main__":
    # Double Integrator example implementation

    # x' = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    # Position, velocity
    max_initial_state = np.array([[10], [11]])

    model = StateSpaceModel(A, B)

    # Write identity matrices out, to easily modify them

    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    settings = LQRDiscreteSettings(
        model=model,
        Q=Q,
        R=R,
        noise_settings=None,
        resample_initial_state=False,
        max_initial_state=max_initial_state,
        dt=0.1,
    )

    lqr = LQRDiscrete(settings)
    lqr.reset()

    K_0 = np.array([[0.1, 0.5]])

    settings = MFStructuredPolicySettings(
        K_0=K_0,
        beta=0.5,
        lambda_=3000,
        epsilon=1e-3,
        limit=1000,
        m=100,
        l=200,
        r=0.5,
        max_linesearch_iter=10,
        stepsize=3.7e-5
    )

    base_K, base_cost = baseline_solution(lqr)
    ks, costs = structured_policy_iteration_mf(lqr=lqr, settings=settings)

    cost_dict = {
        "Structured Policy Iteration (Model-Free)": [true_cost(lqr, K) for K in ks],
    }

    lqr.reset()

    simulation_dict = {
        #"Baseline LQR": simulate(lqr=lqr, K=base_K, n_steps=2500),
        "Structured Policy Iteration (Model-Free)": simulate(lqr=lqr, K=ks[-1], n_steps=2500),
    }


    visualize_multiple_results(simulation_dict, cost_dict, base_cost, dt=lqr.dt)
