import numpy as np
from dataclasses import dataclass

from lqr_base.simple_lqr import (
    LQRDiscrete
)

@dataclass
class ModelFreePolicyGradientSettings:
    K_0: np.ndarray  # Initial policy
    m: int  # Number of trajectories
    l: int  # Length of steps in each trajectory
    r: float  # Smoothing parameter
    gradient_steps: int  # Number of gradient steps
    gradient_lr: float  # Gradient step size

DIVERGENCE_THRESHOLD = 1e5
DIVERGENCE_PENALTY = 1e10


def run_trajectory(lqr: LQRDiscrete, K: np.ndarray, l: int) -> tuple[np.float64, np.float64]:
    y_traj = []
    v_traj = []
    costs = []
    sigma = np.zeros((K.shape[1], K.shape[1]))

    total_cost = 0.0

    for _ in range(l):
        res = lqr.next_state(K)

        diverged = (
            not np.isfinite(res.cost) 
            or not np.all(np.isfinite(res.y_t))
            or np.linalg.norm(res.y_t) > DIVERGENCE_THRESHOLD
        )

        if diverged:
            total_cost = sum(costs) + DIVERGENCE_PENALTY
            return total_cost, sigma

        y_traj.append(res.y_t)
        v_traj.append(res.v_t)
        costs.append(res.cost)
        sigma += np.outer(res.y_t, res.y_t)

    return sum(costs), sigma

def estimate_gradient(lqr: LQRDiscrete, K: np.ndarray, m: int, l: int, r: float) -> np.ndarray:
    pertubations = []
    cost_estimates = []
    sigma_estimates = []

    d = K.shape[0] * K.shape[1]

    for _ in range(m):

        # Reset the state of the LQR system to the initial state before each trajectory
        lqr.reset()
        lqr.apply_noise()

        # Sample a random matrix with K's shape
        U = np.random.randn(*K.shape)
    
        # Normalize to have unit frobenius norm
        U /= np.linalg.norm(U, ord='fro')  
        U *= r

        pertubations.append(U)

        # SHOW: Two point estimator
        K_i_plus = K + U
        K_i_minus = K - U

        # Run a trajectory with K_i_plus policy
        rng_state = np.random.get_state()
        cost_estimate_plus, sigma_estimate_plus = run_trajectory(lqr, K_i_plus, l)

        # Run a trajectory with K_i_minus policy
        np.random.set_state(rng_state)
        cost_estimate_minus, sigma_estimate_minus = run_trajectory(lqr, K_i_minus, l)

        # TODO: Check validity
        overall_cost_estimate = (cost_estimate_plus + cost_estimate_minus)/2
        cost_estimates.append(overall_cost_estimate)
        sigma_estimates.append(sigma_estimate_plus + sigma_estimate_minus)

    # Centering the cost estimates by subtracting the mean to reduce variance
    mean_cost = np.mean(cost_estimates)
    costs = cost_estimates - mean_cost

    cost_gradient_estimate = sum((d / (r ** 2)) * C * U for C, U in zip(costs, pertubations))
    cost_gradient_estimate /= m

    sigma_estimate = sum(sig for sig in sigma_estimates) / m

    return cost_gradient_estimate, sigma_estimate, mean_cost

def model_free_policy_gradient(lqr: LQRDiscrete, settings: ModelFreePolicyGradientSettings):
    K = settings.K_0

    costs = []
    ks = [K]

    for _ in range(settings.gradient_steps):
        cost_gradient_estimate, sigma_estimate, mean_cost = estimate_gradient(lqr, K, settings.m, settings.l, settings.r)

        # By section D.3, above lemma 29, the expected cost of the pertubed policy is used for the cost estimate
        costs.append(mean_cost)

        K = K - settings.gradient_lr * cost_gradient_estimate

        ks.append(K)

    return ks, costs

def model_free_natural_policy_gradient(lqr: LQRDiscrete, settings: ModelFreePolicyGradientSettings):
    K = settings.K_0

    costs = []
    ks = [K]

    for _ in range(settings.gradient_steps):
        cost_gradient_estimate, sigma_estimate, mean_cost = estimate_gradient(lqr, K, settings.m, settings.l, settings.r)

        # By section D.3, above lemma 29, the expected cost of the pertubed policy is used for the cost estimate
        costs.append(mean_cost)

        # SHOW: Use a pseudoinverse with truncation to remain stable for inflated inputs
        K = K - settings.gradient_lr * (cost_gradient_estimate @ np.linalg.pinv(sigma_estimate, rcond=1e-6))

        ks.append(K)

    return ks, costs