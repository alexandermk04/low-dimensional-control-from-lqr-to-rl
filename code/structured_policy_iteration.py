import numpy as np
from dataclasses import dataclass

from scipy.linalg import solve_discrete_lyapunov

from simple_lqr import (
    StateSpaceModel,
    LQRDiscrete,
    simulate,
    get_optimal_K_and_cost_discrete as baseline_solution,
)

from model_free_policy_gradient import (
    estimate_gradient
)

from visualization import visualize_result

from policy_evaluation import visualize_Ks


# All implementation adapted from Structured Policy Iteration paper

@dataclass
class StructuredPolicySettings:
    K_0: np.ndarray  # Initial policy
    beta: float # linesearch parameter
    lambda_: float # regularization parameter for low-rank structure
    epsilon: float # stopping criterion for convergence

def solve_lyapunov(lqr: LQRDiscrete, K_t: np.ndarray, sigma_0: np.ndarray) -> np.ndarray:
    
    lya_a = lqr.A - lqr.B @ K_t

    lya_q = lqr.Q + K_t.T @ lqr.R @ K_t
    
    P_k = solve_discrete_lyapunov(lya_a.T, lya_q)

    sigma_k = solve_discrete_lyapunov(lya_a, sigma_0)

    return P_k, sigma_k

def compute_gradient(lqr: LQRDiscrete, K_t: np.ndarray, P_k: np.ndarray, sigma_k: np.ndarray) -> np.ndarray:

    # Minus, as we use u(t) = - K @ x(t)
    E_k = (lqr.R + lqr.B.T @ P_k @ lqr.B) @ K_t - lqr.B.T @ P_k @ lqr.A

    grad = 2 * E_k @ sigma_k

    return grad

def soft_thresholding(K: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(K) * np.maximum(np.abs(K) - threshold, 0.0)

# Nuclear norm for low-rank matrices
def prox_grad_nuclear_norm(K: np.ndarray, grad: np.ndarray, stepsize: float, lambda_: float) -> np.ndarray:
    
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

    # Check stability (with minus, due to our convention for u(t) = - K @ x(t))
    spectral_radius = np.max(np.abs(np.linalg.eigvals(lqr.A - lqr.B @ K_plus)))

    if spectral_radius >= 1:
        return False

    # Sufficient decrease condition
    G_K = 1/stepsize * (K - K_plus)

    part_1 = f_K - stepsize * np.linalg.trace(grad_K.T @ G_K)

    part_2 = stepsize/2 * (np.linalg.norm(G_K, 'fro') ** 2)

    return f_K_plus <= part_1 + part_2

def caculate_next_K(
        lqr: LQRDiscrete,
        K_t: np.ndarray,
        sigma_0: np.ndarray,
        beta: float,
        lambda_: float,
    ) -> np.ndarray:

    P_k, sigma_k = solve_lyapunov(lqr, K_t, sigma_0=sigma_0)

    grad_K = compute_gradient(lqr, K_t, P_k, sigma_k)

    # Paper suggests O(1/lambda) for initial stepsize
    stepsize = 1 / lambda_

    f_K = np.linalg.trace(sigma_0 @ P_k)

    while True:
        K_plus = prox_grad_nuclear_norm(K_t, grad_K, stepsize, lambda_=lambda_)

        P_plus, _ = solve_lyapunov(lqr, K_plus, sigma_0=sigma_0)
        f_K_plus = np.linalg.trace(sigma_0 @ P_plus)

        if linesearch_criterion_satisfied(lqr, K_t, K_plus, f_K, f_K_plus, grad_K, stepsize):
            break
        
        # TODO: Check if is correct to descrease AFTER first iteration, or if should be before
        stepsize *= beta

    return K_plus


def structured_policy_iteration(lqr: LQRDiscrete, settings: StructuredPolicySettings) -> tuple[list[np.ndarray], list[float]]:
    K_t = settings.K_0

    ks = [K_t]
    # TODO: Costs
    costs = []
    
    # Initial state covariance
    sigma_0 = np.eye(lqr.A.shape[0])

    i = 0

    while True:
        i += 1
        K_plus = caculate_next_K(lqr, K_t, sigma_0, settings.beta, settings.lambda_)

        ks.append(K_plus)

        if np.linalg.norm(K_plus - K_t, ord='fro') < settings.epsilon:
            break

        K_t = K_plus

    print(f"Converged after {i} iterations")
    return ks, costs


@dataclass
class MFStructuredPolicySettings:
    K_0: np.ndarray  # Initial policy
    beta: float # linesearch parameter
    lambda_: float # regularization parameter for low-rank structure
    stepsize: float # fixed stepsize as mentioned in paper
    epsilon: float # stopping criterion for convergence
    m: int  # Number of trajectories
    l: int  # Length of steps in each trajectory
    r: float  # Smoothing parameter

def caculate_next_K_mf(
        K_t: np.ndarray,
        grad_K: np.ndarray,
        lambda_: float,
        stepsize: float
    ) -> np.ndarray:

    K_plus = prox_grad_nuclear_norm(K_t, grad_K, stepsize, lambda_=lambda_)

    return K_plus

def structured_policy_iteration_mf(lqr: LQRDiscrete, settings: MFStructuredPolicySettings) -> tuple[list[np.ndarray], list[float]]:
    K_t = settings.K_0

    ks = [K_t]
    # TODO: Costs
    costs = []
    
    # Initial state covariance
    sigma_0 = np.eye(lqr.A.shape[0])

    i = 0

    while True:
        i += 1
        grad_K, _, _ = estimate_gradient(lqr, K_t, settings.m, settings.l, settings.r)

        K_plus = caculate_next_K_mf(K_t, grad_K, settings.lambda_, stepsize=settings.stepsize)

        ks.append(K_plus)

        if np.linalg.norm(K_plus - K_t, ord='fro') < settings.epsilon:
            break

        K_t = K_plus

    print(f"Converged after {i} iterations")
    return ks, costs


if __name__ == "__main__":
    # Double Integrator example implementation

    # x' = Ax + Bu; x = [position, velocity], u = [acceleration]
    # Hence: Ax + Bu = [velocity, acceleration]
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    # Position, velocity
    x_0 = np.array([10, 0])

    model = StateSpaceModel(A, B, x_0)

    # Write identity matrices out, to easily modify them

    # Q[0, 0] penalizes position error, Q[1, 1] penalizes high velocities
    Q = np.array([[1, 0], [0, 1]])

    # Penalizes gigantic control input
    R = np.array([[1]])

    lqr = LQRDiscrete(model, Q, R)

    K_0 = np.array([[0.1, 0.5]])

    settings = MFStructuredPolicySettings(
        K_0=K_0,
        beta=0.5,
        lambda_=0.1,
        stepsize=1e-4,
        epsilon=1e-3, # SHOW: 1e-3 doesn't exactly display convergence
        m=100,
        l=100,
        r=0.1
    )

    ks, costs = structured_policy_iteration_mf(lqr=lqr, settings=settings)

    result = simulate(lqr=lqr, K=ks[-1], n_steps=1000)

    visualize_result(result, dt=lqr.dt)

    # Compare to baseline solution
    baseline_K, baseline_cost = baseline_solution(lqr)
    visualize_Ks(baseline_K, {'Structured Policy Iteration': ks})


