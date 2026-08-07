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

from evaluation.policy_gradient_evaluation import visualize_Ks

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
    K_0: np.ndarray  # Initial policy
    beta: float # linesearch parameter
    lambda_: float # regularization parameter for low-rank structure
    epsilon: float # stopping criterion for convergence
    limit: int # maximum number of iterations to prevent infinite loop in case of non-convergence
    max_linesearch_iter: int 

def solve_lyapunov(lqr: LQRDiscrete, K_t: np.ndarray, sigma_0: np.ndarray) -> np.ndarray:
    
    K_eff = lqr.effective_gain(K_t)

    lya_a = lqr.A - lqr.B @ K_eff

    lya_q = lqr.Q + K_eff.T @ lqr.R @ K_eff
    
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
    spectral_radius = np.max(np.abs(np.linalg.eigvals(lqr.A - lqr.B @ lqr.effective_gain(K_plus))))

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
        settings: StructuredPolicySettings
    ) -> np.ndarray:

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
        K_plus = caculate_next_K(lqr, K_t, sigma_0, settings)

        ks.append(K_plus)

        if np.linalg.norm(K_plus - K_t, ord='fro') < settings.epsilon:
            print(f"Converged after {i} iterations")
            break

        K_t = K_plus

    print(f"Ended after {i} iterations")

    return ks, costs


@dataclass
class MFStructuredPolicySettings:
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

def caculate_next_K_mf(
        lqr: LQRDiscrete,
        K_t: np.ndarray,
        grad_K: np.ndarray,
        settings: MFStructuredPolicySettings
    ) -> np.ndarray:

    # Paper suggests O(1/lambda) for initial stepsize
    step = 1.0 / settings.lambda_

    linesearch_enabled = True
    if settings.stepsize is not None:
        linesearch_enabled = False
        step = settings.stepsize

    _, _, f_K = estimate_gradient(lqr, K_t, settings.m, settings.l, settings.r)

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

    K_t = settings.K_0

    ks = [K_t]
    costs = []

    i = 0

    while i < settings.limit:
        i += 1
        print(f"Iteration {i}")
        grad_K, _, mean_cost = estimate_gradient(lqr, K_t, settings.m, settings.l, settings.r)

        K_plus = caculate_next_K_mf(lqr, K_t, grad_K, settings)

        ks.append(K_plus)
        costs.append(mean_cost)
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

