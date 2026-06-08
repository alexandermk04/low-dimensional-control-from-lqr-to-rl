from dataclasses import dataclass
import numpy as np

from scipy.linalg import solve_discrete_lyapunov

from basetypes import StateSpaceModel
from visualization import visualize_result

from simple_lqr import LQRDiscrete, simulate

@dataclass
class ModelBasedPolicyGradientSettings:
    K_0: np.ndarray  # Initial policy
    gradient_steps: int  # Number of gradient steps
    gradient_lr: float  # Gradient step size

# Global Convergence... Paper, page 5
def calculate_gradient_model_based_discrete(lqr: LQRDiscrete, K_t: np.ndarray) -> np.ndarray:

    lya_a = lqr.A - lqr.B @ K_t

    lya_q = lqr.Q + K_t.T @ lqr.R @ K_t
    
    P_k = solve_discrete_lyapunov(lya_a.T, lya_q)

    # Initial state covariance
    sigma_0 = np.eye(lqr.A.shape[0])

    # Expected cost using current policy K_t
    expected_cost = lqr.x_0.T @ P_k @ lqr.x_0 # SHOW: np.trace(P_k @ sigma_0)

    # Compute the state covariance under the current policy K_t
    sigma_k = solve_discrete_lyapunov(lya_a, sigma_0)

    E_k = (lqr.R + lqr.B.T @ P_k @ lqr.B) @ K_t - lqr.B.T @ P_k @ lqr.A

    grad = 2 * E_k @ sigma_k

    return grad, expected_cost

# Runs gradient descent for a defined number of steps, returns K after
def gradient_descent_discrete(lqr: LQRDiscrete, settings: ModelBasedPolicyGradientSettings) -> tuple[list[np.ndarray], list[float]]:
    K = settings.K_0

    costs = []
    ks = [K]

    for _ in range(settings.gradient_steps):
        grad, expected_cost = calculate_gradient_model_based_discrete(lqr, K)
        costs.append(expected_cost)

        K = K - settings.gradient_lr * grad
        ks.append(K)

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

    settings = ModelBasedPolicyGradientSettings(
        K_0=K_0,
        gradient_steps=100,
        gradient_lr=0.001
    )

    ks, costs = gradient_descent_discrete(lqr=lqr, settings=settings)

    result = simulate(lqr=lqr, K=ks[-1], n_steps=1000)

    visualize_result(result, dt=lqr.dt)