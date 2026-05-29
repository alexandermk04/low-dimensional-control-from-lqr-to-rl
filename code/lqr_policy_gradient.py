import numpy as np

from scipy.linalg import solve_discrete_lyapunov
from simple_lqr import LQRDiscrete, StateSpaceModel, visualize

# Global Convergence... Paper, page 5
def calculate_gradient_model_based_discrete(lqr: LQRDiscrete, K_t: np.ndarray) -> np.ndarray:

    lya_a = lqr.A - lqr.B @ K_t

    lya_q = lqr.Q + K_t.T @ lqr.R @ K_t
    
    P_k = solve_discrete_lyapunov(lya_a.T, lya_q)

    # Initial state covariance
    sigma_0 = np.eye(lqr.A.shape[0])

    # Compute the state covariance under the current policy K_t
    sigma_k = solve_discrete_lyapunov(lya_a, sigma_0)

    E_k = (lqr.R + lqr.B.T @ P_k @ lqr.B) @ K_t - lqr.B.T @ P_k @ lqr.A

    grad = 2 * E_k @ sigma_k

    return grad

# Runs gradient descent for a defined number of steps, returns K after
def gradient_descent_discrete(lqr: LQRDiscrete, K_t: np.ndarray, stepsize: float, steps: int) -> np.ndarray:
    K = K_t

    for _ in range(steps):
        new_K = calculate_gradient_model_based_discrete(lqr, K)
        K = K - stepsize * new_K

    return K

def simulate_gradient_model_based_discrete(lqr: LQRDiscrete, K_0: np.ndarray, n_steps: int, gradient_lr: float, gradient_steps: int):
    x_traj = []
    u_traj = []

    K = gradient_descent_discrete(lqr=lqr, K_t=K_0, stepsize=gradient_lr, steps=gradient_steps)

    for _ in range(n_steps):
        x_t, u_t = lqr.next_state(K=K)
        x_traj.append(x_t)
        u_traj.append(u_t)

    return np.array(x_traj), np.array(u_traj)


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
    gradient_lr = 0.001
    gradient_steps = 10

    x_traj, u_traj = simulate_gradient_model_based_discrete(lqr=lqr, K_0=K_0, n_steps=1000, gradient_lr=gradient_lr, gradient_steps=gradient_steps)

    visualize(x_traj, u_traj, dt=lqr.dt)