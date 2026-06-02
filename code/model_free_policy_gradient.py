import numpy as np
from dataclasses import dataclass

from basetypes import StateSpaceModel, SimulationResultPlotType
from visualization import visualize_result, visualize_result_single

from simple_lqr import StateSpaceModel, LQRDiscrete, simulate

@dataclass
class ModelFreePolicyGradientSettings:
    K_0: np.ndarray  # Initial policy
    m: int  # Number of trajectories
    l: int  # Length of steps in each trajectory
    r: float  # Smoothing parameter
    gradient_steps: int  # Number of gradient steps
    gradient_lr: float  # Gradient step size


def run_trajectory(lqr: LQRDiscrete, K: np.ndarray, l: int) -> tuple[np.float64, np.float64]:
    x_traj = []
    u_traj = []

    total_cost = 0.0

    for _ in range(l):
        x_t, u_t = lqr.next_state(K)

        # TODO: Determine if this threshold is sufficient
        if np.linalg.norm(x_t) > 1e6:
            # If the system becomes unstable, add high cost and terminate the trajectory early
            print("Trajectory diverged, terminating early.")
            total_cost += 1e10
            break

        x_traj.append(x_t)
        u_traj.append(u_t)

    total_cost = sum(lqr.get_cost(x_t, u_t) for x_t, u_t in zip(x_traj, u_traj))

    sigma = sum(np.outer(x_t, x_t) for x_t in x_traj)

    return total_cost, sigma

def estimate_gradient(lqr: LQRDiscrete, K: np.ndarray, m: int, l: int, r: float, d: int) -> np.ndarray:
    pertubations = []
    cost_estimates = []
    sigma_estimates = []

    for _ in range(m):

        # Reset the state of the LQR system to the initial state before each trajectory
        lqr.reset_state()

        # Sample a random matrix with K's shape
        U = np.random.randn(*K.shape)
    
        # Normalize to have unit frobenius norm
        U /= np.linalg.norm(U, ord='fro')  
        U *= r

        pertubations.append(U)

        K_i = K + U

        # Run a trajectory with K_i policy
        cost_estimate, sigma_estimate = run_trajectory(lqr, K_i, l)

        cost_estimates.append(cost_estimate)
        sigma_estimates.append(sigma_estimate)

    # Centering the cost estimates by subtracting the mean to reduce variance
    mean_cost = np.mean(cost_estimates)
    centered_costs = cost_estimates - mean_cost

    cost_gradient_estimate = sum((d / (r ** 2)) * C * U for C, U in zip(centered_costs, pertubations))
    cost_gradient_estimate /= m

    sigma_estimate = sum(sig for sig in sigma_estimates) / m

    return cost_gradient_estimate, sigma_estimate, mean_cost

def model_free_policy_gradient(lqr: LQRDiscrete, settings: ModelFreePolicyGradientSettings):
    K = settings.K_0

    costs = []
    ks = []

    # TODO: Verify correctness
    d = K.shape[0] * K.shape[1]

    for _ in range(settings.gradient_steps):
        cost_gradient_estimate, sigma_estimate, mean_cost = estimate_gradient(lqr, K, settings.m, settings.l, settings.r, d)

        K = K - settings.gradient_lr * cost_gradient_estimate

        # By section D.3, above lemma 29, the expected cost of the pertubed policy is used for the cost estimate
        costs.append(mean_cost)
        ks.append(K)

    return ks, costs

def model_free_natural_policy_gradient(lqr: LQRDiscrete, settings: ModelFreePolicyGradientSettings):
    K = settings.K_0

    costs = []
    ks = []

    # TODO: Verify correctness
    d = K.shape[0] * K.shape[1]

    for _ in range(settings.gradient_steps):
        cost_gradient_estimate, sigma_estimate, mean_cost = estimate_gradient(lqr, K, settings.m, settings.l, settings.r, d)

        K = K - settings.gradient_lr * (cost_gradient_estimate @ np.linalg.inv(sigma_estimate))

        # By section D.3, above lemma 29, the expected cost of the pertubed policy is used for the cost estimate
        costs.append(mean_cost)
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

    settings = ModelFreePolicyGradientSettings(
        K_0=K_0, 
        m=40,  # Number of trajectories
        l=100,  # Length of steps in each trajectory
        r=0.1,  # Smoothing parameter
        gradient_steps=100,
        gradient_lr=1e-4
    )

    ks, costs = model_free_natural_policy_gradient(lqr=lqr, settings=settings)

    result = simulate(lqr=lqr, K=ks[-1], n_steps=2500)

    visualize_result(result, dt=lqr.dt)
    visualize_result_single(result, dt=lqr.dt, type=SimulationResultPlotType.POSITION)