from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from scipy.signal import cont2discrete

import matplotlib.pyplot as plt

from basetypes import StateSpaceModel, SimulationResult
from visualization import visualize_result


def continuous_to_discrete(A, B, dt):
    # TODO: Add documentation zoh

    # y=Cx+Du; for LQR we have full state feedback and no feedthrough, so C=I and D=0
    system = (A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1])))

    A_d, B_d, _, _, _ = cont2discrete(system, dt, method='zoh')

    return A_d, B_d

class LQRBase:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    P: np.ndarray
    K: np.ndarray

    x_0: np.ndarray
    x: np.ndarray

    def __init__(self, model: StateSpaceModel, Q, R):
        self.A = model.A
        self.B = model.B

        self.x_0 = model.x_0
        self.x = model.x_0

        self.Q = Q
        self.R = R

    def get_current_state(self):
        return self.x
    
    def reset_state(self):
        self.x = self.x_0

    def next_state(self):
        raise NotImplementedError


class LQRDiscrete(LQRBase):
    dt: float

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    P: np.ndarray
    K: np.ndarray

    x_0: np.ndarray
    x: np.ndarray

    def __init__(self, model: StateSpaceModel, Q, R, dt=0.01):
        self.dt = dt
        
        A_d, B_d = continuous_to_discrete(model.A, model.B, dt=dt)

        self.A = A_d
        self.B = B_d
        self.x_0 = model.x_0
        self.x = model.x_0

        self.Q = Q
        self.R = R
    
    def get_cost(self, x_t, u_t):
        return x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t
    
    def next_state(self, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        x_t = self.x
        u_t = - K @ x_t

        self.x = self.A @ x_t + self.B @ u_t

        return x_t, u_t
    

def get_optimal_K_and_cost_discrete(lqr: LQRDiscrete):

    P = solve_discrete_are(lqr.A, lqr.B, lqr.Q, lqr.R)

    K = np.linalg.inv(lqr.R) @ lqr.B.T @ P

    cost = lqr.x_0.T @ P @ lqr.x_0

    return K, cost

def simulate(lqr: LQRDiscrete, K: np.ndarray, n_steps: int):
    # Reset lqr state to initial state before simulation
    lqr.reset_state()

    x_traj = []
    u_traj = []

    for _ in range(n_steps):
        x_t, u_t = lqr.next_state(K=K)
        x_traj.append(x_t)
        u_traj.append(u_t)

    results = SimulationResult(x_traj=np.array(x_traj), u_traj=np.array(u_traj))

    return results
    

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

    K, cost = get_optimal_K_and_cost_discrete(lqr=lqr)

    results = simulate(lqr=lqr, K=K, n_steps=1000)

    visualize_result(results, dt=lqr.dt)