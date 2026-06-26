from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

from scipy.signal import cont2discrete

import matplotlib.pyplot as plt

from basetypes import StateSpaceModel, SimulationResult, StepResult
from visualization import visualize_result
from lqr_base.noise import (
    ParametricNoiseSettings,
    ControlNoiseSettings,
    MeasurementNoiseSettings,
    DisturbanceNoiseSettings,
    LQRNoiseSettings
)

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

    noise: bool
    parametric_noise: ParametricNoiseSettings | None
    control_noise: ControlNoiseSettings | None
    measurement_noise: MeasurementNoiseSettings | None
    disturbance_noise: DisturbanceNoiseSettings | None

    def __init__(self, model: StateSpaceModel, Q, R, noise_settings: LQRNoiseSettings = None):
        self.A = model.A
        self.B = model.B

        self.x_0 = model.x_0
        self.x = model.x_0

        self.Q = Q
        self.R = R

        self.noise = False
        self.setup_noise(noise_settings)
    
    def reset(self):
        self.x = self.x_0
        self.noise = False

    def setup_noise(self, noise_settings: LQRNoiseSettings | None):
        if noise_settings is None:
            self.parametric_noise = None
            self.control_noise = None
            self.measurement_noise = None
            self.disturbance_noise = None
            return
        self.parametric_noise = noise_settings.parametric_noise
        self.control_noise = noise_settings.control_noise
        self.measurement_noise = noise_settings.measurement_noise
        self.disturbance_noise = noise_settings.disturbance_noise

    def apply_noise(self):
        self.noise = True

    def next_state(self, K: np.ndarray) -> StepResult:
        raise NotImplementedError


class LQRDiscrete(LQRBase):
    dt: float

    def __init__(self, model: StateSpaceModel, Q, R, noise_settings: LQRNoiseSettings = None, dt=0.01):
        self.dt = dt
        
        A_d, B_d = continuous_to_discrete(model.A, model.B, dt=dt)

        self.A = A_d
        self.B = B_d
        self.x_0 = model.x_0
        self.x = model.x_0

        self.Q = Q
        self.R = R

        self.noise = False
        self.setup_noise(noise_settings)

    def effective_gain(self, K: np.ndarray) -> np.ndarray:
        return K
    
    def project_gradient(self, grad_K: np.ndarray) -> np.ndarray:
        return grad_K
    
    def next_state(self, K: np.ndarray) -> StepResult:
        if self.noise:
            x_t = self.x

            # Observation
            y_t = x_t
            if self.measurement_noise is not None:
                noise = self.measurement_noise.sample_measurement_noise(y_t)
                y_t = y_t + noise

            # Control input based on the observation
            u_t = - K @ y_t

            # The actual control input applied
            v_t = u_t
            if self.control_noise is not None:
                control_noise = self.control_noise.sample_control_noise(u_t)
                v_t = v_t + control_noise
            
            # Parametric noise applied to state space
            A = self.A
            B = self.B
            if self.parametric_noise is not None:
                noise_A, noise_B = self.parametric_noise.sample_parametric_noise(A, B)
                A = A + noise_A
                B = B + noise_B

            # Here we use the actual state, with our control
            x_next = A @ x_t + B @ v_t

            # Disturbance noise also lives in the state space, not the measured space
            if self.disturbance_noise is not None:
                disturbance_noise = self.disturbance_noise.sample_disturbance(x_next)
                x_next = x_next + disturbance_noise

            self.x = x_next

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t

            return StepResult(y_t, v_t, x_t, u_t, cost)
        
        else:
            x_t = self.x
            u_t = - K @ x_t

            self.x = self.A @ x_t + self.B @ u_t

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t

            return StepResult(x_t, u_t, x_t, u_t, cost)
    

def get_optimal_K_and_cost_discrete(lqr: LQRDiscrete):

    P = solve_discrete_are(lqr.A, lqr.B, lqr.Q, lqr.R)

    # SHOW: K = np.linalg.inv(lqr.R) @ lqr.B.T @ P

    part_1 = np.linalg.inv(lqr.B.T @ P @ lqr.B + lqr.R)

    # SHOW: Minus was here for both papers, but our convention is u(t) = - K @ x(t), so we don't need the minus here
    K = part_1 @ lqr.B.T @ P @ lqr.A

    cost = lqr.x_0.T @ P @ lqr.x_0

    return K, cost

def true_cost(lqr: LQRDiscrete, K: np.ndarray) -> float:
    
    x_0 = lqr.x_0

    A_cl = lqr.A - lqr.B @ K

    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1:
        return np.inf

    P = solve_discrete_lyapunov(A_cl.T, lqr.Q + K.T @ lqr.R @ K)

    return float(x_0 @ P @ x_0)


def simulate(lqr: LQRDiscrete, K: np.ndarray, n_steps: int):
    # Reset lqr state to initial state before simulation
    lqr.reset()
    lqr.apply_noise()

    x_traj = []
    u_traj = []

    for _ in range(n_steps):
        res = lqr.next_state(K=K)
        
        # Add the ACTUAL observation/actuation
        x_traj.append(res.x_t)
        u_traj.append(res.u_t)

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

    # Noise
    parametric_noise = ParametricNoiseSettings(noise_std=0.05, time_invariant=False)
    control_noise = ControlNoiseSettings(noise_std=0.05)
    measurement_noise = MeasurementNoiseSettings(noise_std=0.01)
    disturbance_noise = DisturbanceNoiseSettings(noise_std=0.05, disturbance_type='gaussian')

    noise_settings = LQRNoiseSettings(
        #parametric_noise=parametric_noise,
        #control_noise=control_noise,
        measurement_noise=measurement_noise,
        #disturbance_noise=disturbance_noise
    )

    lqr = LQRDiscrete(model, Q, R, noise_settings=noise_settings, dt=0.01)

    K, cost = get_optimal_K_and_cost_discrete(lqr=lqr)

    results = simulate(lqr=lqr, K=K, n_steps=2500)

    visualize_result(results, dt=lqr.dt)