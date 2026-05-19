import numpy as np
from scipy.linalg import solve_discrete_are

from scipy.signal import cont2discrete

import matplotlib.pyplot as plt

class StateSpaceModel:
    A: np.ndarray
    B: np.ndarray
    x_0: np.ndarray

    def __init__(self, A, B, x_0):
        self.A = A
        self.B = B
        self.x_0 = x_0

def continuous_to_discrete(A, B, dt):

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

    x: np.ndarray

    def __init__(self, model: StateSpaceModel, Q, R):
        self.A = model.A
        self.B = model.B
        self.x = model.x_0

        self.Q = Q
        self.R = R

        self.P = self.calculate_P()
        self.K = self.calculate_K()

    def calculate_P(self):
        raise NotImplementedError
    
    def calculate_K(self):
        raise NotImplementedError
    
    def calculate_cost(self, x_t):
        raise NotImplementedError
    
    def get_optimal_control(self, x_t):
        raise NotImplementedError
    
    def next_step(self):
        raise NotImplementedError
    
    def simulate(self, n_steps):
        x_traj = []
        u_traj = []

        for _ in range(n_steps):
            x_traj.append(self.x)
            u_t = self.get_optimal_control(self.x)
            u_traj.append(u_t)

            self.x = self.A @ self.x + self.B @ u_t

        return np.array(x_traj), np.array(u_traj)
    
    def visualize(self, x_traj, u_traj, dt=1):
        time_sec = np.arange(x_traj.shape[0]) * dt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time_sec, x_traj[:, 0], label='Position')
        plt.plot(time_sec, x_traj[:, 1], label='Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(time_sec, u_traj[:, 0], label='Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Control')
        plt.legend()

        plt.show()


class LQRDiscrete(LQRBase):
    dt: float

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    P: np.ndarray
    K: np.ndarray

    x: np.ndarray

    def __init__(self, model: StateSpaceModel, Q, R, dt=0.01):
        self.dt = dt
        
        A_d, B_d = continuous_to_discrete(model.A, model.B, dt=dt)

        self.A = A_d
        self.B = B_d
        self.x = model.x_0

        self.Q = Q
        self.R = R

        self.P = self.calculate_P()
        self.K = self.calculate_K()

    # Optimal cost V(x(t), t) = x(t)'*P*x(t)
    def calculate_P(self):
        P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        return P

    # Optimal and robust control page 15
    def calculate_K(self):
        K = - np.linalg.inv(self.R) @ self.B.T @ self.P
        return K

    def calculate_cost(self, x_t):
        cost = x_t.T @ self.P @ x_t
        return cost

    def get_optimal_control(self, x_t):
        return self.K @ x_t
    
    def next_step(self): 
        u_t = self.get_optimal_control(self.x)
        self.x = self.A @ self.x + self.B @ u_t

        return self.x, u_t
    

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

    x_traj, u_traj = lqr.simulate(n_steps=1000)

    lqr.visualize(x_traj, u_traj, dt=lqr.dt)