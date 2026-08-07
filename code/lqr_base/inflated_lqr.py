from dataclasses import dataclass
import numpy as np

from basetypes import StepResult

from lqr_base.simple_lqr import (
    continuous_to_discrete,
    StateSpaceModel,
    LQRDiscrete,
    LQRDiscreteSettings,
)

from lqr_base.noise import (
    LQRNoiseSettings,
)

@dataclass
class LQRInflatedSettings(LQRDiscreteSettings):
    U_1: np.ndarray
    U_2: np.ndarray

class LQRInflated(LQRDiscrete):
    dt: float

    def __init__(self, settings: LQRInflatedSettings):
        self.dt = settings.dt

        A_d, B_d = continuous_to_discrete(settings.model.A, settings.model.B, dt=settings.dt)

        self.A = A_d
        self.B = B_d

        self.max_initial_state = settings.max_initial_state
        self.resample_initial_state = settings.resample_initial_state
        self.x_0 = None
        self.initialized = False

        self.Q = settings.Q
        self.R = settings.R
        self.U_1 = settings.U_1
        self.U_2 = settings.U_2
        self.validate_dimensions()

        self.noise = False
        self.setup_noise(settings.noise_settings)

    def validate_dimensions(self):
        assert self.U_1.shape[1] == self.A.shape[0], "U_1 must have the same number of columns as A has rows"
        assert self.U_2.shape[0] == self.B.shape[1], "U_2 must have the same number of rows as B has columns"

    def effective_gain(self, K: np.ndarray) -> np.ndarray:
        return self.U_2 @ K @ self.U_1
    
    def project_gradient(self, grad_K: np.ndarray) -> np.ndarray:
        return self.U_2.T @ grad_K @ self.U_1.T

    def next_state(self, K: np.ndarray) -> StepResult:
        if not self.initialized:
            raise ValueError("LQRInflated must be initialized before calling next_state. Call reset() first.")

        if self.noise:
            x_t = self.x
            y_t = self.U_1 @ x_t

            if self.measurement_noise is not None:
                noise = self.measurement_noise.sample_measurement_noise(y_t)
                y_t = y_t + noise
            v_t = - K @ y_t

            if self.control_noise is not None:
                control_noise = self.control_noise.sample_control_noise(v_t)
                v_t = v_t + control_noise
            
            u_t = self.U_2 @ v_t

            # Parametric noise lives in the state space, not the measured space
            A = self.A
            B = self.B
            if self.parametric_noise is not None:
                noise_A, noise_B = self.parametric_noise.sample_parametric_noise(A, B)
                A = A + noise_A
                B = B + noise_B

            # Use the actual state, with the calculated control, based on the observations
            x_next = A @ x_t + B @ u_t

            # Disturbance noise also lives in the state space, not the measured space
            if self.disturbance_noise is not None:
                disturbance_noise = self.disturbance_noise.sample_disturbance(x_next)
                x_next = x_next + disturbance_noise
            
            self.x = x_next

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t

            return StepResult(y_t, v_t, x_t, u_t, cost)
        
        else:
            x_t = self.x
            y_t = self.U_1 @ x_t
            v_t = - K @ y_t

            u_t = self.U_2 @ v_t

            self.x = self.A @ x_t + self.B @ u_t

            cost = x_t.T @ self.Q @ x_t + u_t.T @ self.R @ u_t
            return StepResult(y_t, v_t, x_t, u_t, cost)