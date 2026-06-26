import numpy as np
from dataclasses import dataclass


class ParametricNoiseSettings:
    noise_std: float
    time_invariant: bool

    sampled_A: np.ndarray | None
    sampled_B: np.ndarray | None

    def __init__(self, noise_std: float, time_invariant: bool):
        self.noise_std = noise_std
        self.time_invariant = time_invariant
        self.sampled_A = None
        self.sampled_B = None

    def sample_parametric_noise(self, A: np.ndarray, B: np.ndarray):
        if self.time_invariant and self.sampled_A is not None and self.sampled_B is not None:
            return self.sampled_A, self.sampled_B
        
        noise_A = np.random.normal(0, self.noise_std, size=A.shape)
        noise_B = np.random.normal(0, self.noise_std, size=B.shape)

        if self.time_invariant:
            self.sampled_A = noise_A
            self.sampled_B = noise_B

        return noise_A, noise_B

class ControlNoiseSettings:
    control_noise_std: float

    def __init__(self, noise_std: float):
        self.control_noise_std = noise_std

    def sample_control_noise(self, u: np.ndarray):
        
        control_noise = np.random.normal(0, self.control_noise_std, size=u.shape)
        
        # Scaled noise based on the control input magnitude
        scaled_noise = u * control_noise

        return scaled_noise

class MeasurementNoiseSettings:
    measurement_noise_std: float

    def __init__(self, noise_std: float):
        self.measurement_noise_std = noise_std

    def sample_measurement_noise(self, x: np.ndarray):
        
        measurement_noise = np.random.normal(0, self.measurement_noise_std, size=x.shape)

        return measurement_noise


class DisturbanceNoiseSettings:
    disturbance_noise_std: float
    disturbance_type: str

    const_val: np.ndarray | None
    step_count: int

    def __init__(self, noise_std: float, disturbance_type: str):
        self.disturbance_noise_std = noise_std
        self.disturbance_type = disturbance_type
        self.const_val = None
        self.step_count = 0

    def sample_disturbance(self, x: np.ndarray):
        if self.disturbance_type == "gaussian":
            disturbance = np.random.normal(0, self.disturbance_noise_std, size=x.shape)
            return disturbance
        elif self.disturbance_type == "sinusoidal":
            disturbance = self.disturbance_noise_std * np.sin(0.1 * self.step_count) * np.ones_like(x)
            self.step_count += 1
            return disturbance
        elif self.disturbance_type == "constant":
            if self.const_val is None:
                self.const_val = np.random.normal(0, self.disturbance_noise_std, size=x.shape)
            return self.const_val
        else:
            raise ValueError(f"Unknown disturbance type: {self.disturbance_type}")


@dataclass
class LQRNoiseSettings:
    parametric_noise: ParametricNoiseSettings | None
    control_noise: ControlNoiseSettings | None
    measurement_noise: MeasurementNoiseSettings | None
    disturbance_noise: DisturbanceNoiseSettings | None

    def __init__(
        self, 
        parametric_noise: ParametricNoiseSettings | None = None, 
        control_noise: ControlNoiseSettings | None = None, 
        measurement_noise: MeasurementNoiseSettings | None = None, 
        disturbance_noise: DisturbanceNoiseSettings | None = None
    ):
        self.parametric_noise = parametric_noise
        self.control_noise = control_noise
        self.measurement_noise = measurement_noise
        self.disturbance_noise = disturbance_noise